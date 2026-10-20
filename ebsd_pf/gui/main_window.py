"""Главное окно: навигация по этапам, файлы проекта, автосохранение и восстановление."""
from __future__ import annotations

import json
import logging
import os
import uuid

from PySide6.QtCore import QSettings, QStandardPaths, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QButtonGroup, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QStackedWidget, QVBoxLayout,
                               QWidget)

from .. import APP_NAME, APP_NAME_RU, ORG_NAME, PROJECT_EXT, __version__
from ..core.imageio_utils import IMAGE_EXTENSIONS, load_image, load_image_b64
from ..core.models import Project, SourceImage, new_id
from ..core.project_io import ProjectError, load_project, project_from_dict, project_to_dict, save_project
from .export_page import ExportPage
from .group_page import GroupPage
from .markup_page import MarkupPage
from .qt_utils import error_box
from .recon_page import ReconPage
from .start_page import StartPage
from .state import AppState

log = logging.getLogger(__name__)

PROJECT_FILTER = f"Проект EBSD (*{PROJECT_EXT})"
OPEN_FILTER = (f"Изображения и проекты (*.jpg *.jpeg *.png *.tif *.tiff *{PROJECT_EXT});;"
               f"Изображения (*.jpg *.jpeg *.png *.tif *.tiff);;{PROJECT_FILTER};;Все файлы (*)")
AUTOSAVE_MS = 2 * 60 * 1000


def app_data_dir() -> str:
    d = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    if not d:
        d = os.path.join(os.path.expanduser("~"), ".ebsd_pf")
    os.makedirs(d, exist_ok=True)
    return d


def load_images_job(entries: list, progress=None, is_cancelled=None) -> list:
    """entries: [(image_id, path, embedded_b64|None)] -> [(image_id, path, array|None, error|None)]"""
    out = []
    for k, (iid, path, emb) in enumerate(entries):
        if is_cancelled and is_cancelled():
            break
        try:
            if path and os.path.isfile(path):
                arr = load_image(path)
            elif emb:
                arr = load_image_b64(emb)
            else:
                raise FileNotFoundError(f"Файл не найден: {path}")
            out.append((iid, path, arr, None))
        except Exception as e:  # noqa: BLE001
            log.exception("Ошибка загрузки %s", path)
            out.append((iid, path, None, str(e)))
        if progress:
            progress(int(100 * (k + 1) / len(entries)))
    return out


class MainWindow(QMainWindow):
    STEPS = ["1. Изображения", "2. Разметка и атрибуты", "3. Группировка", "4. Реконструкция", "5. Экспорт"]

    def __init__(self, auto_start: bool = True):
        super().__init__()
        self.state = AppState(self)
        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.recovery_path = os.path.join(app_data_dir(), "recovery" + PROJECT_EXT)
        self.resize(1500, 920)
        self.setAcceptDrops(True)

        central = QWidget()
        vl = QVBoxLayout(central)
        vl.setContentsMargins(4, 4, 4, 0)
        steps = QHBoxLayout()
        self.step_group = QButtonGroup(self)
        self.step_group.setExclusive(True)
        self.step_buttons = []
        for i, name in enumerate(self.STEPS):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setMinimumHeight(30)
            b.setStyleSheet("QPushButton:checked { background: #2f6fbf; color: white; font-weight: bold; }")
            self.step_group.addButton(b, i)
            steps.addWidget(b)
            self.step_buttons.append(b)
        vl.addLayout(steps)
        self.stack = QStackedWidget()
        vl.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        st = self.state
        self.start_page = StartPage(st)
        self.markup_page = MarkupPage(st)
        self.group_page = GroupPage(st)
        self.recon_page = ReconPage(st)
        self.export_page = ExportPage(st)
        self.pages = [self.start_page, self.markup_page, self.group_page, self.recon_page, self.export_page]
        for p in self.pages:
            self.stack.addWidget(p)
        self.step_group.idClicked.connect(self.goto)

        self.start_page.openImagesRequested.connect(self.open_dialog)
        self.start_page.openProjectRequested.connect(self.open_project_dialog)
        self.start_page.openRecentRequested.connect(self.open_project)
        self.start_page.newProjectRequested.connect(self.new_project)
        self.start_page.removeImageRequested.connect(self.remove_image)
        self.start_page.nextRequested.connect(lambda: self.goto(1))
        self.markup_page.nextRequested.connect(lambda: self.goto(2))
        self.group_page.backRequested.connect(lambda: self.goto(1))
        self.group_page.nextRequested.connect(lambda: self.goto(3))
        self.recon_page.backRequested.connect(lambda: self.goto(2))
        self.recon_page.nextRequested.connect(lambda: self.goto(4))
        self.export_page.backRequested.connect(lambda: self.goto(3))

        # строка состояния
        self.busy_label = QLabel("")
        self.busy_bar = QProgressBar()
        self.busy_bar.setMaximumWidth(220)
        self.busy_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.busy_label)
        self.statusBar().addPermanentWidget(self.busy_bar)
        st.runner.busyChanged.connect(self._on_busy)
        st.runner.progress.connect(lambda name, v: (self.busy_bar.setRange(0, 100), self.busy_bar.setValue(v)))
        st.runner.error_handler = self._task_error
        st.statusMessage.connect(lambda m: self.statusBar().showMessage(m, 10000))
        st.dirtyChanged.connect(lambda _: self._update_title())
        st.projectReplaced.connect(self._update_title)

        self._build_menu()
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(AUTOSAVE_MS)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start()

        geo = self.settings.value("geometry")
        if geo is not None:
            try:
                self.restoreGeometry(geo)
            except Exception:  # noqa: BLE001
                pass
        self.start_page.set_recent(self.recent_projects())
        self.goto(0)
        self._update_title()
        if auto_start:
            QTimer.singleShot(250, self._startup)

    # ------------------------------------------------------------------ меню
    def _build_menu(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&Файл")

        def act(menu, text, slot, shortcut=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            menu.addAction(a)
            return a

        act(m, "Новый проект", self.new_project, QKeySequence.StandardKey.New)
        act(m, "Открыть изображения или проект…", self.open_dialog, QKeySequence.StandardKey.Open)
        act(m, "Открыть проект…", self.open_project_dialog)
        act(m, "Добавить изображения…", self.add_images_dialog)
        m.addSeparator()
        act(m, "Сохранить проект", self.save, QKeySequence.StandardKey.Save)
        act(m, "Сохранить проект как…", self.save_as, QKeySequence.StandardKey.SaveAs)
        self.embed_action = QAction("Встраивать изображения в файл проекта", self)
        self.embed_action.setCheckable(True)
        self.embed_action.toggled.connect(self._toggle_embed)
        m.addAction(self.embed_action)
        m.addSeparator()
        self.recent_menu = m.addMenu("Недавние проекты")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        m.addSeparator()
        act(m, "Выход", self.close, QKeySequence.StandardKey.Quit)

        v = mb.addMenu("&Этапы")
        for i, name in enumerate(self.STEPS):
            act(v, name, lambda checked=False, i=i: self.goto(i), f"Ctrl+{i + 1}")

        h = mb.addMenu("&Справка")
        act(h, "Краткая инструкция", self.show_help, QKeySequence.StandardKey.HelpContents)
        act(h, "Открыть папку журнала", self.open_log_dir)
        act(h, "О программе", self.show_about)

    def _fill_recent_menu(self) -> None:
        self.recent_menu.clear()
        paths = self.recent_projects()
        if not paths:
            a = self.recent_menu.addAction("(пусто)")
            a.setEnabled(False)
        for p in paths:
            a = self.recent_menu.addAction(p)
            a.triggered.connect(lambda checked=False, p=p: self.open_project(p))

    def _toggle_embed(self, on: bool) -> None:
        if self.state.project.embed_images != on:
            self.state.project.embed_images = on
            self.state.mark_dirty()

    # ------------------------------------------------------------------ навигация
    def goto(self, i: int) -> None:
        if i > 0 and not self.state.project.images:
            self.statusBar().showMessage("Сначала загрузите изображения", 5000)
            i = 0
        if i >= 2 and not self.state.project.figures:
            self.statusBar().showMessage("Сначала отметьте окружности полюсных фигур", 5000)
            i = 1
        self.stack.setCurrentIndex(i)
        self.step_buttons[i].setChecked(True)
        page = self.pages[i]
        if hasattr(page, "on_enter"):
            page.on_enter()

    def _update_title(self) -> None:
        mark = "*" if self.state.dirty else ""
        self.setWindowTitle(f"{APP_NAME_RU} — {self.state.project_name()}{mark}")
        self.embed_action.blockSignals(True)
        self.embed_action.setChecked(self.state.project.embed_images)
        self.embed_action.blockSignals(False)

    def _on_busy(self, busy: bool, name: str) -> None:
        self.busy_bar.setVisible(busy)
        self.busy_label.setText(name if busy else "")
        if busy:
            self.busy_bar.setRange(0, 0)

    def _task_error(self, name: str, msg: str, tb: str) -> None:
        error_box(self, "Ошибка", f"{name}: {msg}", tb)

    # ------------------------------------------------------------------ запуск
    def _startup(self) -> None:
        if os.path.isfile(self.recovery_path):
            if self._offer_recovery():
                return
        if not self.state.project.images:
            self.open_dialog()

    def _offer_recovery(self) -> bool:
        try:
            with open(self.recovery_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            self._remove_recovery()
            return False
        orig = data.get("recovery_of")
        when = data.get("saved_at", "")
        text = (f"Обнаружена автосохранённая копия несохранённой работы ({when})"
                + (f"\nпроекта «{os.path.basename(orig)}»" if orig else "") + ".\n\nВосстановить?")
        res = QMessageBox.question(self, "Восстановление", text,
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if res != QMessageBox.StandardButton.Yes:
            self._remove_recovery()
            return False
        try:
            project = project_from_dict(data, orig)
        except ProjectError as e:
            error_box(self, "Восстановление", str(e))
            self._remove_recovery()
            return False
        self._install_project(project, orig if orig and os.path.isfile(orig) else None, dirty=True)
        return True

    # ------------------------------------------------------------------ файлы
    def recent_projects(self) -> list:
        val = self.settings.value("recent", [])
        if isinstance(val, str):
            val = [val]
        return [p for p in (val or []) if os.path.isfile(p)][:10]

    def _add_recent(self, path: str) -> None:
        paths = [os.path.abspath(path)] + [p for p in self.recent_projects() if os.path.abspath(p) != os.path.abspath(path)]
        self.settings.setValue("recent", paths[:10])
        self.start_page.set_recent(paths[:10])

    def _last_dir(self) -> str:
        return str(self.settings.value("last_dir", os.path.expanduser("~")))

    def open_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Открыть изображения или проект", self._last_dir(), OPEN_FILTER)
        if paths:
            self.settings.setValue("last_dir", os.path.dirname(paths[0]))
            self.open_paths(paths)

    def add_images_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Добавить изображения", self._last_dir(),
            "Изображения (*.jpg *.jpeg *.png *.tif *.tiff);;Все файлы (*)")
        if paths:
            self.settings.setValue("last_dir", os.path.dirname(paths[0]))
            self.add_images(paths)

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", self._last_dir(), PROJECT_FILTER)
        if path:
            self.settings.setValue("last_dir", os.path.dirname(path))
            self.open_project(path)

    def open_paths(self, paths: list) -> None:
        projects = [p for p in paths if p.lower().endswith(PROJECT_EXT)]
        images = [p for p in paths if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS]
        other = [p for p in paths if p not in projects and p not in images]
        if other:
            QMessageBox.warning(self, "Открытие", "Неподдерживаемые файлы пропущены:\n" +
                                "\n".join(os.path.basename(p) for p in other))
        if projects:
            self.open_project(projects[0])
            return
        if images:
            self.add_images(images)

    def add_images(self, paths: list) -> None:
        known = {os.path.normcase(os.path.abspath(i.path)) for i in self.state.project.images}
        new = [p for p in paths if os.path.normcase(os.path.abspath(p)) not in known]
        if len(new) < len(paths):
            self.statusBar().showMessage("Уже загруженные изображения пропущены", 5000)
        if not new:
            return
        entries = [(new_id(), os.path.abspath(p), None) for p in new]

        def done(results):
            added, errors = [], []
            for iid, path, arr, err in results:
                if arr is None:
                    errors.append(f"{os.path.basename(path)}: {err}")
                    continue
                self.state.store.set(iid, arr)
                self.state.project.images.append(
                    SourceImage(iid, path, os.path.basename(path), arr.shape[1], arr.shape[0]))
                added.append(iid)
            if errors:
                error_box(self, "Загрузка изображений", "Не удалось загрузить:\n" + "\n".join(errors))
            if added:
                self.state.mark_dirty()
                self.state.imagesChanged.emit()
                self.goto(1)
                self.markup_page.select_image(added[0])
                self.markup_page.detect(added, silent_if_none=True)

        self.state.runner.run("Загрузка изображений", load_images_job, entries, with_progress=True, on_done=done)

    def remove_image(self, image_id: str) -> None:
        pr = self.state.project
        img = pr.image(image_id)
        if img is None:
            return
        n = len(pr.figures_on(image_id))
        if QMessageBox.question(self, "Удаление изображения",
                                f"Удалить изображение «{img.name}» из проекта"
                                + (f" вместе с {n} фигурами" if n else "") + "?"
                                ) != QMessageBox.StandardButton.Yes:
            return
        pr.figures = [f for f in pr.figures if f.image_id != image_id]
        pr.images = [i for i in pr.images if i.id != image_id]
        self.state.store.remove(image_id)
        self.state.pipeline.invalidate()
        self.state.mark_dirty()
        self.state.imagesChanged.emit()
        self.state.figuresChanged.emit()

    def new_project(self) -> None:
        if not self.maybe_save():
            return
        self._install_project(Project(), None)
        self._remove_recovery()

    def open_project(self, path: str) -> None:
        if not self.maybe_save():
            return
        try:
            project = load_project(path)
        except ProjectError as e:
            error_box(self, "Открытие проекта", str(e))
            return
        self._install_project(project, path)
        self._add_recent(path)
        self._remove_recovery()

    def _install_project(self, project: Project, path, dirty: bool = False) -> None:
        self.state.set_project(project, path)
        if dirty:
            self.state.mark_dirty()
        self._update_title()
        page = int(project.ui.get("page", 1)) if project.images else 0
        entries = [(i.id, i.path, i.embedded_png) for i in project.images]
        if not entries:
            self.goto(0)
            return

        def done(results):
            missing = []
            for iid, p, arr, err in results:
                if arr is not None:
                    self.state.store.set(iid, arr)
                    img = project.image(iid)
                    img.width, img.height = arr.shape[1], arr.shape[0]
                else:
                    missing.append(iid)
            for iid in missing:
                self._locate_missing(project.image(iid))
            self.state.imagesChanged.emit()
            self.state.figuresChanged.emit()
            if dirty:
                self.state.mark_dirty()
            self.goto(max(0, min(page, 4)))
            self.statusBar().showMessage(f"Проект загружен: {len(project.figures)} фигур, "
                                         f"{len(project.images)} изображений", 8000)

        self.state.runner.run("Загрузка изображений проекта", load_images_job, entries,
                              with_progress=True, on_done=done)

    def _locate_missing(self, img: SourceImage) -> None:
        res = QMessageBox.question(
            self, "Изображение не найдено",
            f"Исходное изображение «{img.name}» не найдено:\n{img.path}\n\nУказать его расположение?")
        if res != QMessageBox.StandardButton.Yes:
            return
        path, _ = QFileDialog.getOpenFileName(self, f"Где находится «{img.name}»?", self._last_dir(),
                                              "Изображения (*.jpg *.jpeg *.png *.tif *.tiff);;Все файлы (*)")
        if not path:
            return
        try:
            arr = load_image(path)
        except Exception as e:  # noqa: BLE001
            error_box(self, "Загрузка", str(e))
            return
        if img.width and (arr.shape[1], arr.shape[0]) != (img.width, img.height):
            if QMessageBox.question(self, "Размер изображения",
                                    "Размер выбранного изображения отличается от исходного. "
                                    "Координаты окружностей могут не совпасть. Использовать?"
                                    ) != QMessageBox.StandardButton.Yes:
                return
        img.path = path
        img.width, img.height = arr.shape[1], arr.shape[0]
        self.state.store.set(img.id, arr)
        self.state.mark_dirty()

    def save(self) -> bool:
        if not self.state.path:
            return self.save_as()
        return self._save_to(self.state.path)

    def save_as(self) -> bool:
        start = self.state.path or os.path.join(self._last_dir(), "проект" + PROJECT_EXT)
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", start, PROJECT_FILTER)
        if not path:
            return False
        if not path.lower().endswith(PROJECT_EXT):
            path += PROJECT_EXT
        return self._save_to(path)

    def _save_to(self, path: str) -> bool:
        pr = self.state.project
        pr.ui["page"] = self.stack.currentIndex()
        try:
            self.setCursor(Qt.CursorShape.WaitCursor)
            save_project(pr, path, self.state.store)
        except ProjectError as e:
            error_box(self, "Сохранение", str(e))
            return False
        finally:
            self.unsetCursor()
        self.state.path = path
        self.state.mark_dirty(False)
        self._update_title()
        self._add_recent(path)
        self._remove_recovery()
        self.statusBar().showMessage(f"Проект сохранён: {path}", 6000)
        return True

    def maybe_save(self) -> bool:
        if not self.state.dirty:
            return True
        res = QMessageBox.question(
            self, "Несохранённые изменения", "В проекте есть несохранённые изменения. Сохранить?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if res == QMessageBox.StandardButton.Save:
            return self.save()
        return res == QMessageBox.StandardButton.Discard

    def autosave(self) -> None:
        if not self.state.dirty or not self.state.project.images:
            return
        try:
            data = project_to_dict(self.state.project, self.state.path, None)
            data["recovery_of"] = self.state.path
            for d in data["images"]:
                d["embedded_png"] = None
            tmp = self.recovery_path + f".{uuid.uuid4().hex[:6]}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(tmp, self.recovery_path)
            log.info("Автосохранение: %s", self.recovery_path)
        except Exception:  # noqa: BLE001
            log.exception("Ошибка автосохранения")

    def _remove_recovery(self) -> None:
        try:
            if os.path.exists(self.recovery_path):
                os.remove(self.recovery_path)
        except OSError:
            pass

    # ------------------------------------------------------------------ события окна
    def closeEvent(self, event) -> None:
        if self.state.runner.is_busy():
            if QMessageBox.question(self, "Выход", "Выполняется фоновая операция. Всё равно выйти?"
                                    ) != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if not self.maybe_save():
            event.ignore()
            return
        self._remove_recovery()
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.open_paths(paths)

    # ------------------------------------------------------------------ справка
    def open_log_dir(self) -> None:
        from ..app import log_dir
        QDesktopServices.openUrl(QUrl.fromLocalFile(log_dir()))

    def show_about(self) -> None:
        QMessageBox.about(self, "О программе",
                          f"<b>{APP_NAME_RU}</b><br>{APP_NAME} {__version__}<br><br>"
                          "Полуавтоматическая реконструкция полюсных фигур EBSD по растровым изображениям. "
                          "Вся обработка выполняется локально.<br><br>"
                          "Python, PySide6, OpenCV, NumPy, Pillow, scikit-image.")

    def show_help(self) -> None:
        QMessageBox.information(self, "Краткая инструкция", HELP_TEXT)


HELP_TEXT = """<h3>Порядок работы</h3>
<ol>
<li><b>Изображения.</b> Откройте один или несколько файлов JPG/PNG/TIF (можно перетащить в окно).</li>
<li><b>Разметка.</b> Окружности ищутся автоматически. Проверьте их: перетаскивайте, меняйте радиус за
квадратный маркер, удаляйте лишние (Delete), добавляйте пропущенные кнопкой «Добавить окружность»
(щелчок в центре, затем щелчок на окружности). Выделите фигуры и задайте образец, зерно и индекс
плоскости (без скобок: 111 → {111}). Для массового назначения используйте блок «Массовое назначение»
или «Автогруппировку по расположению».</li>
<li><b>Группировка.</b> Проверьте состав зёрен, порядок фигур (↑/↓, перетаскивание), индексы,
расстояние между окружностями.</li>
<li><b>Реконструкция.</b> Выберите режим (точный или повышение читаемости), настройте параметры —
для всех фигур или для отдельной. Сравнивайте с оригиналом; при необходимости правьте маску вручную.
Удалённые фильтром объекты всегда подсвечиваются.</li>
<li><b>Экспорт.</b> Фигура, зерно, образец или все образцы; PNG, TIFF, SVG, PDF; 300/600/1200 DPI.</li>
</ol>
<p>Проект сохраняется в файл .ebsdproj (Ctrl+S). Несохранённая работа автоматически сохраняется
каждые 2 минуты и предлагается к восстановлению после сбоя.</p>
<p><b>Мышь:</b> колесо — масштаб; средняя кнопка или Пробел+перетаскивание — перемещение.</p>"""
