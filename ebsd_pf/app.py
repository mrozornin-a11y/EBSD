"""Запуск приложения: журналирование, обработка исключений, главное окно."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import traceback

from . import APP_NAME, ORG_NAME, __version__

log = logging.getLogger("ebsd_pf")


def log_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "EBSDPoleFigures", "logs")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = os.path.join(os.path.expanduser("~"), ".ebsd_pf_logs")
        os.makedirs(d, exist_ok=True)
    return d


def setup_logging(level=logging.INFO) -> str:
    path = os.path.join(log_dir(), "ebsd_pf.log")
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s")
    try:
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass
    if sys.stderr is not None:   # в EXE без консоли stderr отсутствует
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.WARNING)
        root.addHandler(sh)
    return path


def _excepthook(exc_type, exc, tb):
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    log.critical("Необработанное исключение:\n%s", text)
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is not None:
            box = QMessageBox(QMessageBox.Icon.Critical, "Непредвиденная ошибка",
                              f"Произошла ошибка: {exc}\n\nРабота может быть продолжена; рекомендуется "
                              f"сохранить проект. Подробности записаны в журнал:\n{log_dir()}")
            box.setDetailedText(text)
            box.exec()
    except Exception:  # noqa: BLE001
        pass


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    setup_logging()
    log.info("Запуск %s %s, Python %s", APP_NAME, __version__, sys.version.split()[0])
    sys.excepthook = _excepthook

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(argv)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")

    from .gui.main_window import MainWindow
    files = [a for a in argv[1:] if os.path.isfile(a)]
    win = MainWindow(auto_start=not files)
    win.show()
    if files:
        win.open_paths(files)
    code = app.exec()
    win.state.runner.wait(5000)
    # явное удаление окна до разрушения QApplication и выхода из интерпретатора
    from PySide6.QtCore import QCoreApplication, QEvent
    win.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    del win
    log.info("Завершение, код %s", code)
    return code
