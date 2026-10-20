"""Фоновое выполнение длительных операций без блокировки интерфейса."""
from __future__ import annotations

import logging
import traceback
from typing import Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from ..core.detection import Cancelled

log = logging.getLogger(__name__)


class _Signals(QObject):
    finished = Signal(object)
    error = Signal(str, str)
    progress = Signal(int)
    cancelled = Signal()


class Task(QObject):
    """Дескриптор задачи; живёт в GUI-потоке, поэтому колбэки вызываются в GUI-потоке."""

    def __init__(self, runner: "TaskRunner", name: str, on_done, on_error, on_progress):
        super().__init__()
        self.runner = runner
        self.name = name
        self._on_done = on_done
        self._on_error = on_error
        self._on_progress = on_progress
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    @Slot(object)
    def _done(self, result) -> None:
        self.runner._finish(self)
        if self._on_done:
            self._on_done(result)

    @Slot(str, str)
    def _error(self, msg: str, tb: str) -> None:
        self.runner._finish(self)
        if self._on_error:
            self._on_error(msg, tb)
        else:
            self.runner.default_error(self.name, msg, tb)

    @Slot(int)
    def _progress(self, v: int) -> None:
        if self._on_progress:
            self._on_progress(v)
        self.runner.progress.emit(self.name, v)

    @Slot()
    def _cancelled(self) -> None:
        self.runner._finish(self)


class _Runnable(QRunnable):
    def __init__(self, fn, args, kwargs, signals: _Signals, task: Task, with_progress: bool):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = signals
        self.task = task
        self.with_progress = with_progress
        self.setAutoDelete(True)

    def _emit(self, sig, *args) -> None:
        try:
            sig.emit(*args)
        except RuntimeError:   # получатель уже уничтожен (например, при закрытии приложения)
            pass

    def run(self) -> None:
        try:
            kw = dict(self.kwargs)
            if self.with_progress:
                kw["progress"] = lambda v: self._emit(self.signals.progress, int(v))
                kw["is_cancelled"] = lambda: self.task.cancel_requested
            res = self.fn(*self.args, **kw)
            if self.task.cancel_requested:
                self._emit(self.signals.cancelled)
            else:
                self._emit(self.signals.finished, res)
        except Cancelled:
            self._emit(self.signals.cancelled)
        except MemoryError:
            log.exception("Нехватка памяти в задаче %s", self.task.name)
            self._emit(self.signals.error, "Недостаточно памяти для выполнения операции.", traceback.format_exc())
        except Exception as e:  # noqa: BLE001
            log.exception("Ошибка в фоновой задаче %s", self.task.name)
            self._emit(self.signals.error, str(e) or e.__class__.__name__, traceback.format_exc())


class TaskRunner(QObject):
    busyChanged = Signal(bool, str)
    progress = Signal(str, int)

    def __init__(self, parent=None, max_threads: Optional[int] = None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        if max_threads:
            self.pool.setMaxThreadCount(max_threads)
        self._tasks: set[Task] = set()
        self._silent: set[Task] = set()
        self.error_handler: Optional[Callable[[str, str, str], None]] = None

    def run(self, name: str, fn, *args, on_done=None, on_error=None, on_progress=None,
            with_progress: bool = False, silent: bool = False, **kwargs) -> Task:
        # Task и _Signals принадлежат C++ (родитель — runner) и удаляются только в GUI-потоке через
        # deleteLater(); иначе последняя Python-ссылка могла бы освободиться в рабочем потоке.
        task = Task(self, name, on_done, on_error, on_progress)
        task.setParent(self)
        sig = _Signals()
        sig.setParent(task)
        sig.finished.connect(task._done)
        sig.error.connect(task._error)
        sig.progress.connect(task._progress)
        sig.cancelled.connect(task._cancelled)
        task._signals = sig  # держим ссылку
        self._tasks.add(task)
        if silent:
            self._silent.add(task)
        self._emit_busy()
        self.pool.start(_Runnable(fn, args, kwargs, sig, task, with_progress))
        return task

    def _finish(self, task: Task) -> None:
        if task in self._tasks:
            self._tasks.discard(task)
            self._silent.discard(task)
            task.deleteLater()
        self._emit_busy()

    def _emit_busy(self) -> None:
        visible = [t for t in self._tasks if t not in self._silent]
        self.busyChanged.emit(bool(visible), visible[0].name if visible else "")

    def is_busy(self) -> bool:
        return any(t not in self._silent for t in self._tasks)

    def default_error(self, name: str, msg: str, tb: str) -> None:
        if self.error_handler:
            self.error_handler(name, msg, tb)

    def wait(self, msecs: int = 30000) -> bool:
        return self.pool.waitForDone(msecs)
