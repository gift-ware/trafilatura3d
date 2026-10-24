"""The mouse wheel over value widgets (sliders, spin boxes, dropdowns) changes their value only while they have the
focus (a click gives it); otherwise the wheel goes on to what contains them, so scrolling a panel never edits the
value that passes under the pointer. One filter on the application, for every panel of both workspaces (the
blending dropdown, too: the wheel cycles modes only once it has the focus)."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QAbstractSlider, QAbstractSpinBox, QApplication, QComboBox, QScrollBar

VALUE_WIDGETS = (QAbstractSlider, QAbstractSpinBox, QComboBox)


_GUARDED: dict[type, bool] = {}  # per class: whether its widgets are guarded (worked out once)


def _guarded(obj) -> bool:
    # Scroll bars are sliders too, but a scroll area scrolls by sending them its wheel events: never theirs.
    cls = type(obj)
    if cls not in _GUARDED:
        _GUARDED[cls] = issubclass(cls, VALUE_WIDGETS) and not issubclass(cls, QScrollBar)
    return _GUARDED[cls]


class WheelGuard(QObject):
    def eventFilter(self, obj, event):
        # Every event of the application passes here: the widget's class first (cheap), the event's type (a Python
        # enum made each time) only for the value widgets.
        if not _guarded(obj):
            return False
        kind = event.type()
        if kind == QEvent.Wheel and not obj.hasFocus():
            event.ignore()  # Qt passes an ignored wheel event on to the parent widgets (the scroll area)
            return True
        if kind == QEvent.Polish and obj.focusPolicy() == Qt.WheelFocus:
            obj.setFocusPolicy(Qt.StrongFocus)  # the wheel itself would give them the focus
        return False


_GUARD: WheelGuard | None = None


def install() -> None:
    """Install the filter on the application (once)."""
    global _GUARD
    app = QApplication.instance()
    if _GUARD is None and app is not None:
        _GUARD = WheelGuard(app)
        app.installEventFilter(_GUARD)
