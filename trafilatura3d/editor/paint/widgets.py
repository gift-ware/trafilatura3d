"""The Paint widgets: sliders with the value field above the slider (drag, Shift drag for
precision, click to jump, arrows step, type in the field), section headers in capitals, channel toggle
buttons."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QToolButton, QVBoxLayout,
                               QWidget)

from trafilatura3d.editor.widgets import NumberField


class Slider(QSlider):
    """A slider that jumps to the clicked value and drags finer with Shift (Sliders page)."""

    def __init__(self):
        super().__init__(Qt.Horizontal)
        self._shift = None

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            if e.modifiers() & Qt.ShiftModifier:
                self._shift = (e.position().x(), self.value())
                self.setSliderDown(True)
                return
            span = max(self.width() - 10, 1)
            self.setValue(self.minimum() + round((e.position().x() - 5) / span * (self.maximum() - self.minimum())))
            self.setSliderDown(True)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        span = max(self.width() - 10, 1)
        if self._shift is not None:
            x0, v0 = self._shift
            self.setValue(v0 + round((e.position().x() - x0) / span * (self.maximum() - self.minimum()) / 10))
            return
        if self.isSliderDown():
            self.setValue(self.minimum() + round((e.position().x() - 5) / span * (self.maximum() - self.minimum())))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._shift = None
        if self.isSliderDown():
            self.setSliderDown(False)
            self.sliderReleased.emit()
            return
        super().mouseReleaseEvent(e)


class PSlider(QWidget):
    """Name and value on top, the slider below. `edited(value, final)`."""
    edited = Signal(float, bool)
    STEPS = 1000

    def __init__(self, name: str, value: float, lo: float, hi: float, integer: bool = False, compact: bool = False):
        super().__init__()
        self.lo, self.hi, self.integer = lo, hi, integer
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(name)
        top.addWidget(self.label, 1)
        self.number = NumberField(integer)
        self.number.setFixedWidth(48 if compact else 60)
        self.number.setStyleSheet("background: transparent; border: none;")
        top.addWidget(self.number)
        lay.addLayout(top)
        self.slider = Slider()
        self.slider.setRange(0, self.STEPS)
        lay.addWidget(self.slider)
        self.set_value(value)
        self.slider.valueChanged.connect(self._slid)
        self.slider.sliderReleased.connect(lambda: self.edited.emit(self.value(), True))
        self.number.committed.connect(lambda v: (self._place(v), self.edited.emit(self.value(), True)))

    def value(self) -> float:
        return float(round(self.number.value)) if self.integer else float(self.number.value)

    def set_value(self, v: float):
        self.number.set_value(v)
        self._place(v)

    def _place(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round((v - self.lo) / (self.hi - self.lo) * self.STEPS)))
        self.slider.blockSignals(False)

    def _slid(self, s):
        v = self.lo + s / self.STEPS * (self.hi - self.lo)
        if self.integer:
            v = round(v)
        self.number.set_value(v)
        self.edited.emit(self.value(), not self.slider.isSliderDown())


def header(text: str) -> QLabel:
    h = QLabel(text)
    h.setStyleSheet("font-weight: 600; padding: 8px 0 4px 0;")
    h.setProperty("section", True)  # its panel draws the divider under it, edge to edge (theme.section_lines)
    return h


class ChannelButtons(QWidget):
    """The material's channel buttons (color, height, rough…): toggled on are used. `toggled(channel, on)`."""
    toggled = Signal(str, bool)

    def __init__(self, channels: list[tuple[str, str]], on: set, per_row: int = 4):
        super().__init__()
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        self.buttons = {}
        for i, (cid, short) in enumerate(channels):
            b = QPushButton(short)
            b.setCheckable(True)
            b.setChecked(cid in on)
            b.setStyleSheet("QPushButton { padding: 2px 6px; } QPushButton:checked { background: #3a3a3a; "
                            "border: 1px solid #c8c8c8; color: white; }")
            b.toggled.connect(lambda v, cid=cid: self.toggled.emit(cid, v))
            grid.addWidget(b, i // per_row, i % per_row)
            self.buttons[cid] = b


class PressureButton(QToolButton):
    """The pen pressure toggle next to a pressure capable parameter (Size, Flow: Pen pressure page), in the
    contextual toolbar and the Properties; lit when the pen's pressure drives the parameter."""

    def __init__(self, on: bool, name: str):
        super().__init__()
        from trafilatura3d.editor.paint.icons import picon
        from trafilatura3d.editor.theme import ACCENT
        icon = picon("pen_pressure")
        on_icon = picon("pen_pressure", "#ffffff")
        icon.addPixmap(on_icon.pixmap(16, 16), QIcon.Normal, QIcon.On)
        self.setIcon(icon)
        self.setCheckable(True)
        self.setChecked(on)
        self.setAutoRaise(True)
        self.setToolTip(f"Pen pressure drives the {name}")
        self.setStyleSheet(f"QToolButton:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}")

    def set_on(self, on: bool):
        """Follow the brush without emitting `toggled`."""
        self.blockSignals(True)
        self.setChecked(on)
        self.blockSignals(False)


class Swatch(QWidget):
    """A color field: click opens the color picker (`color_picker.py`), the eyedropper beside it picks a
    color directly (Color picker page: color fields have their own). `title`: the picker's ("<title> color");
    `channel`: what the field edits (the eyedropper's Shift pick). `edited(rgb, final)`."""
    edited = Signal(list, bool)

    def __init__(self, rgb, title: str = "Color", channel: str | None = None):
        super().__init__()
        from trafilatura3d.editor.paint.color_picker import DropperButton
        self.rgb = list(rgb[:3])
        self.title, self.channel = title, channel
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.button = QPushButton()
        self.button.setFixedHeight(18)
        self.button.clicked.connect(self._pick)
        lay.addWidget(self.button, 1)
        self.dropper = DropperButton(channel)
        self.dropper.setFixedSize(20, 18)
        self.dropper.picked.connect(lambda rgb: (self.set_rgb(rgb), self.edited.emit(list(rgb), True)))
        lay.addWidget(self.dropper)
        self._paint()

    def set_rgb(self, rgb):
        self.rgb = list(rgb[:3])
        self._paint()

    def _paint(self):
        c = QColor.fromRgbF(*[min(max(x, 0.0), 1.0) for x in self.rgb])
        self.button.setStyleSheet(f"background: {c.name()}; border: 1px solid #111;")

    def _picked(self, rgb, final):
        self.set_rgb(rgb)
        self.edited.emit(list(rgb), final)

    def _pick(self):
        from trafilatura3d.editor.paint.color_picker import picker
        picker().open(self.title, self.rgb, self._picked, self.channel, self.window())


class ColorField(Swatch):
    """A color parameter of a node (Properties of effects, materials, procedurals) as a Paint color field:
    `value_edited(value, final)` keeps the value's alpha (the picker has none)."""
    value_edited = Signal(list, bool)

    def __init__(self, value, title: str = "Color"):
        self.alpha = list(value[3:])
        super().__init__(value, title)
        self.edited.connect(lambda rgb, final: self.value_edited.emit(list(rgb) + self.alpha, final))


class _DropFilter(QObject):
    """Accepts drags of assets of some kinds on a widget and hands them to a callback."""

    def __init__(self, widget, kinds, on_drop):
        super().__init__(widget)
        self.kinds, self.on_drop = set(kinds), on_drop

    def _asset(self, e):
        from trafilatura3d.editor.paint.assets import asset_of
        a = asset_of(e.mimeData())
        return a if a is not None and a.get("kind") in self.kinds else None

    def eventFilter(self, obj, e):
        t = e.type()
        if t in (QEvent.DragEnter, QEvent.DragMove):
            if self._asset(e) is not None:
                e.acceptProposedAction()
                return True
        elif t == QEvent.Drop:
            a = self._asset(e)
            if a is not None:
                e.acceptProposedAction()
                self.on_drop(a)
                return True
        return False


def accept_assets(widget, kinds, on_drop):
    """Let `widget` take assets dragged from the Assets panel of the given kinds: `on_drop(asset)`."""
    widget.setAcceptDrops(True)
    widget.installEventFilter(_DropFilter(widget, kinds, on_drop))
    return widget
