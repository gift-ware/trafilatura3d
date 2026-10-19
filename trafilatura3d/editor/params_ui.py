"""Parameter editors by kind and Editor property (the input parameter editors), and the grouping
of parameters into collapsible groups ("/" separates sub groups)."""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QDial, QHBoxLayout, QLineEdit, QPlainTextEdit, QToolButton, QVBoxLayout,
                               QWidget)

from trafilatura3d.editor.icons import icon
from trafilatura3d.editor.widgets import (BoolToggle, ColorButton, CurveEditor, GradientBar, NumberField, SliderField,
                                    VectorField)
from trafilatura3d.nodes.base import Param, ParamKind


class AngleField(QWidget):
    """An angle in turns (0..1 = a full turn) with a dial and a number in degrees."""
    edited = Signal(object, bool)

    def __init__(self, value):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.dial = QDial()
        self.dial.setRange(0, 3600)
        self.dial.setWrapping(True)
        self.dial.setFixedSize(40, 40)
        self.number = NumberField(False)
        lay.addWidget(self.dial)
        lay.addWidget(self.number, 1)
        self.set_value(value)
        self.dial.valueChanged.connect(lambda v: self._changed(v / 3600.0, not self.dial.isSliderDown()))
        self.dial.sliderReleased.connect(lambda: self.edited.emit(self.value, True))
        self.number.committed.connect(lambda deg: (self.set_value(deg / 360.0), self.edited.emit(self.value, True)))

    def set_value(self, v):
        self.value = float(v)
        self.dial.blockSignals(True)
        self.dial.setValue(int(round((self.value % 1.0) * 3600)))
        self.dial.blockSignals(False)
        self.number.set_value(self.value * 360.0)

    def _changed(self, turns, final):
        self.value = turns
        self.number.set_value(turns * 360.0)
        self.edited.emit(turns, final)


class PositionPad(QWidget):
    """A 2D position in [min, max]^2: drag in the pad, or type the numbers."""
    edited = Signal(object, bool)

    def __init__(self, value, lo, hi):
        super().__init__()
        self.lo, self.hi = (lo if lo is not None else 0.0), (hi if hi is not None else 1.0)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.pad = _Pad(self)
        self.fields = VectorField(value, self.lo, self.hi)
        self.fields.edited.connect(lambda v, final: (self._set(v), self.edited.emit(v, final)))
        lay.addWidget(self.pad)
        lay.addWidget(self.fields, 1)
        self._set(value)

    def _set(self, v):
        self.value = [float(v[0]), float(v[1])]
        self.pad.update()

    def drag(self, fx, fy, final):
        v = [self.lo + fx * (self.hi - self.lo), self.lo + fy * (self.hi - self.lo)]
        self._set(v)
        self.fields.set_values(v)
        self.edited.emit(v, final)


class _Pad(QWidget):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.setFixedSize(64, 64)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1a1a1a"))
        p.setPen(QPen(QColor("#444"), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        p.drawLine(32, 0, 32, 64)
        p.drawLine(0, 32, 64, 32)
        o = self.owner
        span = (o.hi - o.lo) or 1.0
        x = (o.value[0] - o.lo) / span * 63
        y = (o.value[1] - o.lo) / span * 63
        p.setPen(QPen(QColor("#e8a33a"), 2))
        p.drawEllipse(QPointF(x, y), 4, 4)

    def _at(self, e, final):
        self.owner.drag(min(max(e.position().x() / 63, 0), 1), min(max(e.position().y() / 63, 0), 1), final)

    def mousePressEvent(self, e):
        self._at(e, False)

    def mouseMoveEvent(self, e):
        self._at(e, False)

    def mouseReleaseEvent(self, e):
        self._at(e, True)


class Pow2Field(QComboBox):
    """A size as a power of two (stored as its log2), 1 to 8192."""
    edited = Signal(object, bool)

    def __init__(self, value):
        super().__init__()
        self.addItems([str(1 << i) for i in range(14)])
        self.setCurrentIndex(max(0, min(13, int(value))))
        self.currentIndexChanged.connect(lambda i: self.edited.emit(i, True))


class SeedField(QWidget):
    """The Random seed editor: a slider and a button for a new random value."""
    edited = Signal(object, bool)

    def __init__(self, value, lo, hi):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = SliderField(value, lo if lo is not None else 0, hi if hi is not None else 10000, integer=True)
        dice = QToolButton()
        dice.setIcon(icon("random"))
        dice.setAutoRaise(True)
        dice.setToolTip("Random seed")
        dice.clicked.connect(lambda: (self.slider.set_value(v := random.randint(0, 10000)), self.edited.emit(v, True)))
        self.slider.edited.connect(self.edited.emit)
        lay.addWidget(self.slider, 1)
        lay.addWidget(dice)


def make_editor(param: Param, value, on_edit, color_field=None) -> QWidget:
    """The editor widget of a parameter; `on_edit(value, final)` is called on changes. `color_field(value)`: the
    workspace's own color field for color parameters (Paint's, with `value_edited`), the Graph one by default."""
    k, ed = param.kind, param.editor_name()
    if color_field is not None and (k is ParamKind.COLOR or (k in (ParamKind.FLOAT3, ParamKind.FLOAT4)
                                                             and ed == "Color")):
        w = color_field(list(value))
        w.value_edited.connect(lambda v, final: on_edit(list(v[:3]) if k is ParamKind.FLOAT3 else list(v), final))
        return w
    step = param.step
    if k is ParamKind.FLOAT and ed == "Angle":
        w = AngleField(value)
    elif k is ParamKind.INT and ed == "Drop down list" and param.items:
        w = QComboBox()
        values = [v for v, _ in param.items]
        w.addItems([lbl for _, lbl in param.items])
        if value in values:
            w.setCurrentIndex(values.index(value))
        w.currentIndexChanged.connect(lambda i: on_edit(values[i], True))
        return w
    elif k is ParamKind.INT and ed == "Random seed":
        w = SeedField(value, param.min, param.max)
    elif k is ParamKind.INT and ed == "Size pow 2":
        w = Pow2Field(value)
    elif k in (ParamKind.FLOAT, ParamKind.INT):
        w = SliderField(value, param.min, param.max, integer=(k is ParamKind.INT))
        if step:
            w.number.step = step
    elif k is ParamKind.FLOAT2 and ed == "Position":
        w = PositionPad(value, param.min, param.max)
    elif k in (ParamKind.FLOAT3, ParamKind.FLOAT4) and ed == "Color":
        w = ColorButton(list(value) + ([1.0] if k is ParamKind.FLOAT3 else []))
        w.edited.connect(lambda v, final: on_edit(list(v[:3]) if k is ParamKind.FLOAT3 else list(v), final))
        return w
    elif k is ParamKind.INT2 and ed == "Size pow 2":
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        cur = list(value)
        for axis in range(2):
            c = Pow2Field(cur[axis])
            c.edited.connect(lambda v, final, axis=axis: (cur.__setitem__(axis, int(v)), on_edit(list(cur), True)))
            lay.addWidget(c)
        return w
    elif k in (ParamKind.FLOAT2, ParamKind.FLOAT3, ParamKind.FLOAT4, ParamKind.INT2):
        w = VectorField(value, param.min, param.max, integer=(k is ParamKind.INT2))
    elif k is ParamKind.COLOR:
        w = ColorButton(value)
    elif k is ParamKind.BOOL:
        w = BoolToggle(value)
    elif k is ParamKind.ENUM:
        w = QComboBox()
        w.addItems(param.choices)
        w.setCurrentText(value)
        w.currentTextChanged.connect(lambda t: on_edit(t, True))
        return w
    elif k is ParamKind.GRADIENT:
        w = GradientBar(value)
    elif k is ParamKind.CURVE:
        w = CurveEditor(value)
    else:  # STRING
        if param.label in ("Text", "Description"):
            w = QPlainTextEdit(value)
            w.setFixedHeight(60)
            w.textChanged.connect(lambda: on_edit(w.toPlainText(), True))
        else:
            w = QLineEdit(value)
            w.editingFinished.connect(lambda: on_edit(w.text(), True))
        return w
    w.edited.connect(on_edit)
    return w


def grouped(items, group_of):
    """Items in group order: ungrouped first, then each top group in order of first
    appearance with its sub groups right after it. Returns [(group path tuple, item)]."""
    path_of = lambda it: tuple(p for p in (group_of(it) or "").split("/") if p)
    tops = []
    for it in items:
        path = path_of(it)
        if path and path[0] not in tops:
            tops.append(path[0])
    key = lambda it: (bool(path_of(it)), tops.index(path_of(it)[0]) if path_of(it) else 0, path_of(it))
    return [(path_of(it), it) for it in sorted(items, key=key)]
