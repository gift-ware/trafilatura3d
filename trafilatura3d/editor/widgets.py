"""Parameter widgets: a slider with its number on the right (the number accepts
formulas like (4+2)*3), vector rows, color swatches, False/True toggles, gradient, levels and curve editors.
Every widget emits `edited(value, final)`: `final` is False while a slider is being dragged."""

from __future__ import annotations

import ast
import operator

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QVBoxLayout, QWidget)

from trafilatura3d.editor import theme
from trafilatura3d.nodes.base import GRADIENT_INTERPOLATIONS

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def evaluate_formula(text: str) -> float:
    """A number from a basic math formula (+ - * / ** %, parentheses); raises ValueError."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(text)
    try:
        return float(ev(ast.parse(text.strip(), mode="eval")))
    except (SyntaxError, ZeroDivisionError, TypeError) as e:
        raise ValueError(text) from e


class NumberField(QLineEdit):
    """A number that accepts formulas; Enter validates, an invalid formula reverts."""
    committed = Signal(float)

    def __init__(self, integer: bool):
        super().__init__()
        self.integer = integer
        self.value = 0.0
        self.setFixedWidth(84)
        self.setAlignment(Qt.AlignRight)
        self.editingFinished.connect(self._commit)

    def set_value(self, v: float):
        self.value = v
        self.setText(str(int(round(v))) if self.integer else f"{v:.4g}" if abs(v) < 1e5 else f"{v:.6g}")

    def _commit(self):
        try:
            v = evaluate_formula(self.text())
        except ValueError:
            self.set_value(self.value)
            return
        v = float(round(v)) if self.integer else v
        if v != self.value:
            self.set_value(v)
            self.committed.emit(v)


class SliderField(QWidget):
    """Slider over [min, max] plus a number field (values outside the range can be typed)."""
    edited = Signal(object, bool)
    STEPS = 1000

    def __init__(self, value, lo, hi, integer=False, suffix=""):
        super().__init__()
        self.integer = integer
        self.lo, self.hi = (lo if lo is not None else 0.0), (hi if hi is not None else (100 if integer else 1.0))
        if self.hi <= self.lo:
            self.hi = self.lo + 1
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(self.hi - self.lo) if integer else self.STEPS)
        self.number = NumberField(integer)
        lay.addWidget(self.slider, 1)
        if suffix:
            lay.addWidget(QLabel(suffix))
        lay.addWidget(self.number)
        self.set_value(value)
        self.slider.valueChanged.connect(self._slid)
        self.slider.sliderReleased.connect(lambda: self.edited.emit(self.value(), True))
        self.number.committed.connect(lambda v: (self._place(v), self.edited.emit(self._cast(v), True)))

    def _cast(self, v):
        return int(round(v)) if self.integer else float(v)

    def value(self):
        return self._cast(self.number.value)

    def set_value(self, v):
        self.number.set_value(v)
        self._place(v)

    def _place(self, v):
        self.slider.blockSignals(True)
        if self.integer:
            self.slider.setValue(int(round(v - self.lo)))
        else:
            self.slider.setValue(int(round((v - self.lo) / (self.hi - self.lo) * self.STEPS)))
        self.slider.blockSignals(False)

    def _slid(self, s):
        v = self.lo + s if self.integer else self.lo + s / self.STEPS * (self.hi - self.lo)
        self.number.set_value(v)
        self.edited.emit(self._cast(v), not self.slider.isSliderDown())


class VectorField(QWidget):
    """One slider row per component, labelled x y z w."""
    edited = Signal(object, bool)

    def __init__(self, values, lo, hi, integer=False):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        self.fields = []
        for i, v in enumerate(values):
            row = QHBoxLayout()
            name = QLabel("xyzw"[i])
            name.setObjectName("dim")
            name.setFixedWidth(10)
            f = SliderField(v, lo, hi, integer)
            f.edited.connect(lambda _v, final: self.edited.emit(self.values(), final))
            row.addWidget(name)
            row.addWidget(f, 1)
            lay.addLayout(row)
            self.fields.append(f)

    def values(self):
        return [f.value() for f in self.fields]

    def set_values(self, values):
        for f, v in zip(self.fields, values):
            f.set_value(v)


class ColorButton(QPushButton):
    edited = Signal(object, bool)

    def __init__(self, rgba):
        super().__init__()
        self.rgba = list(rgba)
        self.setFixedHeight(22)
        self.clicked.connect(self._pick)
        self._paint()

    def _paint(self):
        c = QColor.fromRgbF(*[min(max(x, 0.0), 1.0) for x in self.rgba[:3]])
        text = "#fff" if c.lightnessF() < 0.5 else "#000"
        self.setText(f"{c.name()}   α {self.rgba[3]:.2f}")
        self.setStyleSheet(f"background: {c.name()}; color: {text}; border: 1px solid #111;")

    def set_value(self, rgba):
        self.rgba = list(rgba)
        self._paint()

    def _pick(self):
        start = QColor.fromRgbF(*[min(max(x, 0.0), 1.0) for x in self.rgba])
        dlg = QColorDialog(start, self)
        dlg.setOption(QColorDialog.ShowAlphaChannel)
        dlg.currentColorChanged.connect(lambda c: self._live(c))
        if dlg.exec():
            c = dlg.currentColor()
            self.rgba = [c.redF(), c.greenF(), c.blueF(), c.alphaF()]
            self._paint()
            self.edited.emit(self.rgba, True)
        else:
            self.edited.emit(self.rgba, True)

    def _live(self, c):
        self.edited.emit([c.redF(), c.greenF(), c.blueF(), c.alphaF()], False)


class BoolToggle(QWidget):
    """A False / True pair of buttons."""
    edited = Signal(object, bool)

    def __init__(self, value: bool):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.no, self.yes = QPushButton("False"), QPushButton("True")
        for b, v in ((self.no, False), (self.yes, True)):
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, v=v: (self.set_value(v), self.edited.emit(v, True)))
            lay.addWidget(b)
        self.set_value(value)

    def set_value(self, v: bool):
        self.no.setChecked(not v)
        self.yes.setChecked(v)


class GradientBar(QWidget):
    """The gradient drawn as a bar with its keys; click to edit."""
    edited = Signal(object, bool)

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.setFixedHeight(26)
        self.setCursor(Qt.PointingHandCursor)

    def set_value(self, v):
        self.value = v
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        r = self.rect().adjusted(1, 1, -1, -8)
        g = QLinearGradient(r.left(), 0, r.right(), 0)
        for k in self.value["keys"]:
            g.setColorAt(min(max(k["pos"], 0.0), 1.0), QColor.fromRgbF(*[min(max(x, 0), 1) for x in k["color"][:3]]))
        p.fillRect(r, g)
        p.setPen(QColor("#ddd"))
        for k in self.value["keys"]:
            x = r.left() + k["pos"] * r.width()
            p.drawLine(QPointF(x, r.bottom() + 1), QPointF(x, r.bottom() + 6))

    def mousePressEvent(self, e):
        dlg = GradientDialog(self.value, self)
        dlg.changed.connect(lambda v: self.edited.emit(v, False))
        if dlg.exec():
            self.value = dlg.result_value()
            self.update()
        self.edited.emit(self.value, True)


class GradientDialog(QDialog):
    """Edit gradient keys (position + color), interpolation, midpoint and smoothness."""
    changed = Signal(object)

    def __init__(self, value, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gradient editor")
        self.value = {**value, "keys": [dict(k) for k in value["keys"]]}
        lay = QVBoxLayout(self)
        self.bar = GradientBar(self.value)
        self.bar.mousePressEvent = lambda e: None
        lay.addWidget(self.bar)
        form = QFormLayout()
        self.interp = QComboBox()
        self.interp.addItems(GRADIENT_INTERPOLATIONS)
        self.interp.setCurrentText(self.value["interpolation"])
        self.interp.currentTextChanged.connect(lambda t: self._set("interpolation", t))
        self.mid = SliderField(self.value["midpoint"], 0.0, 1.0)
        self.mid.edited.connect(lambda v, f: self._set("midpoint", v))
        self.smooth = SliderField(self.value["smoothness"], 0.0, 1.0)
        self.smooth.edited.connect(lambda v, f: self._set("smoothness", v))
        form.addRow("Interpolation", self.interp)
        form.addRow("Midpoint", self.mid)
        form.addRow("Smoothness", self.smooth)
        lay.addLayout(form)
        self.keys_box = QVBoxLayout()
        lay.addLayout(self.keys_box)
        add = QPushButton("Add key")
        add.clicked.connect(self._add)
        lay.addWidget(add)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._rows()

    def _rows(self):
        while self.keys_box.count():
            item = self.keys_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, k in enumerate(self.value["keys"]):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            pos = SliderField(k["pos"], 0.0, 1.0)
            pos.edited.connect(lambda v, f, i=i: self._key(i, "pos", v, f))
            color = ColorButton(k["color"])
            color.edited.connect(lambda v, f, i=i: self._key(i, "color", v, True))
            rm = QPushButton("×")
            rm.setFixedWidth(24)
            rm.setEnabled(len(self.value["keys"]) > 1)
            rm.clicked.connect(lambda _=False, i=i: self._remove(i))
            h.addWidget(pos, 2)
            h.addWidget(color, 1)
            h.addWidget(rm)
            self.keys_box.addWidget(row)

    def _set(self, key, v):
        self.value[key] = v
        self._emit()

    def _key(self, i, field, v, final):
        self.value["keys"][i][field] = v
        if final and field == "pos":
            self.value["keys"].sort(key=lambda k: k["pos"])
            self._rows()
        self._emit()

    def _add(self):
        self.value["keys"].append({"pos": 0.5, "color": [0.5, 0.5, 0.5, 1.0]})
        self.value["keys"].sort(key=lambda k: k["pos"])
        self._rows()
        self._emit()

    def _remove(self, i):
        del self.value["keys"][i]
        self._rows()
        self._emit()

    def _emit(self):
        self.bar.set_value(self.value)
        self.changed.emit(self.result_value())

    def result_value(self):
        return {**self.value, "keys": sorted((dict(k) for k in self.value["keys"]), key=lambda k: k["pos"])}


# A levels histogram's colour channels, drawn over each other: where all three overlap, a neutral gray.
HISTOGRAM_RGB = ("#d84848", "#48d848", "#4848d8")
HISTOGRAM_GRAY = "#7a7a7a"


class LevelsEditor(QWidget):
    """The levels control of the Levels node and of the Paint Levels effect: an optional channel selector, the
    histogram of the image coming in with the input handles above it (black In low, white In high, gray In mid
    between them: a relative position, 0.5 = linear), the black to white output bar with its Out low and Out high
    handles, and the exact values in fields under each bar. Works on the five values of one channel, in that order;
    `edited(values, final)`, `final` False while a handle is dragged; `channel_changed(index)`."""
    edited = Signal(object, bool)
    channel_changed = Signal(int)

    def __init__(self, values, channels=(), channel=0, dark=False):
        super().__init__()
        self.values = [float(v) for v in values]
        self.series = []  # [(counts, color)] drawn over each other
        self.ends = ("", "")  # labels of the histogram's range (0 and the depth's top)
        self.dark = dark
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.combo = None
        if channels:
            self.combo = QComboBox()
            ramp = QPixmap(28, 12)
            painter = QPainter(ramp)
            g = QLinearGradient(0, 0, 28, 0)
            g.setColorAt(0, QColor("black"))
            g.setColorAt(1, QColor("white"))
            painter.fillRect(ramp.rect(), g)
            painter.end()
            for c in channels:
                self.combo.addItem(QIcon(ramp), c)
            self.combo.setCurrentIndex(channel)
            self.combo.currentIndexChanged.connect(self.channel_changed)
            lay.addWidget(self.combo)
        self.input = _LevelsBar(self, (0, 1, 2), output=False)
        self.output = _LevelsBar(self, (3, 4), output=True)
        self.fields = [NumberField(False) for _ in range(5)]
        for i, f in enumerate(self.fields):
            f.setFixedWidth(56)
            f.committed.connect(lambda v, i=i: self._set(i, v, True))
        lay.addWidget(self.input)
        lay.addLayout(self._row(self.fields[:3]))
        lay.addWidget(self.output)
        lay.addLayout(self._row(self.fields[3:]))
        self.set_levels(self.values)

    @staticmethod
    def _row(fields):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for i, f in enumerate(fields):
            if i:
                row.addStretch(1)
            row.addWidget(f)
        return row

    def set_levels(self, values):
        self.values = [float(v) for v in values]
        for f, v in zip(self.fields, self.values):
            f.set_value(v)
        self.input.update()
        self.output.update()

    def set_histogram(self, series, ends=("", "")):
        """`series`: [(counts per bin, color)], drawn over each other (none: an empty box); `ends`: the labels of
        its range."""
        self.series, self.ends = list(series), ends
        self.input.update()

    def _set(self, i, v, final):
        """Value `i` set by a handle or a field: In low stays under In high (and the reverse)."""
        lo, _, hi = self.values[:3]
        if i == 0:
            v = min(v, hi)
        elif i == 2:
            v = max(v, lo)
        self.values[i] = float(v)
        self.set_levels(self.values)
        self.edited.emit(list(self.values), final)

    def position(self, i) -> float:
        """Where handle `i` is along its bar, 0-1."""
        lo, mid, hi = self.values[:3]
        return lo + mid * (hi - lo) if i == 1 else self.values[i]

    def drag(self, i, x, final):
        """Handle `i` dragged to `x` (0-1 along its bar)."""
        x = min(max(x, 0.0), 1.0)
        if i == 1:
            lo, _, hi = self.values[:3]
            x = min(max((x - lo) / (hi - lo), 0.0), 1.0) if hi > lo else 0.5
        self._set(i, x, final)


class _LevelsBar(QWidget):
    """A bar of the levels editor: the histogram with the input handles above it, or the output gradient with its
    handles under it."""
    MARGIN = 7  # the handles' half width: the end ones stay inside
    HANDLE = 9

    def __init__(self, owner: LevelsEditor, handles, output: bool):
        super().__init__()
        self.owner, self.handles, self.out = owner, handles, output
        self.setFixedHeight(26 if output else 118)
        self._drag = None

    def _x(self, t) -> float:
        return self.MARGIN + t * (self.width() - 2 * self.MARGIN)

    def _t(self, x) -> float:
        return (x - self.MARGIN) / max(self.width() - 2 * self.MARGIN, 1)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        o, m, w, h = self.owner, self.MARGIN, self.width(), self.height()
        if self.out:
            bar = QRectF(m, 2, w - 2 * m, 8)
            g = QLinearGradient(bar.left(), 0, bar.right(), 0)
            g.setColorAt(0, QColor("black"))
            g.setColorAt(1, QColor("white"))
            p.fillRect(bar, g)
            p.setPen(QColor("#111"))
            p.drawRect(bar)
        else:
            box = QRectF(m, self.HANDLE + 3, w - 2 * m, h - self.HANDLE - 4)
            self._histogram(p, box)
        for i in self.handles:
            self._handle(p, i, self._x(o.position(i)))

    def _histogram(self, p, box):
        o = self.owner
        p.fillRect(box, QColor("#1e1e1e" if o.dark else "#d2d2d2"))
        inner = box.adjusted(1, 1, -1, -14 if any(o.ends) else -1)  # the range's labels under the bars
        if o.series:
            top = max(float(np.max(c[1:-1])) for c, _ in o.series) or max(float(np.max(c)) for c, _ in o.series) or 1.0
            if len(o.series) > 1:  # where they overlap gray: the lightest on the dark box, the darkest on the light one
                p.setCompositionMode(QPainter.CompositionMode_Lighten if o.dark else QPainter.CompositionMode_Darken)
            for counts, color in o.series:
                n = len(counts)
                path = QPainterPath(QPointF(inner.left(), inner.bottom()))
                for b, c in enumerate(counts):
                    y = inner.bottom() - min(c / top, 1.0) * inner.height()
                    path.lineTo(inner.left() + b / n * inner.width(), y)
                    path.lineTo(inner.left() + (b + 1) / n * inner.width(), y)
                path.lineTo(inner.right(), inner.bottom())
                p.fillPath(path, QColor(color))
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        hatch = QColor("#3a3a3a" if o.dark else "#9a9a9a")
        for a, b in ((0.0, o.values[0]), (o.values[2], 1.0)):  # outside In low - In high: hatched
            if b > a:
                p.fillRect(QRectF(inner.left() + a * inner.width(), inner.top(), (b - a) * inner.width(),
                                  inner.height()), QBrush(hatch, Qt.BDiagPattern))
        p.setPen(QColor("#555" if o.dark else "#8a8a8a"))
        p.drawRect(box)
        p.setPen(QColor(theme.TEXT_DIM))
        text = box.adjusted(3, 0, -3, -1)
        p.drawText(text, Qt.AlignLeft | Qt.AlignBottom, o.ends[0])
        p.drawText(text, Qt.AlignRight | Qt.AlignBottom, o.ends[1])

    def _handle(self, p, i, x):
        """A triangle pointing at its bar: black (lows), gray (mid), white (highs)."""
        s = self.HANDLE
        fill = {0: "#000", 1: "#808080", 2: "#fff", 3: "#000", 4: "#fff"}[i]
        if self.out:
            y = 12
            pts = [QPointF(x, y), QPointF(x + s / 2 + 1, y + s), QPointF(x - s / 2 - 1, y + s)]
        else:
            pts = [QPointF(x, s + 1), QPointF(x + s / 2 + 1, 1), QPointF(x - s / 2 - 1, 1)]
        p.setPen(QPen(QColor("#c8c8c8" if fill == "#000" else "#222"), 1))
        p.setBrush(QColor(fill))
        p.drawPolygon(pts)

    def _hit(self, x):
        """The handle under `x`: the nearest; on a tie the one the click is past (a low on its high: the side
        tells which one to move)."""
        o = self.owner
        best = None
        for i in self.handles:
            d = abs(self._x(o.position(i)) - x)
            if d <= self.HANDLE and (best is None or d < best[0] - 0.5
                                     or (abs(d - best[0]) <= 0.5 and x > self._x(o.position(i)))):
                best = (d, i)
        return None if best is None else best[1]

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = self._hit(e.position().x())

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.owner.drag(self._drag, self._t(e.position().x()), False)

    def mouseReleaseEvent(self, e):
        if self._drag is not None:
            self.owner.drag(self._drag, self._t(e.position().x()), True)
            self._drag = None


class CurveEditor(QWidget):
    """Points in [0,1]^2: drag to move (with their Bezier handles), drag a handle to bend the curve, double click to
    add a point, right click to remove one."""
    edited = Signal(object, bool)

    def __init__(self, points):
        super().__init__()
        self.points = [list(p) for p in points]
        self.setFixedHeight(150)
        self._drag = None  # (point index, 0 the point / 1 its left handle / 2 its right handle)

    def set_value(self, pts):
        self.points = [list(p) for p in pts]
        self.update()

    def _to_screen(self, pt):
        r = self.rect().adjusted(6, 6, -6, -6)
        return QPointF(r.left() + pt[0] * r.width(), r.bottom() - pt[1] * r.height())

    def _to_curve(self, pos):
        r = self.rect().adjusted(6, 6, -6, -6)
        return [min(max((pos.x() - r.left()) / r.width(), 0.0), 1.0),
                min(max((r.bottom() - pos.y()) / r.height(), 0.0), 1.0)]

    def _sorted(self):
        return sorted(self.points, key=lambda q: q[0])

    def paintEvent(self, e):
        from trafilatura3d.evaluator import curve_values
        import numpy as np
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#1e1e1e"))
        p.setPen(QPen(QColor("#333"), 1))
        r = self.rect().adjusted(6, 6, -6, -6)
        for i in range(1, 4):
            p.drawLine(QPointF(r.left() + r.width() * i / 4, r.top()), QPointF(r.left() + r.width() * i / 4, r.bottom()))
            p.drawLine(QPointF(r.left(), r.top() + r.height() * i / 4), QPointF(r.right(), r.top() + r.height() * i / 4))
        pts = self._sorted()
        xs = np.linspace(0, 1, 200)
        ys = curve_values(pts, xs)
        path = QPainterPath(self._to_screen((xs[0], ys[0])))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(self._to_screen((x, y)))
        p.setPen(QPen(QColor("#e0e0e0"), 1.6))
        p.drawPath(path)
        p.setPen(QPen(QColor("#888"), 1))
        for q in pts:  # Bezier handles: a line from the point, a small square at the end
            for h in (q[2:4], q[4:6]) if len(q) == 6 else ():
                if h != q[:2]:
                    p.drawLine(self._to_screen(q), self._to_screen(h))
                    p.drawRect(self._to_screen(h).x() - 3, self._to_screen(h).y() - 3, 6, 6)
        p.setBrush(QColor("#f0a830"))
        p.setPen(Qt.NoPen)
        for q in pts:
            p.drawEllipse(self._to_screen(q), 4, 4)

    def _hit(self, pos):
        """(point index, part) under the mouse: handles first (they may sit on their point), then points."""
        for i, q in enumerate(self.points):
            if len(q) == 6:
                for part, h in ((1, q[2:4]), (2, q[4:6])):
                    if h != q[:2] and (self._to_screen(h) - pos).manhattanLength() < 8:
                        return i, part
        for i, q in enumerate(self.points):
            if (self._to_screen(q) - pos).manhattanLength() < 9:
                return i, 0
        return None

    def mousePressEvent(self, e):
        hit = self._hit(e.position())
        if e.button() == Qt.RightButton and hit is not None and hit[1] == 0 and len(self.points) > 2:
            del self.points[hit[0]]
            self.update()
            self.edited.emit(self._sorted(), True)
        elif e.button() == Qt.LeftButton:
            self._drag = hit

    def mouseDoubleClickEvent(self, e):
        self.points.append(self._to_curve(e.position()))
        self.update()
        self.edited.emit(self._sorted(), True)

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        i, part = self._drag
        q, at = self.points[i], self._to_curve(e.position())
        if part == 0:  # the point, its handles moving with it
            dx, dy = at[0] - q[0], at[1] - q[1]
            self.points[i] = at + ([q[2] + dx, q[3] + dy, q[4] + dx, q[5] + dy] if len(q) == 6 else [])
        else:
            q[2 * part:2 * part + 2] = at
        self.update()
        self.edited.emit(self._sorted(), False)

    def mouseReleaseEvent(self, e):
        if self._drag is not None:
            self._drag = None
            self.edited.emit(self._sorted(), True)
