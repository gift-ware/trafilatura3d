"""The Paint color picker: the window a color field opens, floating, not dockable,
semi-persistent (it closes when the context changes: another layer or effect selected, another tool). From top:
the spectrum (saturation/value field with a crosshair, vertical hue slider), current and previous color (click the
previous to restore it), hexadecimal field, eyedropper; the sliders (RGB or HSV, with the "..." menu for Dynamic
sliders and Floating point values); the swatches (Add swatch, click loads, hover shows the hex value, right click
deletes, "..." > Delete all swatches), kept across sessions as sRGB in `<user dir>/swatches.json`.

The eyedropper picks anywhere on screen: every screen is captured and covered by an overlay showing the capture
(so the click reaches us over other applications too), with a magnified preview of the pixels around the pointer,
the color and its hex value; Shift over the 3D view takes the edited channel's value on the mesh (no display
conversion); Esc or a right click cancels. macOS needs the Screen Recording permission to capture the other
applications: the overlay says so when it is missing.
Not offered, there is no color management: the Display selector, the Eyedropper color space, the working color
space readout. Defaults the page does not give (ours, from its screenshot): RGB sliders, dynamic, 0-255."""

from __future__ import annotations

import colorsys
import ctypes
import json
import sys

from PySide6.QtCore import QPointF, QRectF, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMenu,
                               QPushButton, QToolButton, QVBoxLayout, QWidget)
from shiboken6 import isValid

from trafilatura3d.editor.paint.icons import picon
from trafilatura3d.editor.paint.widgets import Slider

PREFS = "paint_color_picker"
HUE_STOPS = [(0.0, "#ff0000"), (1 / 6, "#ffff00"), (2 / 6, "#00ff00"), (3 / 6, "#00ffff"), (4 / 6, "#0000ff"),
             (5 / 6, "#ff00ff"), (1.0, "#ff0000")]


def _hex(rgb) -> str:
    return "".join(f"{round(min(max(c, 0.0), 1.0) * 255):02X}" for c in rgb[:3])


def _qcolor(rgb) -> QColor:
    return QColor.fromRgbF(*[min(max(float(c), 0.0), 1.0) for c in rgb[:3]])


def _swatches_file():
    from trafilatura3d.project import user_dir
    return user_dir() / "swatches.json"


def load_swatches() -> list:
    try:
        return [list(map(float, c))[:3] for c in json.loads(_swatches_file().read_text(encoding="utf-8"))]
    except (OSError, ValueError, TypeError):
        return []


def save_swatches(colors) -> None:
    f = _swatches_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps([list(c) for c in colors]), encoding="utf-8")


def _more_button() -> QToolButton:
    """The sections' "..." menu button."""
    b = QToolButton()
    b.setText("…")
    b.setAutoRaise(True)
    b.setPopupMode(QToolButton.InstantPopup)
    b.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0; }")
    return b


class Spectrum(QWidget):
    """Saturation (x) and value (y) of the current hue, the crosshair on the color. `moved(s, v, final)`."""
    moved = Signal(float, float, bool)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        self.hue, self.sat, self.val = 0.0, 0.0, 0.0

    def paintEvent(self, _e):
        p = QPainter(self)
        r = QRectF(self.rect())
        g = QLinearGradient(r.topLeft(), r.topRight())
        g.setColorAt(0, QColor("white"))
        g.setColorAt(1, QColor.fromHsvF(self.hue, 1.0, 1.0))
        p.fillRect(r, g)
        g = QLinearGradient(r.topLeft(), r.bottomLeft())
        g.setColorAt(0, QColor(0, 0, 0, 0))
        g.setColorAt(1, QColor(0, 0, 0, 255))
        p.fillRect(r, g)
        x, y = self.sat * r.width(), (1 - self.val) * r.height()
        p.setPen(QPen(QColor("black" if self.val > 0.5 else "white"), 1.5))
        p.drawLine(QPointF(x - 8, y), QPointF(x + 8, y))
        p.drawLine(QPointF(x, y - 8), QPointF(x, y + 8))
        p.end()

    def _at(self, e, final):
        s = min(max(e.position().x() / max(self.width(), 1), 0.0), 1.0)
        v = 1 - min(max(e.position().y() / max(self.height(), 1), 0.0), 1.0)
        self.moved.emit(s, v, final)

    def mousePressEvent(self, e):
        self._at(e, False)

    def mouseMoveEvent(self, e):
        if e.buttons():
            self._at(e, False)

    def mouseReleaseEvent(self, e):
        self._at(e, True)


class HueBar(QWidget):
    """The vertical hue slider (red at the bottom, as the page's screenshot). `moved(hue, final)`."""
    moved = Signal(float, bool)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(14)
        self.hue = 0.0

    def paintEvent(self, _e):
        p = QPainter(self)
        r = QRectF(self.rect()).adjusted(3, 0, -3, 0)
        g = QLinearGradient(r.bottomLeft(), r.topLeft())
        for t, c in HUE_STOPS:
            g.setColorAt(t, QColor(c))
        p.fillRect(r, g)
        y = (1 - self.hue) * self.height()
        p.setPen(QPen(QColor("white"), 1.5))
        p.setBrush(QColor.fromHsvF(self.hue % 1.0, 1.0, 1.0))
        p.drawEllipse(QPointF(self.width() / 2, min(max(y, 5), self.height() - 5)), 5, 5)
        p.end()

    def _at(self, e, final):
        self.moved.emit(1 - min(max(e.position().y() / max(self.height(), 1), 0.0), 1.0), final)

    def mousePressEvent(self, e):
        self._at(e, False)

    def mouseMoveEvent(self, e):
        if e.buttons():
            self._at(e, False)

    def mouseReleaseEvent(self, e):
        self._at(e, True)


class ColorPair(QWidget):
    """Current (left) and previous color (right, click restores it)."""
    restore = Signal()

    def __init__(self):
        super().__init__()
        self.setFixedSize(72, 22)
        self.current, self.previous = [0, 0, 0], [0, 0, 0]

    def paintEvent(self, _e):
        p = QPainter(self)
        w = self.width() / 2
        p.fillRect(QRectF(0, 0, w, self.height()), _qcolor(self.current))
        p.fillRect(QRectF(w, 0, w, self.height()), _qcolor(self.previous))
        p.end()

    def mousePressEvent(self, e):
        if e.position().x() > self.width() / 2:
            self.restore.emit()


class ColorPicker(QWidget):
    """The picker window. `open(title, rgb, on_edit, channel)`: `on_edit(rgb, final)` receives every change;
    `channel`: what the field edits (the eyedropper's Shift pick over the 3D view)."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.resize(280, 520)
        self.rgb, self.previous = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        self.hsv = [0.0, 0.0, 0.0]  # kept apart: the hue of a gray is not lost
        self.on_edit = None
        s = QSettings()
        self.mode = s.value(f"{PREFS}/mode", "RGB")
        self.dynamic = s.value(f"{PREFS}/dynamic", True) in (True, "true")
        self.floating = s.value(f"{PREFS}/floating", False) in (True, "true")
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.spectrum = Spectrum()
        self.spectrum.moved.connect(lambda sat, val, final: self._set_hsv([self.hsv[0], sat, val], final))
        self.hue_bar = HueBar()
        self.hue_bar.moved.connect(lambda h, final: self._set_hsv([h, self.hsv[1], self.hsv[2]], final))
        top.addWidget(self.spectrum, 1)
        top.addWidget(self.hue_bar)
        lay.addLayout(top, 1)
        row = QHBoxLayout()
        self.pair = ColorPair()
        self.pair.restore.connect(lambda: self.set_rgb(self.previous, True))
        row.addWidget(self.pair)
        row.addStretch(1)
        row.addWidget(QLabel("#"))
        self.hex = QLineEdit()
        self.hex.setFixedWidth(72)
        self.hex.setMaxLength(7)
        self.hex.editingFinished.connect(self._hex_edited)
        row.addWidget(self.hex)
        self.dropper = DropperButton()
        self.dropper.picked.connect(lambda rgb: self.set_rgb(rgb, True))
        row.addWidget(self.dropper)
        lay.addLayout(row)
        head = QHBoxLayout()
        self.mode_box = QComboBox()
        self.mode_box.addItems(["RGB", "HSV"])
        self.mode_box.setCurrentText(self.mode)
        self.mode_box.currentTextChanged.connect(self._mode_changed)
        head.addWidget(self.mode_box)
        head.addStretch(1)
        more = _more_button()
        menu = QMenu(more)
        self.dynamic_action = menu.addAction("Dynamic sliders")
        self.dynamic_action.setCheckable(True)
        self.dynamic_action.setChecked(self.dynamic)
        self.dynamic_action.toggled.connect(lambda on: self._option("dynamic", on))
        self.floating_action = menu.addAction("Floating point values")
        self.floating_action.setCheckable(True)
        self.floating_action.setChecked(self.floating)
        self.floating_action.toggled.connect(lambda on: self._option("floating", on))
        more.setMenu(menu)
        head.addWidget(more)
        lay.addLayout(head)
        grid = QGridLayout()
        self.sliders, self.fields, self.letters = [], [], []
        for i in range(3):
            letter = QLabel()
            s = Slider()
            s.setRange(0, 1000)
            s.valueChanged.connect(lambda v, i=i: self._slid(i, v / 1000, False))
            s.sliderReleased.connect(lambda i=i: self._slid(i, self.sliders[i].value() / 1000, True))
            f = QLineEdit()
            f.setFixedWidth(52)
            f.editingFinished.connect(lambda i=i: self._field(i))
            grid.addWidget(letter, i, 0)
            grid.addWidget(s, i, 1)
            grid.addWidget(f, i, 2)
            self.letters.append(letter)
            self.sliders.append(s)
            self.fields.append(f)
        lay.addLayout(grid)
        sw_head = QHBoxLayout()
        sw_head.addWidget(QLabel("<b>Swatches</b>"))
        sw_head.addStretch(1)
        sw_more = _more_button()
        sw_menu = QMenu(sw_more)
        sw_menu.addAction("Delete all swatches", lambda: self._set_swatches([]))
        sw_more.setMenu(sw_menu)
        sw_head.addWidget(sw_more)
        lay.addLayout(sw_head)
        self.swatch_grid = QGridLayout()
        self.swatch_grid.setSpacing(4)
        lay.addLayout(self.swatch_grid)
        self.swatches = load_swatches()
        self._fill_swatches()

    # --- opening and closing ------------------------------------------------------------------------------

    def open(self, title: str, rgb, on_edit, channel=None, parent=None):
        if parent is not None and self.parentWidget() is not parent:  # above a modal dialog that opened it
            self.setParent(parent, Qt.Tool)
        self.setWindowTitle(f"{title} color")
        self.on_edit = on_edit
        self.dropper.channel = channel
        self.previous = [float(c) for c in rgb[:3]]
        self.set_rgb(self.previous, None)
        if not self.isVisible():
            pos = QCursor.pos()
            self.move(pos.x() + 16, max(pos.y() - 120, 0))
        self.show()
        self.raise_()

    def close_context(self):
        """The context changed (another layer, effect or tool): the picker closes (semi-persistent)."""
        self.on_edit = None
        self.hide()

    # --- the color -----------------------------------------------------------------------------------------

    def set_rgb(self, rgb, final):
        """A new color from outside the spectrum (None: no edit sent)."""
        self.rgb = [float(c) for c in rgb[:3]]
        h, s, v = colorsys.rgb_to_hsv(*[min(max(c, 0.0), 1.0) for c in self.rgb])
        self.hsv = [self.hsv[0] if s == 0 or v == 0 else h, s, v]
        self._show()
        if final is not None:
            self._send(final)

    def _set_hsv(self, hsv, final):
        self.hsv = [float(x) for x in hsv]
        self.rgb = list(colorsys.hsv_to_rgb(self.hsv[0] % 1.0, *self.hsv[1:]))
        self._show()
        self._send(final)

    def _send(self, final):
        if self.on_edit is None:
            return
        try:
            self.on_edit(list(self.rgb), final)
        except RuntimeError:  # the color field is gone: the context changed
            self.close_context()

    def _components(self):
        return self.rgb if self.mode == "RGB" else self.hsv

    def _show(self):
        self.spectrum.hue, self.spectrum.sat, self.spectrum.val = self.hsv
        self.hue_bar.hue = self.hsv[0]
        self.pair.current, self.pair.previous = self.rgb, self.previous
        self.hex.setText(_hex(self.rgb))
        letters = "RGB" if self.mode == "RGB" else "HSV"
        comps = self._components()
        for i, (s, f) in enumerate(zip(self.sliders, self.fields)):
            self.letters[i].setText(letters[i])
            s.blockSignals(True)
            s.setValue(round(min(max(comps[i], 0.0), 1.0) * 1000))
            s.blockSignals(False)
            f.setText(self._format(i, comps[i]))
            s.setStyleSheet(self._groove(i) if self.dynamic else "")
        for w in (self.spectrum, self.hue_bar, self.pair):
            w.update()

    def _format(self, i, x):
        if self.floating:
            return f"{x:.3f}"
        if self.mode == "HSV":
            return str(round(x * 360)) if i == 0 else str(round(x * 100))  # degrees, percent
        return str(round(x * 255))

    def _parse(self, i, text):
        x = float(text)
        if self.floating:
            return x
        if self.mode == "HSV":
            return x / 360 if i == 0 else x / 100
        return x / 255

    def _groove(self, i):
        """Dynamic sliders: the groove shows the colors the slider reaches from the current color."""
        stops = []
        for t in (0.0, 0.25, 0.5, 0.75, 1.0) if not (self.mode == "HSV" and i == 0) else [k / 6 for k in range(7)]:
            c = list(self._components())
            c[i] = t
            rgb = c if self.mode == "RGB" else colorsys.hsv_to_rgb(c[0] % 1.0, c[1], c[2])
            stops.append(f"stop:{t:.3f} {_qcolor(rgb).name()}")
        return ("QSlider::groove:horizontal { height: 6px; border-radius: 3px; background: "
                f"qlineargradient(x1:0, y1:0, x2:1, y2:0, {', '.join(stops)}); }}")

    def _slid(self, i, x, final):
        c = list(self._components())
        c[i] = x
        if self.mode == "RGB":
            self.set_rgb(c, final)
        else:
            self._set_hsv(c, final)

    def _field(self, i):
        try:
            x = self._parse(i, self.fields[i].text())
        except ValueError:
            self._show()
            return
        self._slid(i, x, True)

    def _hex_edited(self):
        t = self.hex.text().strip().lstrip("#")
        if len(t) != 6:
            self._show()
            return
        try:
            self.set_rgb([int(t[k:k + 2], 16) / 255 for k in (0, 2, 4)], True)
        except ValueError:
            self._show()

    def _mode_changed(self, mode):
        self.mode = mode
        QSettings().setValue(f"{PREFS}/mode", mode)
        self._show()

    def _option(self, name, on):
        setattr(self, name, on)
        QSettings().setValue(f"{PREFS}/{name}", on)
        self._show()

    # --- swatches ------------------------------------------------------------------------------------------

    def add_swatch(self):
        """Add swatch: skipped when the newest swatch already holds the current color."""
        if self.swatches and _hex(self.swatches[-1]) == _hex(self.rgb):
            return
        self._set_swatches(self.swatches + [list(self.rgb)])

    def _set_swatches(self, colors):
        self.swatches = [list(c) for c in colors]
        save_swatches(self.swatches)
        self._fill_swatches()

    def _fill_swatches(self):
        while self.swatch_grid.count():
            w = self.swatch_grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        add = QPushButton("+")
        add.setFixedSize(24, 24)
        add.setStyleSheet("padding: 0; font-size: 15px;")
        add.setToolTip("Add swatch")
        add.clicked.connect(self.add_swatch)
        self.swatch_grid.addWidget(add, 0, 0)
        for k, c in enumerate(self.swatches, start=1):
            b = QPushButton()
            b.setFixedSize(24, 24)
            b.setToolTip(f"#{_hex(c)}")
            b.setStyleSheet(f"background: {_qcolor(c).name()}; border: 1px solid #111;")
            b.clicked.connect(lambda _=False, c=c: self.set_rgb(c, True))
            b.setContextMenuPolicy(Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _p, k=k - 1, b=b: self._swatch_menu(k, b))
            self.swatch_grid.addWidget(b, k // 8, k % 8)

    def _swatch_menu(self, k, button):
        m = QMenu(self)
        m.addAction("Delete", lambda: self._set_swatches(self.swatches[:k] + self.swatches[k + 1:]))
        m.exec(QCursor.pos())



# The workspace's answer to a pick over its views: sampler(global point, shift, channel) -> rgb or None.
SAMPLER = None


def set_sampler(fn) -> None:
    global SAMPLER
    SAMPLER = fn


def screen_capture_allowed() -> bool:
    """macOS: whether the Screen Recording permission lets a capture see the other applications (without it the
    capture shows only the desktop and our windows). The first denied check asks the system for it."""
    if sys.platform != "darwin":
        return True
    try:
        cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        cg.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
        cg.CGRequestScreenCaptureAccess.restype = ctypes.c_bool
        if cg.CGPreflightScreenCaptureAccess():
            return True
        global _ASKED
        if not _ASKED:
            _ASKED = True
            cg.CGRequestScreenCaptureAccess()
        return False
    except (OSError, AttributeError):
        return True


_ASKED = False
MAGNIFIER = 11  # pixels around the pointer in the preview (ours), each drawn ZOOM times larger
ZOOM = 10


class EyedropperOverlay(QWidget):
    """One screen while the eyedropper picks: its capture drawn over it (so clicks on other applications come
    here), the magnified pixels around the pointer with the color and its hex value next to it."""

    def __init__(self, session, screen):
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.session, self.screen_ = session, screen
        self.shot = screen.grabWindow(0)
        self.image = self.shot.toImage()
        self.setGeometry(screen.geometry())
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.pointer = None

    def color_at(self, local) -> QColor | None:
        ratio = self.image.width() / max(self.width(), 1)
        x, y = int(local.x() * ratio), int(local.y() * ratio)
        if not (0 <= x < self.image.width() and 0 <= y < self.image.height()):
            return None
        return self.image.pixelColor(x, y)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.drawPixmap(self.rect(), self.shot)
        if not self.session.allowed:
            msg = ("Screen Recording permission needed to pick other applications: System Settings > "
                   "Privacy & Security > Screen Recording. Esc cancels.")
            p.fillRect(QRectF(0, 0, self.width(), 28), QColor(30, 30, 30, 230))
            p.setPen(QColor("#f0a030"))
            p.drawText(QRectF(0, 0, self.width(), 28), Qt.AlignCenter, msg)
        if self.pointer is not None:
            self._magnifier(p, self.pointer)
        p.end()

    def _magnifier(self, p, pos):
        ratio = self.image.width() / max(self.width(), 1)
        cx, cy = int(pos.x() * ratio), int(pos.y() * ratio)
        half = MAGNIFIER // 2
        side = MAGNIFIER * ZOOM
        ox = pos.x() + 20 if pos.x() + 20 + side < self.width() else pos.x() - 20 - side
        oy = pos.y() + 20 if pos.y() + 20 + side + 24 < self.height() else pos.y() - 20 - side - 24
        p.fillRect(QRectF(ox - 2, oy - 2, side + 4, side + 28), QColor(25, 25, 25))
        for j in range(MAGNIFIER):
            for i in range(MAGNIFIER):
                x, y = cx + i - half, cy + j - half
                c = (self.image.pixelColor(x, y) if 0 <= x < self.image.width() and 0 <= y < self.image.height()
                     else QColor(0, 0, 0))
                p.fillRect(QRectF(ox + i * ZOOM, oy + j * ZOOM, ZOOM, ZOOM), c)
        p.setPen(QPen(QColor("white"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(ox + half * ZOOM, oy + half * ZOOM, ZOOM, ZOOM))
        c = self.color_at(pos)
        if c is not None:
            p.fillRect(QRectF(ox, oy + side + 4, 20, 18), c)
            p.setPen(QColor("#dddddd"))
            p.drawText(QRectF(ox + 26, oy + side + 4, side - 26, 18), Qt.AlignVCenter,
                       f"#{_hex([c.redF(), c.greenF(), c.blueF()])}")

    def mouseMoveEvent(self, e):
        self.session.moved(self, e.position())

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self.session.finish(None)
            return
        shift = bool(e.modifiers() & Qt.ShiftModifier)
        self.session.pick(self, e.position(), e.globalPosition().toPoint(), shift)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.session.finish(None)


class EyedropperSession:
    """A pick: an overlay on every screen until a click (the pixel of the capture; with Shift over the
    workspace's 3D view, the channel's value from SAMPLER) or Esc / right click (nothing)."""

    def __init__(self, channel, done):
        self.channel, self.done = channel, done
        self.allowed = screen_capture_allowed()
        self.overlays = [EyedropperOverlay(self, s) for s in QGuiApplication.screens()]
        for o in self.overlays:
            o.show()
            o.raise_()
            o.activateWindow()
        under = QGuiApplication.screenAt(QCursor.pos())
        for o in self.overlays:
            if o.screen_ is under:
                o.setFocus()

    def moved(self, overlay, pos):
        for o in self.overlays:
            if o.pointer is not None and o is not overlay:
                o.pointer = None
                o.update()
        overlay.pointer = pos
        overlay.update()

    def pick(self, overlay, local, global_pos, shift):
        rgb = SAMPLER(global_pos, True, self.channel) if shift and SAMPLER is not None else None
        if rgb is None:
            c = overlay.color_at(local)
            rgb = None if c is None else [c.redF(), c.greenF(), c.blueF()]
        self.finish(rgb)

    def finish(self, rgb):
        for o in self.overlays:
            o.close()
            o.deleteLater()
        self.overlays = []
        global _SESSION
        _SESSION = None
        if rgb is not None:
            self.done([float(c) for c in rgb[:3]])


_SESSION: EyedropperSession | None = None


def start_eyedropper(done, channel=None) -> EyedropperSession:
    """Start a pick; `done(rgb)` receives the color (not called when cancelled)."""
    global _SESSION
    if _SESSION is not None:
        _SESSION.finish(None)
    _SESSION = EyedropperSession(channel, done)
    return _SESSION


class DropperButton(QToolButton):
    """An eyedropper (the picker's, and every color field's): click the icon, move (a magnified preview follows
    the pointer, anywhere on screen), click again; Shift over the 3D view takes the channel's value, Esc or a
    right click cancels. `picked(rgb)`."""
    picked = Signal(list)

    def __init__(self, channel=None):
        super().__init__()
        self.channel = channel
        self.setIcon(picon("picker"))
        self.setToolTip("Eyedropper: click, move, click again (Shift over the 3D view: the channel's value; "
                        "Esc cancels)")
        self.setAutoRaise(True)
        self.clicked.connect(lambda: start_eyedropper(self._done, self.channel))

    def _done(self, rgb):
        if isValid(self):
            self.picked.emit(rgb)


_PICKER: ColorPicker | None = None


def picker() -> ColorPicker:
    """The one picker window of the application."""
    global _PICKER
    if _PICKER is None or not isValid(_PICKER):  # its window may have been closed with the one it lived in
        _PICKER = ColorPicker(QApplication.activeWindow())
    return _PICKER


def close_context() -> None:
    if _PICKER is not None and isValid(_PICKER):
        _PICKER.close_context()
