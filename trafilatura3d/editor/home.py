"""The Home screen (ours, Unreal Engine style: a tile per kind of document, the recent files with their kind) and
the New graph dialog: it picks a template, a name, a size and the package to create the graph
in."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import (QComboBox, QDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QToolButton, QVBoxLayout, QWidget)

from trafilatura3d import shaders
from trafilatura3d.editor import theme
from trafilatura3d.editor.icons import icon
from trafilatura3d.package import TEMPLATES

SIZES = [str(1 << k) for k in range(4, 14)]


def template_icon(name: str) -> QIcon:
    pm = QPixmap(96, 96)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor("#3a3a3a"))
    p.setPen(QColor("#555"))
    p.drawRoundedRect(4, 4, 88, 88, 8, 8)
    t = shaders.TEMPLATES.get(name)  # a chip per output at the channel's neutral value (Empty: none)
    for i, usage in enumerate(t.channels if t else ()):
        r, g, b, _ = shaders.neutral(t.shader, usage)
        p.setBrush(QColor.fromRgbF(r, g, b))
        p.drawRect(12 + (i % 6) * 12, 14 + (i // 6) * 17, 10, 14)
    p.end()
    return QIcon(pm)


class NewGraphDialog(QDialog):
    def __init__(self, packages: list, names: list[str], parent=None, preselect=None):
        super().__init__(parent)
        self.setWindowTitle("New graph")
        self.resize(980, 560)
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Templates</b>"))
        self.templates = QListWidget()
        self.templates.setViewMode(QListWidget.IconMode)
        self.templates.setIconSize(QSize(96, 96))
        self.templates.setGridSize(QSize(150, 140))
        self.templates.setMovement(QListWidget.Static)
        self.templates.setWordWrap(True)
        for name in TEMPLATES:
            it = QListWidgetItem(template_icon(name), name)
            self.templates.addItem(it)
        self.templates.setCurrentRow(TEMPLATES.index(shaders.DEFAULT_TEMPLATE))
        left.addWidget(self.templates)
        lay.addLayout(left, 2)
        right = QVBoxLayout()
        form = QFormLayout()
        self.name = QLineEdit("graph")
        form.addRow("Graph name", self.name)
        size_row = QHBoxLayout()
        self.w, self.h = QComboBox(), QComboBox()
        for c in (self.w, self.h):
            c.addItems(SIZES)
            c.setCurrentText("2048")
        self.link = QToolButton()
        self.link.setIcon(icon("chain"))
        self.link.setCheckable(True)
        self.link.setChecked(True)
        self.w.currentTextChanged.connect(lambda t: self.link.isChecked() and self.h.setCurrentText(t))
        self.h.currentTextChanged.connect(lambda t: self.link.isChecked() and self.w.setCurrentText(t))
        size_row.addWidget(self.w)
        size_row.addWidget(self.link)
        size_row.addWidget(self.h)
        form.addRow("Size", size_row)
        self.target = QComboBox()
        self.target.addItem("New package")
        self.packages = packages
        for n in names:
            self.target.addItem(n)
        if preselect is not None and preselect in packages:
            self.target.setCurrentIndex(packages.index(preselect) + 1)
        form.addRow("Create graph in", self.target)
        right.addLayout(form)
        right.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel, create = QPushButton("Cancel"), QPushButton("Create")
        create.setDefault(True)
        create.setStyleSheet(f"background: {theme.ACCENT}; color: white;")
        cancel.clicked.connect(self.reject)
        create.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(create)
        right.addLayout(buttons)
        lay.addLayout(right, 1)

    def result_values(self):
        """(graph id, (w log2, h log2), package or None for a new one, template name)."""
        size = (int(self.w.currentText()).bit_length() - 1, int(self.h.currentText()).bit_length() - 1)
        i = self.target.currentIndex()
        pkg = None if i == 0 else self.packages[i - 1]
        name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in self.name.text()) or "graph"
        return name, size, pkg, self.templates.currentItem().text() if self.templates.currentItem() else "Empty"


GRAPH_COLOR = "#f0a830"  # the two kinds of documents in the Home (ours)
PAINT_COLOR = "#4fc3a1"


def _wire(p: QPainter, a: QPointF, b: QPointF):
    path = QPainterPath(a)
    dx = (b.x() - a.x()) * 0.5
    path.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y()), b)
    p.drawPath(path)


def _frame(p: QPainter, r: QRectF):
    """The pictures' 100 x 70 frame fitted in `r`, centered, the same scale on both axes."""
    s = min(r.width() / 100, r.height() / 70)
    p.translate(r.center().x() - 50 * s, r.center().y() - 35 * s)
    p.scale(s, s)


def draw_graph(p: QPainter, r: QRectF, small: bool = False):
    """A node graph: nodes with a colored header, wired output to input (a 100 x 70 frame fitted in `r`)."""
    p.save()
    _frame(p, r)
    nodes = [(8, 10), (8, 42), (40, 24), (72, 24)] if not small else [(4, 8), (4, 40), (60, 22)]
    w, h = (20, 18) if not small else (34, 24)
    wire = QPen(QColor(GRAPH_COLOR), 1.6 if not small else 4)
    p.setPen(wire)
    p.setBrush(Qt.NoBrush)
    links = [(0, 2), (1, 2), (2, 3)] if not small else [(0, 2), (1, 2)]
    for a, b in links:
        (ax, ay), (bx, by) = nodes[a], nodes[b]
        _wire(p, QPointF(ax + w, ay + h * 0.6), QPointF(bx, by + h * 0.6))
    for i, (x, y) in enumerate(nodes):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#3a3a3a"))
        p.drawRoundedRect(QRectF(x, y, w, h), 2.5, 2.5)
        p.setBrush(QColor(GRAPH_COLOR if i == len(nodes) - 1 else "#6d8fb3"))
        p.drawRoundedRect(QRectF(x, y, w, h * 0.28), 2.5, 2.5)
        if not small:
            p.setBrush(QColor("#5a5a5a"))
            p.drawRect(QRectF(x + 3, y + h * 0.4, w - 6, h * 0.45))
    p.restore()


def draw_painting(p: QPainter, r: QRectF, small: bool = False):
    """A sphere with brush strokes and the brush's circle (a 100 x 70 frame fitted in `r`)."""
    p.save()
    _frame(p, r)
    c, rad = QPointF(50, 35), 30 if not small else 33
    g = QRadialGradient(QPointF(c.x() - rad * 0.35, c.y() - rad * 0.4), rad * 1.4)
    g.setColorAt(0, QColor("#d8d8d8"))
    g.setColorAt(1, QColor("#4a4a4a"))
    p.setPen(Qt.NoPen)
    p.setBrush(g)
    p.drawEllipse(c, rad, rad)
    clip = QPainterPath()
    clip.addEllipse(c, rad, rad)
    p.setClipPath(clip)
    for color, pts, width in ((PAINT_COLOR, [(22, 30), (40, 18), (58, 34), (80, 26)], 7 if not small else 12),
                              ("#e0655a", [(26, 50), (46, 44), (66, 54), (84, 46)], 5 if not small else 0)):
        if not width:
            continue
        pen = QPen(QColor(color), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        path = QPainterPath(QPointF(*pts[0]))
        path.cubicTo(QPointF(*pts[1]), QPointF(*pts[2]), QPointF(*pts[3]))
        p.drawPath(path)
    p.setClipping(False)
    if not small:
        p.setPen(QPen(QColor("white"), 1.0, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(72, 22), 8, 8)
    p.restore()


def kind_icon(painting: bool, size: int = 22) -> QPixmap:
    """The small picture telling a recent file's kind at a glance."""
    ratio = 2
    pm = QPixmap(size * ratio, size * ratio)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    frame = QRectF(0, size * ratio * 0.15, size * ratio, size * ratio * 0.7)
    (draw_painting if painting else draw_graph)(p, frame, small=True)
    p.end()
    pm.setDevicePixelRatio(ratio)
    return pm


class TemplateTile(QWidget):
    """A big tile of the Home: a picture, a title and a line; clicking it creates a document of its kind."""
    clicked = Signal()

    def __init__(self, title: str, line: str, draw, color: str):
        super().__init__()
        self.title, self.line, self.draw, self.color = title, line, draw, color
        self.setFixedSize(320, 230)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover)
        self.setToolTip(line)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        hover = self.underMouse()
        p.setPen(QPen(QColor(self.color if hover else "#3a3a3a"), 1.5))
        p.setBrush(QColor("#2a2a2a" if hover else "#262626"))
        p.drawRoundedRect(r, 8, 8)
        art = QRectF(r.left() + 1, r.top() + 1, r.width() - 2, 160)
        clip = QPainterPath()
        clip.addRoundedRect(art, 8, 8)
        p.setClipPath(clip)
        p.fillRect(art, QColor("#1c1c1c"))
        p.setClipping(False)
        self.draw(p, art.adjusted(30, 14, -30, -14))
        p.setPen(QColor("#e6e6e6"))
        f = QFont(self.font())
        f.setPointSizeF(f.pointSizeF() * 1.25)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(r.left() + 16, art.bottom() + 12, r.width() - 32, 24), Qt.AlignLeft | Qt.AlignVCenter,
                   self.title)
        p.setFont(self.font())
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(QRectF(r.left() + 16, art.bottom() + 36, r.width() - 32, 20), Qt.AlignLeft | Qt.AlignVCenter,
                   self.line)
        p.end()

    def enterEvent(self, e):
        self.update()

    def leaveEvent(self, e):
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()


class RecentRow(QWidget):
    """A recent file: its kind's icon, the name, a thin separator, the folder, the date it was last modified."""

    def __init__(self, path: Path):
        super().__init__()
        from trafilatura3d.paint.model import PROJECT_SUFFIX
        painting = path.suffix == PROJECT_SUFFIX
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 12, 4)
        lay.setSpacing(10)
        self.icon = QLabel()
        self.icon.setPixmap(kind_icon(painting))
        self.icon.setToolTip("Trafilatura 3D Paint project" if painting else "Trafilatura 3D Graph package")
        lay.addWidget(self.icon)
        name = QLabel(path.stem)
        name.setStyleSheet("font-weight: bold; background: transparent;")
        lay.addWidget(name)
        sep = QFrame()
        sep.setFixedSize(1, 14)
        sep.setStyleSheet("background: #3c3c3c;")
        lay.addWidget(sep)
        folder = QLabel(str(path.parent))
        folder.setStyleSheet(f"color: {theme.TEXT_DIM}; background: transparent;")
        folder.setToolTip(str(path))
        folder.setMinimumWidth(40)
        lay.addWidget(folder, 1)
        when = QLabel(datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y  %H:%M"))
        when.setStyleSheet(f"color: {theme.TEXT_DIM}; background: transparent;")
        lay.addWidget(when)


class HomeScreen(QWidget):
    """The Home (ours, after Unreal Engine's project browser): two big tiles create a
    texture graph or a texture painting project, each with an Open link below; the recent files with their kind's
    icon, folder and date (double click opens)."""
    new_graph = Signal()
    open_package = Signal()
    new_project = Signal()
    open_project = Signal()
    open_recent = Signal(str)
    close_home = Signal()

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background: #1f1f1f;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(48, 28, 48, 24)
        top = QHBoxLayout()
        logo = QLabel("Tr")
        logo.setFixedSize(40, 40)
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("background: #203a20; color: #8fe08f; font-size: 20px; font-weight: bold; "
                           "border-radius: 6px;")
        top.addWidget(logo)
        top.addSpacing(12)
        titles = QVBoxLayout()
        title = QLabel("Welcome to Trafilatura 3D")
        f = QFont()
        f.setPointSize(20)
        f.setBold(True)
        title.setFont(f)
        titles.addWidget(title)
        sub = QLabel("Procedural texture graphs and texture painting on the GPU.")
        sub.setObjectName("dim")
        titles.addWidget(sub)
        top.addLayout(titles)
        top.addStretch(1)
        b_close = QPushButton("Close home screen")
        b_close.setStyleSheet("border-radius: 14px; padding: 6px 14px;")
        b_close.clicked.connect(self.close_home)
        top.addWidget(b_close, 0, Qt.AlignTop)
        lay.addLayout(top)
        lay.addSpacing(28)
        tiles = QHBoxLayout()
        tiles.setSpacing(28)
        self.tiles = []
        for title_, line, draw, color, new_sig, open_text, open_sig in (
                ("Trafilatura 3D Graph", "Procedural textures from node graphs", draw_graph, GRAPH_COLOR, self.new_graph,
                 "Open graph package…", self.open_package),
                ("Trafilatura 3D Paint", "Paint a mesh's textures with layers", draw_painting, PAINT_COLOR,
                 self.new_project, "Open paint project…", self.open_project)):
            col = QVBoxLayout()
            col.setSpacing(6)
            tile = TemplateTile(title_, line, draw, color)
            tile.clicked.connect(new_sig)
            self.tiles.append(tile)
            col.addWidget(tile)
            link = QPushButton(open_text)
            link.setCursor(Qt.PointingHandCursor)
            link.setStyleSheet(f"background: transparent; border: none; color: {theme.TEXT_DIM}; text-align: left; "
                               "padding: 2px 4px;")
            link.clicked.connect(open_sig)
            col.addWidget(link, 0, Qt.AlignLeft)
            tiles.addLayout(col)
        tiles.addStretch(1)
        lay.addLayout(tiles)
        lay.addSpacing(22)
        rec = QLabel("Recent")
        f2 = QFont()
        f2.setPointSize(13)
        f2.setBold(True)
        rec.setFont(f2)
        lay.addWidget(rec)
        self.recent = QListWidget()
        self.recent.setStyleSheet("QListWidget { background: #232323; border: 1px solid #2c2c2c; border-radius: 6px; }"
                                  "QListWidget::item { border-bottom: 1px solid #292929; }"
                                  "QListWidget::item:hover { background: #2a2a2a; }")
        self.recent.itemDoubleClicked.connect(lambda it: self.open_recent.emit(it.data(Qt.UserRole)))
        lay.addWidget(self.recent, 1)
        self.refresh()

    def refresh(self):
        self.recent.clear()
        for path in QSettings().value("recent", [], type=list):
            p = Path(path)
            if not p.exists():
                continue
            it = QListWidgetItem()
            it.setData(Qt.UserRole, str(p))
            row = RecentRow(p)
            it.setSizeHint(QSize(0, 34))
            self.recent.addItem(it)
            self.recent.setItemWidget(it, row)
