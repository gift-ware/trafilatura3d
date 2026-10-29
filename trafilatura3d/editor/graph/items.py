"""Graph view items: a node is a colored title bar over its thumbnail, with
round connectors on the sides (inputs left, outputs right) and its size/format written below;
a Dot node is a small disc on its link; links are curves (or right-angled), orange for color, gray for
grayscale."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSizeF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem

from trafilatura3d.editor import theme
from trafilatura3d.graphtools import (DOT, MARGIN, PORT_R, THUMB, TITLE, link_controls, node_size,  # the geometry
                                      port_y)  # (the layout uses it too)
from trafilatura3d.nodes.base import PortType

# Title bar colors by library category (first match wins).
CATEGORY_COLORS = [
    ("Texture generators", "#9b1d1d"), ("Filters/Adjustments", "#3c7a3c"), ("Filters/Blending", "#6f6f6f"),
    ("Filters/Blurs", "#2f5f8f"), ("Filters/Channels", "#7a4c9a"), ("Filters/Effects", "#2f6f8f"),
    ("Filters/Normal Map", "#5a4aa0"), ("Filters/Tiling", "#8a6a2a"), ("Filters/Transforms", "#3a4fa0"),
    ("Material filters", "#8a5a1c"), ("Mesh based generators", "#6f6f2a"), ("3D View", "#2a6f6f"),
    ("Graph instances", "#8a1f3a"),
]
ATOMIC_COLORS = {"output": "#3d7fc4", "input_color": "#3d7fc4", "input_grayscale": "#3d7fc4",
                 "blend": "#7a7a7a", "levels": "#3c7a3c", "curve": "#3c7a3c", "hsl": "#3c7a3c",
                 "gradient_map": "#3c7a3c", "grayscale_conversion": "#7a4c9a", "channel_shuffle": "#7a4c9a",
                 "transformation_2d": "#3a4fa0", "blur": "#2f5f8f", "directional_blur": "#2f5f8f",
                 "uniform_color": "#9b1d1d", "bitmap": "#9b1d1d", "text": "#9b1d1d", "svg": "#9b1d1d",
                 "normal": "#5a4aa0", "warp": "#2f6f8f", "directional_warp": "#2f6f8f", "emboss": "#2f6f8f",
                 "sharpen": "#2f6f8f", "distance": "#2f6f8f", "gradient_dynamic": "#3c7a3c"}


def title_color(node_type: str, category: str) -> QColor:
    if node_type in ATOMIC_COLORS:
        return QColor(ATOMIC_COLORS[node_type])
    for prefix, color in CATEGORY_COLORS:
        if category.startswith(prefix):
            return QColor(color)
    return QColor("#5a5a5a")


# Function graph value types (connectors of function nodes).
VALUE_COLORS = {"float": "#cfcfcf", "float2": "#8fd16a", "float3": "#e0a050", "float4": "#e06a6a",
                "int": "#6aa8e0", "int2": "#5fc7c7", "int3": "#7f8fe0", "int4": "#b07ae0", "bool": "#e0d050",
                "string": "#e07ab8"}


def port_color(t) -> QColor:
    if isinstance(t, str):
        return QColor(VALUE_COLORS.get(t, "#e6e6e6"))
    if t is PortType.COLOR:
        return QColor(theme.COLOR_LINK)
    if t is PortType.VALUE:
        return QColor("#b07ae0")
    if t is PortType.GRAY:
        return QColor(theme.GRAY_LINK)
    return QColor("#e6e6e6")


def text_scale(p: QPainter, limit: bool) -> float:
    """"Limit text scaling when zooming out": below a zoom threshold text keeps its size on
    screen instead of shrinking with the graph."""
    zoom = p.worldTransform().m11()
    return max(1.0, LIMIT_ZOOM / zoom) if limit and zoom > 0 else 1.0


LIMIT_ZOOM = 0.6
NAME_ZOOM = 1.2  # connector names appear from this zoom in
NAME_W = 110.0
NODE_R = 8.0  # the rounded corners of a node
THUMB_R = 5.0  # and of its thumbnail


MISSING = QColor("#a03030")  # missing nodes and the Disabled / Missing badges
# Display timings: a node's cook time colored by these upper limits (milliseconds).
TIMING_COLORS = ((10.0, "#6cc46c"), (50.0, "#e0cf4a"), (100.0, "#e8913a"), (math.inf, "#e05050"))


class PortItem(QGraphicsItem):
    def __init__(self, node: NodeItem, name: str, label: str, is_input: bool, index: int):
        super().__init__(node)
        self.node, self.name, self.label, self.is_input = node, name, label, is_input
        self.type: PortType | None = None
        self.connected = False
        self.hover = False
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        x = 0.0 if is_input else node.width
        self.setPos(x, port_y(index, node.dot))
        self.setToolTip(label)

    def radius(self) -> float:
        """The connector's grab radius: a Dot's are small, its disc between them is what moves it."""
        return DOT_PORT_R if self.node.dot else PORT_R + 2

    def named(self) -> bool:
        """Connector names show only where a side has several connectors (Display connector name)."""
        side = getattr(self.node, "inputs" if self.is_input else "outputs", {})  # empty while being built
        return self.node.show_names and len(side) > 1

    def boundingRect(self):
        r = self.radius()
        if self.named():  # the name sits outside the node, beside the connector
            return QRectF(-r - NAME_W - 4, -8, NAME_W + 2 * r + 4, 16) if self.is_input else \
                QRectF(-r, -8, NAME_W + 2 * r + 4, 16)
        return QRectF(-r, -r, 2 * r, 2 * r)

    def shape(self):
        path = QPainterPath()
        r = self.radius()
        path.addEllipse(QPointF(0, 0), r, r)  # only the connector is interactive, not its label
        return path

    def paint(self, p: QPainter, option, widget=None):
        c = port_color(self.type)
        if self.node.dot:  # the disc is the Dot: its connectors show only under the mouse
            if self.hover:
                p.setPen(QPen(QColor("#f0f0f0"), 1.2))
                p.setBrush(c)
                p.drawEllipse(QPointF(0, 0), DOT_PORT_R - 1, DOT_PORT_R - 1)
            return
        p.setPen(QPen(QColor("#151515"), 1.5))
        p.setBrush(c if (self.connected or self.hover) else QColor("#2a2a2a"))
        p.drawEllipse(QPointF(0, 0), PORT_R, PORT_R)
        if not (self.connected or self.hover):
            p.setPen(QPen(c, 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(0, 0), PORT_R - 2, PORT_R - 2)
        if self.named() and p.worldTransform().m11() >= NAME_ZOOM:
            # Only on nodes with several connectors on that side, outside the node, close up.
            f = QFont()
            f.setPointSizeF(6.0)
            p.setFont(f)
            p.setPen(QColor("#cfcfcf"))
            if self.is_input:
                p.drawText(QRectF(-NAME_W - PORT_R - 3, -7, NAME_W, 14), Qt.AlignRight | Qt.AlignVCenter, self.label)
            else:
                p.drawText(QRectF(PORT_R + 3, -7, NAME_W, 14), Qt.AlignLeft | Qt.AlignVCenter, self.label)

    def hoverEnterEvent(self, e):
        self.hover = True
        self.update()

    def hoverLeaveEvent(self, e):
        self.hover = False
        self.update()


class NodeItem(QGraphicsItem):
    dot = False

    def __init__(self, node_id: str, label: str, node_type: str, category: str,
                 inputs: list[tuple[str, str]], outputs: list[tuple[str, str]]):
        super().__init__()
        self.node_id, self.label, self.node_type = node_id, label, node_type
        self.width, self.height = node_size(len(inputs), len(outputs), self.dot)
        self.body = self.height - TITLE - 2 * MARGIN
        self.color = title_color(node_type, category)
        self.thumbnail: QImage | None = None
        self.thumbnail_key = None  # (job key, output) of the result the thumbnail shows
        self.info = ""  # "2048x2048 - L8"
        self.timing_ms: float | None = None  # its last cook time (Display timings)
        self.show_names = True  # Node information menu: Display connector name
        self.show_info = True  # Display node size
        self.show_badges = True  # Display node badges
        self.show_timing = False  # Display timings
        self.warnings: list[str] = []  # the warning badge (Display node badges)
        self.limit_text = True  # Limit text scaling when zooming out
        self.disabled = False
        self.missing = False  # its definition can't be found (a missing dependency): drawn in red
        self.inputs = {name: PortItem(self, name, label, True, i) for i, (name, label) in enumerate(inputs)}
        self.outputs = {name: PortItem(self, name, label, False, i) for i, (name, label) in enumerate(outputs)}
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable
                      | QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(1)
        self.setToolTip(f"{label}\n{node_id}")

    def boundingRect(self):
        return QRectF(-PORT_R - 42, -12, self.width + 2 * PORT_R + 84, self.height + 64)  # + badge, size, timing

    def shape(self):
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.width, self.height), NODE_R, NODE_R)
        return path

    def paint(self, p: QPainter, option, widget=None):
        r = QRectF(0, 0, self.width, self.height)
        outline = self.shape()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#1c1a18"))
        p.drawPath(outline)
        p.save()
        p.setClipPath(outline)  # the title band follows the rounded corners
        p.setBrush(MISSING if self.missing else self.color)
        p.drawRect(QRectF(0, 0, self.width, TITLE))
        p.restore()
        body = QRectF(MARGIN, TITLE + MARGIN, THUMB, self.body)
        p.setBrush(QColor("#121110"))
        p.drawRoundedRect(body, THUMB_R, THUMB_R)
        if self.thumbnail is not None:
            img = self.thumbnail
            k = min(THUMB / img.width(), THUMB / img.height())  # a square at the top of tall nodes
            w, h = img.width() * k, img.height() * k
            thumb = QPainterPath()
            thumb.addRoundedRect(body, THUMB_R, THUMB_R)
            p.save()
            p.setClipPath(thumb)
            p.drawImage(QRectF(body.x() + (THUMB - w) / 2, body.y() + (THUMB - h) / 2, w, h), img)
            p.restore()
        ts = text_scale(p, self.limit_text)
        f = QFont()
        f.setPointSizeF(6.5 * ts)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.setPen(QColor("#f6f1ea"))
        p.drawText(QRectF(3, 0, self.width - 6, TITLE), Qt.AlignVCenter | Qt.AlignHCenter,
                   p.fontMetrics().elidedText(self.label, Qt.ElideRight, int(self.width - 6)))
        if self.disabled:  # dashed outline and a "Disabled" badge
            p.setPen(QPen(QColor("#e0e0e0"), 1.5, Qt.DashLine))
            p.setBrush(QColor(0, 0, 0, 120))
            p.drawPath(outline)
        if self.missing:  # a node of a missing dependency is drawn in red
            p.setPen(QPen(MISSING, 2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(outline)
        if self.warnings and self.show_badges:  # warning badge, top right
            tri = QPainterPath()
            x0 = self.width - 15
            tri.moveTo(x0 + 6, -9); tri.lineTo(x0 + 13, 3); tri.lineTo(x0 - 1, 3); tri.closeSubpath()
            p.setPen(QPen(QColor("#1b1b1b"), 1))
            p.setBrush(QColor("#e8a33a"))
            p.drawPath(tri)
            p.setPen(QPen(QColor("#1b1b1b"), 1.6))
            p.drawLine(QPointF(x0 + 6, -5), QPointF(x0 + 6, -1))
            p.drawPoint(QPointF(x0 + 6, 1.3))
        if (self.disabled or self.missing) and self.show_badges:
            badge = QRectF(r.center().x() - 30, r.center().y() - 8, 60, 16)
            p.setBrush(MISSING)
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(badge, 3, 3)
            f2 = QFont()
            f2.setPointSizeF(6.5)
            f2.setBold(True)
            p.setFont(f2)
            p.setPen(QColor("#ffffff"))
            p.drawText(badge, Qt.AlignCenter, "Missing" if self.missing else "Disabled")
        p.setBrush(Qt.NoBrush)
        if self.isSelected():
            p.setPen(QPen(QColor(theme.ACCENT), 2))
            p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), NODE_R + 1, NODE_R + 1)
        elif not (self.disabled or self.missing):
            p.setPen(QPen(QColor("#0f0e0d"), 1))
            p.drawPath(outline)
        line = self.height + 2
        f.setPointSizeF(7 * ts)
        f.setWeight(QFont.Normal)
        p.setFont(f)
        if self.show_info and self.info:
            p.setPen(QColor("#9a9087"))
            p.drawText(QRectF(-40, line, self.width + 80, 14 * ts), Qt.AlignHCenter, self.info)
            line += 13 * ts
        if self.show_timing and self.timing_ms is not None:
            ms = self.timing_ms
            p.setPen(QColor(next(c for limit, c in TIMING_COLORS if ms < limit)))
            p.drawText(QRectF(-40, line, self.width + 80, 14 * ts), Qt.AlignHCenter,
                       f"{ms:.2f} ms" if ms < 10 else f"{ms:.1f} ms" if ms < 100 else f"{ms:.0f} ms")

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene() is not None:
            return self.scene().snap(value, self)
        if change == QGraphicsItem.ItemPositionHasChanged and self.scene() is not None:
            self.scene().node_moved_live(self)
        return super().itemChange(change, value)

    def port(self, name: str, is_input: bool) -> PortItem:
        return (self.inputs if is_input else self.outputs)[name]


DOT_PORT_R = 4.0


class DotItem(NodeItem):
    """A Dot node: a disc on its link, colored like what flows through it, that moves like a node."""
    dot = True

    def boundingRect(self):
        return QRectF(-DOT_PORT_R - 2, -2, self.width + 2 * DOT_PORT_R + 4, self.height + 4)

    def shape(self):
        path = QPainterPath()
        path.addEllipse(QRectF(0, 0, self.width, self.height))
        return path

    def paint(self, p: QPainter, option, widget=None):
        port = next(iter(self.outputs.values()), None)
        c = port_color(port.type if port else None)
        p.setPen(QPen(QColor("#f0f0f0") if self.isSelected() else QColor("#151515"), 2 if self.isSelected() else 1.5))
        p.setBrush(c)
        p.drawEllipse(QRectF(1, 1, self.width - 2, self.height - 2))
        if self.disabled:
            p.setPen(QPen(QColor("#e0e0e0"), 1.2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QRectF(-2, -2, self.width + 4, self.height + 4))


class LinkItem(QGraphicsPathItem):
    def __init__(self, src: PortItem, dst: PortItem):
        super().__init__()
        self.src, self.dst = src, dst
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setZValue(0)
        self.rectangular = False
        self.highlight = False
        self.update_path()

    def update_path(self):
        self.setPath(link_path(self.src.scenePos(), self.dst.scenePos(), self.rectangular))

    def shape(self):
        stroker_path = QPainterPath(self.path())
        from PySide6.QtGui import QPainterPathStroker
        s = QPainterPathStroker()
        s.setWidth(10)
        return s.createStroke(stroker_path)

    def paint(self, p: QPainter, option, widget=None):
        c = port_color(self.src.type)
        if self.isSelected():
            c = QColor(theme.ACCENT)
        width = 3.6 if self.isSelected() or self.highlight else 2.8
        p.setPen(QPen(c, width, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        p.drawPath(self.path())


class GraphItem(QGraphicsItem):
    """A frame (titled colored box behind nodes; moving it moves the nodes inside, and the frames, comments and pins
    that lie in it), a comment (text) or a navigation pin."""

    HANDLE = 12
    TITLE = 22  # a frame's title bar
    EDGE = 6  # the band along a frame's or comment's border that resizes it (sides and corners)
    MIN_SIZE = (80.0, 40.0)

    def __init__(self, data: dict):
        super().__init__()
        self.data = data
        self.item_id, self.kind = data["id"], data["kind"]
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable
                      | QGraphicsItem.ItemSendsGeometryChanges)
        w, h = data.get("size", (0.0, 0.0))
        self.setZValue(-5 - w * h * 1e-9 if self.kind == "frame" else 3)  # a frame in a frame: drawn over it
        self._resizing = None  # (mouse at the press, (x, y, w, h) then, the sides moved: left, top, right, bottom)
        self.setAcceptHoverEvents(self.kind != "pin")  # the resize cursor along the border
        self.carried = []  # nodes moved with a frame during a drag
        self.carried_items = []  # frames, comments and pins moved with it
        self.setPos(*data["position"])

    def size(self):
        if self.kind == "pin":
            return 22.0, 22.0
        return tuple(self.data.get("size", [200.0, 100.0]))

    def boundingRect(self):
        w, h = self.size()
        return QRectF(-2, -2, w + 4, h + 4)

    def shape(self):
        """A frame answers by its title bar and its border (resizing it): a drag inside it draws a selection box (and
        a click there selects nothing); comments and pins by all of them."""
        w, h = self.size()
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)  # the bands overlap at the corners: odd-even would leave them out
        if self.kind == "frame":
            e = self.EDGE
            path.addRect(QRectF(0, 0, w, self.TITLE))
            for r in (QRectF(0, 0, e, h), QRectF(w - e, 0, e, h), QRectF(0, h - e, w, e),
                      QRectF(w - self.HANDLE, h - self.HANDLE, self.HANDLE, self.HANDLE)):
                path.addRect(r)
        else:
            path.addRect(QRectF(0, 0, w, h))
        return path

    def _sides(self, pos: QPointF) -> tuple[bool, bool, bool, bool]:
        """The sides a press at `pos` (item coordinates) resizes: (left, top, right, bottom); none for a pin. The
        bottom right corner keeps its larger handle."""
        if self.kind == "pin":
            return False, False, False, False
        w, h, e = *self.size(), self.EDGE
        corner = pos.x() > w - self.HANDLE and pos.y() > h - self.HANDLE
        return pos.x() < e, pos.y() < e, corner or pos.x() > w - e, corner or pos.y() > h - e

    def hoverMoveEvent(self, e):
        left, top, right, bottom = self._sides(e.pos())
        if (left or right) and (top or bottom):
            self.setCursor(Qt.SizeFDiagCursor if (left and top) or (right and bottom) else Qt.SizeBDiagCursor)
        elif left or right:
            self.setCursor(Qt.SizeHorCursor)
        elif top or bottom:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(e)

    def hoverLeaveEvent(self, e):
        self.unsetCursor()
        super().hoverLeaveEvent(e)

    def paint(self, p: QPainter, option, widget=None):
        w, h = self.size()
        sel = self.isSelected()
        if self.kind == "frame":
            c = QColor.fromRgbF(*[min(max(x, 0.0), 1.0) for x in self.data["color"]])
            p.setPen(QPen(QColor("#f0f0f0") if sel else c.darker(140), 2 if sel else 1))
            p.setBrush(c)
            p.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)
            bar = QColor(c)
            bar.setAlpha(min(255, c.alpha() + 90))
            p.setBrush(bar)
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(0, 0, w, self.TITLE), 4, 4)
            f = QFont()
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor("#f4f4f4"))
            p.drawText(QRectF(8, 0, w - 16, self.TITLE), Qt.AlignVCenter, self.data.get("title", ""))
            if self.data.get("text"):
                p.setPen(QColor("#dddddd"))
                p.drawText(QRectF(8, 26, w - 16, h - 30), Qt.TextWordWrap, self.data["text"])
            p.setPen(QPen(QColor("#dddddd"), 1))
            p.drawLine(QPointF(w - 9, h - 3), QPointF(w - 3, h - 9))
            p.drawLine(QPointF(w - 5, h - 3), QPointF(w - 3, h - 5))
        elif self.kind == "comment":
            p.setPen(QPen(QColor("#f0f0f0") if sel else QColor("#8a7a3a"), 1.5 if sel else 1))
            p.setBrush(QColor(60, 55, 35, 230))
            p.drawRect(QRectF(0, 0, w, h))
            p.setPen(QColor("#f0e6b0"))
            p.drawText(QRectF(6, 4, w - 12, h - 8), Qt.TextWordWrap, self.data.get("text", ""))
            p.setPen(QPen(QColor("#bbb"), 1))
            p.drawLine(QPointF(w - 9, h - 3), QPointF(w - 3, h - 9))
        else:  # pin
            p.setPen(QPen(QColor("#f0f0f0") if sel else QColor("#222"), 1.5))
            p.setBrush(QColor("#e05050"))
            p.drawEllipse(QRectF(3, 0, 16, 16))
            p.drawLine(QPointF(11, 16), QPointF(11, 22))
            p.setPen(QColor("#f0f0f0"))
            p.drawText(QRectF(24, 0, 200, 16), Qt.AlignVCenter, self.data.get("title", ""))

    def mousePressEvent(self, e):
        w, h = self.size()
        sides = self._sides(e.pos())
        if any(sides):
            self._resizing = (e.scenePos(), (self.pos().x(), self.pos().y(), w, h), sides)
            e.accept()
            return
        if self.kind == "frame" and self.scene() is not None:
            r = QRectF(self.scenePos(), QPointF(self.scenePos().x() + w, self.scenePos().y() + h))
            self.carried = [(n, n.pos() - self.pos()) for n in self.scene().nodes.values()
                            if r.contains(n.sceneBoundingRect().center())]
            self.carried_items = [(gi, gi.pos() - self.pos()) for gi in self.scene().graph_items
                                  if gi is not self and r.contains(QRectF(gi.pos(), QSizeF(*gi.size())))]
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._resizing is not None:  # the sides grabbed follow the mouse, the others stay
            start, (x, y, w, h), (left, top, right, bottom) = self._resizing
            d = e.scenePos() - start
            min_w, min_h = self.MIN_SIZE
            x0, x1 = x + (d.x() if left else 0.0), x + w + (d.x() if right else 0.0)
            y0, y1 = y + (d.y() if top else 0.0), y + h + (d.y() if bottom else 0.0)
            if x1 - x0 < min_w:
                x0, x1 = (x1 - min_w, x1) if left else (x0, x0 + min_w)
            if y1 - y0 < min_h:
                y0, y1 = (y1 - min_h, y1) if top else (y0, y0 + min_h)
            self.prepareGeometryChange()
            self.data["size"] = [x1 - x0, y1 - y0]
            self.setPos(x0, y0)
            self.update()
            return
        super().mouseMoveEvent(e)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for n, offset in self.carried + self.carried_items:
                n.setPos(self.pos() + offset)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, e):
        resized = self._resizing is not None
        self._resizing = None
        super().mouseReleaseEvent(e)
        if self.scene() is not None:
            moved = {n.node_id: (n.pos().x(), n.pos().y()) for n, _ in self.carried}
            moved_items = {gi.item_id: (gi.pos().x(), gi.pos().y()) for gi, _ in self.carried_items}
            self.carried, self.carried_items = [], []
            self.scene().item_changed(self, moved, resized, moved_items)

    def mouseDoubleClickEvent(self, e):
        if self.scene() is not None:
            self.scene().ctl.edit_item_text(self.item_id)


def link_path(a: QPointF, b: QPointF, rectangular: bool = False) -> QPainterPath:
    path = QPainterPath(a)
    if rectangular:
        mid = (a.x() + b.x()) / 2
        path.lineTo(mid, a.y())
        path.lineTo(mid, b.y())
        path.lineTo(b)
        return path
    c1, c2 = link_controls((a.x(), a.y()), (b.x(), b.y()))
    path.cubicTo(QPointF(*c1), QPointF(*c2), b)
    return path
