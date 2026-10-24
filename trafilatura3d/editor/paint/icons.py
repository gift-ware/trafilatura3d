"""Icons of the Paint workspace drawn in code (no image assets), 16x16 line glyphs like the editor's."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from trafilatura3d.editor.icons import icon as editor_icon


def _pen(p: QPainter, color: str, width: float = 1.4):
    p.setPen(QPen(QColor(color), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(Qt.NoBrush)


def _draw(name: str, p: QPainter, c: str) -> bool:
    _pen(p, c)
    P, R = QPointF, QRectF
    if name == "brush":
        p.drawLine(P(13.5, 2.5), P(7, 9))
        path = QPainterPath()
        path.moveTo(7.5, 8.5); path.cubicTo(4, 8, 3, 11, 2.5, 13.5); path.cubicTo(5, 13.5, 8.5, 12.5, 7.5, 8.5)
        p.drawPath(path)
    elif name == "eraser":
        p.drawPolygon([P(9, 2.5), P(14, 7.5), P(8, 13.5), P(3, 8.5)])
        p.drawLine(P(5.5, 6), P(10.5, 11)); p.drawLine(P(8, 13.5), P(14, 13.5))
    elif name == "geometry_mask":  # a cube with a part cut away
        p.drawPolygon([P(3, 5.5), P(8, 3), P(13, 5.5), P(13, 11), P(8, 13.5), P(3, 11)])
        p.drawLine(P(3, 5.5), P(8, 8)); p.drawLine(P(8, 8), P(13, 5.5)); p.drawLine(P(8, 8), P(8, 13.5))
        p.setBrush(QColor(c)); p.drawPolygon([P(8, 8), P(13, 5.5), P(13, 11), P(8, 13.5)])
    elif name == "projection":  # a slide projector's beam onto a surface
        p.drawRect(R(1.5, 6, 4, 4)); p.drawLine(P(5.5, 6), P(14, 2.5)); p.drawLine(P(5.5, 10), P(14, 13.5))
        p.drawLine(P(14, 2.5), P(14, 13.5))
    elif name == "polygon_fill":  # a filled triangle in a wireframe
        p.drawPolygon([P(2.5, 13.5), P(8, 2.5), P(13.5, 13.5)]); p.drawLine(P(5.2, 8), P(10.8, 8))
        p.setBrush(QColor(c)); p.drawPolygon([P(5.2, 8), P(10.8, 8), P(8, 2.5)])
    elif name == "smudge":  # a finger tip dragging a smear
        p.drawEllipse(P(11, 5), 2.5, 2.5)
        path = QPainterPath()
        path.moveTo(9.5, 7); path.cubicTo(7, 10, 5, 11, 2, 13.5)
        p.drawPath(path); p.drawLine(P(12.5, 7), P(4, 14))
    elif name == "clone":  # a stamp
        p.drawRect(R(3, 10, 10, 3.5)); p.drawLine(P(6.5, 10), P(6.5, 6)); p.drawLine(P(9.5, 10), P(9.5, 6))
        p.drawEllipse(P(8, 4.5), 2.5, 2)
    elif name == "quick_mask":  # a dashed rectangle with a dot
        _pen(p, c, 1.2)
        p.setPen(QPen(QColor(c), 1.2, Qt.DashLine)); p.drawRect(R(2.5, 3.5, 11, 9))
        p.setBrush(QColor(c)); p.drawEllipse(P(8, 8), 2, 2)
    elif name == "path":  # a curve between two vertices
        path = QPainterPath()
        path.moveTo(2.5, 12); path.cubicTo(5, 2, 11, 14, 13.5, 4)
        p.drawPath(path); p.drawRect(R(1, 10.5, 3, 3)); p.drawRect(R(12, 2.5, 3, 3))
    elif name == "eye":
        path = QPainterPath()
        path.moveTo(1.5, 8); path.cubicTo(5, 2.5, 11, 2.5, 14.5, 8); path.cubicTo(11, 13.5, 5, 13.5, 1.5, 8)
        p.drawPath(path); p.drawEllipse(P(8, 8), 2.2, 2.2)
    elif name == "eye_off":
        path = QPainterPath()
        path.moveTo(1.5, 8); path.cubicTo(5, 2.5, 11, 2.5, 14.5, 8); path.cubicTo(11, 13.5, 5, 13.5, 1.5, 8)
        p.drawPath(path); p.drawLine(P(3, 13), P(13, 3))
    elif name == "paint_layer":
        p.drawRect(R(2.5, 3.5, 11, 9)); p.drawLine(P(5, 10), P(11, 6))
    elif name == "fill_layer":  # a bucket
        p.drawPolygon([P(3, 7), P(8, 2.5), P(12.5, 7), P(8, 11.5)])
        p.drawLine(P(3, 7), P(12.5, 7))
        path = QPainterPath()
        path.moveTo(13.5, 9); path.cubicTo(12, 11.5, 12.5, 13.5, 13.5, 13.5); path.cubicTo(14.5, 13.5, 15, 11.5, 13.5, 9)
        p.drawPath(path)
    elif name == "mask":
        p.drawRect(R(2.5, 2.5, 11, 11)); p.setBrush(QColor(c)); p.drawEllipse(P(8, 8), 3, 3)
    elif name == "delete":
        p.drawLine(P(3, 4), P(13, 4)); p.drawLine(P(6.5, 4), P(6.5, 2.5)); p.drawLine(P(6.5, 2.5), P(9.5, 2.5))
        p.drawLine(P(9.5, 2.5), P(9.5, 4)); p.drawPolyline([P(4, 4), P(5, 14), P(11, 14), P(12, 4)])
        p.drawLine(P(7, 6.5), P(7, 11.5)); p.drawLine(P(9, 6.5), P(9, 11.5))
    elif name == "folder":
        p.drawPolyline([P(2, 13), P(2, 4), P(6, 4), P(7.5, 5.5), P(14, 5.5), P(14, 13), P(2, 13)])
    elif name == "display_settings":  # a sun over a horizon
        p.drawEllipse(P(8, 7), 2.5, 2.5)
        for a, b in ((P(8, 1.5), P(8, 3)), (P(2.5, 7), P(4, 7)), (P(12, 7), P(13.5, 7)), (P(4, 3), P(5, 4)),
                     (P(12, 3), P(11, 4))):
            p.drawLine(a, b)
        p.drawLine(P(2, 12), P(14, 12))
    elif name == "history":
        p.drawEllipse(P(8.5, 8.5), 5.5, 5.5); p.drawPolyline([P(8.5, 5), P(8.5, 8.5), P(11, 10)])
        p.drawPolyline([P(1.5, 6), P(3, 8.5), P(5, 6.5)])
    elif name == "log":
        p.drawRect(R(3, 2, 10, 12))
        for y in (5, 8, 11):
            p.drawLine(P(5, y), P(11, y))
    elif name == "assets":
        for x, y in ((2.5, 2.5), (9, 2.5), (2.5, 9), (9, 9)):
            p.drawRect(R(x, y, 4.5, 4.5))
    elif name == "texture_sets":
        p.drawRect(R(2.5, 2.5, 7, 7)); p.drawPolyline([P(6.5, 13.5), P(13.5, 13.5), P(13.5, 6.5)])
        p.drawPolyline([P(4.5, 11.5), P(11.5, 11.5), P(11.5, 4.5)])
    elif name == "focus_mode":
        p.drawEllipse(P(8, 8), 5.5, 5.5); p.setBrush(QColor(c)); p.drawEllipse(P(8, 8), 2, 2)
    elif name == "dots":
        p.setBrush(QColor(c))
        for x in (4, 8, 12):
            p.drawEllipse(P(x, 8), 1, 1)
    elif name == "lines_menu":
        for y in (4, 8, 12):
            p.drawLine(P(3, y), P(13, y))
    elif name == "lock":
        p.drawRect(R(3.5, 7, 9, 7)); p.drawArc(R(5, 2.5, 6, 9), 0, 180 * 16)
    elif name == "unlock":
        p.drawRect(R(3.5, 7, 9, 7)); p.drawArc(R(8, 2.5, 6, 9), 0, 180 * 16)
    elif name == "effect":  # a wand with sparks
        p.drawLine(P(2.5, 13.5), P(10, 6))
        for a, b in ((P(12, 2), P(12, 5)), (P(10.5, 3.5), P(13.5, 3.5)), (P(13.5, 7.5), P(13.5, 9.5)),
                     (P(12.5, 8.5), P(14.5, 8.5)), (P(6.5, 2), P(6.5, 4)), (P(5.5, 3), P(7.5, 3))):
            p.drawLine(a, b)
    elif name == "anchor":  # a ship's anchor: ring, shank, stock, arms
        p.drawEllipse(P(8, 3), 1.5, 1.5)
        p.drawLine(P(8, 4.5), P(8, 13.5)); p.drawLine(P(5, 6.5), P(11, 6.5))
        p.drawArc(R(3, 6, 10, 7.5), 200 * 16, 140 * 16)
    elif name == "picker":
        p.drawLine(P(3, 13), P(9, 7)); p.drawEllipse(P(11, 5), 2.5, 2.5)
    elif name == "thumb_size":  # four squares: the Assets' thumbnail size
        for x, y in ((2.5, 2.5), (9, 2.5), (2.5, 9), (9, 9)):
            p.drawRect(R(x, y, 4.5, 4.5))
    elif name == "uv_tiles":  # a 2 x 2 grid of tiles, one numbered (UV Tile information)
        for x, y in ((2, 2), (8.5, 2), (2, 8.5), (8.5, 8.5)):
            p.drawRect(R(x, y, 5.5, 5.5))
        p.drawLine(P(4.8, 3.6), P(4.8, 6.2))
    elif name == "pen_pressure":  # a stylus, tip down, over the line it presses (Pen pressure toggle)
        p.drawPolygon([P(13.5, 1.5), P(15, 3), P(7, 11), P(5.5, 9.5)])
        p.drawPolygon([P(5.5, 9.5), P(7, 11), P(4, 12.5)])
        path = QPainterPath()
        path.moveTo(1.5, 14.5); path.cubicTo(5, 13, 9, 15.5, 14.5, 14)
        p.drawPath(path)
    else:
        return False
    return True


_ICONS: dict[tuple[str, str], QIcon] = {}


def picon(name: str, color: str = "#cfcfcf") -> QIcon:
    """A Paint icon, or the editor's icon of that name; drawn once (the layer rows ask for theirs at every row
    made)."""
    if (name, color) not in _ICONS:
        _ICONS[name, color] = _picon(name, color)
    return _ICONS[name, color]


def _picon(name: str, color: str) -> QIcon:
    result = QIcon()
    for scale in (1, 2):
        pm = QPixmap(16 * scale, 16 * scale)
        pm.setDevicePixelRatio(scale)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        ok = _draw(name, p, color)
        p.end()
        if not ok:
            return editor_icon(name, color)
        result.addPixmap(pm)
    return result
