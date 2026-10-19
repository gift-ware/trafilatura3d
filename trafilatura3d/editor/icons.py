"""Toolbar and panel icons drawn in code (no image assets): simple 16x16 line glyphs; the application's icon
puts its artwork (app_icon.png) on a tile."""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def _pen(p: QPainter, color: str, width: float = 1.4):
    p.setPen(QPen(QColor(color), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(Qt.NoBrush)


def _draw(name: str, p: QPainter, c: str):
    _pen(p, c)
    R = QRectF
    if name == "new":
        p.drawPolyline([QPointF(4, 2), QPointF(10, 2), QPointF(13, 5), QPointF(13, 14), QPointF(4, 14), QPointF(4, 2)])
        p.drawLine(QPointF(10, 2), QPointF(10, 5)); p.drawLine(QPointF(10, 5), QPointF(13, 5))
    elif name == "open":
        p.drawPolyline([QPointF(2, 13), QPointF(2, 4), QPointF(6, 4), QPointF(7.5, 5.5), QPointF(13, 5.5), QPointF(13, 7)])
        p.drawPolyline([QPointF(2, 13), QPointF(4.5, 7), QPointF(15, 7), QPointF(12.5, 13), QPointF(2, 13)])
    elif name == "save":
        p.drawRect(R(2.5, 2.5, 11, 11)); p.drawRect(R(5, 2.5, 6, 4)); p.drawRect(R(4.5, 9, 7, 4.5))
    elif name in ("undo", "redo"):
        path = QPainterPath()
        if name == "undo":
            path.moveTo(4, 6); path.lineTo(10, 6); path.cubicTo(14.5, 6, 14.5, 13, 10, 13); path.lineTo(6, 13)
            p.drawPath(path); p.drawPolyline([QPointF(6.5, 3.5), QPointF(4, 6), QPointF(6.5, 8.5)])
        else:
            path.moveTo(12, 6); path.lineTo(6, 6); path.cubicTo(1.5, 6, 1.5, 13, 6, 13); path.lineTo(10, 13)
            p.drawPath(path); p.drawPolyline([QPointF(9.5, 3.5), QPointF(12, 6), QPointF(9.5, 8.5)])
    elif name == "focus":
        for x, y, dx, dy in [(2, 2, 1, 1), (14, 2, -1, 1), (2, 14, 1, -1), (14, 14, -1, -1)]:
            p.drawPolyline([QPointF(x, y + 4 * dy), QPointF(x, y), QPointF(x + 4 * dx, y)])
        p.drawRect(R(6, 6, 4, 4))
    elif name == "reset_zoom":
        p.drawLine(QPointF(4, 4), QPointF(4, 12)); p.drawLine(QPointF(12, 4), QPointF(12, 12))
        p.drawPoint(QPointF(8, 6.5)); p.drawPoint(QPointF(8, 9.5))
    elif name == "camera":
        p.drawRoundedRect(R(2, 5, 12, 8), 1.5, 1.5); p.drawEllipse(QPointF(8, 9), 2.5, 2.5); p.drawRect(R(5.5, 3, 5, 2))
    elif name == "environment":  # a globe
        p.drawEllipse(QPointF(8, 8), 6, 6); p.drawEllipse(QPointF(8, 8), 2.6, 6); p.drawLine(QPointF(2, 8), QPointF(14, 8))
    elif name == "renderer":  # sliders
        for y, x in ((4, 10), (8, 5), (12, 9)):
            p.drawLine(QPointF(2, y), QPointF(14, y)); p.drawEllipse(QPointF(x, y), 1.6, 1.6)
    elif name == "shadows":  # a sphere and its shadow
        p.drawEllipse(QPointF(7, 6), 4, 4)
        p.setBrush(QColor(c)); p.drawEllipse(R(6, 12, 8, 2.5)); p.setBrush(Qt.NoBrush)
    elif name == "ground":  # a plane in perspective
        p.drawPolyline([QPointF(1, 13), QPointF(5, 6), QPointF(15, 6), QPointF(11, 13), QPointF(1, 13)])
    elif name == "displacement":  # a wave over a line
        path = QPainterPath(); path.moveTo(2, 9); path.cubicTo(5, 2, 7, 2, 8, 7); path.cubicTo(9, 12, 12, 4, 14, 8)
        p.drawPath(path); p.drawLine(QPointF(2, 13), QPointF(14, 13))
    elif name == "warning":  # an orange triangle with "!"
        path = QPainterPath(); path.moveTo(8, 2); path.lineTo(15, 14); path.lineTo(1, 14); path.closeSubpath()
        p.setPen(QPen(QColor("#e8a33a"), 1.4)); p.setBrush(QColor("#e8a33a")); p.drawPath(path)
        p.setPen(QPen(QColor("#1b1b1b"), 1.8)); p.drawLine(QPointF(8, 6), QPointF(8, 10)); p.drawPoint(QPointF(8, 12.3))
    elif name == "info":
        p.drawEllipse(QPointF(8, 8), 6, 6); p.drawLine(QPointF(8, 7), QPointF(8, 11.5)); p.drawPoint(QPointF(8, 4.8))
    elif name == "search":
        p.drawEllipse(QPointF(7, 7), 4, 4); p.drawLine(QPointF(10, 10), QPointF(14, 14))
    elif name == "highlight":
        for x in (3, 8, 13):
            p.drawEllipse(QPointF(x, 8), 1.6, 1.6)
        p.drawLine(QPointF(4.6, 8), QPointF(6.4, 8)); p.drawLine(QPointF(9.6, 8), QPointF(11.4, 8))
    elif name == "palette":
        for i in range(3):
            for j in range(2):
                p.drawRect(R(2 + i * 4.5, 4 + j * 4.5, 3, 3))
    elif name == "links":
        path = QPainterPath(); path.moveTo(2, 12); path.lineTo(8, 12); path.lineTo(8, 4); path.lineTo(14, 4); p.drawPath(path)
    elif name == "align":
        p.drawLine(QPointF(3, 2), QPointF(3, 14)); p.drawRect(R(5, 4, 8, 3)); p.drawRect(R(5, 9, 5, 3))
    elif name == "pin":
        p.drawLine(QPointF(8, 9), QPointF(8, 14)); p.drawRect(R(5, 3, 6, 6)); p.drawLine(QPointF(3.5, 9), QPointF(12.5, 9))
    elif name == "undock":
        p.drawRect(R(2.5, 5.5, 8, 8)); p.drawPolyline([QPointF(6, 2.5), QPointF(13.5, 2.5), QPointF(13.5, 10)])
    elif name == "maximize":
        p.drawRect(R(3, 3, 10, 10)); p.drawRect(R(5.5, 5.5, 5, 5))
    elif name == "close":
        p.drawLine(QPointF(4, 4), QPointF(12, 12)); p.drawLine(QPointF(12, 4), QPointF(4, 12))
    elif name == "explorer":
        p.drawLine(QPointF(3, 3), QPointF(3, 13)); p.drawLine(QPointF(3, 5), QPointF(7, 5))
        p.drawLine(QPointF(3, 11), QPointF(7, 11)); p.drawRect(R(8, 3, 6, 4)); p.drawRect(R(8, 9, 6, 4))
    elif name == "graph":
        p.drawEllipse(QPointF(3.5, 8), 1.8, 1.8); p.drawEllipse(QPointF(12.5, 4), 1.8, 1.8)
        p.drawEllipse(QPointF(12.5, 12), 1.8, 1.8)
        p.drawLine(QPointF(5, 7), QPointF(11, 4.5)); p.drawLine(QPointF(5, 9), QPointF(11, 11.5))
    elif name == "properties":
        p.drawPolyline([QPointF(4, 2), QPointF(10, 2), QPointF(13, 5), QPointF(13, 14), QPointF(4, 14), QPointF(4, 2)])
        for y in (7, 9.5, 12):
            p.drawLine(QPointF(6, y), QPointF(11, y))
    elif name == "view2d":
        p.drawRect(R(2.5, 2.5, 11, 11)); p.drawLine(QPointF(2.5, 8), QPointF(13.5, 8)); p.drawLine(QPointF(8, 2.5), QPointF(8, 13.5))
    elif name == "view3d":
        path = QPainterPath(); path.moveTo(8, 2); path.lineTo(14, 5); path.lineTo(14, 11); path.lineTo(8, 14)
        path.lineTo(2, 11); path.lineTo(2, 5); path.closeSubpath(); p.drawPath(path)
        p.drawPolyline([QPointF(2, 5), QPointF(8, 8), QPointF(14, 5)]); p.drawLine(QPointF(8, 8), QPointF(8, 14))
    elif name == "library":
        for x in (3, 6.5):
            p.drawRect(R(x, 3, 2.5, 10))
        p.save(); p.translate(11.5, 3); p.rotate(-15); p.drawRect(R(0, 0, 2.5, 10)); p.restore()
    elif name == "package":
        path = QPainterPath(); path.moveTo(8, 2); path.lineTo(14, 5); path.lineTo(14, 11); path.lineTo(8, 14)
        path.lineTo(2, 11); path.lineTo(2, 5); path.closeSubpath(); p.drawPath(path)
        p.drawPolyline([QPointF(2, 5), QPointF(8, 8), QPointF(14, 5)]); p.drawLine(QPointF(8, 8), QPointF(8, 14))
        p.drawLine(QPointF(5, 3.5), QPointF(11, 6.5))
    elif name == "folder":
        p.drawPolyline([QPointF(2, 13), QPointF(2, 4), QPointF(6, 4), QPointF(7.5, 5.5), QPointF(14, 5.5),
                        QPointF(14, 13), QPointF(2, 13)])
    elif name == "bitmap":
        p.drawRect(R(2.5, 3.5, 11, 9)); p.drawPolyline([QPointF(3, 12), QPointF(7, 8), QPointF(9, 10), QPointF(11, 8), QPointF(13, 10)])
    elif name == "copy":
        p.drawRect(R(5, 5, 8, 9)); p.drawPolyline([QPointF(3, 11), QPointF(3, 2.5), QPointF(10, 2.5)])
    elif name == "histogram":
        for i, h in enumerate((4, 8, 11, 6, 9, 3)):
            p.drawLine(QPointF(3 + i * 2, 14), QPointF(3 + i * 2, 14 - h))
    elif name == "invert":  # a square half filled across its diagonal
        p.drawRect(R(2.5, 2.5, 11, 11))
        p.setBrush(QColor(c))
        p.drawPolygon([QPointF(2.5, 13.5), QPointF(13.5, 2.5), QPointF(13.5, 13.5)])
    elif name == "auto_level":  # two arrows pushing toward a bar: )|(
        p.drawLine(QPointF(8, 2), QPointF(8, 14))
        p.drawLine(QPointF(1.5, 8), QPointF(5.5, 8)); p.drawPolyline([QPointF(3.5, 6), QPointF(5.5, 8), QPointF(3.5, 10)])
        p.drawLine(QPointF(14.5, 8), QPointF(10.5, 8)); p.drawPolyline([QPointF(12.5, 6), QPointF(10.5, 8), QPointF(12.5, 10)])
    elif name == "checker":
        p.setBrush(QColor(c))
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    p.drawRect(R(2 + i * 3, 2 + j * 3, 3, 3))
    elif name == "tile":
        for i in range(3):
            for j in range(3):
                p.drawRect(R(2 + i * 4, 2 + j * 4, 3, 3))
    elif name == "fit":
        p.drawRect(R(4, 4, 8, 8))
        for x, y, dx, dy in [(2, 2, 1, 1), (14, 2, -1, 1), (2, 14, 1, -1), (14, 14, -1, -1)]:
            p.drawLine(QPointF(x, y), QPointF(x + 2 * dx, y + 2 * dy))
    elif name == "random":
        p.drawPolyline([QPointF(2, 4), QPointF(5, 4), QPointF(11, 12), QPointF(14, 12)])
        p.drawPolyline([QPointF(2, 12), QPointF(5, 12), QPointF(11, 4), QPointF(14, 4)])
        p.drawPolyline([QPointF(12, 2), QPointF(14, 4), QPointF(12, 6)]); p.drawPolyline([QPointF(12, 10), QPointF(14, 12), QPointF(12, 14)])
    elif name == "chain":
        p.drawRoundedRect(R(2, 6, 7, 4), 2, 2); p.drawRoundedRect(R(7, 6, 7, 4), 2, 2)
    elif name == "dropdown":  # the arrow of a dropdown (theme.dropdown_arrows)
        p.drawPolyline([QPointF(4.5, 6.5), QPointF(8, 10), QPointF(11.5, 6.5)])
    elif name == "menu":
        for y in (4, 8, 12):
            p.drawLine(QPointF(3, y), QPointF(13, y))
    elif name == "inherit":
        p.drawRect(R(2, 2, 5, 4)); p.drawRect(R(9, 10, 5, 4)); p.drawPolyline([QPointF(4.5, 6), QPointF(4.5, 12), QPointF(9, 12)])
    elif name == "add":
        p.drawLine(QPointF(8, 3), QPointF(8, 13)); p.drawLine(QPointF(3, 8), QPointF(13, 8))
    elif name == "export":
        p.drawPolyline([QPointF(5, 6), QPointF(8, 3), QPointF(11, 6)]); p.drawLine(QPointF(8, 3), QPointF(8, 10))
        p.drawPolyline([QPointF(3, 9), QPointF(3, 13.5), QPointF(13, 13.5), QPointF(13, 9)])
    elif name == "channels":
        p.setBrush(QColor(220, 60, 60)); p.drawEllipse(QPointF(6, 6), 3.5, 3.5)
        p.setBrush(QColor(60, 200, 80)); p.drawEllipse(QPointF(10, 6), 3.5, 3.5)
        p.setBrush(QColor(70, 120, 230)); p.drawEllipse(QPointF(8, 10), 3.5, 3.5)
    elif name == "home":
        p.drawPolyline([QPointF(2, 8), QPointF(8, 2.5), QPointF(14, 8)]); p.drawPolyline([QPointF(4, 6.5), QPointF(4, 13.5), QPointF(12, 13.5), QPointF(12, 6.5)])
    elif name == "tools":  # a wrench
        p.drawLine(QPointF(3.5, 12.5), QPointF(9, 7))
        p.drawArc(R(8, 2.5, 6, 6), 200 * 16, 280 * 16)
    else:
        p.drawRect(R(3, 3, 10, 10))


def app_icon(size: int = 512) -> QIcon:
    """The application's icon (window, Dock, shortcut): the artwork (app_icon.png) on a light blue rounded tile,
    proportioned as macOS icons are (the tile 824 of 1024 wide, its corners 185)."""
    from pathlib import Path
    from PySide6.QtGui import QImage, QLinearGradient
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
    k = size / 1024.0
    p.scale(k, k)
    tile = QRectF(100, 100, 824, 824)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(0, 0, 0, 60))  # a soft shadow below the tile
    p.drawRoundedRect(tile.translated(0, 10), 185, 185)
    blue = QLinearGradient(0, 100, 0, 924)
    blue.setColorAt(0, QColor("#a9dcff"))
    blue.setColorAt(1, QColor("#5aa9e6"))
    p.setBrush(blue)
    p.drawRoundedRect(tile, 185, 185)
    art = QImage(str(Path(__file__).with_name("app_icon.png")))
    p.drawImage(QRectF(150, 150, 724, 724), art)
    p.end()
    return QIcon(pm)


@lru_cache(maxsize=None)
def icon(name: str, color: str = "#cfcfcf") -> QIcon:
    result = QIcon()
    for scale in (1, 2):
        pm = QPixmap(16 * scale, 16 * scale)
        pm.setDevicePixelRatio(scale)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        _draw(name, p, color)
        p.end()
        result.addPixmap(pm)
    return result
