"""Graph view navigation: wheel or Alt+right drag zooms, middle drag or Ctrl+right
drag pans (two-finger swipe on a trackpad pans, pinch zooms), F frames the selection (or all),
Z resets the zoom to 1:1, a drag on empty space draws a selection box. Dragging a link near or past the
view's border scrolls the view that way."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPointF, Qt, QTimer
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QApplication, QGraphicsView

ZOOM_MIN, ZOOM_MAX = 0.05, 4.0
EDGE = 24  # pixels from the border where a dragged link starts scrolling the view
EDGE_STEP = 40  # the most it scrolls per tick (pixels), reached far past the border


class GraphView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.StrongFocus)
        self.viewport().grabGesture(Qt.PinchGesture)
        self._pan = None
        self._zoom_drag = None
        self.resized = None  # callback after a resize (overlays placed on the view)
        self.cursor_scene_pos = QPointF()
        self._edge_pos = None  # the mouse in the viewport while a link is dragged
        self._edge_timer = QTimer(self, interval=16, timeout=self._edge_scroll)

    def _edge_scroll(self) -> None:
        """While a link is dragged with the mouse near or past a border, scroll that way and keep the link's end
        under the mouse."""
        scene = self.scene()
        if self._edge_pos is None or not scene.dragging_link() or not QApplication.mouseButtons() & Qt.LeftButton:
            self._edge_timer.stop()
            return
        p, r = self._edge_pos, self.viewport().rect()

        def step(v, lo, hi):
            d = v - (hi - EDGE) if v > hi - EDGE else v - (lo + EDGE) if v < lo + EDGE else 0.0
            return 0 if not d else int(max(-EDGE_STEP, min(EDGE_STEP, d * 0.5 + (4 if d > 0 else -4))))
        dx, dy = step(p.x(), r.left(), r.right()), step(p.y(), r.top(), r.bottom())
        if dx or dy:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + dx)
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + dy)
            scene.drag_link_to(self.mapToScene(p.toPoint()))


    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.resized:
            self.resized()
    def zoom(self) -> float:
        return self.transform().m11()

    def zoom_by(self, factor: float, anchor=None) -> None:
        z = self.zoom()
        factor = max(ZOOM_MIN / z, min(ZOOM_MAX / z, factor))
        if anchor is not None:
            self.setTransformationAnchor(QGraphicsView.NoAnchor)
            before = self.mapToScene(anchor)
            self.scale(factor, factor)
            after = self.mapToScene(anchor)
            d = after - before
            self.translate(d.x(), d.y())
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        else:
            self.scale(factor, factor)

    def wheelEvent(self, e):
        # A trackpad swipe (pixel deltas, no Ctrl/Cmd) pans; a mouse wheel zooms.
        if not e.pixelDelta().isNull() and not (e.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)):
            d = e.pixelDelta()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            return
        from trafilatura3d.editor.preferences import zoom_sign
        self.zoom_by(1.0015 ** (zoom_sign() * e.angleDelta().y()))

    def viewportEvent(self, e):
        if e.type() == QEvent.Gesture:
            pinch = e.gesture(Qt.PinchGesture)
            if pinch is not None:
                self.zoom_by(pinch.scaleFactor(), self.mapFromGlobal(pinch.centerPoint().toPoint()))
                return True
        return super().viewportEvent(e)

    def mousePressEvent(self, e):
        b, m = e.button(), e.modifiers()
        if b == Qt.MiddleButton or (b == Qt.RightButton and m & Qt.ControlModifier):
            self._pan = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if b == Qt.RightButton and m & Qt.AltModifier:
            self._zoom_drag = (e.position(), e.position())
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        self.cursor_scene_pos = self.mapToScene(e.position().toPoint())
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if self._zoom_drag is not None:
            start, last = self._zoom_drag
            dx = e.position().x() - last.x()
            self._zoom_drag = (start, e.position())
            self.zoom_by(1.01 ** dx, start.toPoint())
            return
        if self.scene().dragging_link():
            self._edge_pos = e.position()
            if not self._edge_timer.isActive():
                self._edge_timer.start()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._pan is not None or self._zoom_drag is not None:
            self._pan = self._zoom_drag = None
            self.unsetCursor()
            return
        super().mouseReleaseEvent(e)

    def contextMenuEvent(self, e):
        if e.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
            return  # right drags navigate
        super().contextMenuEvent(e)

    def focus(self) -> None:
        """Frame the selected nodes, or the whole graph when nothing is selected."""
        scene = self.scene()
        items = scene.selectedItems() or list(scene.nodes.values())
        if not items:
            return
        rect = items[0].sceneBoundingRect()
        for i in items[1:]:
            rect = rect.united(i.sceneBoundingRect())
        self.fitInView(rect.adjusted(-80, -80, 80, 80), Qt.KeepAspectRatio)
        if self.zoom() > 1.5:
            self.reset_zoom(rect.center())

    def reset_zoom(self, center=None) -> None:
        c = center or self.mapToScene(self.viewport().rect().center())
        self.resetTransform()
        self.centerOn(c)

    # Drops from the Library (node types) and the Explorer (graphs, as instances).
    on_drop = None  # callable(mime data, scene position) -> bool

    def dragEnterEvent(self, e):
        if self.on_drop and any(f.startswith("application/x-trafilatura3d") for f in e.mimeData().formats()):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if any(f.startswith("application/x-trafilatura3d") for f in e.mimeData().formats()):
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if self.on_drop and self.on_drop(e.mimeData(), self.mapToScene(e.position().toPoint())):
            e.acceptProposedAction()
        else:
            super().dropEvent(e)

    def enterEvent(self, e):
        """The view takes the keyboard when the mouse enters (its shortcuts are the graph's: Tab, Space,
        Ctrl+C/V/D...), unless a text field is being edited."""
        from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QLineEdit, QPlainTextEdit, QTextEdit
        if not isinstance(QApplication.focusWidget(), (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)):
            self.setFocus(Qt.MouseFocusReason)
        super().enterEvent(e)

    def keyPressEvent(self, e):
        if e.modifiers() == Qt.NoModifier and e.key() == Qt.Key_F:
            self.focus()
            return
        if e.modifiers() == Qt.NoModifier and e.key() == Qt.Key_Z:
            self.reset_zoom()
            return
        super().keyPressEvent(e)
