"""Graph view navigation: wheel or Alt+right drag zooms, middle drag or Ctrl+right
drag pans (two-finger swipe on a trackpad pans, pinch zooms), F frames the selection (or all),
Z resets the zoom to 1:1, a drag on empty space (inside a frame too) draws a selection box, Shift adds to the
selection. Dragging a link, nodes or a selection box near or past the view's border scrolls the view that way."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPointF, Qt, QTimer
from PySide6.QtGui import QMouseEvent, QPainter
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
        self._edge_pos = None  # the mouse in the viewport during a left drag (`_edge_scroll`)
        self._shift_select = False  # a left press with Shift away from connectors: Shift taken as Control
        self._edge_timer = QTimer(self, interval=16, timeout=self._edge_scroll)

    def _left_drag(self) -> bool:
        """A left drag that follows the mouse across the scene: a link, nodes or frames moved, a selection box."""
        scene = self.scene()
        return bool(QApplication.mouseButtons() & Qt.LeftButton) and (
            scene.dragging_link() or scene.mouseGrabberItem() is not None or not self.rubberBandRect().isNull())

    def _edge_scroll(self) -> None:
        """While a left drag (`_left_drag`) has the mouse near or past a border, scroll that way; the drag follows (a
        mouse move where the mouse is: the link's end, the moved items, the selection box stay under it)."""
        if self._edge_pos is None or not self._left_drag():
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
            QApplication.sendEvent(self.viewport(), QMouseEvent(
                QEvent.MouseMove, p, self.viewport().mapToGlobal(p), Qt.NoButton, QApplication.mouseButtons(),
                QApplication.keyboardModifiers()))

    def _shift_as_control(self, e) -> QMouseEvent:
        """Shift adds to the selection (a click on nodes or frames, a selection box): what Qt does with Control."""
        mods = (e.modifiers() & ~Qt.ShiftModifier) | Qt.ControlModifier
        return QMouseEvent(e.type(), e.position(), e.globalPosition(), e.button(), e.buttons(), mods)


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
        # Shift on a connector moves its links (the scene's); elsewhere it adds to the selection
        self._shift_select = b == Qt.LeftButton and bool(m & Qt.ShiftModifier) and \
            self.scene()._port_at(self.mapToScene(e.position().toPoint())) is None
        super().mousePressEvent(self._shift_as_control(e) if self._shift_select else e)

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
        if self._left_drag():
            self._edge_pos = e.position()
            if not self._edge_timer.isActive():
                self._edge_timer.start()
        super().mouseMoveEvent(self._shift_as_control(e) if self._shift_select and e.modifiers() & Qt.ShiftModifier
                               else e)

    def mouseReleaseEvent(self, e):
        if self._pan is not None or self._zoom_drag is not None:
            self._pan = self._zoom_drag = None
            self.unsetCursor()
            return
        shift, self._shift_select = self._shift_select, False
        super().mouseReleaseEvent(self._shift_as_control(e) if shift else e)

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
