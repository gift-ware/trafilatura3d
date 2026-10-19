"""The editor's render canvases (2D View, 3D View, Paint viewport) share this base."""

from __future__ import annotations

import shiboken6
from rendercanvas.qt import QRenderWidget


class Canvas(QRenderWidget):
    """A render widget that counts as closed once its Qt object is gone: a canvas inside a window that is destroyed gets
    no close event, so the draw scheduler would keep drawing it, and a present already under way would finish on it.
    Every backend hook that touches the widget does nothing then."""

    def _alive(self) -> bool:
        return shiboken6.isValid(self)

    def _rc_get_closed(self):
        return not self._alive() or super()._rc_get_closed()

    def _rc_request_draw(self):
        if self._alive():
            super()._rc_request_draw()

    def _rc_request_paint(self):
        if self._alive():
            super()._rc_request_paint()

    def _rc_force_paint(self):
        if self._alive():
            super()._rc_force_paint()

    def _rc_present_bitmap(self, **kwargs):
        if self._alive():
            super()._rc_present_bitmap(**kwargs)

    def _rc_set_logical_size(self, width, height):
        if self._alive():
            super()._rc_set_logical_size(width, height)

    def _rc_set_title(self, title):
        if self._alive():
            super()._rc_set_title(title)

    def _rc_set_cursor(self, cursor):
        if self._alive():
            super()._rc_set_cursor(cursor)

    def _rc_close(self):
        if self._alive():
            super()._rc_close()
