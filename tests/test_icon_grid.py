"""The icon grids (the Library, the Assets): in icon mode the mouse wheel scrolls by rows of the grid, as the list
mode scrolls by lines, whatever the number of items on a row."""

import os
import unittest


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class IconGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def grid(self, cls):
        from PySide6.QtCore import QSize
        from PySide6.QtWidgets import QListWidget, QListWidgetItem
        g = cls()
        g.setViewMode(QListWidget.IconMode)
        g.setResizeMode(QListWidget.Adjust)
        g.setUniformItemSizes(True)
        g.setIconSize(QSize(72, 72))
        g.setGridSize(QSize(100, 120))
        for i in range(300):
            g.addItem(QListWidgetItem(f"item {i}"))
        g.resize(340, 600)  # three items a row
        g.show()
        self.app.processEvents()
        self.addCleanup(g.deleteLater)
        return g

    def notch(self, g, n=1):
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent
        from PySide6.QtWidgets import QApplication
        at = QPoint(50, 50)
        for _ in range(n):
            QApplication.sendEvent(g.viewport(), QWheelEvent(
                QPointF(at), QPointF(g.viewport().mapToGlobal(at)), QPoint(0, 0), QPoint(0, -120), Qt.NoButton,
                Qt.NoModifier, Qt.NoScrollPhase, False))
        self.app.processEvents()
        return g.verticalScrollBar().value()

    def test_a_wheel_notch_scrolls_rows_of_the_grid(self):
        from trafilatura3d.editor.widgets import IconGrid
        g = self.grid(IconGrid)
        lines = self.app.wheelScrollLines()
        self.assertEqual(self.notch(g), lines * 120)  # whole rows, every item on them
        self.assertEqual(self.notch(g, 2), 3 * lines * 120)

    def test_the_list_mode_keeps_its_lines(self):
        from PySide6.QtWidgets import QListWidget
        from trafilatura3d.editor.widgets import IconGrid
        g = self.grid(IconGrid)
        g.setViewMode(QListWidget.ListMode)
        g.setGridSize(g.gridSize().__class__())
        self.app.processEvents()
        self.assertEqual(g.verticalScrollMode(), QListWidget.ScrollPerItem)
        self.assertEqual(self.notch(g), self.app.wheelScrollLines())  # lines, as before


if __name__ == "__main__":
    unittest.main()
