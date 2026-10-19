"""The look: a warm graphite dark theme (amber accent, teal color links, Inter), as a Fusion palette plus a
stylesheet."""

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSize, Qt
from PySide6.QtGui import QColor, QFontDatabase, QPainter, QPalette
from PySide6.QtWidgets import QApplication, QSizePolicy, QToolButton, QWidget

BG = "#292624"  # panels, panel titles, rollout headers
BG_DARK = "#201e1c"  # viewports, fields
BG_HEADER = "#33302d"  # selected tabs, table headers
BG_ROLLOUT = "#3a3633"
BG_BUTTON = "#3f3a36"
BG_SELECTED = "#5a3a22"  # selected rows, checked tool buttons
BORDER = "#151312"
BORDER_LIGHT = "#4a443f"  # the lit top edge of buttons
GROOVE = "#5a524b"  # slider grooves
TEXT = "#e2dbd3"
TEXT_DIM = "#948a80"
ACCENT = "#e8833a"  # selection, focus, the active panel
WARNING = "#e8a33a"
COLOR_LINK = "#3fc1c9"  # color links and ports
GRAY_LINK = "#c9c0b6"  # grayscale links and ports
RADIUS = 5
FONT = "Inter Variable"  # editor/fonts/InterVariable.ttf (SIL OFL 1.1, LICENSE-Inter.txt), loaded by load_font()


def load_font() -> None:
    """The UI font, bundled: the same text on every platform."""
    QFontDatabase.addApplicationFont(str(Path(__file__).with_name("fonts") / "InterVariable.ttf"))


def dropdown_arrows() -> str:
    """The rules that give dropdowns their arrow (a styled drop-down draws none): the "dropdown" icon written as images,
    plain and @2x, into the temporary folder, where a stylesheet can point at them; dimmed when disabled."""
    import tempfile
    from trafilatura3d.editor.icons import icon
    folder = Path(tempfile.gettempdir()) / "trafilatura3d-ui"
    folder.mkdir(exist_ok=True)
    rules = ""
    for state, color in (("", TEXT), (":disabled", GROOVE)):
        name = f"dropdown{'-disabled' if state else ''}"
        for scale in (1, 2):
            icon("dropdown", color).pixmap(QSize(16, 16), scale).save(
                str(folder / f"{name}{'@2x' if scale == 2 else ''}.png"))
        rules += f'QComboBox::down-arrow{state} {{ image: url("{(folder / name).as_posix()}.png"); width: 14px; height: 14px; }}\n'
    return rules


STYLESHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-family: "{FONT}"; font-size: 12px; }}
QMainWindow::separator:vertical {{ width: 1px; background: {BORDER}; border-left: 2px solid {BG};
    border-right: 2px solid {BG}; }}
QMainWindow::separator:horizontal {{ height: 1px; background: {BORDER}; border-top: 2px solid {BG};
    border-bottom: 2px solid {BG}; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget > QAbstractScrollArea {{ border: none; }}  /* a panel's body draws no frame: the dividers do */
QMenuBar {{ border-bottom: 1px solid {BORDER}; }}
QStatusBar {{ border-top: 1px solid {BORDER}; }}
*[divider="top"] {{ border-top: 1px solid {BORDER}; }}
*[divider="bottom"] {{ border-bottom: 1px solid {BORDER}; }}
*[divider="left"] {{ border-left: 1px solid {BORDER}; }}
*[divider="right"] {{ border-right: 1px solid {BORDER}; }}
QToolTip {{ background: {BG_DARK}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: {RADIUS}px;
    padding: 4px 6px; }}
QMenuBar {{ background: {BG}; }}
QMenuBar::item {{ padding: 3px 8px; border-radius: {RADIUS - 2}px; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {ACCENT}; color: white; }}
QMenu {{ background: {BG_DARK}; border: 1px solid {BORDER}; border-radius: {RADIUS + 2}px; padding: 4px; }}
QMenu::item {{ padding: 4px 18px; border-radius: {RADIUS - 2}px; }}
QMenu::separator {{ height: 1px; background: {BG_BUTTON}; margin: 3px 6px; }}
QToolBar {{ background: {BG}; border: none; spacing: 2px; padding: 2px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: {RADIUS - 1}px; padding: 2px; }}
QToolButton:hover {{ background: {BG_BUTTON}; }}
QToolButton:checked {{ background: {BG_SELECTED}; border-color: {ACCENT}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {BG_DARK}; border: 1px solid #312d2a; border-top-color: {BORDER};  /* inset: shaded top edge */
    border-radius: {RADIUS - 1}px; padding: 3px 6px; selection-background-color: {ACCENT}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {BG_DARK}; selection-background-color: {ACCENT}; }}
QPushButton {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #423c37, stop:1 #39342f);
    border: 1px solid {BORDER}; border-top-color: {BORDER_LIGHT};  /* raised: lit top edge */
    border-radius: {RADIUS}px; padding: 4px 12px; }}
QPushButton:hover {{ background: #48413b; }}
QPushButton:checked {{ background: {ACCENT}; color: white; }}
QPushButton:disabled {{ color: {TEXT_DIM}; }}
QTreeView, QListView, QTableView {{ background: {BG}; border: none; outline: none;
    selection-background-color: transparent; }}
QTreeView::item:selected, QListView::item:selected {{ background: {BG_SELECTED}; border: none; color: white; }}
QTreeView::item:hover, QListView::item:hover {{ background: #322e2b; }}
QHeaderView::section {{ background: {BG_HEADER}; border: none; padding: 3px; }}
QScrollBar:vertical {{ background: {BG}; width: 8px; margin: 0; }}
QScrollBar:horizontal {{ background: {BG}; height: 8px; margin: 0; }}
QScrollBar::handle {{ background: #544c46; border-radius: 4px; min-height: 20px; min-width: 20px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QSlider::groove:horizontal {{ height: 4px; border-radius: 2px; background: {GROOVE}; }}
QSlider::sub-page:horizontal {{ height: 4px; border-radius: 2px; background: {ACCENT}; }}
QSlider::handle:horizontal {{ background: {TEXT}; border: none; width: 12px; height: 12px; margin: -4px 0;
    border-radius: 6px; }}
QSplitter::handle:horizontal {{ width: 5px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {BG},
    stop:0.399 {BG}, stop:0.4 {BORDER}, stop:0.6 {BORDER}, stop:0.601 {BG}, stop:1 {BG}); }}
QSplitter::handle:vertical {{ height: 5px; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {BG},
    stop:0.399 {BG}, stop:0.4 {BORDER}, stop:0.6 {BORDER}, stop:0.601 {BG}, stop:1 {BG}); }}
QStatusBar {{ background: {BG}; color: {TEXT_DIM}; }}
QStatusBar QLabel {{ color: {TEXT}; padding: 0 8px; }}
QTabBar::tab {{ background: {BG}; padding: 5px 12px; border: none; }}
QTabBar::tab:selected {{ background: {BG_HEADER}; border-bottom: 2px solid {ACCENT}; }}
QMainWindow > QTabBar {{ border-top: 1px solid {BORDER}; }}  /* tabbed panels: the line over their tabs */
QMainWindow > QTabBar::tab:!selected {{ border-top: 1px solid {BORDER}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 4px; border: 1px solid {BORDER};
    background: {BG_DARK}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QLabel#dim {{ color: {TEXT_DIM}; }}
"""


def divider(widget, side: str):
    """The UI's one divider: a 1 px BORDER line on the `side` ("top", "bottom", "left", "right") of a bar where it
    meets the content next to it (panel titles, toolbars, a panel's own bars, a viewport's header). Splitters and
    the dock separators draw the same line centered in their 5 px grip; panels have no frame of their own."""
    widget.setProperty("divider", side)
    widget.setAttribute(Qt.WA_StyledBackground, True)  # a plain QWidget draws no stylesheet border without it
    # The line's pixel kept from the children (a stylesheet border takes no room from a plain widget's layout,
    # so children with no margin painted over it): the widget's own margins add to its layout's.
    m = widget.contentsMargins()
    widget.setContentsMargins(m.left() + (side == "left"), m.top() + (side == "top"),
                              m.right() + (side == "right"), m.bottom() + (side == "bottom"))
    return widget


class Overflow(QWidget):
    """A bar that does not set the window's minimum width: as wide as it needs, the bar is laid out as it is; narrower,
    it keeps its minimum width and is cut on the right, where "»" shows the cut end ("«" back to the start)."""

    def __init__(self, bar: QWidget):
        super().__init__()
        self.bar = bar
        bar.setParent(self)
        bar.installEventFilter(self)  # its contents change (shown, hidden, resized): placed again
        self.at_end = False
        self.cap = QWidget(self)  # the button's ground, over the cut bar (the bar's dividers left visible)
        self.cap.setAttribute(Qt.WA_StyledBackground, True)
        self.cap.setStyleSheet(f"background: {BG};")
        self.button = QToolButton(self.cap)
        self.button.setAutoRaise(True)
        self.button.clicked.connect(self._toggle)
        self.setSizePolicy(QSizePolicy.Preferred, bar.sizePolicy().verticalPolicy())

    def sizeHint(self):
        return self.bar.sizeHint()

    def minimumSizeHint(self):
        return QSize(self.button.sizeHint().width(), self.bar.minimumSizeHint().height())

    def eventFilter(self, obj, event):
        if event.type() == QEvent.LayoutRequest:
            self.updateGeometry()
            self._place()
        return False

    def resizeEvent(self, event):
        self._place()

    def _toggle(self):
        self.at_end = not self.at_end
        self._place()

    def _place(self):
        w, h, need = self.width(), self.height(), self.bar.minimumSizeHint().width()
        cut = w < need
        self.bar.setGeometry(w - need if cut and self.at_end else 0, 0, need if cut else w, h)
        self.cap.setVisible(cut)
        if cut:
            m = self.bar.contentsMargins()
            bw = self.button.sizeHint().width()
            self.cap.setGeometry(0 if self.at_end else w - bw, m.top(), bw, h - m.top() - m.bottom())
            self.button.setGeometry(0, 0, bw, self.cap.height())
            self.button.setText("«" if self.at_end else "»")
            self.button.setToolTip("Show the start of the bar" if self.at_end else "Show the rest of the bar")
            self.cap.raise_()


class _SectionLines(QObject):
    """Paints, after a panel body's own painting, the divider under each of its section titles (`header` labels
    that are its direct children) across the body's whole width, in the gap below the title: the panel's margins
    would cut a line drawn by the title itself short of the panel's edges."""

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Paint:
            return False
        obj.removeEventFilter(self)
        QApplication.sendEvent(obj, event)  # the body paints itself first
        obj.installEventFilter(self)
        p = QPainter(obj)
        p.setPen(QColor(BORDER))
        for child in obj.children():
            if getattr(child, "property", None) and child.property("section") and child.isVisible():
                y = child.geometry().bottom() + 1
                p.drawLine(0, y, obj.width(), y)
        p.end()
        return True


_SECTION_LINES = None


def section_lines(body):
    """A panel body whose section titles get the divider under them, edge to edge."""
    global _SECTION_LINES
    if _SECTION_LINES is None:
        _SECTION_LINES = _SectionLines()
    body.installEventFilter(_SECTION_LINES)
    return body


def palette() -> QPalette:
    p = QPalette()
    for role, value in [(QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, BG_DARK),
                        (QPalette.AlternateBase, BG), (QPalette.Text, TEXT), (QPalette.Button, BG_BUTTON),
                        (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT),
                        (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, BG_DARK),
                        (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, TEXT_DIM)]:
        p.setColor(role, QColor(value))
    return p
