"""The node menu (Space in the Graph view): fuzzy search over every node, arrows move through the
results (looping), Enter inserts. Opened from a connector or with one node selected, it lists only
nodes whose primary input accepts that output's type."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout

from trafilatura3d.nodes import NODES
from trafilatura3d.nodes.base import PortType


def graph_nodes() -> dict:
    """The node types the Graph library offers: everything with a category except the Paint workspace's own effects and
    generators (they belong to the Paint workspace's shelf)."""
    return {t: d for t, d in NODES.items() if d.category and not d.category.startswith("Paint/")}


def fuzzy_score(query: str, text: str) -> int | None:
    """Higher is better; None when the query's letters don't appear in order."""
    q, t = query.lower().replace(" ", ""), text.lower()
    if not q:
        return 0
    if q in t.replace(" ", ""):
        return 1000 - t.find(q[0]) - len(t)
    pos, score, last = 0, 0, -2
    for ch in q:
        i = t.find(ch, pos)
        if i < 0:
            return None
        score += 10 if i == last + 1 else 1
        last, pos = i, i + 1
    return score - len(t) // 4


def library_entries(accepts: PortType | None = None) -> list[tuple[str, str, str]]:
    """(label, category, node type) of every library node, optionally only those whose primary input
    accepts `accepts`."""
    out = []
    for t, d in graph_nodes().items():
        if accepts is not None:
            port = d.primary_input()
            if port is None or d.inputs[port].type not in (PortType.ANY, accepts):
                continue
        out.append((d.label, d.category, t))
    return sorted(out)


class NodeMenu(QFrame):
    chosen = Signal(str)

    def __init__(self, parent, accepts: PortType | None = None, favorites=()):
        super().__init__(parent, Qt.Popup)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(300, 340)
        self.entries = library_entries(accepts)
        self.favorites = set(favorites)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search nodes…")
        self.list = QListWidget()
        lay.addWidget(self.search)
        lay.addWidget(self.list)
        self.search.textChanged.connect(self.refresh)
        self.search.installEventFilter(self)
        self.list.itemActivated.connect(lambda item: self._choose(item))
        self.refresh("")
        self.search.setFocus()

    def refresh(self, text: str):
        scored = []
        for label, category, t in self.entries:
            s = fuzzy_score(text, label)
            if s is None:
                s = fuzzy_score(text, category + " " + label)
                if s is None:
                    continue
                s -= 50
            scored.append((-(s + (500 if t in self.favorites else 0)), label, category, t))
        scored.sort()
        self.list.clear()
        from trafilatura3d.editor.library import library_icon  # (the Library imports this module)
        for _, label, category, t in scored[:200]:
            item = QListWidgetItem(f"{label}    —  {category}")
            if (i := library_icon(t)) is not None:
                item.setIcon(i)
            item.setData(Qt.UserRole, t)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def eventFilter(self, obj, e):
        if e.type() == e.Type.KeyPress:
            n = self.list.count()
            if e.key() in (Qt.Key_Down, Qt.Key_Up) and n:
                step = 1 if e.key() == Qt.Key_Down else -1
                self.list.setCurrentRow((self.list.currentRow() + step) % n)
                return True
            if e.key() in (Qt.Key_Return, Qt.Key_Enter):
                if self.list.currentItem():
                    self._choose(self.list.currentItem())
                return True
            if e.key() == Qt.Key_Escape:
                self.close()
                return True
        return super().eventFilter(obj, e)

    def _choose(self, item):
        self.chosen.emit(item.data(Qt.UserRole))
        self.close()
