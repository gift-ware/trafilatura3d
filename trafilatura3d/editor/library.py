"""The Library: categories on the left (Favorites first), labelled thumbnails on the
right, a search over the whole library, a display size, label toggle and favorites; items are dragged
into the Graph view. Generators show their cooked image (a few at a time so the interface stays responsive),
other filters their initials (thumbnails.cooked)."""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QByteArray, QMimeData, QRectF, QSettings, QSize, QStandardPaths, Qt, QTimer
from PySide6.QtGui import QColor, QDrag, QFont, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QSplitter,
                               QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from trafilatura3d.editor import theme
from trafilatura3d.editor.graph.items import title_color
from trafilatura3d.editor.graph.nodemenu import graph_nodes, fuzzy_score
from trafilatura3d.editor.icons import icon
from trafilatura3d.nodes import NODES
from trafilatura3d.package import SUFFIX
from trafilatura3d.sample import sample_job
from trafilatura3d.thumbnails import cooked

MIME = "application/x-trafilatura3d-node"
# Thumbnails shipped with the package (tools/library_thumbnails.py makes them) and the user cache for
# nodes added since.
SHIPPED_DIR = Path(__file__).parent / "thumbnails"
CACHE_DIR = Path(QStandardPaths.writableLocation(QStandardPaths.CacheLocation) or Path.home() / ".cache") \
    / "trafilatura3d" / "thumbnails"


def thumbnail_file(node_type: str) -> Path | None:
    for folder in (SHIPPED_DIR, CACHE_DIR):
        p = folder / f"{node_type}.png"
        if p.exists():
            return p
    return None


def initials_icon(node_type: str) -> QIcon:
    """A filter's Library icon: the initials of its label (two words: their first letters, one: its first two) on its
    category's title colour."""
    d = NODES[node_type]
    words = [w for w in re.split(r"[\s_/()-]+", d.label) if w]
    text = "".join(w[0] for w in words[:2]).upper() if len(words) > 1 else (words[0][:2].title() if words else "?")
    pix = QPixmap(96, 96)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(title_color(node_type, d.category).darker(130))
    p.drawRoundedRect(QRectF(4, 4, 88, 88), 10, 10)
    f = QFont()
    f.setPixelSize(38)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor("#f2f2f2"))
    p.drawText(QRectF(0, 0, 96, 96), Qt.AlignCenter, text)
    p.end()
    return QIcon(pix)


def library_icon(node_type: str) -> QIcon | None:
    """A node's icon in the Library, the Paint Assets and the Add node menus: its initials (a filter) or its
    shipped/cached thumbnail; None while the thumbnail is still to be cooked. Kept once made."""
    found = _ICONS.get(node_type)
    if found is None:
        if not cooked(node_type):
            found = initials_icon(node_type)
        else:
            f = thumbnail_file(node_type)
            found = QIcon(str(f)) if f else None
        if found is not None:
            _ICONS[node_type] = found
    return found


_ICONS: dict[str, QIcon] = {}


FAVORITES = "Favorites"
PROJECT = "Project"  # graphs of the packages in the project's watched folders
SIZES = {"Small Icons": 48, "Medium Icons": 72, "Large Icons": 104}


class LibraryList(QListWidget):
    def startDrag(self, actions):
        types = [i.data(Qt.UserRole) for i in self.selectedItems()]
        if not types:
            return
        mime = QMimeData()
        mime.setData(MIME, QByteArray("\n".join(types).encode()))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(self.selectedItems()[0].icon().pixmap(48, 48))
        drag.exec(Qt.CopyAction)


class LibraryPanel(QWidget):
    def __init__(self, evaluator):
        super().__init__()
        self.ev = evaluator
        self.settings = QSettings()  # the application's organization and name (app.py; tests use their own)
        self._scanned = {}
        self.favorites = set(self.settings.value("library/favorites", [], type=list))
        self.thumbs: dict[str, QIcon] = {}
        self._queue: list[str] = []
        self.timer = QTimer(self, interval=30, timeout=self._cook_some)
        # Search and display controls on top, categories and content side by side below.
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar_widget = theme.divider(QWidget(), "bottom")  # search and display controls over the content
        bar = QHBoxLayout(bar_widget)
        bar.setContentsMargins(4, 2, 4, 2)
        lay.addWidget(bar_widget)
        split = QSplitter()
        lay.addWidget(split, 1)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(110)
        self.tree.currentItemChanged.connect(lambda *_: self.refresh())
        split.addWidget(self.tree)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search")
        self.search.textChanged.connect(lambda *_: self.refresh())
        self.size_box = QComboBox()
        self.size_box.addItems(list(SIZES) + ["List"])
        self.size_box.setCurrentText(self.settings.value("library/display", "Medium Icons"))
        self.size_box.currentTextChanged.connect(self._display)
        self.labels_btn = QToolButton()
        self.labels_btn.setText("Aa")
        self.labels_btn.setToolTip("Toggle Labels")
        self.labels_btn.setCheckable(True)
        self.labels_btn.setChecked(True)
        self.labels_btn.toggled.connect(lambda *_: self.refresh())
        self.fav_btn = QToolButton()
        self.fav_btn.setText("★")
        self.fav_btn.setToolTip("Add to Favorites / remove from Favorites")
        self.fav_btn.clicked.connect(self._toggle_favorite)
        bar.addWidget(self.search, 1)
        bar.addWidget(self.size_box)
        bar.addWidget(self.labels_btn)
        bar.addWidget(self.fav_btn)
        self.list = LibraryList()
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setWordWrap(True)
        rv.addWidget(self.list)
        split.addWidget(right)
        split.setSizes([120, 280])
        self._fill_tree()
        self._display(self.size_box.currentText())

    def _fill_tree(self):
        self.tree.clear()
        fav = QTreeWidgetItem([FAVORITES])
        fav.setData(0, Qt.UserRole, FAVORITES)
        self.tree.addTopLevelItem(fav)
        cats = sorted({d.category for d in graph_nodes().values()},
                      key=lambda c: (_ORDER.index(c.split("/")[0]) if c.split("/")[0] in _ORDER else 99, c))
        items = {}
        for cat in cats:
            parent = None
            path = ""
            for part in cat.split("/"):
                path = f"{path}/{part}" if path else part
                if path not in items:
                    it = QTreeWidgetItem([part])
                    it.setData(0, Qt.UserRole, path)
                    (parent.addChild(it) if parent else self.tree.addTopLevelItem(it))
                    items[path] = it
                parent = items[path]
        project = self._project_graphs()
        if project:  # Project settings > Library: graphs of the packages in the watched folders
            root = QTreeWidgetItem([PROJECT])
            root.setData(0, Qt.UserRole, PROJECT)
            self.tree.addTopLevelItem(root)
            for cat in sorted({c for _, _, _, c in project}):
                it = QTreeWidgetItem([cat])
                it.setData(0, Qt.UserRole, f"{PROJECT}/{cat}")
                root.addChild(it)
        self.tree.expandAll()
        self.tree.setCurrentItem(items.get("Atomic nodes") or fav)

    def _project_graphs(self):
        """(package path, graph id, label, category) of the graphs in the project's watched folders,
        read again only when a package file changes."""
        from pathlib import Path
        from trafilatura3d.package import Package
        from trafilatura3d.project import configuration
        out = []
        for folder in configuration().library_paths():
            for f in sorted(Path(folder).rglob(f"*{SUFFIX}")) if Path(folder).is_dir() else []:
                key = (str(f), f.stat().st_mtime)
                if key not in self._scanned:
                    try:
                        pkg = Package.load(f)
                        self._scanned[key] = [(str(f), gid, g.label or gid, g.attributes.get("category") or f.stem)
                                              for gid, g in pkg.graphs.items()]
                    except Exception:  # noqa: BLE001 - a broken package simply doesn't show
                        self._scanned[key] = []
                out += self._scanned[key]
        return out

    def _display(self, mode):
        self.settings.setValue("library/display", mode)
        if mode == "List":
            self.list.setViewMode(QListWidget.ListMode)
            self.list.setIconSize(QSize(24, 24))
            self.list.setGridSize(QSize())
        else:
            s = SIZES[mode]
            self.list.setViewMode(QListWidget.IconMode)
            self.list.setIconSize(QSize(s, s))
            self.list.setGridSize(QSize(s + 28, s + 34))
        self.list.setMovement(QListWidget.Static)  # items stay in place...
        self.list.setDragEnabled(True)  # ...and drag out: setViewMode and setMovement(Static) turn it off
        self.refresh()

    def refresh(self):
        text = self.search.text().strip()
        cur = self.tree.currentItem()
        cat = cur.data(0, Qt.UserRole) if cur else ""
        entries = []
        for t, d in graph_nodes().items():
            if text:  # the search covers the whole library, ignoring the category
                s = fuzzy_score(text, d.label)
                if s is None and fuzzy_score(text, d.category) is None:
                    continue
                entries.append((-(s or 0), d.label, t))
            elif cat == FAVORITES:
                if t in self.favorites:
                    entries.append((0, d.label, t))
            elif d.category == cat or d.category.startswith(cat + "/"):
                entries.append((0, d.label, t))
        if cat == PROJECT or cat.startswith(PROJECT + "/") or text:
            for path, gid, label, gcat in self._project_graphs():
                if text and fuzzy_score(text, label) is None:
                    continue
                if text or cat == PROJECT or cat == f"{PROJECT}/{gcat}":
                    entries.append((0, label, f"graph:{path}#{gid}"))
        entries.sort()
        self.list.clear()
        self._queue = []
        for _, label, t in entries:
            if t.startswith("graph:"):
                item = QListWidgetItem(icon("graph"), label if self.labels_btn.isChecked() else "")
                item.setData(Qt.UserRole, t)
                item.setToolTip(f"{label}\n{t[6:]}")
                self.list.addItem(item)
                continue
            if t not in self.thumbs:
                found = library_icon(t)  # initials, shipped or cached: instant
                if found is not None:
                    self.thumbs[t] = found
            item = QListWidgetItem(self.thumbs.get(t, icon("graph")), label if self.labels_btn.isChecked() else "")
            item.setData(Qt.UserRole, t)
            item.setToolTip(f"{label}\n{NODES[t].category}\n{(NODES[t].__doc__ or '').strip()}")
            self.list.addItem(item)
            if t not in self.thumbs:
                self._queue.append(t)
        if self._queue:
            self.timer.start()

    def _cook_some(self):
        """One missing thumbnail per tick (a node added since the shipped set): the interface stays
        responsive; the result is saved in the user cache for next time."""
        if not self._queue:
            self.timer.stop()
            return
        t = self._queue.pop(0)
        try:
            job = sample_job(t, False, 7)
            img = self.ev.thumbnails([(job, next(iter(job.node.outputs)))], 96)[0]
            q = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 4, QImage.Format_RGBA8888).copy()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            q.save(str(CACHE_DIR / f"{t}.png"))
            self.thumbs[t] = QIcon(QPixmap.fromImage(q))
        except Exception:  # noqa: BLE001 - a failing node keeps the placeholder
            self.thumbs[t] = icon("graph")
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.UserRole) == t:
                item.setIcon(self.thumbs[t])

    def _toggle_favorite(self):
        for item in self.list.selectedItems():
            t = item.data(Qt.UserRole)
            self.favorites.symmetric_difference_update({t})
        self.settings.setValue("library/favorites", sorted(self.favorites))
        self.refresh()


_ORDER = ["Atomic nodes", "Texture generators", "Filters", "Material filters", "Mesh based generators", "3D View"]
