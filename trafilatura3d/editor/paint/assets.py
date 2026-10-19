"""The Assets panel (Assets pages): the resources usable in the project, as a searchable grid of thumbnails
filtered by type: Alphas (generated, paint/alphas.py), Textures (imported into the project), Procedurals
(Trafilatura 3D Graph's texture generators), Materials (Trafilatura 3D graphs with channel outputs), Smart materials,
Filters and Generators (the Paint ones, then Trafilatura 3D Graph's filters and mask
generators).

Every asset drags (the layer stack and the Properties' resource slots take what they can use, as in the
application) and double clicks: an alpha goes to the tool, a material becomes a fill layer, a smart material
is added above the selection, a filter or a generator is added to the selected layer (its mask when the mask is
selected).

A resource slot (ResourceSlot) shows its resource's thumbnail and name; a click opens a picker (ResourcePicker:
search, type, the same thumbnails). Ours: the documentation does not show the picker.

Not offered yet: libraries on disk and their configuration, saved searches, sub-libraries, online assets,
smart masks, environments, presets."""

from __future__ import annotations

import json

import numpy as np
from PySide6.QtCore import QByteArray, QMimeData, QSettings, QSize, Qt
from PySide6.QtGui import QActionGroup, QDrag, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
                               QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from trafilatura3d.editor import theme
from trafilatura3d.editor.library import SIZES as LIBRARY_SIZES
from trafilatura3d.editor.paint.icons import picon
from trafilatura3d.editor.paint.properties import alpha_icon
from trafilatura3d.paint.alphas import ALPHAS

MIME = "application/x-trafilatura3d-paint-asset"
TYPES = ("All", "Alphas", "Textures", "Procedurals", "Materials", "Smart materials", "Filters", "Generators")
TYPE_OF = {"alpha": "Alphas", "texture": "Textures", "procedural": "Procedurals", "material": "Materials",
           "smart": "Smart materials", "filter": "Filters", "generator": "Generators",
           "anchor": "Anchor points"}  # asset kind -> type (anchor points: in the slots of the stack only)
ANCHOR = "anchor:"  # an anchor point's reference from a slot's picker: "anchor:<effect uid>"


def image_icon(pixels: np.ndarray, size: int = 64) -> QIcon:
    h, w = pixels.shape[:2]
    step = max(1, max(h, w) // size)
    a = pixels[::step, ::step, :3]
    if a.shape[2] == 1:  # a grayscale image (Resources: one channel)
        a = a.repeat(3, 2)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    a = np.ascontiguousarray(a)
    img = QImage(a.data, a.shape[1], a.shape[0], a.shape[1] * 3, QImage.Format_RGB888).copy()
    return QIcon(QPixmap.fromImage(img))


def node_icon(node_type: str) -> QIcon:
    from trafilatura3d.editor.library import library_icon
    return library_icon(node_type) or picon("effect")


def effects_of(kind: str) -> list[tuple[str, str, str]]:
    """(procedural id, label, group) of the filters or generators: the Paint ones, then Trafilatura 3D Graph's."""
    from trafilatura3d.paint import graph_nodes as dn
    nodes = dn.filters() if kind == "filter" else dn.generators()
    return [(dn.PREFIX + n.type, n.label, dn.group_of(n)) for n in nodes]


def procedurals() -> list[tuple[str, str, str]]:
    from trafilatura3d.paint import graph_nodes as dn
    return [(dn.PREFIX + n.type, n.label, n.category.split("/")[-1]) for n in dn.procedurals()]


def asset_ref(asset: dict) -> str:
    """A texture's, procedural's or anchor point's reference in a channel source: ANCHOR + the anchor's uid,
    the procedural's id, else the name."""
    if asset["kind"] == "anchor":
        return ANCHOR + asset["uid"]
    return asset.get("id") or asset.get("name", "")


def source_asset(src) -> dict | None:
    """The asset of a channel source (a ChannelSource or a stencil's): its anchor point, procedural or texture."""
    if src is None:
        return None
    if getattr(src, "anchor", ""):
        return {"kind": "anchor", "uid": src.anchor}
    if src.procedural:
        return {"kind": "procedural", "id": src.procedural}
    return {"kind": "texture", "name": src.image} if src.image else None


class AssetGrid(QListWidget):
    """The grid: items drag their asset reference (MIME)."""

    def startDrag(self, actions):
        it = self.currentItem()
        if it is None:
            return
        mime = QMimeData()
        mime.setData(MIME, QByteArray(json.dumps(it.data(Qt.UserRole)).encode()))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(it.icon().pixmap(48, 48))
        drag.exec(Qt.CopyAction)


def asset_of(mime) -> dict | None:
    """The asset a drag carries, or None."""
    if not mime.hasFormat(MIME):
        return None
    try:
        return json.loads(bytes(mime.data(MIME)).decode())
    except ValueError:
        return None


# Thumbnail sizes of the Assets: the Graph Library's (one source for the two)
THUMB_SIZES = {"Small": LIBRARY_SIZES["Small Icons"], "Medium": LIBRARY_SIZES["Medium Icons"],
               "Large": LIBRARY_SIZES["Large Icons"]}

class AssetsPanel(QWidget):
    def __init__(self, workspace):
        super().__init__()
        self.ws = workspace
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        head = theme.divider(QWidget(), "bottom")  # path, search and type over the grid
        lay = QVBoxLayout(head)
        lay.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(head)
        crumb = QLabel("›  All libraries")
        crumb.setStyleSheet("font-weight: bold; padding: 2px;")
        lay.addWidget(crumb)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search")
        self.search.addAction(picon("search"), QLineEdit.LeadingPosition)
        self.search.textChanged.connect(lambda _t: self.refresh())
        lay.addWidget(self.search)
        self.kind_box = QComboBox()
        self.kind_box.addItems(TYPES)
        self.kind_box.currentTextChanged.connect(lambda _t: self.refresh())
        lay.addWidget(self.kind_box)
        self.grid = AssetGrid()
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setWordWrap(True)
        self.grid.setUniformItemSizes(True)
        self.grid.itemClicked.connect(self._clicked)
        self.grid.itemDoubleClicked.connect(self._double_clicked)
        self._materials = None  # (package, graph, label) list, found once
        self._icons: dict = {}
        outer.addWidget(self.grid, 1)
        foot = theme.divider(QWidget(), "top")  # refresh and import under the grid
        bottom = QHBoxLayout(foot)
        bottom.setContentsMargins(4, 2, 4, 2)
        # Thumbnail size (Customizing the layout page): the left-most icon, Small, Medium (default), Large, List
        self.size_button = QToolButton()
        self.size_button.setIcon(picon("thumb_size"))
        self.size_button.setToolTip("Thumbnail size")
        self.size_button.setAutoRaise(True)
        self.size_button.setPopupMode(QToolButton.InstantPopup)
        size_menu = QMenu(self.size_button)
        group = QActionGroup(size_menu)
        current = QSettings().value("paint/assets_size", "Medium")
        current = current if current in THUMB_SIZES or current == "List" else "Medium"
        for mode in list(THUMB_SIZES) + ["List"]:
            a = size_menu.addAction(mode, lambda mode=mode: self.set_thumbnail_size(mode))
            a.setCheckable(True)
            a.setChecked(mode == current)
            group.addAction(a)
        self.size_button.setMenu(size_menu)
        bottom.addWidget(self.size_button)
        refresh = QToolButton()
        refresh.setIcon(picon("random"))
        refresh.setToolTip("Refresh (search the libraries again)")
        refresh.setAutoRaise(True)
        refresh.clicked.connect(self.rescan)
        bottom.addWidget(refresh)
        bottom.addStretch(1)
        imp = QToolButton()
        imp.setIcon(picon("add"))
        imp.setToolTip("Import resources")
        imp.setAutoRaise(True)
        imp.clicked.connect(self.ws.import_resources_dialog)
        bottom.addWidget(imp)
        outer.addWidget(foot)
        self.set_thumbnail_size(current, save=False)
        self.refresh()

    @property
    def kind(self) -> str:
        return self.kind_box.currentText()

    def set_kind(self, kind: str):
        self.kind_box.setCurrentText(kind)

    def rescan(self):
        self._materials = None
        self._icons.clear()
        self.refresh()

    # --- listing --------------------------------------------------------------------------------------

    def _icon(self, key, make):
        if key not in self._icons:
            try:
                self._icons[key] = make()
            except Exception:  # noqa: BLE001 - an asset whose thumbnail fails shows without one
                self._icons[key] = QIcon()
        return self._icons[key]

    def icon_of(self, asset: dict) -> QIcon:
        """An asset's thumbnail (made once)."""
        kind = asset["kind"]
        if kind == "alpha":
            return self._icon(("alpha", asset["name"]), lambda: alpha_icon(asset["name"], 64))
        if kind == "texture":
            if self.ws.doc is None:
                return QIcon()
            res = self.ws.doc.session.resources  # an image sequence: its first image
            image = dict(res.listing()).get(asset["name"], asset["name"])
            if image not in res.images:
                return QIcon()
            return self._icon(("tex", image, res.stamps.get(image)), lambda: image_icon(res.images[image]))
        if kind in ("procedural", "filter", "generator"):
            return self._icon(("node", asset["id"]), lambda: node_icon(asset["id"][5:]))
        if kind == "material":
            return self._icon(("mat", asset["package"], asset["graph"]),
                              lambda: self._material_icon(asset["package"], asset["graph"]))
        if kind == "anchor":
            return picon("anchor")
        return picon("folder")

    def entries(self, types: tuple | None = None, owner: str | None = None) -> list[tuple[str, QIcon, dict, str]]:
        """(label, icon, asset, tooltip) of the `types` (default: the type shown); the anchor points a slot of the
        layer `owner` can reference (those under it)."""
        from trafilatura3d.editor.paint import settings as prefs
        from trafilatura3d.paint import smart
        types, doc, out = types or (self.kind,), self.ws.doc, []
        want = lambda k: "All" in types or k in types
        if want("Alphas"):
            out += [(n, {"kind": "alpha", "name": n}, "Alpha") for n in ALPHAS]
        if doc is not None and want("Textures"):  # an image sequence once
            out += [(n, {"kind": "texture", "name": n}, "Texture" if n == i else "Image sequence")
                    for n, i in doc.session.resources.listing()]
        if want("Procedurals"):
            out += [(label, {"kind": "procedural", "id": i}, f"Procedural ({group})") for i, label, group in procedurals()]
        if doc is not None and want("Materials"):
            out += [(label, {"kind": "material", "package": path, "graph": gid, "label": label}, f"{path}#{gid}")
                    for path, gid, label in self.materials()]
        if want("Smart materials"):
            out += [(smart.name_of(p), {"kind": "smart", "path": str(p)}, str(p))
                    for p in smart.library(prefs.folders("smart_materials"))]
        for k, title in (("filter", "Filters"), ("generator", "Generators")):
            if want(title):
                out += [(label, {"kind": k, "id": i, "label": label}, f"{title[:-1]} ({group})")
                        for i, label, group in effects_of(k)]
        if doc is not None and owner and want("Anchor points"):
            out += [(fx.name, {"kind": "anchor", "uid": fx.uid},
                     f"Anchor point in {layer.name}" + (" (mask)" if m else ""))
                    for layer, fx, m in doc.ts.anchors(below=owner)]
        return [(label, self.icon_of(asset), asset, tip) for label, asset, tip in out]

    def set_thumbnail_size(self, mode: str, save: bool = True):
        """Small, Medium, Large thumbnails (the Graph Library's sizes) or a List with the full names."""
        if save:
            QSettings().setValue("paint/assets_size", mode)
        if mode == "List":
            self.grid.setViewMode(QListWidget.ListMode)
            self.grid.setIconSize(QSize(24, 24))
            self.grid.setGridSize(QSize())
        else:
            s = THUMB_SIZES[mode]
            self.grid.setViewMode(QListWidget.IconMode)
            self.grid.setIconSize(QSize(s, s))
            self.grid.setGridSize(QSize(s + 28, s + 34))
        self.grid.setMovement(QListWidget.Static)
        self.grid.setDragEnabled(True)  # after setViewMode / setMovement: Qt turns it off there
        self.thumbnail_size = mode

    def refresh(self):
        self.grid.clear()
        q = self.search.text().strip().lower()
        for label, icon, asset, tip in self.entries():
            if q and q not in label.lower():
                continue
            it = QListWidgetItem(icon, label)
            it.setData(Qt.UserRole, asset)
            it.setToolTip(f"{label}\n{tip}")  # the full name, then what it is
            self.grid.addItem(it)

    def materials(self) -> list:
        """The material graphs found in the libraries (Settings > Libraries configuration) and the project's
        library folders (searched once; Refresh searches again)."""
        if self._materials is None:
            from trafilatura3d.paint.materials import Materials
            lib = self.ws.doc.session.material_library if self.ws.doc else Materials()
            from trafilatura3d.editor.paint import settings as prefs
            self._materials = lib.available(prefs.material_folders())
        return self._materials

    def _material_icon(self, path, gid):
        doc = self.ws.doc
        outs = doc.session.material_library.outputs({"package": path, "graph": gid, "values": {}}, (128, 128),
                                                    doc.session.ev)
        if "basecolor" not in outs:
            return QIcon()
        img = doc.session.ev.thumbnails([outs["basecolor"]], 64)[0]
        return image_icon(img.astype(np.float32) / 255.0)

    # --- using assets ---------------------------------------------------------------------------------

    def _clicked(self, it):
        a = it.data(Qt.UserRole)
        if a["kind"] == "alpha":
            self.ws.tools.alphas[self.ws.tools.tool] = a["name"]
            self.ws.properties.rebuild()

    def _double_clicked(self, it):
        self.ws.use_asset(it.data(Qt.UserRole))



class ResourcePicker(QFrame):
    """A resource slot's picker, under the slot: a search field (focused; Enter takes the first match), the type
    when the slot takes several, the Assets' thumbnails of those types (a click takes one), `none` first when the
    slot can be emptied, Import under the textures. `use(asset)`, None for `none`."""

    def __init__(self, panel: AssetsPanel, kinds: tuple, use, none: str | None = None, owner: str | None = None):
        super().__init__(None, Qt.Popup)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setFrameShape(QFrame.StyledPanel)
        self.panel, self.use, self.none, self.owner = panel, use, none, owner
        self.types = tuple(TYPE_OF[k] for k in kinds)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search")
        self.search.addAction(picon("search"), QLineEdit.LeadingPosition)
        self.search.textChanged.connect(lambda _t: self.refresh())
        self.search.returnPressed.connect(self._first)
        lay.addWidget(self.search)
        self.type_box = QComboBox()
        self.type_box.addItems(("All", *self.types))
        self.type_box.currentTextChanged.connect(lambda _t: self.refresh())
        self.type_box.setVisible(len(self.types) > 1)
        lay.addWidget(self.type_box)
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setMovement(QListWidget.Static)
        self.grid.setWordWrap(True)
        self.grid.setUniformItemSizes(True)
        s = THUMB_SIZES.get(panel.thumbnail_size, THUMB_SIZES["Medium"])
        self.grid.setIconSize(QSize(s, s))
        self.grid.setGridSize(QSize(s + 28, s + 34))
        self.grid.itemClicked.connect(lambda it: self._take(it.data(Qt.UserRole)))
        lay.addWidget(self.grid, 1)
        if "Textures" in self.types:
            imp = QToolButton()
            imp.setIcon(picon("add"))
            imp.setText("Import…")
            imp.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            imp.setAutoRaise(True)
            imp.clicked.connect(self._import)
            lay.addWidget(imp)
        self.resize(4 * (s + 28) + 40, 480)
        self.refresh()

    def refresh(self):
        self.grid.clear()
        q = self.search.text().strip().lower()
        if self.none and not q:
            it = QListWidgetItem(QIcon(), self.none)
            it.setData(Qt.UserRole, None)
            self.grid.addItem(it)
        types = self.types if self.type_box.currentText() == "All" else (self.type_box.currentText(),)
        for label, icon, asset, tip in self.panel.entries(types, self.owner):
            if q and q not in label.lower():
                continue
            it = QListWidgetItem(icon, label)
            it.setData(Qt.UserRole, asset)
            it.setToolTip(f"{label}\n{tip}")
            self.grid.addItem(it)

    def _first(self):
        for i in range(self.grid.count()):
            asset = self.grid.item(i).data(Qt.UserRole)
            if asset is not None:
                self._take(asset)
                return

    def _take(self, asset):
        self.close()
        self.use(asset)

    def _import(self):
        self.close()
        name = self.panel.ws.import_resources_dialog()
        if name:
            self.use({"kind": "texture", "name": name})

    def popup(self, under: QWidget):
        self.move(under.mapToGlobal(under.rect().bottomLeft()))
        self.show()
        self.search.setFocus()


class ResourceSlot(QToolButton):
    """A resource slot of the Properties: the resource's thumbnail and name (`none` when empty); a click opens the
    ResourcePicker of the `kinds` it takes, the Assets drop those kinds on it. `use(asset)`, None for `none`;
    `owner`: the layer of a slot of the stack (the anchor points it can reference)."""

    def __init__(self, panel: AssetsPanel, kinds: tuple, use, current: dict | None, label: str = "",
                 none: str | None = None, owner: str | None = None):
        super().__init__()
        from trafilatura3d.editor.paint.widgets import accept_assets
        self.picker_args = (panel, kinds, use, none, owner)
        self.setIconSize(QSize(48, 48))
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)  # centered, the name under the thumbnail
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setIcon(panel.icon_of(current) if current else QIcon())
        self.setText(label or (none or ""))
        self.setToolTip(self.text())
        self.clicked.connect(self.open_picker)
        accept_assets(self, kinds, use)

    def open_picker(self) -> ResourcePicker:
        p = ResourcePicker(*self.picker_args)
        p.popup(self)
        return p
