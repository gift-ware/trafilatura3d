"""The Layer stack panel (Layer stack, Creating layers, Managing layers, Masks pages): the viewmode channel
dropdown, the action buttons (mask menu, new Paint layer, new Fill layer, new folder, delete), one row per
layer, top first, with visibility, thumbnails of the content (and of the mask) in the viewmode channel, the
name, the channel's blending mode and opacity. Drag to reorder or to put into a folder; right click for the
layer's menu. Shift + click on the mask thumbnail toggles the mask; Ctrl/Cmd + D duplicates, Ctrl/Cmd + G
groups into a folder.

Effects (Effects page): Add Effect puts a Fill, Paint or Levels effect on the selected layer's content, or on
its mask when the mask is selected; they show under the layer (the mask's when the mask is selected), with
their blending mode and opacity (Levels has none); Alt + Up / Down or the right click menu moves them.

Add Effect also adds a Generator or a Filter (chosen in the Properties, paint/graph_nodes.py).

Not offered yet: anchor points, compare mask, color selection; Smart Materials, bitmap /
color selection / height combination masks, instancing, geometry masks, flattening."""

from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPixmap, QShortcut, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QHBoxLayout, QInputDialog, QLabel, QMenu,
                               QSpinBox, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from trafilatura3d.editor import theme
from trafilatura3d.editor.paint.icons import picon
from trafilatura3d.editor.paint.widgets import Slider
from trafilatura3d.paint import blending
from trafilatura3d.paint.channels import CHANNELS
from trafilatura3d.paint.model import (GEOMETRY_TYPES, MASK_KEY, ChannelBlend, ChannelSource, Effect, Levels, Mask,
                                   new_layer, new_uid, unique_name)

THUMB = 30
_BLEND_MODELS: dict = {}


def blend_model(normal: bool) -> QStandardItemModel:
    """The blending modes of the rows' dropdowns (short label, mode identifier, full name as tooltip), made once and
    shared by every row: filling ~30 items per row was most of making one. The Normal map modes only on the Normal
    channel (`normal`)."""
    if normal not in _BLEND_MODELS:
        model = QStandardItemModel(QApplication.instance())
        for m in blending.IDS:
            if normal or m not in blending.NORMAL_ONLY:
                item = QStandardItem(blending.SHORT[m])
                item.setData(m, Qt.UserRole)
                item.setToolTip(blending.LABEL[m])
                model.appendRow(item)
        _BLEND_MODELS[normal] = model
    return _BLEND_MODELS[normal]


# The Add effect menus (layer, mask, toolbar): Effects page
ADD_EFFECTS = (("fill", "Add fill"), ("paint", "Add paint"), ("levels", "Add levels"), ("generator", "Add generator"),
               ("filter", "Add filter"), ("anchor", "Add anchor point"))


def _pixmap(rgba: np.ndarray) -> QPixmap:
    """A thumbnail over a checkerboard where it is transparent (unpainted)."""
    h, w = rgba.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    checker = np.where(((x // 5) + (y // 5)) % 2 == 0, 90, 60).astype(np.float32)[..., None]
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * alpha + checker * (1 - alpha)
    a = np.ascontiguousarray(np.concatenate([rgb, np.full((h, w, 1), 255.0)], 2).astype(np.uint8))
    return QPixmap.fromImage(QImage(a.data, w, h, w * 4, QImage.Format_RGBA8888).copy())


class EyeButton(QToolButton):
    """A layer's visibility. Pressed, it turns the layer on or off; dragged with the button held over the other
    layers' eyes, it sets them the same way."""

    def __init__(self, panel, uid):
        super().__init__()
        self.panel, self.uid = panel, uid
        self.setAutoRaise(True)

    def show_visible(self, on: bool) -> None:
        self.setIcon(picon("eye" if on else "eye_off"))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.panel.eye_press(self.uid)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):  # the press grabs the mouse: the layer under the cursor is looked up
        uid = self.panel.eye_at(e.globalPosition().toPoint())
        if uid is not None:
            self.panel.eye_over(uid)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.panel.eye_release()
            e.accept()
            return
        super().mouseReleaseEvent(e)


class Row(QWidget):
    """A layer's row."""

    def __init__(self, panel, layer):
        super().__init__()
        self.panel, self.uid = panel, layer.uid
        self.setObjectName("stackRow")
        doc = panel.doc
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 4, 2)
        lay.setSpacing(4)
        self.eye = EyeButton(panel, layer.uid)
        self.eye.show_visible(layer.visible)
        lay.addWidget(self.eye)
        if layer.kind == "folder":
            f = QLabel()
            f.setPixmap(picon("folder").pixmap(16, 16))
            lay.addWidget(f)
        self.thumb = QLabel()
        self.thumb.setObjectName("layerThumb")  # its look in the theme (a style sheet per widget is slow to make)
        self.thumb.setFixedSize(THUMB, THUMB)
        self.thumb.mousePressEvent = self._thumb_clicked
        lay.addWidget(self.thumb)
        self.mask = None
        if layer.mask is not None:
            self.mask = QLabel()
            self.mask.setObjectName("layerMask")
            self.mask.setFixedSize(THUMB, THUMB)
            self.mask_enabled = layer.mask.enabled
            self.show_mask_selected(doc.selected == layer.uid and doc.mask_selected)
            self.mask.mousePressEvent = self._mask_clicked
            self.mask.setContextMenuPolicy(Qt.CustomContextMenu)
            self.mask.customContextMenuRequested.connect(
                lambda pos: panel.mask_menu(self.uid, self.mask.mapToGlobal(pos)))
            self.mask.setToolTip("Click: select the mask · Shift+click: toggle it · Alt+click: show it alone in "
                                 "the viewport · Right click: its menu")
            lay.addWidget(self.mask)
        self.name = QLabel(layer.name)
        lay.addWidget(self.name, 1)
        # The Geometry mask: its state on the icon (what its Masking type includes), its menu on a click or a
        # right click.
        if layer.geometry_type == "UV Tiles":
            names, excluded, what = doc.ts.tiles, [t for t in layer.geometry_tiles if t in doc.ts.tiles], "UV tiles"
        else:
            names = doc.session.mesh_names(doc.ts)
            excluded, what = [n for n in layer.geometry_excluded if n in names], "meshes"
        geo = QToolButton()
        geo.setAutoRaise(True)
        geo.setIcon(picon("geometry_mask", "#cfcfcf" if not excluded else "#e0a040"))
        if not excluded:
            geo.setToolTip("Geometry mask: nothing excluded")
        elif len(excluded) == len(names):
            geo.setToolTip("Geometry mask: nothing included (no effect)")
            geo.setText("0")
        else:
            geo.setToolTip(f"Geometry mask: {len(names) - len(excluded)} of {len(names)} {what} included")
            geo.setText(str(len(names) - len(excluded)))
        geo.setToolButtonStyle(Qt.ToolButtonTextBesideIcon if excluded else Qt.ToolButtonIconOnly)
        geo.clicked.connect(lambda: panel.geometry_menu(self.uid, geo.mapToGlobal(geo.rect().bottomLeft())))
        geo.setContextMenuPolicy(Qt.CustomContextMenu)
        geo.customContextMenuRequested.connect(lambda pos: panel.geometry_menu(self.uid, geo.mapToGlobal(pos)))
        lay.addWidget(geo)
        right = QVBoxLayout()
        right.setSpacing(0)
        b = layer.blend_of(doc.viewmode)
        self.mode = QComboBox()  # the wheel edits it only with the focus (editor/wheel_guard.py)
        modes = [m for m in blending.IDS if doc.viewmode == "normal" or m not in blending.NORMAL_ONLY]
        self.mode.setModel(blend_model(doc.viewmode == "normal"))
        self.mode.setCurrentIndex(modes.index(b.mode))
        self.mode.setFixedWidth(64)
        self.mode.currentIndexChanged.connect(lambda i: panel.set_blend(self.uid, mode=self.mode.itemData(i)))
        self.mode.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mode.customContextMenuRequested.connect(
            lambda _p: self._apply_all_menu())
        right.addWidget(self.mode)
        self.opacity = QSpinBox()
        self.opacity.setRange(0, 100)
        self.opacity.setValue(round(b.opacity * 100))
        self.opacity.setFixedWidth(64)
        self.opacity.setButtonSymbols(QSpinBox.NoButtons)
        self.opacity.valueChanged.connect(lambda v: panel.set_blend(self.uid, opacity=v / 100, merge=True))
        right.addWidget(self.opacity)
        self.opacity_slider = Slider()  # under the opacity field: drag it, the field follows (and the reverse)
        self.opacity_slider.setObjectName("rowOpacity")  # its thin look in the theme
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(self.opacity.value())
        self.opacity_slider.setFixedSize(64, 10)
        self.opacity_slider.valueChanged.connect(self.opacity.setValue)
        self.opacity.valueChanged.connect(lambda v: (self.opacity_slider.blockSignals(True),
                                                     self.opacity_slider.setValue(v),
                                                     self.opacity_slider.blockSignals(False)))
        right.addWidget(self.opacity_slider)
        lay.addLayout(right)

    def show_mask_selected(self, selected: bool):
        """The mask thumbnail's border: light when the mask is the selection, red when the mask is off (the theme's
        `state` rules; the style worked out again only when it changes)."""
        if self.mask is not None:
            state = "selected" if selected else ("" if self.mask_enabled else "off")
            if self.mask.property("state") != state:
                self.mask.setProperty("state", state)
                self.mask.style().unpolish(self.mask)
                self.mask.style().polish(self.mask)

    def _apply_all_menu(self):
        """Right click on the blending mode: Apply to all channels (Flatten layers page)."""
        menu = QMenu(self)
        menu.addAction("Apply to all channels", lambda: self.panel.blend_all_channels(self.uid))
        menu.exec(self.mode.mapToGlobal(self.mode.rect().bottomLeft()))

    def _thumb_clicked(self, e):
        """A click on the layer's thumbnail: out of a mask shown alone (Alt+click) back to the material; the row
        takes the press as usual (selection, drag)."""
        ws = self.panel.ws
        if e.button() == Qt.LeftButton and ws.display.startswith("mask:"):
            ws.set_display("material")
        e.ignore()

    def _mask_clicked(self, e):
        if e.button() != Qt.LeftButton:
            return
        if e.modifiers() & Qt.ShiftModifier:
            self.panel.toggle_mask(self.uid)
        elif e.modifiers() & Qt.AltModifier:  # the mask alone in the viewport (Alt+click again: back)
            self.panel.ws.isolate_mask(self.uid)
        else:
            self.panel.doc.select(self.uid, mask=True)


class EffectRow(QWidget):
    """An effect's row under its layer."""

    def __init__(self, panel, effect, key):
        super().__init__()
        self.panel, self.uid = panel, effect.uid
        self.setObjectName("stackRow")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 4, 1)
        lay.setSpacing(4)
        eye = QToolButton()
        eye.setIcon(picon("eye" if effect.visible else "eye_off"))
        eye.setAutoRaise(True)
        eye.clicked.connect(lambda: panel.set_effect_visible(self.uid))
        lay.addWidget(eye)
        ic = QLabel()
        ic.setPixmap(picon({"fill": "fill_layer", "paint": "brush", "levels": "histogram", "generator": "effect",
                            "filter": "effect", "anchor": "anchor"}[effect.kind]).pixmap(16, 16))
        lay.addWidget(ic)
        name = effect.name
        if effect.kind in ("filter", "generator"):
            from trafilatura3d.paint.graph_nodes import node_of
            node = node_of(effect.procedural)
            name = node.label if node else f"{effect.name} (none)"
        if effect.kind == "levels" and key != MASK_KEY:
            name = f"Levels - {CHANNELS[effect.levels.channel].label}"
        self.name = QLabel(name)
        lay.addWidget(self.name, 1)
        if effect.kind not in ("levels", "anchor"):  # no blending: Levels changes, an anchor only exposes
            b = effect.blend_of(key)
            self.mode = QComboBox()  # the wheel edits it only with the focus (editor/wheel_guard.py)
            modes = [m for m in blending.IDS if key == "normal" or m not in blending.NORMAL_ONLY]
            self.mode.setModel(blend_model(key == "normal"))
            self.mode.setCurrentIndex(modes.index(b.mode))
            self.mode.setFixedWidth(64)
            self.mode.currentIndexChanged.connect(
                lambda i: panel.set_effect_blend(self.uid, key, mode=self.mode.itemData(i)))
            lay.addWidget(self.mode)
            op = QSpinBox()
            op.setRange(0, 100)
            op.setValue(round(b.opacity * 100))
            op.setFixedWidth(46)
            op.setButtonSymbols(QSpinBox.NoButtons)
            op.valueChanged.connect(lambda v: panel.set_effect_blend(self.uid, key, opacity=v / 100, merge=True))
            lay.addWidget(op)


class LayerTree(QTreeWidget):
    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.setHeaderHidden(True)
        self.setIndentation(14)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)  # Ctrl/Cmd and Shift click (Managing layers)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        # The rows' look (transparent parts, so a selected row is highlighted whole) is in the theme's style sheet:
        # one here made every row added cost ~0.5 ms more.

    def drawRow(self, painter, option, index):
        """Every row (layer or effect) closed by the UI's divider line, across the whole width (the indentation
        included)."""
        super().drawRow(painter, option, index)
        painter.save()
        painter.setPen(QColor(theme.BORDER))
        y = option.rect.bottom()
        painter.drawLine(0, y, self.viewport().width(), y)
        painter.restore()

    def dragEnterEvent(self, e):
        from trafilatura3d.editor.paint.assets import MIME
        if e.mimeData().hasFormat(MIME):
            e.acceptProposedAction()
            return
        super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        from trafilatura3d.editor.paint.assets import MIME
        if e.mimeData().hasFormat(MIME):
            e.acceptProposedAction()
            return
        super().dragMoveEvent(e)

    def dropEvent(self, e):
        """Assets dropped from the Assets panel become layers there; moves go through the document (one History
        step), then the tree is rebuilt."""
        from trafilatura3d.editor.paint.assets import asset_of
        asset = asset_of(e.mimeData())
        if asset is not None:
            e.acceptProposedAction()
            target = self.itemAt(e.position().toPoint())
            self.panel.drop_asset(asset, target.data(0, Qt.UserRole) if target is not None else None)
            return
        item = self.currentItem()
        target = self.itemAt(e.position().toPoint())
        pos = self.dropIndicatorPosition()
        e.ignore()
        if item is None or target is None or target is item:
            return
        where = {QAbstractItemView.AboveItem: "above", QAbstractItemView.BelowItem: "below",
                 QAbstractItemView.OnItem: "into"}.get(pos)
        if not where:
            return
        uid, target_uid = item.data(0, Qt.UserRole), target.data(0, Qt.UserRole)
        if self.panel.doc.ts.owner_of(uid)[1] is not None:  # an effect: by the rows around the drop line
            self.drop_effect_at(uid, target, where)
        else:  # the selected layers together, in their order
            ts = self.panel.doc.ts
            if ts.find(target_uid) is None or (where == "into" and ts.find(target_uid).kind != "folder"):
                return  # among effects, or into a layer that is not a folder: not a place for layers
            uids = self.panel._selected_layers()
            self.panel.move_many(uids if uid in uids else [uid], target_uid, where)


    def drop_effect_at(self, uid, target, where):
        """An effect dropped "above", "below" or "into" a row: the rows on each side of the drop line decide
        (LayerStackPanel.drop_effect)."""
        if where == "into":
            self.panel.drop_effect(uid, into=target.data(0, Qt.UserRole))
            return
        above, below = (self.itemAbove(target), target) if where == "above" else (target, self.itemBelow(target))
        self.panel.drop_effect(uid, above.data(0, Qt.UserRole) if above else None,
                               below.data(0, Qt.UserRole) if below else None)


class LayerStackPanel(QWidget):
    title_changed = Signal(str)

    def __init__(self, workspace):
        super().__init__()
        self.ws = workspace
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar_widget = theme.divider(QWidget(), "bottom")  # the channel shown and the stack's buttons
        bar = QHBoxLayout(bar_widget)
        bar.setContentsMargins(6, 4, 6, 4)
        self.viewmode = QComboBox()
        self.viewmode.setFixedWidth(100)
        self.viewmode.currentIndexChanged.connect(self._viewmode)
        bar.addWidget(self.viewmode)
        bar.addStretch(1)

        def button(name, tip, slot=None, menu=None):
            b = QToolButton()
            b.setIcon(picon(name))
            b.setToolTip(tip)
            b.setAutoRaise(True)
            if slot:
                b.clicked.connect(slot)
            if menu:
                b.setMenu(menu)
                b.setPopupMode(QToolButton.InstantPopup)
            bar.addWidget(b)
            return b

        fx_menu = QMenu(self)
        for kind, label in ADD_EFFECTS:
            fx_menu.addAction(label, lambda k=kind: self.add_effect(k))
        button("effect", "Add Effect", menu=fx_menu)
        mask_menu = QMenu(self)
        mask_menu.addAction("Add white mask", lambda: self.add_mask(1.0))
        mask_menu.addAction("Add black mask", lambda: self.add_mask(0.0))
        button("mask", "Create Mask", menu=mask_menu)
        button("paint_layer", "Create New Paint Layer", lambda: self.add_layer("paint"))
        button("fill_layer", "Create New Fill Layer", lambda: self.add_layer("fill"))
        button("folder", "Add New Folder", lambda: self.add_layer("folder"))
        button("delete", "Delete Layer", self.delete)
        lay.addWidget(bar_widget)
        self.tree = LayerTree(self)
        self.tree.itemSelectionChanged.connect(self._selection)
        self.tree.itemPressed.connect(self._pressed)  # back from a layer's mask to the layer: the same row
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.itemDoubleClicked.connect(lambda it, _c: self.rename(it.data(0, Qt.UserRole)))
        # the rows the user collapsed (folders, layers' effects): a rebuild keeps them so
        self.tree.itemCollapsed.connect(lambda it: self._expanded(it, False))
        self.tree.itemExpanded.connect(lambda it: self._expanded(it, True))
        lay.addWidget(self.tree, 1)
        QShortcut(QKeySequence("Ctrl+D"), self.tree, activated=self.duplicate, context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("Ctrl+G"), self.tree, activated=self.group, context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence.Delete, self.tree, activated=self.delete, context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("Backspace"), self.tree, activated=self.delete, context=Qt.WidgetWithChildrenShortcut)
        # Shortcuts page: Copy / Cut / Paste layer, Copy / Paste layer content.
        for keys, slot in (("Ctrl+C", self.copy), ("Ctrl+X", self.cut), ("Ctrl+V", self.paste),
                           ("Ctrl+Shift+C", self.copy_content), ("Ctrl+Shift+V", self.paste_content)):
            QShortcut(QKeySequence(keys), self.tree, activated=slot, context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("Alt+Up"), self.tree, activated=lambda: self.move_effect(1),
                  context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("Alt+Down"), self.tree, activated=lambda: self.move_effect(-1),
                  context=Qt.WidgetWithChildrenShortcut)
        self._thumb_timer = QTimer(self, singleShot=True, interval=150, timeout=self.update_thumbnails)
        self._thumbs: dict = {}  # job key -> its thumbnail (QPixmap), for the results the rows show
        self.rows, self.items = {}, {}
        self._built, self._sigs = None, {}  # the tree's `_frame` and each row's signature, when made
        self._rebuilding = False
        self._eye, self._eye_drag = None, 0  # the eye press being dragged: (visible, layers done), and its count
        self._collapsed: set = set()  # uids of the rows collapsed by the user

    @property
    def doc(self):
        return self.ws.doc

    # --- building -------------------------------------------------------------------------------------

    def sync_selection(self):
        """The document's selection on the tree without making its rows again (rows dragged or clicked stay alive:
        a rebuild under a press ended drags and made the selection jump); a layer whose mask gets (or loses) the
        selection shows the other effects (`_update`)."""
        self._update()

    def _select(self):
        """The document's selection on the rows: the current one, those selected, the mask selected."""
        doc = self.doc
        cur = self.items.get(doc.selected)
        if cur is not None and self.tree.currentItem() is not cur:
            self.tree.setCurrentItem(cur)
        for uid, it in self.items.items():
            it.setSelected(uid == doc.selected or uid in doc.multi)
        for uid, row in self.rows.items():
            row.show_mask_selected(doc.selected == uid and doc.mask_selected)

    def _expanded(self, it, on: bool) -> None:
        if self._rebuilding:
            return
        uid = it.data(0, Qt.UserRole)
        if on:
            self._collapsed.discard(uid)
        else:
            self._collapsed.add(uid)

    def _frame(self):
        """What every row depends on (the tree made again whole when it changes): the document, the Texture Set,
        its channels, the channel shown."""
        doc = self.doc
        return id(doc), doc.ts_name, tuple(doc.ts.channel_ids()), doc.viewmode

    def _children(self, layer) -> list:
        """A layer row's children, top first, as (uid, effect or layer, the effect's channel key): its effects shown
        (the mask's while the mask is selected, else the content's; a folder has only its mask's), then the layers a
        folder holds."""
        doc = self.doc
        in_mask = bool(doc.mask_selected and doc.layer() is layer)
        out = []
        if layer.kind != "folder" or in_mask:
            key = MASK_KEY if in_mask else doc.viewmode
            out += [(fx.uid, fx, key) for fx in reversed(layer.mask.effects if in_mask else layer.effects)]
        if layer.kind == "folder":
            out += [(x.uid, x, None) for x in reversed(layer.children)]
        return out

    def _row_signature(self, layer) -> str:
        """What a layer's row shows (`Row`): made again when it changes."""
        b = layer.blend_of(self.doc.viewmode)
        mask = None if layer.mask is None else layer.mask.enabled
        return repr((layer.kind, layer.visible, layer.name, mask, layer.geometry_type, layer.geometry_tiles,
                     layer.geometry_excluded, b.mode, b.opacity))

    def _effect_signature(self, fx, key) -> str:
        """What an effect's row shows (`EffectRow`)."""
        b = fx.blend_of(key) if fx.kind not in ("levels", "anchor") else None
        channel = fx.levels.channel if fx.kind == "levels" else None
        return repr((fx.visible, fx.kind, fx.name, fx.procedural, channel, key, b and (b.mode, b.opacity)))

    def rebuild(self):
        """The tree brought to the document (`_update`) and the thumbnails."""
        self._update()
        self.update_thumbnails()

    def _update(self):
        """The tree brought to the document, the rows that did not change kept: in each row's children (and at the
        top) those before and after the first difference stay, the ones between are made again; a kept row is made
        again only if what it shows changed (its signature). Made a row at a time, an edit cost ~40 ms on 19 layers;
        clearing the tree also scrolled it and reopened the collapsed folders. Made whole when what every row
        depends on changes (`_frame`)."""
        doc = self.doc
        self._rebuilding = True
        if doc is None or doc.ts is None or self._built != self._frame():
            scroll = self.tree.verticalScrollBar().value()
            self.viewmode.blockSignals(True)
            self.viewmode.clear()
            self.tree.clear()
            self.rows, self.items, self._built, self._sigs = {}, {}, None, {}
            if doc is None or doc.ts is None:
                self.viewmode.blockSignals(False)
                self._rebuilding = False
                return
            ids = doc.ts.channel_ids()
            for c in ids:
                self.viewmode.addItem(CHANNELS[c].label, c)
            self.viewmode.setCurrentIndex(ids.index(doc.viewmode) if doc.viewmode in ids else 0)
            self.viewmode.blockSignals(False)
            self._reconcile(None, [(x.uid, x, None) for x in reversed(doc.ts.layers)])
            self.tree.doItemsLayout()
            self.tree.verticalScrollBar().setValue(scroll)
            self._built = self._frame()
        else:
            self._reconcile(None, [(x.uid, x, None) for x in reversed(doc.ts.layers)])
        self._select()
        self._rebuilding = False

    def _reconcile(self, parent, wanted: list) -> None:
        """The children of `parent` (None: the top rows) made `wanted` ((uid, layer or effect, effect key), top
        first): the rows the same before and after the first difference kept (and brought up to date), those
        between taken away and made again (a layer added or removed: that row alone)."""
        if parent is None:
            have = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        else:
            have = [parent.child(i) for i in range(parent.childCount())]
        uids = [w[0] for w in wanted]
        old = [it.data(0, Qt.UserRole) for it in have]
        start = 0
        while start < min(len(old), len(uids)) and old[start] == uids[start]:
            start += 1
        end = 0
        while end < min(len(old), len(uids)) - start and old[-1 - end] == uids[-1 - end]:
            end += 1
        for i in reversed(range(start, len(old) - end)):
            self._forget(parent.takeChild(i) if parent is not None else self.tree.takeTopLevelItem(i))
        for k, (uid, obj, key) in enumerate(wanted[start:len(uids) - end]):
            self._new_row(parent, start + k, obj, key)
        for i in list(range(start)) + list(range(len(uids) - end, len(uids))):
            self._refresh_row(have[i] if i < start else have[len(old) - len(uids) + i], *wanted[i][1:])

    def _new_row(self, parent, index, obj, key) -> None:
        """A row made at `index` of `parent`'s children: a layer (and its children) or an effect."""
        it = QTreeWidgetItem()
        it.setData(0, Qt.UserRole, obj.uid)
        layer = key is None
        if layer:
            it.setSizeHint(0, QSize(10, THUMB + 20))
            # Every layer row takes drops: Qt allows a drop line between two rows only when their parent row does,
            # and a layer's effects are its children (moved between them); layers dropped onto a layer that is not
            # a folder are refused in dropEvent.
            it.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled)
        else:
            it.setSizeHint(0, QSize(10, 24))
            it.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled)
        if parent is None:
            self.tree.insertTopLevelItem(index, it)
        else:
            parent.insertChild(index, it)
        self.items[obj.uid] = it
        self._set_widget(it, obj, key)
        if layer:
            self._reconcile(it, self._children(obj))
            it.setExpanded(obj.uid not in self._collapsed)

    def _refresh_row(self, it, obj, key) -> None:
        """A kept row: made again if what it shows changed, its children brought up to date."""
        sig = self._row_signature(obj) if key is None else self._effect_signature(obj, key)
        if self._sigs.get(obj.uid) != sig:
            self._set_widget(it, obj, key)
        if key is None:
            self._reconcile(it, self._children(obj))

    def _set_widget(self, it, obj, key) -> None:
        if key is None:
            self.rows[obj.uid] = row = Row(self, obj)
            self._sigs[obj.uid] = self._row_signature(obj)
        else:
            row = EffectRow(self, obj, key)
            self._sigs[obj.uid] = self._effect_signature(obj, key)
        self.tree.setItemWidget(it, 0, row)

    def _forget(self, it) -> None:
        """A row taken away, and its children: no longer known."""
        for i in range(it.childCount()):
            self._forget(it.child(i))
        uid = it.data(0, Qt.UserRole)
        for d in (self.items, self.rows, self._sigs):
            d.pop(uid, None)

    def schedule_thumbnails(self):
        self._thumb_timer.start()

    def update_thumbnails(self):
        """The rows' thumbnails now where they are known (icons, results already computed); the others are
        computed by the viewports' refresh, its last step (`thumbnail_steps`), asked for here."""
        if self.stale_thumbnails():
            self.ws.live.schedule()

    def stale_thumbnails(self) -> list:
        """The rows' thumbnails set where they are known: icons (General preferences' simplified thumbnails, UV Tile
        projects), the images of results already computed (`_thumbs`, by job key); the others returned, (layer uid,
        "thumb" or "mask", job) for `thumbnail_steps`."""
        doc = self.doc
        if doc is None or doc.ts is None or not self.rows:
            return []
        from trafilatura3d.editor.paint import settings as prefs
        # General preferences: icons instead of computed thumbnails. UV Tile projects: computed only for fill
        # layers (the same content on every tile), icons for the rest (paint, folders, masks differ per tile).
        simplified, tiled = prefs.get("simplified_thumbnails"), bool(doc.project.uv_tiles)
        for layer in doc.ts.walk():
            row = self.rows.get(layer.uid)
            if row is None:
                continue
            if layer.kind == "folder" and (simplified or tiled):
                row.thumb.clear()  # the row's own folder icon says it already
            elif simplified or (tiled and layer.kind != "fill"):
                kind = {"paint": "paint_layer", "fill": "fill_layer"}.get(layer.kind, "folder")
                row.thumb.setPixmap(picon(kind).pixmap(THUMB - 8, THUMB - 8))
            if row.mask is not None and (simplified or tiled):
                row.mask.setPixmap(picon("mask").pixmap(THUMB - 8, THUMB - 8))
        if simplified:
            return []
        from trafilatura3d.editor.live import PREVIEW_SIZE
        # the first tile, at the preview's resolution: a thumbnail needs no more (its cache partition, not the
        # full resolution's)
        ts, sc = doc.ts, doc.session.compiler(doc.ts.tiles[0], PREVIEW_SIZE)
        ch = doc.viewmode if doc.viewmode in ts.channel_ids() else ts.channel_ids()[0]
        wanted = []
        for layer in ts.walk():
            row = self.rows.get(layer.uid)
            if row is None or (tiled and layer.kind != "fill"):
                continue
            job = sc.content(ts, layer, ch, ts.format_of(ch)) if layer.kind != "folder" or layer.children else None
            if job is not None:
                wanted.append((layer.uid, "thumb", job))
            else:
                row.thumb.clear()
            if row.mask is not None and not tiled:
                m = sc.mask(ts, layer) if layer.mask.enabled else None
                if m is None:  # a turned off mask still shows its content
                    on = copy.copy(layer.mask)
                    on.enabled = True
                    saved, layer.mask = layer.mask, on
                    m = sc.mask(ts, layer)
                    layer.mask = saved
                wanted.append((layer.uid, "mask", m))
        stale = []
        for uid, part, job in wanted:
            if job.key in self._thumbs:
                getattr(self.rows[uid], part).setPixmap(self._thumbs[job.key])
            else:
                stale.append((uid, part, job))
        self._thumbs = {job.key: self._thumbs[job.key] for _, _, job in wanted if job.key in self._thumbs}
        return stale

    def thumbnail_steps(self, stale: list, related=(), precious: frozenset = frozenset()):
        """The thumbnails of `stale_thumbnails` computed and shown, a step per wait of their cook for the GPU (the
        viewports' refresh runs it); `related`, `precious`: the refresh's (`LiveCook`), kept by the budget. Shown
        on the rows there are when done (a rebuild in between made new ones)."""
        if not stale:
            return
        try:
            images = yield from self.doc.session.ev.thumbnail_steps([(job, "output") for _, _, job in stale],
                                                                    THUMB - 2, related=related, precious=precious)
        except Exception as e:  # noqa: BLE001 - thumbnails never break the panel
            self.ws.log(f"Layer thumbnails: {e}", True)
            return
        for (uid, part, job), img in zip(stale, images):
            self._thumbs[job.key] = pix = _pixmap(img)
            row = self.rows.get(uid)
            if row is not None and getattr(row, part) is not None:
                getattr(row, part).setPixmap(pix)

    # --- selection ------------------------------------------------------------------------------------

    def _selection(self):
        """The tree's selection to the document: the current row is the selected layer (or effect), the selected
        layer rows are selected together."""
        if self._rebuilding:
            return
        cur = self.tree.currentItem()
        if cur is None:
            return
        uid = cur.data(0, Qt.UserRole)
        ts = self.doc.ts
        layers = [it.data(0, Qt.UserRole) for it in self.tree.selectedItems()]
        layers = [u for u in layers if ts.find(u) is not None]
        if not cur.isSelected() and layers:
            uid = layers[-1]
        if uid == self.doc.selected and layers == self.doc.multi and not self.doc.mask_selected:
            return
        self.doc.select(uid, multi=layers if ts.find(uid) is not None else None)

    def _pressed(self, item, _column):
        """A press on a layer's row (not on its mask thumbnail, which selects the mask): the layer itself. With its
        mask selected the row is current already and Qt reports no selection change, so the click did nothing."""
        doc = self.doc
        if item is None:  # a press on no row
            return
        uid = item.data(0, Qt.UserRole)
        if doc is not None and doc.mask_selected and uid == doc.selected and doc.ts.find(uid) is not None:
            doc.select(uid, multi=doc.multi or None)

    def _selected_layers(self) -> list[str]:
        """The selected layers, outermost only (a folder carries its content), bottom first."""
        doc = self.doc
        ts = doc.ts
        uids = [u for u in (doc.multi or ([doc.selected] if doc.selected else [])) if ts.find(u) is not None]
        inside = {c.uid for u in uids for c in ts.find(u).walk() if c.uid != u}
        order = [layer.uid for layer in ts.walk()]
        return sorted((u for u in uids if u not in inside), key=order.index)

    def _viewmode(self, i):
        if self._rebuilding or i < 0:
            return
        self.doc.viewmode = self.viewmode.itemData(i)
        self.rebuild()

    # --- edits ----------------------------------------------------------------------------------------

    def _insert_point(self, ts):
        """The list and index of a new layer: above the selected one."""
        layer = self.doc.layer()
        sel = layer.uid if layer is not None else None
        lst = ts.parent_list(sel) if sel else None
        if lst is None:
            return ts.layers, len(ts.layers)
        return lst, [layer.uid for layer in lst].index(sel) + 1

    def add_layer(self, kind):
        doc = self.doc
        base = {"paint": "Layer", "fill": "Fill layer", "folder": "Folder"}[kind]
        created = {}

        def do():
            ts = doc.ts
            layer = new_layer(ts, kind, unique_name(ts, base))
            lst, i = self._insert_point(ts)
            lst.insert(i, layer)
            created["uid"] = layer.uid
        label = {"paint": "New paint layer", "fill": "New fill layer", "folder": "New folder"}[kind]
        doc.edit(label, do)
        if "uid" in created:
            doc.select(created["uid"])

    def _insert(self, label, layer_fn, above_uid=None):
        """Insert the layer `layer_fn(ts)` makes above `above_uid` (or the selection) as one History step and
        select it."""
        doc = self.doc
        made = {}

        def do():
            ts = doc.ts
            layer = layer_fn(ts)
            lst = ts.parent_list(above_uid) if above_uid else None
            if lst is None:
                lst, i = self._insert_point(ts)
            else:
                i = [x.uid for x in lst].index(above_uid) + 1
            lst.insert(i, layer)
            made["uid"] = layer.uid
        doc.edit(label, do)
        if made:
            doc.select(made["uid"])

    def add_smart_material(self, path, above_uid=None):
        """A smart material's folder above the selection (or `above_uid`)."""
        from trafilatura3d.paint import smart
        try:
            folder, warnings = smart.load(path)
        except Exception as e:  # noqa: BLE001 - a broken file is reported
            self.ws.log(str(e), True)
            return
        for w in warnings:
            self.ws.log(f"Warning: {w}")
        self._insert("Add smart material", lambda ts: folder, above_uid)

    def add_effect_layer(self, kind, procedural, label, above_uid=None):
        """An effect dropped into the stack: a layer with every channel in Passthrough holding the effect
        (Effects page), so it works on the whole stack below."""
        def make(ts):
            layer = new_layer(ts, "fill", unique_name(ts, label))
            layer.material.clear()
            for ch in ts.channel_ids():
                layer.blend[ch] = ChannelBlend("passthrough", 1.0)
            layer.effects.append(Effect(new_uid(), kind, label, procedural=procedural))
            return layer
        self._insert(f"Add {kind}", make, above_uid)

    def drop_asset(self, asset: dict, on_uid=None):
        """An asset dropped on the layer stack (on a row: above that layer)."""
        layer_uid = None
        if on_uid is not None:
            owner = self.doc.ts.owner_of(on_uid)[0]
            layer_uid = owner.uid if owner is not None else None
        kind = asset["kind"]
        if kind == "material":
            self.add_material_layer(asset["package"], asset["graph"], asset.get("label", asset["graph"]), layer_uid)
        elif kind == "smart":
            self.add_smart_material(asset["path"], layer_uid)
        elif kind in ("filter", "generator"):
            self.add_effect_layer(kind, asset["id"], asset.get("label", kind.capitalize()), layer_uid)
        else:
            self.ws.log(f"A {kind} goes on a slot of the Properties, not in the layer stack")

    def create_smart_material(self, uid):
        """Right click on a folder > Create smart material: saved in the Default library (Settings > Libraries
        configuration), named after it."""
        from trafilatura3d.editor.paint import settings as prefs
        from trafilatura3d.paint import smart
        folder = self.doc.ts.find(uid)
        name = "".join(c if c.isalnum() or c in " _-" else "_" for c in folder.name)
        path = prefs.default_folder("smart_materials") / name
        try:
            saved = smart.save(folder, path)
        except Exception as e:  # noqa: BLE001
            self.ws.log(str(e), True)
            return
        self.ws.assets.refresh()
        self.ws.log(f"Smart material saved: {saved}")

    def add_material_layer(self, path, gid, label, above_uid=None):
        """A fill layer with a material (dropping a material into the stack, in the application)."""
        def make(ts):
            from trafilatura3d.editor.paint import settings as prefs
            layer = new_layer(ts, "fill", unique_name(ts, label))
            layer.resource = {"package": path, "graph": gid, "values": {}}
            layer.projection.tiling = prefs.get("uv_scale")  # General preferences: Default UV scale for materials
            return layer
        self._insert("New fill layer", make, above_uid)

    def delete(self):
        doc = self.doc
        uid = doc.selected
        if not uid:
            return
        if doc.effect() is not None:
            def do_fx():
                layer, fx, in_mask = doc.ts.owner_of(uid)
                lst = layer.mask.effects if in_mask else layer.effects
                lst.remove(fx)
            doc.edit("Delete effect", do_fx)
            return
        uids = self._selected_layers()
        if len(uids) > 1:
            def do_many():
                for u in uids:
                    lst = doc.ts.parent_list(u)
                    lst[:] = [x for x in lst if x.uid != u]
            doc.edit("Delete layers", do_many)
            return

        def do():
            lst = doc.ts.parent_list(uid)
            lst[:] = [x for x in lst if x.uid != uid]
        doc.edit("Delete layer", do)

    def duplicate(self):
        """Ctrl/Cmd+D: the selected layers (or the selected effect) duplicated above themselves."""
        doc, uid = self.doc, self.doc.selected
        if not uid:
            return
        if doc.effect() is not None:
            self.copy()
            self.paste()
            return
        if len(self._selected_layers()) > 1:
            self.copy()
            self.paste()
            return
        new_uids = {}

        def do():
            ts = doc.ts
            src = ts.find(uid)
            dup = copy.deepcopy(src)
            self._fresh(dup, new_uids)  # layers, effects (their paintings copied below), anchor references
            dup.name = f"{src.name} copy"
            lst = ts.parent_list(uid)
            lst.insert([x.uid for x in lst].index(uid) + 1, dup)
        doc.edit("Duplicate layer", do)
        for old, new in new_uids.items():
            doc.session.store.duplicate(old, new)
        if uid in new_uids:
            doc.content_changed.emit()
            doc.select(new_uids[uid])

    def group(self):
        doc, uid = self.doc, self.doc.selected
        if not uid:
            return
        made = {}

        uids = self._selected_layers()

        def do():
            ts = doc.ts
            lst = ts.parent_list(uids[-1])  # the folder goes where the topmost selected layer is
            at = [x.uid for x in lst].index(uids[-1])
            folder = new_layer(ts, "folder", unique_name(ts, "Folder"))
            for u in uids:  # bottom first, keeping their order
                src = ts.parent_list(u)
                layer = next(x for x in src if x.uid == u)
                if src is lst and src.index(layer) < at:
                    at -= 1
                src.remove(layer)
                folder.children.append(layer)
            lst.insert(at, folder)
            made["uid"] = folder.uid
        doc.edit("Group layers", do)
        if made:
            doc.select(made["uid"])

    # --- copy and paste (Shortcuts page: Copy / Cut / Paste layer, Copy / Paste layer content) ----------

    def _keys_of(self, layer) -> list:
        """The painted entries of a layer and its effects (content and mask)."""
        store = self.doc.session.store
        uids = [x.uid for x in layer.walk()]
        for x in layer.walk():
            uids += [fx.uid for fx in x.effects] + [fx.uid for fx in (x.mask.effects if x.mask else [])]
        return [k for u in uids for k in store.keys_of(u)]

    def copy(self):
        """The selected layers (or effect), with what is painted on them, to the clipboard."""
        import copy as cp
        doc = self.doc
        store = doc.session.store
        fx = doc.effect()
        if fx is not None:
            self.ws.clipboard = ("effect", cp.deepcopy(fx), store.snapshot(store.keys_of(fx.uid)))
            self.ws.log(f"Copied the effect {fx.name}")
            return
        layers = [cp.deepcopy(doc.ts.find(u)) for u in self._selected_layers()]
        if not layers:
            return
        keys = [k for layer in layers for k in self._keys_of(layer)]
        self.ws.clipboard = ("layers", layers, store.snapshot(keys))
        self.ws.log(f"Copied {len(layers)} layer(s)")

    def cut(self):
        self.copy()
        self.delete()

    def _fresh(self, obj, remap: dict):
        """New uids for a copied layer (and its content and effects) or effect; old -> new in `remap` (the
        references to its anchor points follow)."""
        from trafilatura3d.paint.model import Effect, remap_anchors
        effects = [obj] if isinstance(obj, Effect) else []
        if not isinstance(obj, Effect):
            for x in obj.walk():
                remap[x.uid] = x.uid = new_uid()
                effects += x.effects + (x.mask.effects if x.mask else [])
        for fx in effects:
            remap[fx.uid] = fx.uid = new_uid()
        remap_anchors([obj], remap)

    def _paste_pixels(self, snap: dict, remap: dict):
        store = self.doc.session.store
        store.restore({(remap.get(u, u), key, t): tex for (u, key, t), tex in snap.items() if u in remap})

    def paste(self):
        """Ctrl/Cmd+V: copied layers above the selection; a copied effect on the selected layer (its mask when the
        mask is selected)."""
        import copy as cp
        clip = self.ws.clipboard
        if clip is None or self.doc.ts is None:
            return
        doc, remap = self.doc, {}
        if clip[0] == "effect":
            fx = cp.deepcopy(clip[1])
            self._fresh(fx, remap)
            layer = doc.layer()
            if layer is None or (layer.kind == "folder" and not doc.mask_selected):
                self.ws.log("Paste an effect on a paint or fill layer (or a mask)")
                return
            in_mask = doc.mask_selected
            self._paste_pixels(clip[2], remap)

            def do_fx():
                lay = doc.ts.find(layer.uid)
                (lay.mask.effects if in_mask and lay.mask else lay.effects).append(fx)
            doc.edit("Paste effect", do_fx)
            doc.select(fx.uid)
            doc.content_changed.emit()
            return
        layers = [cp.deepcopy(x) for x in clip[1]]
        for layer in layers:
            self._fresh(layer, remap)
        self._paste_pixels(clip[2], remap)
        anchor = self._selected_layers()[-1] if self._selected_layers() else None

        def do():
            ts = doc.ts
            lst = ts.parent_list(anchor) if anchor else None
            if lst is None:
                lst, i = ts.layers, len(ts.layers)
            else:
                i = [x.uid for x in lst].index(anchor) + 1
            for k, layer in enumerate(layers):
                lst.insert(i + k, layer)
        doc.edit("Paste layers", do)
        doc.select(layers[-1].uid, multi=[x.uid for x in layers])
        doc.content_changed.emit()

    def copy_content(self):
        """Ctrl/Cmd+Shift+C: the selected layer's content (its painting and effects)."""
        import copy as cp
        layer = self.doc.layer()
        if layer is None or layer.kind == "folder":
            return
        store = self.doc.session.store
        keys = [k for k in store.keys_of(layer.uid) if k[1] != "mask"]
        keys += [k for fx in layer.effects for k in store.keys_of(fx.uid)]
        self.ws.content_clipboard = (layer.uid, cp.deepcopy(layer.effects), store.snapshot(keys))
        self.ws.log(f"Copied the content of {layer.name}")

    def paste_content(self):
        """Ctrl/Cmd+Shift+V: the copied content replaces the selected layer's painting and effects."""
        import copy as cp
        clip = self.ws.content_clipboard
        layer = self.doc.layer()
        if clip is None or layer is None or layer.kind == "folder":
            return
        src_uid, effects, snap = clip
        effects = cp.deepcopy(effects)
        remap = {src_uid: layer.uid}
        for fx in effects:
            self._fresh(fx, remap)
        store = self.doc.session.store
        store.restore({k: None for k in store.keys_of(layer.uid) if k[1] != "mask"})
        self._paste_pixels(snap, remap)
        uid = layer.uid
        self.doc.edit("Paste layer content", lambda: setattr(self.doc.ts.find(uid), "effects", effects))
        self.doc.content_changed.emit()

    def rename(self, uid):
        layer, fx, _m = self.doc.ts.owner_of(uid)
        if fx is not None:
            name, ok = QInputDialog.getText(self, "Rename", "Name", text=fx.name)
            if ok and name.strip():
                self.doc.edit("Rename effect", lambda: setattr(self.doc.ts.owner_of(uid)[1], "name", name.strip()))
            return
        if layer is None:
            return
        name, ok = QInputDialog.getText(self, "Rename", "Name", text=layer.name)
        if ok and name.strip():
            self.doc.edit("Rename layer", lambda: setattr(self.doc.ts.find(uid), "name", name.strip()))

    def eye_press(self, uid):
        """A press on a layer's eye: that layer turned on or off, then every eye the drag passes over the same way,
        one History step (`eye_over`, `eye_release`)."""
        self._eye_drag += 1
        self._eye = (not self.doc.ts.find(uid).visible, set())
        self.eye_over(uid)

    def eye_at(self, pos):
        """The layer whose eye column is under the global point `pos` (anywhere across the eye's width), or None."""
        it = self.tree.itemAt(self.tree.viewport().mapFromGlobal(pos))
        row = self.rows.get(it.data(0, Qt.UserRole)) if it is not None else None
        if row is None:
            return None
        x = row.eye.mapFromGlobal(pos).x()
        return row.uid if 0 <= x < row.eye.width() else None

    def eye_over(self, uid):
        if self._eye is None or uid in self._eye[1]:
            return
        on, done = self._eye
        done.add(uid)
        # merged edits (one step, the viewports follow at the preview's resolution) without rebuilding the tree,
        # whose eye under the press must stay alive: the icons change in place
        self.doc.edit("Visibility", lambda: setattr(self.doc.ts.find(uid), "visible", on),
                      merge=f"eyes.{self._eye_drag}", structure=False)
        self.rows[uid].eye.show_visible(on)

    def eye_release(self):
        """The end of the press: the full resolution back (an edit without merge, Document.edit)."""
        if self._eye is not None:
            self._eye = None
            self.doc.edit("Visibility", lambda: None, structure=False)

    def set_blend(self, uid, mode=None, opacity=None, merge=False):
        doc = self.doc
        ch = doc.viewmode

        def do():
            layer = doc.ts.find(uid)
            b = layer.blend_of(ch)
            layer.blend[ch] = ChannelBlend(mode if mode is not None else b.mode,
                                           opacity if opacity is not None else b.opacity)
        doc.edit("Blending mode" if mode is not None else "Opacity", do,
                 merge=f"{uid}.{ch}.opacity" if merge else None, structure=mode is not None)

    def blend_all_channels(self, uid):
        doc = self.doc
        b = doc.ts.find(uid).blend_of(doc.viewmode)

        def do():
            layer = doc.ts.find(uid)
            for ch in doc.ts.channel_ids():
                if b.mode in blending.NORMAL_ONLY and ch != "normal":
                    continue
                layer.blend[ch] = ChannelBlend(b.mode, layer.blend_of(ch).opacity)
        doc.edit("Apply to all channels", do)

    # --- effects --------------------------------------------------------------------------------------

    def add_effect(self, kind, procedural: str = "", label: str = ""):
        """Add Effect: on the selected layer's content, or on its mask when the mask is selected (Levels on
        the viewed channel; a filter or generator given by its `procedural` id). Not on folders."""
        doc = self.doc
        layer = doc.layer()
        in_mask = doc.mask_selected
        if layer is None or (layer.kind == "folder" and not in_mask):  # folders: effects on their mask only
            self.ws.log("Effects go on a paint or fill layer, or on a mask")
            return
        made = {}

        def do():
            lay = doc.ts.find(layer.uid)
            lst = lay.mask.effects if in_mask else lay.effects
            names = {"fill": "Fill", "paint": "Paint", "levels": "Levels", "generator": "Generator",
                     "filter": "Filter", "anchor": "Anchor Point"}
            name = label or names[kind]
            if kind == "anchor":  # numbered: the slots list the anchors by name
                taken = {f.name for _l, f, _m in doc.ts.anchors()}
                name = next(f"{name} {i}" for i in range(1, len(taken) + 2) if f"{name} {i}" not in taken)
            fx = Effect(new_uid(), kind, name, procedural=procedural)
            if kind == "fill":
                if in_mask:
                    fx.material[MASK_KEY] = ChannelSource([1.0, 1.0, 1.0])
                else:
                    for ch in doc.ts.channel_ids():
                        d = CHANNELS[ch].default
                        fx.material[ch] = ChannelSource(color=list(d) * 3 if len(d) == 1 else list(d))
            if kind == "levels" and not in_mask:
                fx.levels = Levels(doc.viewmode)
            lst.append(fx)
            made["uid"] = fx.uid
        doc.edit(f"Add {kind} effect", do)
        if made:
            doc.select(made["uid"])

    def set_effect_visible(self, uid):
        def do():
            fx = self.doc.ts.owner_of(uid)[1]
            fx.visible = not fx.visible
        self.doc.edit("Effect visibility", do)

    def set_effect_blend(self, uid, key, mode=None, opacity=None, merge=False):
        def do():
            fx = self.doc.ts.owner_of(uid)[1]
            b = fx.blend_of(key)
            fx.blend[key] = ChannelBlend(mode if mode is not None else b.mode,
                                         opacity if opacity is not None else b.opacity)
        self.doc.edit("Effect blending" if mode is not None else "Effect opacity", do,
                      merge=f"{uid}.{key}.opacity" if merge else None, structure=mode is not None)

    def drop_effect(self, uid, above=None, below=None, into=None):
        """An effect dragged in the stack, by the rows the drop line falls between (`above`, `below`; rows are top
        first) or the layer row it is dropped onto (`into`). It never changes stack: a mask's effect stays in a
        mask, a content's in a content. Between rows: above the row below it, or below the row above it, when that
        row is an effect of its own stack; just under its layer's row: the top of its stack. Onto another paint or
        fill layer: the top of that layer's same stack (a mask's effect needs a mask there). Anything else is
        refused (one History step either way)."""
        doc = self.doc

        def do():
            ts = doc.ts
            layer, fx, in_mask = ts.owner_of(uid)
            src = layer.mask.effects if in_mask else layer.effects
            if into is not None:
                t = ts.find(into)
                if t is None or t.kind == "folder":
                    raise ValueError("effects go on paint and fill layers")
                if in_mask and t.mask is None:
                    raise ValueError(f"{t.name} has no mask: a mask's effect stays in a mask")
                src.remove(fx)
                (t.mask.effects if in_mask else t.effects).append(fx)
                return

            def same_stack(u):
                if not u or u == uid:
                    return None
                o_layer, o_fx, o_mask = ts.owner_of(u)
                return o_fx if o_fx is not None and o_layer is layer and o_mask == in_mask else None
            src.remove(fx)
            if same_stack(below) is not None:
                src.insert(src.index(same_stack(below)) + 1, fx)  # list bottom first: above that row
            elif same_stack(above) is not None:
                src.insert(src.index(same_stack(above)), fx)
            elif above == layer.uid:
                src.append(fx)  # just under its layer's row: the top of its stack
            else:
                raise ValueError("an effect stays in its stack: drop it between its layer's effects, or onto a layer")
        doc.edit("Move effect", do)

    # --- Geometry mask -------------------------------------------------------------------------------

    def set_geometry_excluded(self, uid, excluded):
        def do():
            self.doc.ts.find(uid).geometry_excluded = sorted(set(excluded))
        self.doc.edit("Geometry mask", do)

    def set_geometry_tiles(self, uid, excluded):
        def do():
            self.doc.ts.find(uid).geometry_tiles = sorted(set(excluded))
        self.doc.edit("Geometry mask", do)

    def set_geometry_type(self, uid, kind):
        self.doc.edit("Geometry mask", lambda: setattr(self.doc.ts.find(uid), "geometry_type", kind))

    def geometry_menu(self, uid, global_pos):
        """The Geometry mask's menu (Geometry mask page): its Masking type (UV Tiles only with the UV Tile
        workflow), the Texture Set's UV tiles or meshes to include (checked) or exclude, Include all, Exclude all,
        Copy geometry mask, Paste into geometry mask, and the edition in the viewport (clicking toggles the tile or
        the mesh under the pointer)."""
        doc = self.doc
        layer = doc.ts.find(uid)
        menu = QMenu(self)
        if doc.project.uv_tiles:
            menu.addSection("Masking type")
            for kind in GEOMETRY_TYPES:
                a = menu.addAction(kind)
                a.setCheckable(True)
                a.setChecked(layer.geometry_type == kind)
                a.triggered.connect(lambda _=False, kind=kind: self.set_geometry_type(uid, kind))
        tiles = layer.geometry_type == "UV Tiles"
        if tiles:
            names, current, setter = doc.ts.tiles, layer.geometry_tiles, self.set_geometry_tiles
            menu.addSection("UV Tiles")
        else:
            names, current, setter = doc.session.mesh_names(doc.ts), layer.geometry_excluded, self.set_geometry_excluded
            menu.addSection("Mesh names")
        for n in names:
            a = menu.addAction(str(n))
            a.setCheckable(True)
            a.setChecked(n not in current)
            a.toggled.connect(lambda on, n=n: setter(uid, [x for x in current if x != n] + ([] if on else [n])))
        menu.addSeparator()
        menu.addAction("Include all", lambda: setter(uid, []))
        menu.addAction("Exclude all", lambda: setter(uid, names))
        menu.addSeparator()
        menu.addAction("Copy geometry mask", lambda: setattr(self.ws, "geometry_clipboard", (tiles, list(current))))
        clip = self.ws.geometry_clipboard
        paste = menu.addAction("Paste into geometry mask", lambda: setter(uid, clip[1]))
        paste.setEnabled(clip is not None and clip[0] == tiles)  # a list of the same Masking type
        menu.addSeparator()
        edit = menu.addAction(f"Edit in the viewport (click {'tiles' if tiles else 'meshes'} to include / exclude, "
                              "Escape to end)", lambda: self.ws.edit_geometry_mask(uid))
        edit.setCheckable(True)
        edit.setChecked(self.ws.geometry_edit == uid)
        menu.exec(global_pos)

    def mask_menu(self, uid, global_pos):
        """Right click on a mask thumbnail (Masks page): its effects, toggle, invert background, copy / paste
        its content, remove."""
        doc = self.doc
        doc.select(uid, mask=True)
        layer = doc.ts.find(uid)
        menu = QMenu(self)
        fx = menu.addMenu("Add effect")
        for kind, label in ADD_EFFECTS:
            fx.addAction(label, lambda k=kind: (doc.select(uid, mask=True), self.add_effect(k)))
        menu.addSeparator()
        menu.addAction("Toggle mask", lambda: self.toggle_mask(uid))
        menu.addAction("Invert mask background", lambda: self.invert_mask_background(uid))
        menu.addAction("Show the mask alone (Alt+click)", lambda: self.ws.isolate_mask(uid))
        menu.addSeparator()
        menu.addAction("Copy mask content", lambda: self.copy_mask(uid))
        paste = menu.addAction("Paste into mask", lambda: self.paste_mask(uid))
        paste.setEnabled(self.ws.mask_clipboard is not None)
        menu.addSeparator()
        menu.addAction("Remove mask", lambda: self.remove_mask(uid))
        if layer is not None:
            menu.exec(global_pos)

    def copy_mask(self, uid):
        """Copy mask content: the mask (background, effects) and what is painted on it and its paint effects."""
        import copy as cp
        layer = self.doc.ts.find(uid)
        store = self.doc.session.store
        owners = [uid] + [fx.uid for fx in layer.mask.effects if fx.kind == "paint"]
        keys = [k for u in owners for k in store.keys_of(u) if k[1] == "mask"]
        self.ws.mask_clipboard = (cp.deepcopy(layer.mask), store.snapshot(keys))
        self.ws.log("Mask copied")

    def paste_mask(self, uid):
        """Paste into mask: the copied mask replaces this layer's (its paint entries copied under new uids)."""
        import copy as cp
        if self.ws.mask_clipboard is None:
            return
        mask, snap = self.ws.mask_clipboard
        store = self.doc.session.store
        new = cp.deepcopy(mask)
        remap = {}
        for fx in new.effects:
            old, fx.uid = fx.uid, new_uid()
            remap[old] = fx.uid
        from trafilatura3d.paint.model import remap_anchors
        remap_anchors(new.effects, remap)
        moved = {}
        for (src_uid, key, tile), tex in snap.items():
            dst = uid if src_uid not in remap else remap[src_uid]
            moved[(dst, key, tile)] = tex
        # the previous painting of this mask goes (as re-adding a mask)
        store.restore({k: None for k in store.keys_of(uid) if k[1] == "mask"})
        store.restore(moved)
        self.doc.edit("Paste into mask", lambda: setattr(self.doc.ts.find(uid), "mask", new))
        self.doc.content_changed.emit()

    def move_effect(self, step):
        """Alt + Up / Down: the selected effect one place up (1) or down (-1) in its stack."""
        doc = self.doc
        uid = doc.selected
        if doc.effect() is None:
            return

        def do():
            layer, fx, in_mask = doc.ts.owner_of(uid)
            lst = layer.mask.effects if in_mask else layer.effects
            i = lst.index(fx)
            j = min(max(i + step, 0), len(lst) - 1)
            lst.insert(j, lst.pop(i))
        doc.edit("Move effect", do)

    def add_mask(self, background):
        doc, uid = self.doc, self.doc.selected
        if not uid:
            return
        # Re-adding a mask destroys the previous one and its content (documentation).
        doc.session.store.restore({k: None for k in doc.session.store.keys_of(uid) if k[1] == "mask"})
        doc.edit("Add mask", lambda: setattr(doc.ts.find(uid), "mask", Mask(background)))
        doc.select(uid, mask=True)

    def remove_mask(self, uid):
        self.doc.edit("Remove mask", lambda: setattr(self.doc.ts.find(uid), "mask", None))

    def toggle_mask(self, uid):
        def do():
            m = self.doc.ts.find(uid).mask
            m.enabled = not m.enabled
        self.doc.edit("Toggle mask", do)

    def invert_mask_background(self, uid):
        def do():
            m = self.doc.ts.find(uid).mask
            m.background = 1.0 - m.background
        self.doc.edit("Invert mask background", do)

    def toggle_fill_channel(self, uid, channel, on):
        """A fill layer's or fill effect's Material channel button."""
        def do():
            owner, fx, _m = self.doc.ts.owner_of(uid)
            layer = fx or owner
            if on:
                d = CHANNELS[channel].default
                layer.material[channel] = ChannelSource(color=list(d) * 3 if len(d) == 1 else list(d))
            else:
                layer.material.pop(channel, None)
        self.doc.edit("Fill channels", do)

    def move_many(self, uids, target, where):
        """Several layers dragged together: they keep their order where the drop puts them (one History step)."""
        doc = self.doc
        if target in uids:
            return

        def do():
            ts = doc.ts
            for u in uids:
                if any(x.uid == target for x in ts.find(u).walk()):
                    raise ValueError("a folder cannot go into itself")
            moved = []
            for u in uids:
                lst = ts.parent_list(u)
                layer = next(x for x in lst if x.uid == u)
                lst.remove(layer)
                moved.append(layer)
            tgt = ts.find(target)
            if where == "into" and tgt.kind == "folder":
                tgt.children.extend(moved)
                return
            lst = ts.parent_list(target)
            i = [x.uid for x in lst].index(target) + (1 if where == "above" else 0)
            lst[i:i] = moved
        doc.edit("Move layers" if len(uids) > 1 else "Move layer", do)

    def move(self, uid, target, where):
        doc = self.doc

        def do():
            ts = doc.ts
            if uid == target or any(x.uid == target for x in ts.find(uid).walk()):
                raise ValueError("a folder cannot go into itself")
            src_list = ts.parent_list(uid)
            layer = next(x for x in src_list if x.uid == uid)
            src_list.remove(layer)
            tgt = ts.find(target)
            if where == "into" and tgt.kind == "folder":
                tgt.children.append(layer)  # on top of the folder's content
                return
            lst = ts.parent_list(target)
            i = [x.uid for x in lst].index(target)
            # Rows are top first: "above" a row is higher in the stack (after it in the list).
            lst.insert(i + 1 if where == "above" else i, layer)
        doc.edit("Move layer", do)

    def _menu(self, pos):
        it = self.tree.itemAt(pos)
        if it is None:
            return
        uid = it.data(0, Qt.UserRole)
        if self.doc.ts.owner_of(uid)[1] is not None:  # an effect
            self.doc.select(uid)
            menu = QMenu(self)
            menu.addAction("Rename", lambda: self.rename(uid))
            menu.addAction("Move up", lambda: self.move_effect(1))
            menu.addAction("Move down", lambda: self.move_effect(-1))
            menu.addAction("Delete", self.delete)
            menu.exec(self.tree.viewport().mapToGlobal(pos))
            return
        layer = self.doc.ts.find(uid)
        self.doc.select(uid)
        menu = QMenu(self)
        menu.addAction("Rename", lambda: self.rename(uid))
        menu.addAction("Duplicate", self.duplicate)
        menu.addAction("Group layers", self.group)
        menu.addAction("Delete", self.delete)
        menu.addSeparator()
        if layer.kind != "folder":
            fx = menu.addMenu("Add effect")
            for kind, label in ADD_EFFECTS:
                fx.addAction(label, lambda k=kind: self.add_effect(k))
        menu.addAction("Add white mask", lambda: self.add_mask(1.0))
        menu.addAction("Add black mask", lambda: self.add_mask(0.0))
        if layer.kind == "folder":
            menu.addAction("Create smart material", lambda: self.create_smart_material(uid))
        if layer.mask is not None:
            menu.addAction("Toggle mask", lambda: self.toggle_mask(uid))
            menu.addAction("Invert mask background", lambda: self.invert_mask_background(uid))
            menu.addAction("Remove mask", lambda: self.remove_mask(uid))
        menu.exec(self.tree.viewport().mapToGlobal(pos))
