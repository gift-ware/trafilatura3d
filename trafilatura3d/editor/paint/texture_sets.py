"""Texture Set list and Texture Set settings panels (Texture Set pages).

List: one row per Texture Set with visibility, name, size and shader instance; click selects (its layer stack
and settings show), double click renames; the visibility menu (Show All, Hide All, Invert Show/Hide), Focus
Mode (only the selected one shown), the settings menu for the descriptions, the context menu (Show/Hide, Edit
name, Reset name to the original, Edit Description). One shader instance exists, "Main shader" (no shader
management, no Import Shader Parameters, no reassignment yet).

Settings: Size (width, height, lock), the channels (Add channel menu: Supported / Unsupported / User channels;
format per channel; remove), Normal mixing, Height to normal method, Ambient occlusion mixing, UV padding and
the Mesh maps slots (images from the project's resources or files). Bake Mesh Maps is not offered yet.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QInputDialog, QLabel, QMenu,
                               QPushButton, QScrollArea, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from trafilatura3d.editor import theme
from trafilatura3d.editor.paint.icons import picon
from trafilatura3d.editor.paint.widgets import header
from trafilatura3d.paint.channels import CHANNELS, FORMATS, USER_CHANNELS, supported, unsupported
from trafilatura3d.paint.model import AO_MIXING, HEIGHT_TO_NORMAL, MESH_MAPS, NORMAL_MIXING, SIZES, UV_PADDING, Channel
from trafilatura3d.paint.session import IMAGE_FILTER

DESCRIPTIONS = ("Hide Empty Descriptions", "Hide All Descriptions", "Show All Descriptions")


TILE_ROLE = Qt.UserRole + 1  # a UV tile row (Baking mode): (Texture Set name, UDIM)


class TextureSetList(QWidget):
    def __init__(self, workspace):
        super().__init__()
        self.ws = workspace
        self.focus_mode = False
        self.descriptions = DESCRIPTIONS[0]
        self.unchecked = set()  # Baking mode: the Texture Sets not to bake
        self.unchecked_tiles = set()  # Baking mode: the (Texture Set, UV tile) not to bake
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar_widget = theme.divider(QWidget(), "bottom")  # the visibility and settings buttons over the list
        bar = QHBoxLayout(bar_widget)
        bar.setContentsMargins(6, 4, 6, 2)
        vis = QToolButton()
        vis.setIcon(picon("eye"))
        vis.setToolTip("Open Menu")
        vis.setAutoRaise(True)
        m = QMenu(vis)
        m.addAction("Show All", lambda: self._all(lambda ts: True))
        m.addAction("Hide All", lambda: self._all(lambda ts: False))
        m.addAction("Invert Show/Hide", lambda: self._all(lambda ts: not ts.visible))
        vis.setMenu(m)
        vis.setPopupMode(QToolButton.InstantPopup)
        bar.addWidget(vis)
        self.focus = QToolButton()
        self.focus.setIcon(picon("focus_mode"))
        self.focus.setToolTip("Focus Mode (Alt+Q)")
        self.focus.setCheckable(True)
        self.focus.setAutoRaise(True)
        self.focus.toggled.connect(self._focus)
        bar.addWidget(self.focus)
        bar.addStretch(1)
        settings = QToolButton()
        settings.setIcon(picon("lines_menu"))
        settings.setToolTip("Settings")
        settings.setAutoRaise(True)
        sm = QMenu(settings)
        for d in DESCRIPTIONS:
            a = sm.addAction(d, lambda d=d: self._descriptions(d))
            a.setCheckable(True)
            a.setChecked(d == self.descriptions)
        self.desc_actions = sm.actions()
        settings.setMenu(sm)
        settings.setPopupMode(QToolButton.InstantPopup)
        bar.addWidget(settings)
        lay.addWidget(bar_widget)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(4)
        self.tree.setRootIsDecorated(False)
        self.tree.itemClicked.connect(self._clicked)
        self.tree.itemChanged.connect(self._check_changed)
        self.tree.itemDoubleClicked.connect(
            lambda it, c: c == 1 and it.data(1, Qt.UserRole) and self.rename(it.data(1, Qt.UserRole)))
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.tree, 1)

    @property
    def doc(self):
        return self.ws.doc

    def rebuild(self):
        self.tree.blockSignals(True)
        self._rebuild()
        self.tree.blockSignals(False)

    def _rebuild(self):
        self.tree.clear()
        doc = self.doc
        if doc is None:
            return
        for ts in doc.project.texture_sets:
            it = QTreeWidgetItem(["", ts.name, f"{ts.size[0]}x{ts.size[1]}", "Main shader"])
            it.setIcon(0, picon("eye" if ts.visible else "eye_off"))
            it.setData(1, Qt.UserRole, ts.name)
            if self.ws.mode == "baking":  # the check box: bake this Texture Set
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(1, Qt.Unchecked if ts.name in self.unchecked else Qt.Checked)
            show = (self.descriptions == "Show All Descriptions"
                    or (self.descriptions == "Hide Empty Descriptions" and ts.description))
            if show:
                it.setToolTip(1, ts.description)
                child = QTreeWidgetItem(["", ts.description or "", "", ""])
                child.setFlags(Qt.ItemIsEnabled)
                it.addChild(child)
                it.setExpanded(True)
            if self.ws.mode == "baking" and len(ts.tiles) > 1:  # a check box per UV tile (UV Tile workflow)
                for udim in ts.tiles:
                    child = QTreeWidgetItem(["", str(udim), "", ""])
                    child.setData(1, TILE_ROLE, (ts.name, udim))
                    child.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    child.setCheckState(1, Qt.Unchecked if (ts.name, udim) in self.unchecked_tiles else Qt.Checked)
                    it.addChild(child)
                it.setExpanded(True)
            self.tree.addTopLevelItem(it)
            if ts.name == doc.ts_name:
                self.tree.setCurrentItem(it)
                f = it.font(1)
                f.setBold(True)
                it.setFont(1, f)
        for c in (0, 2, 3):
            self.tree.resizeColumnToContents(c)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(1, self.tree.header().ResizeMode.Stretch)

    def _check_changed(self, it, col):
        if col != 1 or self.ws.mode != "baking":
            return
        tile = it.data(1, TILE_ROLE)
        if tile is not None:
            (self.unchecked_tiles.discard if it.checkState(1) == Qt.Checked else self.unchecked_tiles.add)(tile)
            return
        name = it.data(1, Qt.UserRole)
        if name is None:
            return
        if it.checkState(1) == Qt.Checked:
            self.unchecked.discard(name)
        else:
            self.unchecked.add(name)

    def checked(self) -> list[tuple[str, list[int]]]:
        """The checked Texture Sets with their checked UV tiles (those with none left out)."""
        out = []
        for ts in self.doc.project.texture_sets:
            tiles = [t for t in ts.tiles if (ts.name, t) not in self.unchecked_tiles]
            if ts.name not in self.unchecked and tiles:
                out.append((ts.name, tiles))
        return out

    def _clicked(self, it, col):
        name = it.data(1, Qt.UserRole)
        if name is None:
            return
        if col == 0:
            self.set_visible(name, not self.doc.project.texture_set(name).visible)
            return
        self.doc.select_texture_set(name)
        if self.ws.isolated():
            self.ws.refresh_materials()

    def set_visible(self, name, on):
        self.doc.edit("Texture Set visibility", lambda: setattr(self.doc.project.texture_set(name), "visible", on))

    def _all(self, fn):
        def do():
            for ts in self.doc.project.texture_sets:
                ts.visible = fn(ts)
        self.doc.edit("Texture Set visibility", do)

    def _focus(self, on):
        self.focus_mode = on
        self.ws.refresh_materials()

    def _descriptions(self, d):
        self.descriptions = d
        for a in self.desc_actions:
            a.setChecked(a.text() == d)
        self.rebuild()

    def rename(self, name):
        if name is None:
            return
        new, ok = QInputDialog.getText(self, "Edit name", "Name (used at export)", text=name)
        if ok and new.strip() and new.strip() != name:
            self._rename(name, new.strip())

    def _rename(self, old, new):
        doc = self.doc

        def do():
            if any(ts.name == new for ts in doc.project.texture_sets):
                raise ValueError(f"a Texture Set is already named {new!r}")
            doc.project.texture_set(old).name = new
        if doc.edit("Rename Texture Set", do) is not False and doc.ts_name == old:
            doc.ts_name = new
            doc.structure_changed.emit()

    def _menu(self, pos):
        it = self.tree.itemAt(pos)
        if it is None or it.data(1, Qt.UserRole) is None:
            return
        name = it.data(1, Qt.UserRole)
        ts = self.doc.project.texture_set(name)
        menu = QMenu(self)
        menu.addAction("Show/Hide texture set", lambda: self.set_visible(name, not ts.visible))
        menu.addAction("Edit name", lambda: self.rename(name))
        menu.addAction(f"Reset name to {ts.material}", lambda: self._rename(name, ts.material))
        menu.addAction("Edit Description", lambda: self._description(name))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _description(self, name):
        ts = self.doc.project.texture_set(name)
        text, ok = QInputDialog.getText(self, "Edit Description", "Description", text=ts.description)
        if ok:
            self.doc.edit("Texture Set description", lambda: setattr(self.doc.project.texture_set(name), "description", text))


class TextureSetSettings(QScrollArea):
    def __init__(self, workspace):
        super().__init__()
        self.ws = workspace
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.lock = True
        self.tile = None  # the UV Tile whose mesh maps the section shows (UV Tile projects)

    @property
    def doc(self):
        return self.ws.doc

    def _set_body(self, body):
        """The panel's new content; the old one is deleted later: a rebuild often comes from a signal of one of its
        controls (setWidget would delete it at once, under the signal: a crash)."""
        old = self.takeWidget()
        if old is not None:
            old.deleteLater()
        self.setWidget(body)

    def rebuild(self):
        body = theme.section_lines(QWidget())
        lay = QVBoxLayout(body)
        lay.setContentsMargins(10, 6, 10, 10)
        doc = self.doc
        if doc is None or doc.ts is None:
            self._set_body(body)
            return
        ts = doc.ts
        name = ts.name

        def edit(label, fn):
            doc.edit(label, lambda: fn(doc.project.texture_set(name)))

        form = QFormLayout()
        size_row = QHBoxLayout()
        w, h = QComboBox(), QComboBox()
        for c, v in ((w, ts.size[0]), (h, ts.size[1])):
            for s in SIZES:
                if s >= 128:
                    c.addItem(str(s), s)
            c.setCurrentText(str(v))
        lock = QToolButton()
        lock.setCheckable(True)
        lock.setChecked(self.lock)
        lock.setIcon(picon("lock" if self.lock else "unlock"))
        lock.setAutoRaise(True)
        lock.toggled.connect(lambda on: (setattr(self, "lock", on), lock.setIcon(picon("lock" if on else "unlock"))))

        def set_size(axis, value):
            def fn(t):
                if self.lock:
                    t.size = [value, value]
                else:
                    t.size[axis] = value
            edit("Texture Set size", fn)
        w.currentIndexChanged.connect(lambda i: set_size(0, w.itemData(i)))
        h.currentIndexChanged.connect(lambda i: set_size(1, h.itemData(i)))
        size_row.addWidget(w)
        size_row.addWidget(lock)
        size_row.addWidget(h)
        form.addRow("Size", size_row)
        if self.tile not in ts.tiles:
            self.tile = ts.tiles[0]
        tile = self.tile
        if len(ts.tiles) > 1:  # UV Tile workflow: the tile the rows below are about (ours: one at a time)
            pick = QComboBox()
            for t in ts.tiles:
                pick.addItem(str(t), t)
            pick.setCurrentText(str(tile))
            pick.currentIndexChanged.connect(lambda i, c=pick: (setattr(self, "tile", c.itemData(i)), self.rebuild()))
            form.addRow("UV Tile", pick)
            own = QComboBox()  # the tile's resolution (ours: square; the documentation names it, no more)
            own.addItem("Texture Set size", None)
            for v in SIZES:
                if v >= 128:
                    own.addItem(f"{v}x{v}", v)
            current = ts.tile_sizes.get(str(tile))
            own.setCurrentIndex(own.findData(current[0]) if current else 0)

            def set_tile_size(i, c=own):
                def fn(t):
                    if c.itemData(i) is None:
                        t.tile_sizes.pop(str(tile), None)
                    else:
                        t.tile_sizes[str(tile)] = [c.itemData(i)] * 2
                edit("UV Tile size", fn)
            own.currentIndexChanged.connect(set_tile_size)
            form.addRow("UV Tile size", own)
        lay.addLayout(form)

        top = QHBoxLayout()
        top.addWidget(header("Channels"))
        top.addStretch(1)
        add = QToolButton()
        add.setIcon(picon("add"))
        add.setToolTip("Add channel")
        add.setAutoRaise(True)
        menu = QMenu(add)
        shader = self.ws.template().shader  # supported: what the project's shader reads
        for title, group in (("Supported channels", supported(shader)), ("Unsupported channels", unsupported(shader)),
                             ("User channels", USER_CHANNELS)):
            sub = menu.addMenu(title)
            for c in group:
                a = sub.addAction(CHANNELS[c].label, lambda c=c: edit("Add channel", lambda t, c=c: t.channels.append(
                    Channel(c, CHANNELS[c].format))))
                a.setEnabled(c not in ts.channel_ids())
        add.setMenu(menu)
        add.setPopupMode(QToolButton.InstantPopup)
        top.addWidget(add)
        lay.addLayout(top)
        for ch in ts.channels:
            row = QHBoxLayout()
            row.addWidget(QLabel(CHANNELS[ch.id].label), 1)
            fmt = QComboBox()
            fmt.addItems(FORMATS)
            fmt.setCurrentText(ch.format)

            def set_fmt(t, cid=ch.id, combo=fmt):
                next(c for c in t.channels if c.id == cid).format = combo.currentText()
            fmt.currentTextChanged.connect(lambda _t, f=set_fmt: edit("Channel format", f))
            row.addWidget(fmt)
            rm = QToolButton()
            rm.setIcon(picon("close"))
            rm.setToolTip("Remove channel (its painting is kept and comes back with the channel)")
            rm.setAutoRaise(True)
            rm.clicked.connect(lambda _=False, cid=ch.id: edit("Remove channel", lambda t: t.channels.__setitem__(
                slice(None), [c for c in t.channels if c.id != cid])))
            row.addWidget(rm)
            lay.addLayout(row)

        lay.addWidget(header("Mixing"))
        form2 = QFormLayout()
        for label, attr, choices, needs in (("Normal mixing", "normal_mixing", NORMAL_MIXING, "normal"),
                                            ("Height to normal method", "height_to_normal", HEIGHT_TO_NORMAL, "height"),
                                            ("Ambient occlusion mixing", "ao_mixing", AO_MIXING, "ambientocclusion"),
                                            ("UV padding", "uv_padding", UV_PADDING, None)):
            c = QComboBox()
            c.addItems(choices)
            c.setCurrentText(getattr(ts, attr))
            c.setEnabled(needs is None or needs in ts.channel_ids())  # disabled without the channel (documentation)
            c.currentTextChanged.connect(lambda t, attr=attr, label=label: edit(label, lambda x: setattr(x, attr, t)))
            form2.addRow(label, c)
        lay.addLayout(form2)

        lay.addWidget(header("Mesh maps"))
        res = doc.session.resources
        for slot, label, _suffix in MESH_MAPS:  # the slots of the UV Tile chosen above
            row = QHBoxLayout()
            row.addWidget(QLabel(label), 1)
            b = QPushButton(ts.mesh_map(slot, tile) or "Select mesh map")
            m = QMenu(b)
            m.addAction("None", lambda slot=slot: edit("Mesh map", lambda t, s=slot: t.set_mesh_map(s, tile, None)))
            for n in sorted(res.images):
                m.addAction(n, lambda n=n, slot=slot: edit("Mesh map", lambda t, s=slot, n=n: t.set_mesh_map(s, tile, n)))
            m.addAction("Import…", lambda slot=slot: self._import_map(slot, tile))
            b.setMenu(m)
            row.addWidget(b, 2)
            lay.addLayout(row)
        lay.addStretch(1)
        self._set_body(body)

    def _import_map(self, slot, tile):
        from trafilatura3d.editor.preferences import suggested_dir
        path, _ = QFileDialog.getOpenFileName(self, "Import mesh map", suggested_dir(), IMAGE_FILTER)
        if not path:
            return
        name = self.ws.import_resource(path)
        if name:  # an image sequence fills the slot of each of its tiles the Texture Set has
            ts_name = self.doc.ts_name
            seq = self.doc.session.resources.sequence(name) or {tile: name}

            def do():
                ts = self.doc.project.texture_set(ts_name)
                for t, image in seq.items():
                    if t in ts.tiles:
                        ts.set_mesh_map(slot, t, image)
            self.doc.edit("Mesh map", do)
