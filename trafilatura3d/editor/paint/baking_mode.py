"""The Baking mode panels (Baking interface pages): Mesh map bakers (check, select, the dropdown's check and
sync actions), Common settings, Mesh map settings (the selected baker's parameters) and Baking log. The
Texture Set list shows check boxes for the Texture Sets to bake; the bar under the viewport has Bake selected
textures and Return to painting mode. Settings are per Texture Set and saved in the project (one History step
per change)."""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QStyle, QStyleOptionViewItem, QToolButton, QVBoxLayout, QWidget)

from trafilatura3d.editor import theme
from trafilatura3d.bake import ANTIALIASING, MATCH, SIZES
from trafilatura3d.editor.bake_form import baker_rows
from trafilatura3d.editor.paint.icons import picon
from trafilatura3d.meshio import MESH_FILTER
from trafilatura3d.paint import baking


class _BakerList(QListWidget):
    """The Mesh map bakers list: a press on a row's icon asks to visualize that map (`visualize(slot)`)."""
    visualize = Signal(str)

    def mousePressEvent(self, e):
        pos = e.position().toPoint()
        it = self.itemAt(pos)
        if it is not None and e.button() == Qt.LeftButton:
            opt = QStyleOptionViewItem()
            self.initViewItemOption(opt)  # the view's icon size, which the item's own options lack
            self.itemDelegate().initStyleOption(opt, self.indexFromItem(it))
            opt.rect = self.visualItemRect(it)
            if self.style().subElementRect(QStyle.SE_ItemViewItemDecoration, opt, self).contains(pos):
                self.visualize.emit(it.data(Qt.UserRole))
                return
        super().mousePressEvent(e)


class BakingPanels:
    """Builds and refreshes the four Baking mode panels of a workspace."""

    def __init__(self, ws):
        self.ws = ws
        self.selected = "normal"
        self.bakers = QWidget()
        self.common = QScrollArea()
        self.settings = QScrollArea()
        for a in (self.common, self.settings):
            a.setWidgetResizable(True)
            a.setFrameShape(QFrame.NoFrame)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        lay = QVBoxLayout(self.bakers)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        top_bar = theme.divider(QWidget(), "bottom")  # the bakers' menu over their list
        top = QHBoxLayout(top_bar)
        top.setContentsMargins(4, 2, 4, 2)
        top.addStretch(1)
        menu_btn = QToolButton()
        menu_btn.setIcon(picon("lines_menu"))
        menu_btn.setAutoRaise(True)
        m = QMenu(menu_btn)
        m.addAction("Sync all settings to all Texture Sets", self._sync_all)
        m.addSeparator()
        m.addAction("Check all", lambda: self._check(lambda on: True))
        m.addAction("Uncheck all", lambda: self._check(lambda on: False))
        m.addAction("Invert checked mesh maps", lambda: self._check(lambda on: not on))
        m.addAction("Apply checked to all Texture sets", self._apply_checked_all)
        menu_btn.setMenu(m)
        menu_btn.setPopupMode(QToolButton.InstantPopup)
        top.addWidget(menu_btn)
        lay.addWidget(top_bar)
        self.list = _BakerList()
        self.list.visualize.connect(self._visualize)
        self.list.itemChanged.connect(self._item_changed)
        self.list.currentRowChanged.connect(self._row)
        lay.addWidget(self.list, 1)

    @property
    def doc(self):
        return self.ws.doc

    def _cfg(self):
        return baking.config(self.doc.ts)

    def _save(self, label, fn, merge=None):
        """Change the current Texture Set's configuration as one History step (`merge`: edits with the same key in
        a row are one step)."""
        name = self.doc.ts_name

        def do():
            ts = self.doc.project.texture_set(name)
            cfg = baking.config(ts)
            fn(cfg)
            baking.set_config(ts, cfg)
        self.doc.edit(label, do, merge=merge, structure=False)
        self.ws._update_bake_overlay()  # the cage follows the distances, the high poly list…

    # --- Mesh map bakers ------------------------------------------------------------------------------

    def rebuild(self):
        if self.doc is None or self.doc.ts is None:
            return
        cfg = self._cfg()
        self.list.blockSignals(True)
        self.list.clear()
        for slot, label, _b, _f in baking.BAKERS:
            it = QListWidgetItem(label)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if cfg["bakers"][slot]["enabled"] else Qt.Unchecked)
            it.setData(Qt.UserRole, slot)
            # visualize: the eye open on the map shown in the viewport; a map not baked yet has none to show
            baked = slot in self.doc.ts.mesh_maps
            it.setIcon(picon("eye" if self.ws.display == f"mesh:{slot}" else "eye_off"))
            it.setToolTip("Show this map in the viewport" if baked else "Not baked yet")
            self.list.addItem(it)
            if slot == self.selected:
                self.list.setCurrentItem(it)
        self.list.blockSignals(False)
        self._common(cfg)
        self._settings(cfg)

    def _visualize(self, slot):
        """A map's visualize icon: that mesh map in the viewport, again: back to the neutral material."""
        if slot not in self.doc.ts.mesh_maps:
            return
        key = f"mesh:{slot}"
        self.ws.set_display("material" if self.ws.display == key else key)
        self.rebuild()

    def _item_changed(self, it):
        slot, on = it.data(Qt.UserRole), it.checkState() == Qt.Checked
        self._save("Check mesh map", lambda cfg: cfg["bakers"][slot].__setitem__("enabled", on))

    def _row(self, i):
        if i >= 0:
            self.selected = self.list.item(i).data(Qt.UserRole)
            self._settings(self._cfg())

    def _check(self, fn):
        def do(cfg):
            for b in cfg["bakers"].values():
                b["enabled"] = fn(b["enabled"])
        self._save("Check mesh maps", do)
        self.rebuild()

    def _apply_checked_all(self):
        checked = {s: b["enabled"] for s, b in self._cfg()["bakers"].items()}

        def do():
            for ts in self.doc.project.texture_sets:
                cfg = baking.config(ts)
                for s, on in checked.items():
                    cfg["bakers"][s]["enabled"] = on
                baking.set_config(ts, cfg)
        self.doc.edit("Apply checked to all Texture sets", do, structure=False)

    def _sync_all(self):
        src = self._cfg()

        def do():
            for ts in self.doc.project.texture_sets:
                cfg = copy.deepcopy(src)
                if ts.size != self.doc.ts.size and src["common"]["size"] == list(self.doc.ts.size):
                    cfg["common"]["size"] = list(ts.size)
                baking.set_config(ts, cfg)
        self.doc.edit("Sync all settings to all Texture Sets", do, structure=False)

    # --- Common settings ------------------------------------------------------------------------------

    def _common(self, cfg):
        c = cfg["common"]
        body = QWidget()
        form = QFormLayout(body)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        def setter(key, label):
            return lambda v: self._save(label, lambda cfg: cfg["common"].__setitem__(key, v))

        size = QHBoxLayout()
        w, h = QComboBox(), QComboBox()
        for combo, v, axis in ((w, c["size"][0], 0), (h, c["size"][1], 1)):
            combo.addItems([str(x) for x in SIZES])
            combo.setCurrentText(str(v))

            def set_size(text, axis=axis):
                def do(cfg):
                    cfg["common"]["size"][axis] = int(text)
                self._save("Output size", do)
            combo.currentTextChanged.connect(set_size)
        size.addWidget(w)
        size.addWidget(h)
        form.addRow("Output size", size)
        dil = QSpinBox()
        dil.setRange(0, 256)
        dil.setValue(int(c["dilation"]))
        dil.editingFinished.connect(lambda: setter("dilation", "Dilation width")(dil.value()))
        form.addRow("Dilation width", dil)
        self._check_row(form, "Apply diffusion", c["apply_diffusion"], setter("apply_diffusion", "Apply diffusion"))
        self._check_row(form, "Use Low poly mesh as high poly mesh", c["use_low_as_high"],
                        setter("use_low_as_high", "Use Low poly mesh as high poly mesh"))
        highs = QListWidget()
        highs.setMaximumHeight(70)
        for p in c["high_meshes"]:
            highs.addItem(p)
        form.addRow("High definition meshes", highs)
        hrow = QHBoxLayout()
        add, rm = QPushButton("Add"), QPushButton("Remove")
        add.clicked.connect(self._add_high)
        rm.clicked.connect(lambda: highs.currentItem() and self._save(
            "High definition meshes", lambda cfg, p=highs.currentItem().text(): cfg["common"]["high_meshes"].remove(p)))
        hrow.addWidget(add)
        hrow.addWidget(rm)
        form.addRow("", hrow)
        cage = QComboBox()
        cage.addItems(baking.CAGES)
        cage.setCurrentText(c.get("cage", "Distance based"))
        cage.currentTextChanged.connect(setter("cage", "Cage"))
        form.addRow("Cage", cage)
        if c.get("cage") == "Custom file":
            crow = QHBoxLayout()
            path = QLineEdit(c["cage_file"])
            path.setReadOnly(True)
            pick = QPushButton("…")
            pick.clicked.connect(self._pick_cage)
            crow.addWidget(path, 1)
            crow.addWidget(pick)
            form.addRow("Cage file", crow)
        else:
            for key, label in (("frontal", "Max Frontal Distance"), ("rear", "Max Rear Distance")):
                sp = QDoubleSpinBox()
                sp.setDecimals(4)
                sp.setRange(0.0, 1000.0)
                sp.setSingleStep(0.005)
                sp.setValue(float(c[key]))
                sp.editingFinished.connect(lambda sp=sp, key=key, label=label: setter(key, label)(sp.value()))
                form.addRow(label, sp)
        self._check_row(form, "Relative to Bounding Box", c["relative_to_bounding_box"],
                        setter("relative_to_bounding_box", "Relative to Bounding Box"))
        self._check_row(form, "Average Normals", c["average_normals"], setter("average_normals", "Average Normals"))
        self._check_row(form, "Ignore backface", c["ignore_backface"], setter("ignore_backface", "Ignore backface"))
        match = QComboBox()
        match.addItems(MATCH)
        match.setCurrentText(c["match"])
        match.currentTextChanged.connect(setter("match", "Match"))
        form.addRow("Match", match)
        for key, label in (("low_suffix", "Low poly mesh suffix"), ("high_suffix", "High poly mesh suffix"),
                           ("ignore_backface_suffix", "Ignore backface suffix")):
            e = QLineEdit(c[key])
            e.editingFinished.connect(lambda e=e, key=key, label=label: setter(key, label)(e.text()))
            form.addRow(label, e)
        aa = QComboBox()
        aa.addItems(list(ANTIALIASING))
        aa.setCurrentText(c["antialiasing"])
        aa.currentTextChanged.connect(setter("antialiasing", "Antialiasing"))
        form.addRow("Antialiasing", aa)
        self.common.setWidget(body)

    @staticmethod
    def _check_row(form, label, value, slot):
        cb = QCheckBox()
        cb.setChecked(bool(value))
        cb.toggled.connect(slot)
        form.addRow(label, cb)

    def _add_high(self):
        from trafilatura3d.editor.preferences import suggested_dir
        names, _ = QFileDialog.getOpenFileNames(self.ws, "High definition meshes", suggested_dir(), MESH_FILTER)
        if names:
            self._save("High definition meshes", lambda cfg: cfg["common"]["high_meshes"].extend(
                n for n in names if n not in cfg["common"]["high_meshes"]))
            self.rebuild()

    def _pick_cage(self):
        from trafilatura3d.editor.preferences import suggested_dir
        name, _ = QFileDialog.getOpenFileName(self.ws, "Cage", suggested_dir(), MESH_FILTER)
        if name:
            self._save("Cage file", lambda cfg: cfg["common"].__setitem__("cage_file", name))
            self.rebuild()

    # --- Mesh map settings ----------------------------------------------------------------------------

    def _settings(self, cfg):
        slot = self.selected
        body = QWidget()
        form = QFormLayout(body)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        title = QLabel(f"<b>{next(lbl for s, lbl, *_ in baking.BAKERS if s == slot)}</b>")
        form.addRow(title)
        values = {**baking.defaults_of(slot), **cfg["bakers"][slot]["params"], **baking.FIXED[slot]}

        def set_param(key, v):  # one History step per parameter edited in a row
            self._save("Mesh map settings", lambda cfg: cfg["bakers"][slot]["params"].__setitem__(key, v),
                       merge=f"bake.{slot}.{key}")
            values[key] = v
        baker_rows(form, baking.BAKER_OF[slot], values, list(baking.editable_params(slot)), set_param,
                   lambda: self._settings(self._cfg()))
        if slot == "normal":
            note = QLabel("Tangent space, in the project's normal map format.")
            note.setObjectName("dim")
            form.addRow(note)
        self.settings.setWidget(body)


class BakingVisualization(QFrame):
    """The Baking visualization settings at the top left of the viewport in the Baking mode (Baking
    visualization settings page), what the viewport's renderer can show: Hide baking meshes, Mesh (HP), Matching
    error (high poly areas outside the cage, in red), Cage surface and its opacity, Project mesh, the Neutral
    material's Roughness, Metallic and AO intensity (the project mesh shaded with its baked normal and AO). The
    colors, the separate cage wireframe, the missing seams, the neutral material's Quality and Bent normal and the
    skew vectors are not offered; the meshes shown are always the selected Texture Set's."""

    def __init__(self, ws, parent):
        super().__init__(parent)
        self.ws = ws
        self.setStyleSheet("QFrame { background: rgba(30, 30, 30, 220); border: 1px solid #111; border-radius: 4px; }"
                           "QCheckBox, QLabel { background: transparent; border: none; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.setSpacing(2)
        title = QLabel("<b>Baking visualization</b>")
        lay.addWidget(title)
        look = ws.view3d.look
        self.hide_all = QCheckBox("Hide baking meshes")
        self.hide_all.toggled.connect(self._changed)
        lay.addWidget(self.hide_all)
        self.boxes = {}
        for label, attr in (("Mesh (HP)", "show_high"), ("Matching error", "outside_red"), ("Cage surface", "show_cage")):
            cb = QCheckBox(label)
            cb.setChecked(True)  # ours: everything shown when entering the Baking mode
            cb.toggled.connect(self._changed)
            lay.addWidget(cb)
            self.boxes[attr] = cb
        from trafilatura3d.editor.paint.widgets import PSlider
        self.opacity = PSlider("Cage surface opacity", look.cage_opacity, 0.0, 1.0, compact=True)
        self.opacity.edited.connect(lambda v, _f: self._changed())
        lay.addWidget(self.opacity)
        self.project = QCheckBox("Project mesh")
        self.project.setChecked(True)
        self.project.toggled.connect(self._changed)
        lay.addWidget(self.project)
        lay.addWidget(QLabel("<b>Neutral material</b>"))
        # ours: the documentation gives no defaults (the painting viewport's missing-channel values, full AO)
        self.roughness = PSlider("Roughness", 0.3, 0.0, 1.0, compact=True)
        self.metallic = PSlider("Metallic", 0.0, 0.0, 1.0, compact=True)
        self.ao_intensity = PSlider("AO intensity", 1.0, 0.0, 1.0, compact=True)
        for s in (self.roughness, self.metallic, self.ao_intensity):
            s.edited.connect(lambda v, _f: self.ws.live.schedule())
            lay.addWidget(s)
        self.adjustSize()

    def neutral_constants(self) -> dict:
        """The neutral material's values for the renderer slots without a texture."""
        r, m = self.roughness.value(), self.metallic.value()
        return {"roughness": (r, r, r, 1.0), "metallic": (m, m, m, 1.0)}

    def _changed(self, *_):
        self.apply()
        self.ws.view3d.refresh()

    def apply(self, on: bool = True):
        """The Look flags for the Baking mode (on) or back to painting (off)."""
        look = self.ws.view3d.look
        show = on and not self.hide_all.isChecked()
        look.show_high = show and self.boxes["show_high"].isChecked()
        look.outside_red = self.boxes["outside_red"].isChecked()
        look.show_cage = show and self.boxes["show_cage"].isChecked()
        look.cage_opacity = self.opacity.value()
        # Hide baking meshes forces the project mesh visible (documentation).
        look.hide_low = on and show and not self.project.isChecked()
        for w in (*self.boxes.values(), self.opacity, self.project):
            w.setEnabled(not self.hide_all.isChecked())
