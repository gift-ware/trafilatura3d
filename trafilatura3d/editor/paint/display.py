"""The Display settings panel (Display settings pages), the parameters the viewport's renderer supports:
Environment (map, opacity over the background colors, exposure, rotation, blur, shadows and their opacity), Camera
(field of view, tone mapping function Linear / ACES) and Viewport (wireframe and grid, each with its color and
opacity). Not offered: environment alignment, shadow computation mode, depth of field, temporal anti-aliasing, post
effects, color profiles, subsurface scattering, anisotropic filtering and mipmap bias, the stencil and HDR display
options, the grid's Axis (the documentation does not list its values), automatic divisions and unit label."""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QScrollArea, QVBoxLayout, QWidget

from trafilatura3d.editor import theme
from trafilatura3d.editor.paint.widgets import PSlider, Swatch, header
from trafilatura3d.editor.render3d import TONEMAPS

ENV_FILTER = "Environment maps (*.hdr *.exr)"


class DisplaySettings(QScrollArea):
    def __init__(self, workspace):
        super().__init__()
        self.ws = workspace
        self.env_name = "Studio"
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.rebuild()

    def rebuild(self):
        view = self.ws.view3d
        look = view.look
        body = theme.section_lines(QWidget())
        lay = QVBoxLayout(body)
        lay.setContentsMargins(10, 6, 10, 10)

        def changed():
            view.refresh()

        lay.addWidget(header("Environment"))
        form = QFormLayout()
        env = QComboBox()
        env.addItems(["Studio", "Load…"])
        if self.env_name != "Studio":
            env.insertItem(1, self.env_name)
            env.setCurrentIndex(1)
        env.activated.connect(lambda i: self._env(env.itemText(i)))
        form.addRow("Environment Map", env)
        lay.addLayout(form)

        def slider(name, get, set_, lo, hi):
            s = PSlider(name, get(), lo, hi)
            s.edited.connect(lambda v, _f: (set_(v), changed()))
            lay.addWidget(s)

        # over the background colors of General preferences (default 1 ours: the page gives none)
        slider("Environment Opacity", lambda: look.env_opacity, lambda v: setattr(look, "env_opacity", v), 0.0, 1.0)
        slider("Environment Exposure", lambda: math.log2(max(look.env_intensity, 1e-6)),
               lambda v: setattr(look, "env_intensity", 2.0 ** v), -6.0, 6.0)
        slider("Environment Rotation", lambda: math.degrees(look.env_rotation) % 360,
               lambda v: setattr(look, "env_rotation", math.radians(v)), 0.0, 360.0)
        slider("Environment Blur", lambda: look.background_blur, lambda v: setattr(look, "background_blur", v), 0.0, 6.0)
        shadows = QCheckBox("Shadows")
        shadows.setChecked(look.shadows)
        shadows.toggled.connect(lambda on: (setattr(look, "shadows", on), changed()))
        lay.addWidget(shadows)
        slider("Shadows opacity", lambda: look.shadow_opacity, lambda v: setattr(look, "shadow_opacity", v), 0.0, 1.0)

        lay.addWidget(header("Camera"))
        slider("Field of View", lambda: view.fov, lambda v: setattr(view, "fov", max(v, 1.0)), 5.0, 120.0)
        form2 = QFormLayout()
        tone = QComboBox()
        tone.addItems(["Linear", "ACES"])
        tone.setCurrentText("ACES" if TONEMAPS[look.tonemap] == "ACES" else "Linear")
        tone.currentTextChanged.connect(lambda t: (setattr(look, "tonemap", TONEMAPS.index(t)), changed()))
        form2.addRow("Tone mapping: Function", tone)
        lay.addLayout(form2)

        lay.addWidget(header("Viewport"))
        for label, attr, name in (("Show mesh wireframe", "wireframe", "Wireframe"), ("Show grid", "grid", "Grid")):
            c = QCheckBox(label)
            c.setChecked(getattr(look, attr))
            c.toggled.connect(lambda on, attr=attr: (setattr(look, attr, on), changed()))
            lay.addWidget(c)
            form3 = QFormLayout()
            color = Swatch(getattr(look, f"{attr}_color"), name)
            color.edited.connect(lambda rgb, _f, attr=attr: (setattr(look, f"{attr}_color", tuple(rgb)), changed()))
            form3.addRow(f"{name} color", color)
            lay.addLayout(form3)
            slider(f"{name} opacity", lambda attr=attr: getattr(look, f"{attr}_opacity"),
                   lambda v, attr=attr: setattr(look, f"{attr}_opacity", v), 0.0, 1.0)
        lay.addStretch(1)
        self.setWidget(body)

    def _env(self, text):
        r = self.ws.view3d.renderer
        if text == "Studio":
            r.set_studio()
            self.env_name = "Studio"
        elif text == "Load…":
            from trafilatura3d.editor.preferences import suggested_dir
            path, _ = QFileDialog.getOpenFileName(self, "Environment map", suggested_dir(), ENV_FILTER)
            if path:
                self.load_environment(path)
        self.rebuild()
        self.ws.view3d.refresh()

    def load_environment(self, path):
        from trafilatura3d import imageio
        try:
            img = imageio.read(Path(path)).astype("float32")
        except OSError as e:
            self.ws.log(f"Environment map: {e}", True)
            return
        self.ws.view3d.renderer.set_environment_image(img)
        self.env_name = Path(path).name
