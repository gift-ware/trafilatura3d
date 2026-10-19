"""The Shader settings window. Shader: one of shaders.TEMPLATES (every renderer model of the Graph workspace's 3D
View with its blending modes; the New Project's template); changing it keeps the Texture Sets' channels (the Add
channel menu lists what the shader reads as Supported). Common Parameters (AO Intensity,
Emissive Intensity, Alpha threshold with an alpha test shader: lib-alpha), Parallax Occlusion Mapping (Enable,
Strength), Displacement (Source Channel Height, Scale unit Normalized, Scale amount) and Tesselation (Uniform,
Subdivision Count). Saved in the project (paint/model.py ShaderSettings), edits are History steps.

Not offered: the toon and pixelated example shaders and custom GLSL shaders (no renderer model), instance name, the
shader's own undo stack, Quality, Horizon Fading, Alpha dithering (it needs the temporal antialiasing the viewport does
not have), the subsurface parameters, POM sample counts, the Displacement channel as source, the Scene and Physical
size units, Edge Length subdivision, Iray (MDL)."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QFrame, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from trafilatura3d import shaders
from trafilatura3d.editor import theme
from trafilatura3d.editor.paint.widgets import PSlider, header
from trafilatura3d.paint.model import ShaderSettings


class ShaderSettingsPanel(QScrollArea):
    def __init__(self, workspace):
        super().__init__()
        self.ws = workspace
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.rebuild()

    def _edit(self, attr, value, final=True):
        doc = self.ws.doc

        def do():
            setattr(doc.project.shader, attr, value)
        doc.edit("Shader settings", do, merge=None if final else f"shader.{attr}", structure=False)
        self.ws.refresh_materials()

    def _shader(self, name):
        """Another shader: its parameters shown (rebuilt after the combo's signal: rebuilding inside it crashes)."""
        self._edit("shader", name)
        QTimer.singleShot(0, self.rebuild)

    def restore_defaults(self):
        doc = self.ws.doc

        def do():
            doc.project.shader = ShaderSettings(shader=doc.project.shader.shader)  # the parameters, not the shader
        doc.edit("Restore shader defaults", do, structure=False)
        self.rebuild()
        self.ws.refresh_materials()

    def rebuild(self):
        body = theme.section_lines(QWidget())
        lay = QVBoxLayout(body)
        lay.setContentsMargins(10, 6, 10, 10)
        self.setWidget(body)
        if self.ws.doc is None:
            lay.addWidget(QLabel("No project"))
            lay.addStretch(1)
            return
        sh = self.ws.doc.project.shader
        form = QFormLayout()
        choice = QComboBox()
        choice.addItems(list(shaders.TEMPLATES))
        choice.setCurrentText(sh.shader)
        choice.currentTextChanged.connect(self._shader)
        form.addRow("Shader", choice)
        lay.addLayout(form)
        restore = QPushButton("Restore defaults")
        restore.clicked.connect(self.restore_defaults)
        lay.addWidget(restore)

        def slider(label, attr, lo, hi, integer=False):
            s = PSlider(label, getattr(sh, attr), lo, hi, integer)
            s.edited.connect(lambda v, f: self._edit(attr, int(v) if integer else v, f))
            lay.addWidget(s)

        def toggle(label, attr):
            box = QCheckBox(label)
            box.setChecked(getattr(sh, attr))
            box.toggled.connect(lambda on: self._edit(attr, on))
            lay.addWidget(box)

        def combo(label, attr, items):
            c = QComboBox()
            c.addItems(items)
            c.setCurrentText(getattr(sh, attr))
            f = QFormLayout()
            f.addRow(label, c)
            lay.addLayout(f)

        lay.addWidget(header("Common Parameters"))
        slider("AO Intensity", "ao_intensity", 0.0, 1.0)
        slider("Emissive Intensity", "emissive_intensity", 0.0, 20.0)
        if shaders.TEMPLATES[sh.shader].blending == shaders.ALPHA_TEST:
            slider("Alpha threshold", "alpha_threshold", 0.0, 1.0)
        lay.addWidget(header("Parallax Occlusion Mapping"))
        toggle("Enable", "pom_enable")
        slider("Strength", "pom_strength", 0.0, 10.0)
        lay.addWidget(header("Displacement"))
        combo("Source Channel", "displacement_source", ["Height"])
        combo("Scale unit", "scale_unit", ["Normalized"])
        slider("Scale amount", "scale_amount", -0.2, 0.2)
        lay.addWidget(header("Tesselation"))
        combo("Subdivision Mode", "subdivision_mode", ["Uniform"])
        slider("Subdivision Count", "subdivision_count", 0, 3, integer=True)
        lay.addStretch(1)
