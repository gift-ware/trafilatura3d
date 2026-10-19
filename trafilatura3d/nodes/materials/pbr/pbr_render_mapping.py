from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, C, MatNode
from trafilatura3d.nodes.materials.pbr._pbr import CATEGORY


class PbrRenderMapping(MatNode):
    """Maps a texture onto the shape of a PBR Render, through its UVs output (red/green = UV, blue =
    coverage of the shape): the texture where the shape is, Background Color elsewhere."""

    type = "pbr_render_mapping"
    label = "PBR Render Mapping"
    category = CATEGORY
    catalog = "pbr_render_mapping"
    inputs = {"texture": Input(label="Texture"), "uvs": Input(COLOR, "UVs", (0.0, 0.0, 0.0, 1.0))}
    main_input = "uvs"
    outputs = {"output": Output()}
    params = {"background_color": C([0.0, 0.0, 0.0], "Background Color")}
