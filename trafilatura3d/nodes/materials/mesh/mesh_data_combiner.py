from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, GRAY, B, MatNode
from trafilatura3d.nodes.materials.mesh._common import CATEGORY

MAPS = {  # port: (label, type, value when disabled)
    "ambient_occlusion": ("Ambient Occlusion", GRAY, (1.0, 1.0, 1.0, 1.0)),
    "uv_masks": ("UV Masks", GRAY, (0.0, 0.0, 0.0, 1.0)),
    "curvature": ("Curvature", GRAY, (0.5, 0.5, 0.5, 1.0)),
    "height": ("Height", GRAY, (0.5, 0.5, 0.5, 1.0)),
    "position_grayscale": ("Position (Grayscale)", GRAY, (0.5, 0.5, 0.5, 1.0)),
    "thickness": ("Thickness", GRAY, (0.5, 0.5, 0.5, 1.0)),
    "normal": ("Normal", COLOR, (0.5, 0.5, 1.0, 1.0)),
    "position_rgb": ("Position (RGB)", COLOR, (0.5, 0.5, 0.5, 1.0)),
    "color_id": ("Color ID", COLOR, (0.0, 0.0, 0.0, 1.0)),
    "world_space_direction": ("World Space Direction", COLOR, (0.5, 1.0, 0.5, 1.0)),
    "world_space_normal": ("World Space Normal", COLOR, (0.5, 1.0, 0.5, 1.0)),
}


class MeshDataCombiner(MatNode):
    """Gathers baked mesh maps for nodes taking many of them: each enabled map passes through to the
    output of the same name, a disabled one outputs its neutral value. Trafilatura 3D has one connection
    per map, so this node is a switchboard."""

    type = "mesh_data_combiner"
    label = "Mesh Data Combiner"
    category = CATEGORY
    catalog = "mesh_data_combiner"
    channels = tuple(MAPS)
    inputs = {k: Input(t, lab, dv) for k, (lab, t, dv) in MAPS.items()}
    outputs = {k: Output(t, lab) for k, (lab, t, _) in MAPS.items()}
    params = {f"use_{k}": B(k != "thickness", lab) for k, (lab, _, _) in MAPS.items()}  # Thickness off

    def wgsl(self):
        body = "".join(
            f"    store_{k}(c, select(vec4f{tuple(dv)}, load_{k}(c), p.use_{k} != 0));\n" for k, (_, _, dv) in MAPS.items())
        return super().wgsl().replace("//$BODY\n", body)
