from trafilatura3d.nodes.materials._base import (MATRIX, NORMAL_FORMATS, B, E, F2, F4, MatNode, channel_toggles,
                                           material_inputs, material_outputs)
from trafilatura3d.nodes.materials.filters._common import BLEND_ON


class MaterialTransform(MatNode):
    """Transformation 2D on every channel of a material at once: the 2x2 Transformation matrix
    (rotation, scale) about the centre, then Offset; the normal map's vectors are turned with the
    rotation (and mirrored with the matrix) so the relief stays lit from the right side. Normal
    Affected by Scale and Skew: the slopes follow the whole matrix (inverse transpose), as the
    Normal Transform node does, instead of its rotation only."""

    type = "material_transform"
    label = "Material Transform"
    category = "Material filters/Transforms (Material)"
    catalog = "material_transform"
    inputs = material_inputs()
    outputs = material_outputs()
    params = {
        **channel_toggles(BLEND_ON + ("ambient_occlusion",)),
        "transformation": F4(MATRIX, "Transformation"),
        "offset": F2([0.0, 0.0], "Offset", -0.5, 0.5),
        "normal_format": E("OpenGL", "Normal Format", NORMAL_FORMATS),
        "normal_affected_by_scale_and_skew_transformations": B(
            False, "Normal Affected by Scale and Skew Transformations"),
    }
