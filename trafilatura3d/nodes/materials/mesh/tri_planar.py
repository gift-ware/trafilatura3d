from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import B, E, F, F2, MatNode, color_in
from trafilatura3d.nodes.materials.mesh._common import CATEGORY


class TriPlanar(MatNode):
    """Triplanar projection from baked Position and World Space Normal: the input (or one input per
    axis) is projected along X, Y and Z of the position, tiled, rotated and offset per axis (plus a
    seeded random offset), and blended by the normal (contrast: soft to hard; advanced mode favours
    the brighter projection like a height blend). Normalization Factor restores the contrast lost
    where projections mix; Fix Mirrored Projection unmirrors faces looking down the negative axes."""

    type = "tri_planar"
    label = "Tri Planar"
    category = CATEGORY
    catalog = "tri_planar"
    inputs = {"position": color_in("Position", (0.5, 0.5, 0.5, 1.0)),
              "world_space_normal": color_in("World Space Normal", (0.5, 1.0, 0.5, 1.0)),
              "input_x": Input(label="Input X"), "input_y": Input(label="Input Y"), "input_z": Input(label="Input Z")}
    main_input = "position"  # the library graph's primary input: Relative to input follows it
    outputs = {"output": Output()}
    params = {
        "projection": E("All axis", "Projection", ("All axis", "X only", "Y only", "Z only")),
        "image_inputs": E("1 input", "Image Inputs", ("1 input", "3 inputs")),
        "blending_mode": E("linear", "Blending Mode", ("linear", "advanced")),
        "blending_contrast": F(0.5, "Blending Contrast", 0.001, 1.0),
        "normalization_factor": F(0.0, "Normalization Factor", 0.0, 1.0),
        "texture_tiling": F(1.0, "Texture Tiling", 0.0, 10.0),
        "global_rotation": F(0.0, "Global Rotation", 0.0, 1.0),
        "fix_mirrored_projection": B(True, "Fix Mirrored Projection"),
        "rotation_x": F(0.0, "Rotation X", 0.0, 1.0),
        "rotation_y": F(0.0, "Rotation Y", 0.0, 1.0),
        "rotation_z": F(0.0, "Rotation Z", 0.0, 1.0),
        "offset_x": F2([0.0, 0.0], "Offset X", 0.0, 1.0),
        "random_offset_x": F(0.0, "Random Offset X", 0.0, 1.0),
        "offset_y": F2([0.0, 0.0], "Offset Y", 0.0, 1.0),
        "random_offset_y": F(0.0, "Random Offset Y", 0.0, 1.0),
        "offset_z": F2([0.0, 0.0], "Offset Z", 0.0, 1.0),
        "random_offset_z": F(0.0, "Random Offset Z", 0.0, 1.0),
    }
