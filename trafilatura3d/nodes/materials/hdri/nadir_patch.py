from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, GRAY, B, E, F, F2
from trafilatura3d.nodes.materials.hdri._common import HdriNode


class NadirPatch(HdriNode):
    """Clones a patch of the panorama over the nadir (or wherever Patch Position points): the patch
    is a square in the plane tangent to the sphere at the target, copied from the same square at
    Source Position (turned by Source Rotation Offset), and blended with a smooth square, gaussian
    or Mask Input alpha. Frames helper draws both squares (target green, source red)."""

    type = "nadir_patch"
    label = "Nadir Patch"
    catalog = "nadir_patch"
    inputs = {"input": Input(COLOR, "Input", (0.0, 0.0, 0.0, 1.0)),
              "mask_input": Input(GRAY, "Mask Input", (1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(COLOR)}
    params = {
        "enable_": B(True, "Enable"),
        "show_frames_helper": B(True, "Show Frames Helper"),
        "frame_thickness": F(0.25, "Frame Thickness", 0.0, 1.0),
        "patch_scale": F(0.25, "Patch Scale", 0.0, 1.0),
        "patch_size": F2([1.0, 1.0], "Patch Size", 0.0, 1.0),
        "patch_rotation": F(0.0, "Patch Rotation", 0.0, 1.0),
        "patch_alpha": E("Smooth Square", "Patch Alpha", ("Smooth Square", "Gaussian", "Mask Input")),
        "patch_hardness": F(0.5, "Patch Hardness", 0.0, 1.0),
        "source_rotation_offset": F(0.0, "Source Rotation Offset", 0.0, 1.0),
        "source_position": F2([0.75, 0.8], "Source Position", group="Position Coordinates"),
        "patch_position": F2([0.5, 1.0], "Patch Position", group="Position Coordinates"),
    }
