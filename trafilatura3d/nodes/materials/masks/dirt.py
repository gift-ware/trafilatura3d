from trafilatura3d.nodes.materials._base import B, F, I, gray_in
from trafilatura3d.nodes.materials.masks._common import AO, MASK, POS, WSN, MaskNode


class Dirt(MaskNode):
    """Dirt in occluded areas and sunken corners (dark AO, concave curvature), removed from raised
    edges by Edges Masking, broken up by a built-in or custom grunge (optionally projected
    triplanar from Position and World Space Normal to hide UV seams)."""

    type = "dirt"
    label = "Dirt"
    catalog = "dirt"
    inputs = {"curvature": gray_in("Curvature", 0.0), "ambient_occlusion": AO, "grunge_input": gray_in("Grunge input", 0.5),
              "mask": MASK, "world_space_normal": WSN, "position": POS}
    params = {
        "dirt_level": F(0.75, "Dirt Level", 0.0, 1.0),
        "dirt_contrast": F(0.5, "Dirt Contrast", 0.0, 1.0),
        "grunge_amount": F(0.3, "Grunge Amount", 0.0, 1.0),
        "edges_masking": F(0.5, "Edges Masking", 0.0, 1.0),
        "use_custom_grunge": B(False, "Use Custom Grunge"),
        "grunge_scale": I(1, "Grunge Scale", 1, 16),
        "use_triplanar": B(False, "Use Triplanar"),
        "triplanar_blending_contrast": F(0.5, "Triplanar Blending Contrast", 0.001, 1.0),
    }
