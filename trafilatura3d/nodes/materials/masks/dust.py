from trafilatura3d.nodes.materials._base import B, F, gray_in
from trafilatura3d.nodes.materials.masks._common import AO, MASK, WSN, MaskNode


class Dust(MaskNode):
    """Dust settled on upward-facing surfaces (World Space Normal) and gathered in occluded areas
    (AO, weighted by Occlusion Amount), mottled by a noise (built-in or the Noise input)."""

    type = "dust"
    label = "Dust"
    catalog = "dust"
    inputs = {"ambient_occlusion": AO, "world_space_normal": WSN, "noise": gray_in("Noise", 0.5), "mask": MASK}
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.5, "Contrast", 0.0, 1.0),
        "occlusion_amount": F(0.5, "Occlusion Amount", 0.0, 1.0),
        "noise_opacity": F(0.0, "Noise Opacity", 0.0, 1.0),
        "override_noise": B(False, "Override Noise"),
    }
