from trafilatura3d.nodes.base import Input
from trafilatura3d.nodes.materials._base import COLOR, B, E, F, gray_in, px256
from trafilatura3d.nodes.materials.masks._common import AO, CURV, MASK, VARIATION, BlurredMaskNode


class Leaks(BlurredMaskNode):
    """Streaks of grime leaking from sharp edges: sources are convex edges (curvature) picked by the
    variation (built-in or the Variation Mask), traced downhill on the baked Position over Length,
    thinned across the run, attenuated by AO on faces turned towards the Occlude axis."""

    type = "leaks"
    label = "Leaks"
    catalog = "leaks"
    unsupported = {"Lenght": "the documentation's typo of Length"}
    inputs = {"position": gray_in("Position", 0.5), "curvature": CURV, "ambient_occlusion": AO,
              "normal_world_space": Input(COLOR, "Normal World Space", (0.0, 0.0, 0.0, 0.0)),
              "variation_mask": VARIATION, "mask": MASK}
    pre_inputs = ("curvature", "variation_mask")
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
        "variation": F(0.5, "Variation", 0.0, 1.0),
        "length": F(4.0, "Length", 0.0, 8.0),
        "occlude": E("None", "Occlude", ("X", "Y", "Z", "None")),
        "override_variation_mask": B(False, "Override variation mask"),
    }

    def blur_sigma(self, job):
        return 1.0 * px256(job)
