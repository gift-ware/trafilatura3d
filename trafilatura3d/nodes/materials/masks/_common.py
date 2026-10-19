from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import GRAY, MatNode, PreBlurNode, color_in, gray_in, px256

CATEGORY = "Mesh based generators/Mask Generators"
OUT = {"output": Output(GRAY)}
MASK = gray_in("Mask (optional)", 1.0)
CURV = gray_in("Curvature", 0.5)
AO = gray_in("Ambient Occlusion", 1.0)
WSN = color_in("World Space Normal", (0.0, 0.0, 0.0, 0.0))  # unconnected: no direction, nothing faces up
POS = color_in("Position", (0.5, 0.5, 0.5, 1.0))
POS_GRAY = gray_in("Position", 0.5)
VARIATION = gray_in("Variation Mask", 0.5)
NOISE_TYPES = ("Spots", "Cloud", "Moisture", "White Noise")


class MaskNode(MatNode):
    category = CATEGORY
    includes = ("_masks.wgsl",)
    outputs = OUT


class BlurredMaskNode(PreBlurNode):
    category = CATEGORY
    includes = ("_masks.wgsl",)
    outputs = OUT


class EdgeWearNode(BlurredMaskNode):
    """Edge wear: the edge mask, twice the convexity times Curvature Weight (`_edge_wear_pre.wgsl`), blurred by Edges
    Smoothness; the node's own pass adds its grunge and the occlusion and applies Wear Level and Wear Contrast."""

    pre_wgsl = "_edge_wear_pre.wgsl"
    pre_inputs = ("curvature",)

    def blur_sigma(self, job):
        # Pixels at 256, growing slowly with Edges Smoothness (fitted on the output: wide smoothing softens the
        # edges more than it spreads them).
        return 0.38 * max(job.values["edges_smoothness"], 0.0) ** 0.25 * px256(job)
