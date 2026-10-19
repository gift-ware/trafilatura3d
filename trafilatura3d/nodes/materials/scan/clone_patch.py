from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import B, MatNode
from trafilatura3d.nodes.materials.scan._clone import clone_encode, clone_params
from trafilatura3d.nodes.materials.scan._common import CATEGORY

# The luminance difference between source and target, as the band of the patch border.
LUMA_BAND = "p.threshold * sat(abs(luma(sample_{0}(su).rgb) - luma(sample_{0}(uv).rgb)) * 5.0 + nz)"


class ClonePatch(MatNode):
    """Parametric clone stamp: the target stamp (the disc or square the Target
    Matrix / Offset place) shows the source stamp (Source Matrix / Offset) with the same stamp
    coordinates. The patch border is analysed on a grid (Grid Resolution): within Threshold of the
    stamp edge it follows where source and target look alike, then Blur softens it. Normal maps blend
    as vectors. Areas: the input with both stamps lightened (Show Patch Areas)."""

    type = "clone_patch"
    label = "Clone Patch"
    category = CATEGORY
    catalog = "clone_patch"
    band_expr = LUMA_BAND.format("input")
    inputs = {"input": Input(label="Input")}
    outputs = {"output": Output(), "areas": Output(label="Areas")}
    params = {"is_normal": B(False, "Is Normal (only for Color)"), **clone_params()}

    def encode(self, ctx, job):
        clone_encode(self, ctx, job, self.inputs, "clone_patch.wgsl", self.inputs, self.outputs)
