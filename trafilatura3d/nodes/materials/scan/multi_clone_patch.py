from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import B, I, MatNode
from trafilatura3d.nodes.materials.scan._clone import clone_encode, clone_params
from trafilatura3d.nodes.materials.scan._common import CATEGORY
from trafilatura3d.nodes.materials.scan.clone_patch import LUMA_BAND

N = 8


class MultiClonePatch(MatNode):
    """The same Clone Patch on up to eight inputs (Input Count): one patch mask, analysed on Input 1,
    applied to all. Areas: Input 1 with both stamps lightened."""
    counted_inputs = {"input_count": "input_"}

    type = "multi_clone_patch"
    label = "Multi Clone Patch"
    category = CATEGORY
    catalog = "multi_clone_patch"
    band_expr = LUMA_BAND.format("input_1")
    channels = tuple(f"input_{i}" for i in range(1, N + 1))
    inputs = {f"input_{i}": Input(label=f"Input {i}") for i in range(1, N + 1)}
    outputs = {**{f"output_{i}": Output(label=f"Output {i}") for i in range(1, N + 1)}, "areas": Output(label="Areas")}
    params = {"input_count": I(1, "Input Count", 1, 8), "is_normal": B(False, "Is Normal (only for Color)"),
              **clone_params()}

    def encode(self, ctx, job):
        clone_encode(self, ctx, job, {"input_1": self.inputs["input_1"]}, "multi_clone_patch.wgsl", self.inputs,
                     self.outputs)
