from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.filters.basic._lib import gaussian
from trafilatura3d.nodes.materials._base import (B, COLOR, GRAY, F, F2, I, MatNode, Pass, color_in, gray_in,
                                           material_inputs, material_outputs, passes_of, px256)
from trafilatura3d.nodes.materials.scan._common import CATEGORY, crop_params


class SmartAutoTile(MatNode):
    """Makes a non-tiling Base Color / Normal / Height set tile. The cropped area (Input Size,
    Transform, Offset as in Crop) is shown as is in the middle and shifted by half a tile near the
    borders, where the shifted copy is continuous across the tile edges. The cut between them sits
    Cut Offset from the borders (X and Y apart) and, analysed on a grid (Grid Resolution), moves in
    steps towards where both copies look alike, by Threshold (global or per channel); Detect Edges
    also favours running along detected shapes. Blur softens the cut, Smoothness smooths its
    analysis; Use Mask takes the Mask as the shape of the shifted area. Unused channels pass the crop."""

    type = "auto_tile"
    label = "Auto Tile"
    category = CATEGORY
    catalog = "smart_auto_tile"
    channels = ("basecolor", "normal", "height")
    inputs = {**material_inputs(("basecolor", "normal", "height")), "mask": gray_in("Mask", 0.0)}
    outputs = material_outputs(("basecolor", "normal", "height"))
    params = {
        **crop_params(group="Crop", background=False, area=False, scale=1.5),
        "detect_edges": B(True, "Detect Edges", group="Edge"),
        "use_threshold_per_channel": B(False, "Use Threshold Per Channel", group="Edge"),
        "threshold": F(0.5, "Threshold", 0.0, 1.0, group="Edge"),
        "threshold_base_color": F(0.5, "Threshold Base Color", 0.0, 1.0, group="Edge"),
        "threshold_normal": F(0.5, "Threshold Normal", 0.0, 1.0, group="Edge"),
        "threshold_height": F(0.5, "Threshold Height", 0.0, 1.0, group="Edge"),
        "cut_offset": F2([0.2, 0.2], "Cut Offset", 0.0, 0.5, group="Edge"),
        "blur": F(0.0, "Blur", 0.0, 2.0, group="Edge"),
        "smoothness": F(0.1, "Smoothness", 0.0, 2.0, group="Edge"),
        "grid_resolution": I(9, "Grid Resolution", 1, 11, group="Edge"),
        "use_base_color": B(True, "Use Base Color", group="Edge"),
        "use_normal": B(True, "Use Normal", group="Edge"),
        "use_height": B(True, "Use Height", group="Edge"),
        "use_mask": B(False, "Use Mask", group="Edge"),
    }

    def encode(self, ctx, job):
        seam, comp = passes_of(self, lambda n: (
            Pass(n, "seam", "auto_tile_seam.wgsl", n.inputs, {"m": Output(GRAY)}, includes=("_crop.wgsl",)),
            Pass(n, "comp", "auto_tile.wgsl", {**{k: n.inputs[k] for k in n.channels}, "m": Input(GRAY)},
                 n.outputs, includes=("_crop.wgsl",))))
        m = ctx.scratch(job, True)
        ctx.dispatch(seam, job, job.values, dsts={"m": m})
        mb = gaussian(ctx, job, m, True, (job.values["blur"] * 4.0 + 0.7) * px256(job))
        ctx.dispatch(comp, job, job.values, srcs={"m": (mb, True)})
