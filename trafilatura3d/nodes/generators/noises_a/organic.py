"""Fur 1-3, Fluid and Creased: strand, drip and crease noises built from shapes scattered on
periodic grids (Creased: a sparse convolution of blades, then a bending domain warp)."""

from trafilatura3d.nodes.base import Input, Node, Param, PortType

from .common import CATEGORY, GRAY_OUT, HERE, LIB, NSE_CAPS, F, F2, I, B, NoiseNode, scale

DISORDER_01 = {"disorder": Param(F, 0.0, "Disorder", 0.0, 1.0)}


class Fur1(NoiseNode):
    """Straight fur: long thin strands, nearly vertical, each brightest in its middle, overlaid with
    Max; Disorder slides them along themselves."""
    type = "fur_straight"
    label = "Fur Straight"
    catalog = "fur_1"
    shader = "fur"
    params = {**scale(2, 1, 8), **DISORDER_01, **NSE_CAPS}
    scale_cells = 2
    variant = 0
    snippets = {"params": "dis = p.disorder;"}


class Fur2(NoiseNode):
    """Wavy fur: short horizontal strands turned by waves across the tile (Waves Scale: fewer
    waves when bigger; Waves Rotation: how far they turn)."""
    type = "fur_wavy"
    label = "Fur Wavy"
    catalog = "fur_2"
    shader = "fur"
    params = {**scale(1, 1, 8), "waves_scale": Param(F, 0.0, "Waves Scale", 0.0, 1.0),
              "waves_rotation": Param(F, 0.125, "Waves Rotation", 0.0, 1.0), **NSE_CAPS}
    scale_cells = 4
    variant = 1
    snippets = {"params": "k = max(p.waves_scale, 0.0); t = p.waves_rotation;"}


class Fur3(NoiseNode):
    """Spreading bristles: short strands fanning out; Waves Amount is the number of fans across the
    tile (upright at their centres, leaning outwards towards their edges)."""
    type = "fur_fanned"
    label = "Fur Fanned"
    catalog = "fur_3"
    shader = "fur"
    params = {**scale(1, 1, 8), **DISORDER_01, "waves_amount": Param(F, 2.0, "Waves Amount", 0.0, 8.0),
              **NSE_CAPS}
    scale_cells = 4
    variant = 2
    snippets = {"params": "dis = p.disorder; k = p.waves_amount;"}


class Fluid(NoiseNode):
    """Falling fluid: dense drips, each a round bright head under a trail darkening upwards, wobbling
    sideways up the trail (Warp Intensity), as wide as Pattern Size x and as long as its y; Disorder makes them
    fall (each at its own speed, looping at 1)."""
    type = "fluid"
    label = "Fluid"
    catalog = "fluid"
    params = {**scale(1, 1, 8), **DISORDER_01, "warp_intensity": Param(F, 0.5, "Warp Intensity", 0.0, 1.0),
              "pattern_size": Param(F2, [0.5, 1.0], "Pattern Size", 0.0, 1.0), **NSE_CAPS}
    scale_cells = 3
    internal_uniforms = {"flow": F, "jitter": F, "gaps": F}  # Fluid (Legacy)'s, none here

    def pack(self, values: dict) -> bytes:
        return super().pack({"flow": 0.0, "jitter": 0.0, "gaps": 0.0, **values})


class _CreasedBlades(Node):
    """Creased's blades (creased_blades.wgsl), at half resolution."""
    type = "creased_blades"
    registered = False
    outputs = GRAY_OUT
    params = {"scale": Param(I, 1, ""), "disorder": Param(F, 0.0, ""), "non_square_expansion": Param(B, True, "")}

    def wgsl(self) -> str:
        return LIB + (HERE / "creased_blades.wgsl").read_text(encoding="utf-8")


class _CreasedWarp(Node):
    """Creased's last pass (creased.wgsl)."""
    type = "creased_warp"
    registered = False
    inputs = {"blades": Input(PortType.GRAY)}
    outputs = GRAY_OUT
    params = {"scale": Param(I, 1, ""), "warp_intensity": Param(F, 0.0, ""),
              "non_square_expansion": Param(B, True, "")}

    def wgsl(self) -> str:
        return LIB + (HERE / "creased.wgsl").read_text(encoding="utf-8")


BLADES, WARP = _CreasedBlades(), _CreasedWarp()


class Creased(Node):
    """Cloth-like creases: a sparse convolution of tilted blades with a crisp edge along one side (8 and 16 x Scale
    blade units across the tile, creased_blades.wgsl; Disorder moves them), read through a domain warp that bends the
    rows sideways by Warp Intensity x (1 - (2y - 1)^2) of the tile (leaning one way in the top half, the other in the
    bottom half) and curls them with a coarse noise (creased.wgsl)."""
    type = "creased"
    label = "Creased"
    catalog = "creased"
    category = CATEGORY
    outputs = GRAY_OUT
    params = {**scale(1, 1, 8), "warp_intensity": Param(F, 0.25, "Warp Intensity", 0.0, 1.0),
              "disorder": Param(F, 0.0, "Disorder", 0.0, 1.0), **NSE_CAPS}

    def encode(self, ctx, job):
        v = job.values
        # The blades are soft across a few pixels: drawn at half resolution, the last pass samples them bilinearly.
        small = (max(job.size[0] // 2, 1), max(job.size[1] // 2, 1))
        blades = ctx.scratch(job, True, "16f", small)
        ctx.dispatch(BLADES, job, {k: v[k] for k in ("scale", "disorder", "non_square_expansion")},
                     dsts={"output": blades}, size=small)
        ctx.dispatch(WARP, job, {k: v[k] for k in ("scale", "warp_intensity", "non_square_expansion")},
                     srcs={"blades": (blades, True)})
