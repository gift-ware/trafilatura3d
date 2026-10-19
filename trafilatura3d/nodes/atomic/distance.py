"""Distance: from the white areas of the mask (binarized at 0.5) outwards, a linear fade that stops
at Maximum distance and where cells meet; with a Source, each cell takes the source value at its
nearest white pixel (Voronoi-like). Nearest white pixels are found by jump flooding on the GPU:
a seed pass, log2(size) flood passes (fewer when only Maximum distance matters), then the final pass."""

from pathlib import Path

from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType
from trafilatura3d.nodes.filters.effects._fx import first_jump

_DIR = Path(__file__).parent
# The seeds packed in 16 bit integers as the effects' jump flooding keeps them, then how this node reads them.
_LIBS = (_DIR.parent / "filters" / "effects" / "_jfa_lib.wgsl", _DIR / "distance_lib.wgsl")
METRICS = ("Euclidean", "Manhattan", "Chebyshev")


class Distance(Node):
    type = "distance"
    label = "Distance"
    category = "Atomic nodes"
    catalog = "distance"
    inputs = {"mask_input": Input(PortType.GRAY, "Mask input"), "source_input": Input(label="Source input")}
    main_input = "mask_input"
    outputs = {"output": Output()}
    color_mode = "color_mode"
    params = {
        "color_mode": Param(ParamKind.ENUM, "Grayscale", "Color mode", choices=("Grayscale", "Color")),
        "maximum_distance": Param(ParamKind.FLOAT, 10.0, "Maximum distance", 0.0, 256.0),  # px at 256
        "combine_source_distance": Param(ParamKind.ENUM, "Combine", "Combine source/distance",
                                         choices=("Combine", "Only Source")),
        "distance_mode": Param(ParamKind.ENUM, "Euclidean", "Distance mode", choices=METRICS),
    }

    def encode(self, ctx, job):
        w, h = job.size
        seeds = [ctx.scratch(job, True, "16", alpha=True), ctx.scratch(job, True, "16", alpha=True)]
        ctx.dispatch(SEED, job, {}, dsts={"seeds": seeds[0]})
        metric = METRICS.index(job.values["distance_mode"])
        # Past Maximum distance the fade is 0 and a pixel with no seed found is black as well: the jumps start at
        # the power of 2 past it. A source is copied from the nearest seed however far it is.
        reach = None if job.inputs.get("source_input") else job.values["maximum_distance"] * w / 256.0 + 1.0
        step, cur = first_jump(w, h, reach), 0
        while step >= 1:
            ctx.dispatch(FLOOD, job, {"step": step, "metric": metric},
                         srcs={"seeds": (seeds[cur], False)}, dsts={"seeds": seeds[1 - cur]})
            cur, step = 1 - cur, step // 2
        ctx.dispatch(FINAL, job, job.values, srcs={"seeds": (seeds[cur], False)})


class _Seed(Node):
    """Each white mask pixel is its own seed (its coordinates); the others have none (-1)."""
    type = "distance_seed"
    registered = False
    inputs = {"mask_input": Input(PortType.GRAY)}
    outputs = {"seeds": Output(PortType.COLOR)}

    def wgsl(self):
        return "\n".join(f.read_text(encoding="utf-8") for f in (_LIBS[0], _DIR / "distance_seed.wgsl"))


class _Flood(Node):
    """Jump flooding: keep the nearest of the seeds found at +-step in every direction."""
    type = "distance_flood"
    registered = False
    inputs = {"seeds": Input(PortType.COLOR)}
    outputs = {"seeds": Output(PortType.COLOR)}
    internal_uniforms = {"step": ParamKind.INT, "metric": ParamKind.INT}

    def wgsl(self):
        return "\n".join(f.read_text(encoding="utf-8") for f in (*_LIBS, _DIR / "distance_flood.wgsl"))


class _Final(Node):
    """The fade and the source copy from each pixel's nearest seed."""
    type = "distance_final"
    registered = False
    inputs = {"source_input": Input(), "seeds": Input(PortType.COLOR)}
    outputs = {"output": Output()}
    params = Distance.params

    def wgsl(self):
        return "\n".join(f.read_text(encoding="utf-8") for f in (*_LIBS, _DIR / "distance.wgsl"))


SEED, FLOOD, FINAL = _Seed(), _Flood(), _Final()
