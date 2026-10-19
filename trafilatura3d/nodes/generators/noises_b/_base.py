"""Shared contract of the noises_b generators: the shader is the node's own `.wgsl` (named after its
`type`) after `noise_lib.wgsl`; grunge maps add a finishing pass (sharpen, Invert, Brush Pattern, then
Balance and Contrast through our Histogram Scan) shared by all of them.

Several recipes were tuned in units of their own, so a node's `internal` maps the parameter's value to the one
its shader reads."""

from pathlib import Path
from typing import ClassVar

from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType
from trafilatura3d.nodes.filters.basic.histogram_scan import HistogramScan

DIR = Path(__file__).parent
LIB = (DIR / "noise_lib.wgsl").read_text(encoding="utf-8")
CATEGORY = "Texture generators/Noises"


def f(default, label, lo=0.0, hi=1.0, group=""):
    return Param(ParamKind.FLOAT, float(default), label, lo, hi, group=group)


def i(default, label, lo=1, hi=64, group=""):
    return Param(ParamKind.INT, int(default), label, lo, hi, group=group)


def b(default, label, group=""):
    return Param(ParamKind.BOOL, bool(default), label, group=group)


def f2(default, label, lo=-1.0, hi=1.0):
    return Param(ParamKind.FLOAT2, list(default), label, lo, hi)


def disorder_params(anisotropy=True):
    """The disorder controls: Disorder, Disorder speed, and the anisotropy pair."""
    d = {"disorder": f(0.0, "Disorder"), "disorder_speed": f(1.0, "Disorder speed", 0.0, 2.0)}
    if anisotropy:
        d["disorder_anisotropy"] = f(0.0, "Disorder anisotropy")
        d["disorder_anisotropy_angle"] = f(0.0, "Disorder anisotropy angle")
    return d


class Generator(Node):
    """A grayscale noise without inputs."""
    category = CATEGORY
    inputs = {}
    outputs = {"output": Output(PortType.GRAY, "Output")}
    libs: ClassVar[tuple[str, ...]] = ()  # extra shared .wgsl files of this folder, before the node's
    # The parameter's value -> the shader's: {param: (offset, factor)}, offset + factor x value (rounded for an Int,
    # a Bool flipped by (1, -1)); the default gives the value the recipe was tuned with.
    internal: ClassVar[dict[str, tuple[float, float]]] = {}

    def wgsl(self) -> str:
        return "\n".join([LIB, *((DIR / name).read_text(encoding="utf-8") for name in self.libs),
                          (DIR / f"{self.type}.wgsl").read_text(encoding="utf-8")])

    def shader_values(self, values: dict) -> dict:
        """The job's values as the shader reads them (`internal`)."""
        out = dict(values)
        for name, (offset, factor) in self.internal.items():
            x = offset + factor * float(values[name])
            kind = self.params[name].kind
            out[name] = round(x) if kind is ParamKind.INT else bool(round(x)) if kind is ParamKind.BOOL else x
        return out

    def encode(self, ctx, job):
        ctx.dispatch(self, job, self.shader_values(job.values))


class _GrungeFinish(Node):
    type = "noises_b_grunge_finish"
    registered = False
    inputs = {"input": Input(PortType.GRAY)}
    outputs = {"output": Output(PortType.GRAY)}
    params = {"sharpen": f(0.0, "Sharpen"), "window": f(0.0, "Window"), "invert": b(False, "Invert"),
              "brush": f(0.0, "Brush")}

    def wgsl(self) -> str:
        return LIB + (DIR / "grunge_finish.wgsl").read_text(encoding="utf-8")


FINISH = _GrungeFinish()
SCAN = HistogramScan()


def grunge_params(advanced: dict, first: dict | None = None, brush=False, extra_main: dict | None = None,
                  contrast=0.0):
    """The common grunge controls, then the node's Advanced group."""
    d = dict(first or {})
    d["balance"] = f(0.5, "Balance")
    d["contrast"] = f(contrast, "Contrast")
    d.update(extra_main or {})
    d["invert"] = b(False, "Invert")
    if brush:
        d["brush_pattern"] = f(0.0, "Brush Pattern")
    d["non_square_expansion"] = b(True, "Non Square Expansion")
    for name, prm in advanced.items():
        d[name] = Param(prm.kind, prm.default, prm.label, prm.min, prm.max, prm.choices, "Advanced")
    return d


class Grunge(Generator):
    """A grunge map: the node's shader writes the raw pattern, the shared finishing passes apply sharpen, Invert
    and Brush Pattern, then Balance and Contrast as a Histogram Scan (Position = Balance) last.
    The raw pattern was tuned to the preview as the output at the default Balance / Contrast; the default scan
    window (Contrast c0: 1 - c0 wide) is undone first, so the default output is the raw pattern."""
    sharpen: ClassVar[str | None] = None  # name of the node's sharpen parameter, if it has one

    def encode(self, ctx, job):
        raw, finished = ctx.scratch(job, True), ctx.scratch(job, True)
        v = self.shader_values(job.values)
        ctx.dispatch(self, job, v, dsts={"output": raw})
        ctx.dispatch(FINISH, job, {"sharpen": float(v[self.sharpen]) if self.sharpen else 0.0,
                                   "window": self.params["contrast"].default, "invert": v["invert"],
                                   "brush": float(v.get("brush_pattern", 0.0))},
                     srcs={"input": (raw, True)}, dsts={"output": finished})
        ctx.dispatch(SCAN, job, {"position": v["balance"], "contrast": v["contrast"], "invert_position": False},
                     srcs={"input": (finished, True)})
