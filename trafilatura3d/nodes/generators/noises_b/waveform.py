"""Waveform: Samples bars side by side, each as tall as a signal at its position, computed per pixel (waveform.wgsl)."""

from trafilatura3d.fxmap import PATTERNS
from trafilatura3d.nodes.base import Node, Output, Param, ParamKind, PortType
from trafilatura3d.nodes.generators.patterns.common import SHAPES_WGSL

from ._base import CATEGORY, DIR, LIB, disorder_params, f, i

SHAPES = PATTERNS[2:]  # the built-in patterns (Square ... Hemisphere): the index is `shape()`'s kind


class Waveform1(Node):
    """The signal is a fractal sum of octaves (Scale x 2^o periods, weighted by Roughness^o) of a cosine with a random
    phase or of a smooth Hermite noise (Value noise; Disorder evolves it), or the Custom function's constant (a
    function graph: here the constant it returns). Each bar is the Pattern, Amplitude max. x
    mix(Amplitude min., 1, signal x (1 - Noise + Noise x random)) tall, centred or standing on the bottom. Defaults:
    those of the documentation's preview (Cosine, Scale 2, Noise 0.5, Gaussian, 256 samples, a clean signal)."""
    type = "waveform"
    label = "Waveform"
    catalog = "waveform_1"
    category = CATEGORY
    inputs = {}
    outputs = {"output": Output(PortType.GRAY, "Output")}
    params = {
        "samples": i(256, "Samples", 1, 1024),
        "function": Param(ParamKind.ENUM, "Cosine", "Function",
                          choices=("Value noise", "Cosine", "Custom function")),
        "custom_function": f(1.0, "Custom function"),
        "roughness": f(0.01, "Roughness", 1e-5, 1.0),
        "scale": i(2, "Scale", 1, 64),
        "amplitude_min": f(0.0, "Amplitude min."),
        "amplitude_max": f(1.0, "Amplitude max."),
        "noise": f(0.5, "Noise"),
        "position": Param(ParamKind.ENUM, "Centered", "Position", choices=("Centered", "Bottom")),
        "pattern": Param(ParamKind.ENUM, "Gaussian", "Pattern", choices=SHAPES),
        "pattern_variation": f(0.0, "Pattern variation"),
        **disorder_params(anisotropy=False),
    }

    def wgsl(self) -> str:
        return LIB + SHAPES_WGSL + (DIR / "waveform.wgsl").read_text(encoding="utf-8")
