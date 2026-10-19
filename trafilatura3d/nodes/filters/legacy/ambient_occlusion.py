"""Deprecated Ambient occlusion filters on a height, found in older materials: Ambient occlusion and Ambient
occlusion (Deprecated); the current library replaces both with HBAO. The documentation doesn't list them. Both are
ours, fitted on the legacy nodes' outputs; the height is resampled to the output's size."""

from __future__ import annotations

import math
from pathlib import Path

from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType
from trafilatura3d.nodes.filters.basic._lib import gaussian, source
from trafilatura3d.nodes.generators.legacy._lib import F2, F, LegacyNode, fit, output, target

DIR = Path(__file__).parent
CATEGORY = "Filters/Legacy"
F3 = ParamKind.FLOAT3


class LegacyFilter(LegacyNode):
    """A legacy filter on a height: `image` also takes the input, `source`."""
    category = CATEGORY
    inputs = {"source": Input(PortType.GRAY, "Height")}

    def encode(self, ctx, job):
        self.image(ctx, job, job.values, job.seed, "output", source=source(ctx, job, "source"))


class _Pass(Node):
    """A pass of the Ambient occlusion filters with its shader in this folder."""
    registered = False
    outputs = {"output": Output(PortType.GRAY)}

    def wgsl(self) -> str:
        return (DIR / f"{self.type}.wgsl").read_text(encoding="utf-8")


class _Occlusion(_Pass):
    type = "ao1_occlusion"
    inputs = {"height": Input(PortType.GRAY), "blurred": Input(PortType.GRAY), "lit": Input(PortType.GRAY)}
    internal_uniforms = {"shift": ParamKind.FLOAT2, "depth": ParamKind.FLOAT, "power": ParamKind.FLOAT,
                         "face": ParamKind.FLOAT, "relief": ParamKind.FLOAT, "light": ParamKind.FLOAT,
                         "light_k": ParamKind.FLOAT, "light_dir": ParamKind.FLOAT2, "range": ParamKind.FLOAT2}


class _Cavity(_Pass):
    type = "ao2_cavity"
    inputs = {"height": Input(PortType.GRAY), "blurred": Input(PortType.GRAY)}
    internal_uniforms = {"depth": ParamKind.FLOAT, "steep": ParamKind.FLOAT}


class _Bands(_Pass):
    type = "ao2_bands"
    inputs = {"ao": Input(PortType.GRAY), "fine": Input(PortType.GRAY), "coarse": Input(PortType.GRAY)}
    internal_uniforms = {"equalizer": F3, "gains": ParamKind.FLOAT2}


OCCLUSION, CAVITY, BANDS = _Occlusion(), _Cavity(), _Bands()


class AmbientOcclusionLegacy(LegacyFilter):
    """Ambient occlusion and a directional light on a height (ours, fitted on the legacy node's outputs). The
    reference is the height's gaussian (sigma SPREAD x AO spread pixels at 256, at least a pixel),
    read SHIFT x Light distance x Light intensity of the image away from the light (Light angle in turns): how far
    the height lies under it darkens as exp(-(under / depth)^POWER), depth = DEPTH + DEPTH_SPREAD x AO spread. The
    light received is RELIEF x what rises above the reference plus the facing 1 / (1 + (FACE x slope)^2) (slope in 256
    px units of the height's gaussian of a pixel), plus Light intensity x the slope facing the light of the height's
    gaussian (sigma LIGHT_SIGMA pixels at 256, scale LIGHT_K); clamped and multiplied. Levels in AO levels."""
    type = "ambient_occlusion_legacy"
    label = "Ambient occlusion"
    params = {
        "ao_spread": Param(F, 0.5, "AO spread", 0.0, 1.0),
        "ao_levels": Param(F2, [0.0, 1.0], "AO levels", 0.0, 1.0),
        "light_intensity": Param(F, 0.0, "Light intensity", 0.0, 1.0),
        "light_angle": Param(F, 0.0, "Light angle", 0.0, 1.0),
        "light_distance": Param(F, 0.0, "Light distance", 0.0, 1.0),
    }
    outputs = {"output": Output(PortType.GRAY, "Output")}
    SPREAD, SHIFT = 1.8, 0.01
    DEPTH, DEPTH_SPREAD, POWER = 0.108, 0.078, 0.588
    FACE, RELIEF = 7.59, 2.01
    LIGHT_SIGMA, LIGHT_K = 0.265, 24.9

    def image(self, ctx, job, v, seed, dst=None, source=None):
        src = fit(ctx, job, source, job.size)
        px = job.size[0] / 256.0
        spread, intensity = v["ao_spread"], v["light_intensity"]
        blurred = gaussian(ctx, job, src[0], True, math.hypot(self.SPREAD * spread * px, 1.0))
        lit = gaussian(ctx, job, src[0], True, max(self.LIGHT_SIGMA * px, 0.5)) if intensity > 0.0 else src[0]
        a = 2.0 * math.pi * v["light_angle"]
        direction = [math.cos(a), math.sin(a)]
        shift = -self.SHIFT * v["light_distance"] * intensity
        out = target(ctx, job, dst)
        ctx.dispatch(OCCLUSION, job, {"shift": [shift * direction[0], shift * direction[1]],
                                      "depth": self.DEPTH + self.DEPTH_SPREAD * spread, "power": self.POWER,
                                      "face": self.FACE, "relief": self.RELIEF, "light": intensity,
                                      "light_k": self.LIGHT_K, "light_dir": direction,
                                      "range": [float(x) for x in v["ao_levels"]]},
                     srcs={"height": src, "blurred": (blurred, True), "lit": (lit, True)}, dsts={"output": out})
        return out, True


class AmbientOcclusion2Legacy(LegacyFilter):
    """Ambient occlusion with an equalizer (ours, fitted on the legacy node's outputs): how far
    the height lies under its gaussian blur (sigma SPREAD x Spreading pixels at 256) darkens by DEPTH per unit, raised
    to 1 + the facing of the height's normal (slope scale STEEP): cavities on flat ground darker than on walls. The
    Equalizer works on three gaussian bands (FINE, COARSE pixels at 256): the coarse one through a gamma, the others
    scaled (GAINS); at 0, 0, 0 the AO unchanged. Levels in low / mid / high from Levels."""
    type = "ambient_occlusion_2_legacy"
    label = "Ambient occlusion (Deprecated)"
    params = {
        "spreading": Param(F, 0.15, "Spreading", 0.0, 1.0),
        "equalizer": Param(F3, [0.0, 0.0, 0.0], "Equalizer", -1.0, 1.0, clamp=True),
        "levels": Param(F3, [0.0, 0.5, 1.0], "Levels", 0.0, 1.0, clamp=True),
    }
    outputs = {"output": Output(PortType.GRAY, "Ambient occlusion")}
    SPREAD, DEPTH, STEEP = 77.0, 2.0, 48.0
    FINE, COARSE, GAINS = 1.6, 2.4, (2.5, 2.0)

    def image(self, ctx, job, v, seed, dst=None, source=None):
        src = fit(ctx, job, source, job.size)
        eq, (low, mid, high) = v["equalizer"], v["levels"]
        px = job.size[0] / 256.0
        blurred = gaussian(ctx, job, src[0], True, self.SPREAD * v["spreading"] * px)
        ao = ctx.scratch(job, True, "32f")
        ctx.dispatch(CAVITY, job, {"depth": self.DEPTH, "steep": self.STEEP},
                     srcs={"height": src, "blurred": (blurred, True)}, dsts={"output": ao})
        if any(eq):
            fine = gaussian(ctx, job, ao, True, self.FINE * px)
            coarse = gaussian(ctx, job, ao, True, self.COARSE * px)
            out = ctx.scratch(job, True, "32f")
            ctx.dispatch(BANDS, job, {"equalizer": list(eq), "gains": list(self.GAINS)},
                         srcs={"ao": (ao, True), "fine": (fine, True), "coarse": (coarse, True)}, dsts={"output": out})
            ao = out
        return output(ctx, job, (ao, True), dst, low=low, mid=mid, high=high)
