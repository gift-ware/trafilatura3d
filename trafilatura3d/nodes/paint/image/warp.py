from dataclasses import replace

from trafilatura3d.nodes.atomic.directional_warp import DirectionalWarp
from trafilatura3d.nodes.atomic.warp import Warp as AtomicWarp
from trafilatura3d.nodes.base import Input, PortType
from trafilatura3d.nodes.filters.effects.multi_directional_warp import MultiDirectionalWarp
from trafilatura3d.nodes.paint._lib import DOC, SEED, E, F, Filter

from ._passes import DIVIDERS, SOURCES, make_source

SRC = "Source Parameters"
REGULAR, DIRECTIONAL, MULTI = AtomicWarp(), DirectionalWarp(), MultiDirectionalWarp()
REGULAR_GAIN = 1 / 16  # the Warp node moves 16 times as far for the same Intensity: matched to the other modes


class Warp(Filter):
    """Deforms the input by a grayscale source: the default noise (tileable fbm, the node's Random Seed), the
    Previous Input (the input's own luminance) or the Custom Noise input, repeated Source Tilling times, balanced,
    contrasted and blurred (Source Blur, pixels at 256). Warp: pushed along the source's slopes (as the Warp node);
    Directional: moved along Angle (turns, clockwise from +X) by source x Intensity (as the Directional Warp node);
    Multi Directional: Directions pairs of opposite moves (1, 2 or 4) averaged or combined by Max / Min (Blending
    Mode) so the content stays in place (as the Multi Directional Warp node). Intensity in pixels at 256, divided
    by the Intensity Divider."""
    type = "paint_warp"
    label = "Warp"
    catalog = "painter.filter.warp"
    doc = DOC + "filters/standard-filters/warp"
    inputs = {"input": Input(label="Input"),
              "custom_noise": Input(PortType.GRAY, "Custom Noise", (0.5, 0.5, 0.5, 1.0))}
    params = {"warp_mode": E("Warp Mode", ("Warp", "Directional", "Multi Directional")),
              "intensity": F("Intensity", 8.0, 0.0, 64.0), "intensity_divider": E("Intensity Divider", DIVIDERS),
              "angle": F("Angle", 0.0, 0.0, 1.0),
              "blending_mode": E("Blending Mode", ("Average", "Max", "Min")),
              "directions": E("Directions", ("1", "2", "4")),
              "source_mode": replace(E("Source Mode", SOURCES), group=SRC),
              "source_blur": replace(F("Source Blur", 1.0, 0.0, 16.0), group=SRC),
              "source_balance": replace(F("Source Balance", 0.5, 0.0, 1.0), group=SRC),
              "source_contrast": replace(F("Source Contrast", 0.0, 0.0, 1.0), group=SRC),
              "source_tilling": replace(F("Source Tilling", 1.0, 1.0, 16.0), group=SRC)}
    unsupported = {**SEED}

    def encode(self, ctx, job):
        v = job.values
        src = (make_source(ctx, job, v["source_mode"], v["source_blur"], v["source_balance"], v["source_contrast"],
                           v["source_tilling"]), True)
        amount = v["intensity"] / float(v["intensity_divider"])
        if v["warp_mode"] == "Warp":
            ctx.dispatch(REGULAR, job, {"intensity": amount * REGULAR_GAIN, "input_filtering_mode": "Bilinear"},
                         srcs={"gradient_input": src})
        elif v["warp_mode"] == "Directional":
            ctx.dispatch(DIRECTIONAL, job, {"intensity": amount, "warp_angle": v["angle"],
                                            "input_filtering_mode": "Bilinear", "intensity_map_offset": 0.0},
                         srcs={"intensity_input": src})
        else:
            ctx.dispatch(MULTI, job, {"intensity": amount, "warp_angle": v["angle"], "mode": v["blending_mode"],
                                      "directions": v["directions"], "pass_angle": -1.0},
                         srcs={"intensity_input": src})
