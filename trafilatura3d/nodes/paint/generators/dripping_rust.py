from trafilatura3d.nodes.base import Input, ParamKind, PortType
from trafilatura3d.nodes.paint._lib import DOC, SEED, B, E, F, I, mesh_inputs, soft_blur
from trafilatura3d.nodes.paint.generators._common import GenNode, extended


class DrippingRust(GenNode):
    """Rust where water stays (cavities of the curvature, occluded areas), spread by Rust Spreading (blur then
    threshold, soft by Spreading Smoothness, broken by a grunge in world space), and drips running down from it:
    each pixel looks up the surface (in texture space, along the slope of the chosen position axis) for
    Drips Samples Amount steps over a length set by Drips Intensity, stopping at UV seams (the position map jumps),
    and takes the rust it finds, fading with the distance and streaked by a noise across the drips; Drips Smoothness
    blurs the drips. Rust Contrast on the result. Invert inverts curvature and ambient occlusion before they are
    combined (documentation). Horizontal faces get no drips (no downhill direction)."""
    type = "paint_dripping_rust"
    label = "Dripping Rust"
    catalog = "painter.generator.dripping_rust"
    doc = DOC + "generators/dripping-rust"
    includes = ("generators/common",)
    inputs = mesh_inputs("curvature", "baked_ambient_occlusion", "position")
    params = {"invert": B("Invert", False), "rust_spreading": F("Rust Spreading", 0.3, 0.0, 1.0),
              "rust_contrast": F("Rust Contrast", 0.3, -1.0, 1.0),
              "spreading_smoothness": F("Spreading Smoothness", 0.3, 0.0, 1.0),
              "drips_intensity": F("Drips Intensity", 0.5, 0.0, 1.0),
              "drips_smoothness": F("Drips Smoothness", 0.2, 0.0, 1.0),
              "drips_samples_amount": I("Drips Samples Amount", 32, 4, 128),
              "position_axis": E("Position axis", ("Y-Green channel", "X-Red channel", "B-Blue channel"))}
    unsupported = {**SEED}
    STAGES = {"sources_": Input(PortType.GRAY), "spread_": Input(PortType.GRAY), "rust_": Input(PortType.GRAY),
              "drips_": Input(PortType.GRAY)}

    def encode(self, ctx, job):
        st, v = extended(self, "stages", self.STAGES, {"stage_": ParamKind.INT}), job.values
        sources = ctx.scratch(job, True)
        ctx.dispatch(st, job, {**v, "stage_": 0}, dsts={"output": sources})
        srcs = {"sources_": (sources, True), "spread_": (soft_blur(ctx, job, sources, True, 1.0 + v["rust_spreading"] * 12.0), True)}
        rust = ctx.scratch(job, True)
        ctx.dispatch(st, job, {**v, "stage_": 1}, srcs=srcs, dsts={"output": rust})
        srcs["rust_"] = (rust, True)
        drips = ctx.scratch(job, True)
        ctx.dispatch(st, job, {**v, "stage_": 2}, srcs=srcs, dsts={"output": drips})
        srcs["drips_"] = (soft_blur(ctx, job, drips, True, v["drips_smoothness"] * 6.0), True)
        ctx.dispatch(st, job, {**v, "stage_": 3}, srcs=srcs)
