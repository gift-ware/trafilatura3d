from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class RTIrradiance(Effect):
    """Ray traced lighting of a height map, lit by an Environment (latitude-longitude, Z up,
    Environment Rotation in turns, x 2^Exposure EV) and by the Emissive input (black = no light,
    x Emissive Intensity, read as sRGB or linear). Quality = rays per pixel, cosine-distributed around
    the surface normal and marched through the height field (Height Scale = height of white, fraction
    of the width): an escaping ray sees the environment, a blocked one the emission where it lands
    (averaged over a footprint that Emissive LOD Bias widens: lower = sharper but noisier). Compute
    Bounces adds one bounce: a second pass where landing rays also see the first pass's light where
    they land. The surface's albedo is the Base Color (sRGB) x (1 - Metallic). Raw Irradiance = the
    light received x the albedo, linear; alpha = the share of rays reaching the environment when IBL
    Shadows in Raw Irradiance Alpha is on. Output = that plus the emission, in sRGB."""

    type = "rt_irradiance"
    label = "RT Irradiance"
    category = "Filters/Effects"
    catalog = "rt_irradiance"
    libs = ("_rays.wgsl",)
    inputs = {"basecolor": Input(PortType.COLOR, "Base Color", default=(1.0, 1.0, 1.0, 1.0)),
              "metallic": Input(PortType.GRAY, "Metallic", default=(0.0, 0.0, 0.0, 1.0)),
              "height": Input(PortType.GRAY, "Height"),
              "emissive": Input(PortType.COLOR, "Emissive", default=(0.0, 0.0, 0.0, 1.0)),
              "environment": Input(PortType.COLOR, "Environment", default=(0.0, 0.0, 0.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR, "Output"),
               "raw_irradiance": Output(PortType.COLOR, "Raw Irradiance")}
    params = {
        "height_scale": Param(ParamKind.FLOAT, 0.1, "Height Scale", 0.0, 1.0),
        "quality": Param(ParamKind.ENUM, "64 rays", "Quality", choices=("32 rays", "64 rays", "128 rays")),
        "compute_bounces": Param(ParamKind.BOOL, True, "Compute Bounces"),
        "environment_rotation": Param(ParamKind.FLOAT, 0.0, "Environment Rotation", 0.0, 1.0),
        "environment_exposure_ev": Param(ParamKind.FLOAT, 0.0, "Environment Exposure (EV)", -4.0, 4.0),
        "emissive_intensity": Param(ParamKind.FLOAT, 10.0, "Emissive Intensity", 0.0, 20.0),
        "emissive_color_space": Param(ParamKind.ENUM, "sRGB", "Emissive Color Space", choices=("sRGB", "Linear")),
        "ibl_shadows_in_raw_irradiance_alpha": Param(ParamKind.BOOL, False, "IBL Shadows in Raw Irradiance Alpha"),
        "emissive_lod_bias": Param(ParamKind.FLOAT, 0.0, "Emissive LOD Bias", -1.0, 1.0),
    }
    internal_uniforms = {"bounce": ParamKind.INT}

    def encode(self, ctx, job):
        v = job.values
        first = ctx.ev.constant((0.0, 0.0, 0.0, 0.0))
        if v["compute_bounces"]:
            first = ctx.scratch(job, False)
            spare = ctx.scratch(job, False)
            ctx.dispatch(PASS, job, {**v, "bounce": 0}, srcs={"previous": (ctx.ev.constant((0, 0, 0, 0)), False)},
                         dsts={"output": spare, "raw_irradiance": first})
        ctx.dispatch(PASS, job, {**v, "bounce": int(v["compute_bounces"])}, srcs={"previous": (first, False)})


class _Pass(Effect):
    type = "fx_rt_irradiance_pass"
    registered = False
    libs = ("_rays.wgsl",)
    inputs = {**RTIrradiance.inputs, "previous": Input(PortType.COLOR)}
    outputs = RTIrradiance.outputs
    params = RTIrradiance.params
    internal_uniforms = RTIrradiance.internal_uniforms


PASS = _Pass()
