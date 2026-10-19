from trafilatura3d.nodes.base import Input, Output, ParamKind, PortType
from trafilatura3d.nodes.materials._base import material_inputs, material_outputs
from trafilatura3d.nodes.paint._lib import F, mesh_inputs
from trafilatura3d.nodes.paint.matfinish._lighting import INCLUDES, LIGHTING_DOC, MESH, Lighting, helper


class BakedLightingEnvironment(Lighting):
    """The material lit by an environment map (lat-long panorama, linear values, +Y up), baked into the base
    color: diffuse irradiance and the reflection prefiltered for the roughness, with the view along the normal;
    ambient occlusion darkens the diffuse (Ambient Occlusion Intensity) and the reflection (Specular Occlusion
    Intensity, Lagarde's specular occlusion). The effect has no environment of its own: the Environment
    Map input is a resource slot (unconnected: a uniform white environment)."""
    type = "paint_baked_lighting_environment"
    label = "Baked Lighting Environment"
    catalog = "painter.filter.baked_lighting_environment"
    doc = LIGHTING_DOC + "baked-lighting-environment"
    inputs = {**material_inputs(("basecolor", "roughness", "metallic", "normal", "height")), **mesh_inputs(*MESH),
              "environment": Input(PortType.COLOR, "Environment Map", (1.0, 1.0, 1.0, 1.0))}
    outputs = material_outputs(("basecolor",))
    params = {"horizontal_rotation": F("Horizontal Rotation", 0.0, 0.0, 360.0),
              "vertical_rotation": F("Vertical Rotation", 0.0, -90.0, 90.0),
              "exposure": F("Exposure", 0.0, -5.0, 5.0),
              "height_intensity": F("Height Intensity", 0.5, 0.0, 1.0),
              "ambient_occlusion_intensity": F("Ambient Occlusion Intensity", 1.0, 0.0, 1.0),
              "specular_occlusion_intensity": F("Specular Occlusion Intensity", 1.0, 0.0, 1.0)}

    def encode(self, ctx, job):
        small = ctx.scratch(job, False, size=(64, 32))
        irr = ctx.scratch(job, False, size=(32, 16))
        pre = ctx.scratch(job, False, size=(64, 128))
        ctx.dispatch(MAPS, job, {"pass_": 0}, dsts={"output": small}, size=(64, 32))
        ctx.dispatch(MAPS, job, {"pass_": 1}, srcs={"environment": (small, False)}, dsts={"output": irr}, size=(32, 16))
        ctx.dispatch(MAPS, job, {"pass_": 2}, srcs={"environment": (small, False)}, dsts={"output": pre},
                     size=(64, 128))
        ctx.dispatch(FINAL, job, job.values, srcs={"irradiance": (irr, False), "prefiltered": (pre, False)})


MAPS = helper("paint_baked_lighting_environment_maps", "env_maps.wgsl", ("matfinish/panorama",),
              {"environment": BakedLightingEnvironment.inputs["environment"]}, {"output": Output(PortType.COLOR)},
              internal={"pass_": ParamKind.INT})
FINAL = helper("paint_baked_lighting_environment_final", "baked_lighting_environment.wgsl", INCLUDES,
               {**BakedLightingEnvironment.inputs, "irradiance": Input(PortType.COLOR),
                "prefiltered": Input(PortType.COLOR)},
               BakedLightingEnvironment.outputs, BakedLightingEnvironment.params, main=True)
