from trafilatura3d.nodes.base import Input, Output, ParamKind
from trafilatura3d.nodes.filters.basic._lib import gaussian
from trafilatura3d.nodes.materials._base import (ANY, COLOR, GRAY, NORMAL_FORMATS, B, C, E, F, F2, F3, I, MatNode, Pass,
                                           color_in, gray_in, passes_of, px256, with_tiling)
from trafilatura3d.nodes.materials.pbr._pbr import CATEGORY

ENV_W, ENV_H = 256, 128
ENV_SIGMAS = (0.0, 1.5, 4.0, 9.0, 18.0, 40.0)  # prefiltered levels, texels at 256 x 128


def _channel(label, default):
    return Input(ANY, label, default)


class PbrRender(MatNode):
    """Renders a PBR material on a sphere, plane or cylinder lit by an HDR environment (image based
    lighting), as a texture. The environment is prefiltered into six blur levels: diffuse light
    integrates Diffuse Quality cosine-weighted samples of a blurred level, specular reflections read
    the level matching the roughness with an analytic environment BRDF (split sum); displacement
    moves the surface along its normal (iterated ray hits), normals can be recomputed from height.
    Clear coat, emissive (optionally lighting the diffuse), approximate subsurface scattering
    (translucency, wrap and color shift), anisotropy (bent reflection normal), dithered or blended
    opacity, self shadowing from the height. Then depth of field (polygonal bokeh, ring, diffraction,
    swirl, custom aperture), bloom, halo, flares and lens dirt, exposure, tone mapping and vignette.
    Outputs the beauty (display-encoded) and the raw irradiance, raw specular, world and tangent
    normals and UVs (blue = coverage, for PBR Render Mapping)."""

    type = "pbr_render"
    label = "PBR Render"
    category = CATEGORY
    catalog = "pbr_render"
    unsupported = {"Aperture Difraction": "the documentation's typo of Aperture Diffraction",
                   "Diffuse Emissive Mulitplier": "the documentation's typo of Diffuse Emissive Multiplier",
                   "Specular Shadow Multiplier": "the documentation's old name of Specular Shadow Intensity"}
    inputs = {
        "base_color": _channel("Base Color", (0.5, 0.5, 0.5, 1.0)),
        "normal": _channel("Normal", (0.5, 0.5, 1.0, 1.0)),
        "emissive": _channel("Emissive", (0.0, 0.0, 0.0, 1.0)),
        "roughness": _channel("Roughness", (0.4, 0.4, 0.4, 1.0)),
        "metallic": _channel("Metallic", (0.0, 0.0, 0.0, 1.0)),
        "specular_level": _channel("Specular Level", (0.5, 0.5, 0.5, 1.0)),
        "height": _channel("Height", (0.5, 0.5, 0.5, 1.0)),
        "ambient_occlusion": _channel("Ambient Occlusion", (1.0, 1.0, 1.0, 1.0)),
        "opacity_mask": _channel("Opacity Mask", (1.0, 1.0, 1.0, 1.0)),
        "anisotropy_level": _channel("Anisotropy Level", (0.0, 0.0, 0.0, 1.0)),
        "anisotropy_angle": _channel("Anisotropy Angle", (0.0, 0.0, 0.0, 1.0)),
        "translucency": _channel("Translucency", (0.0, 0.0, 0.0, 1.0)),
        "scattering_distance_scale": _channel("Scattering Distance Scale", (1.0, 0.5, 0.25, 1.0)),
        "lens_dirt_map": gray_in("Lens Dirt Map", 0.5),
        "lens_aperture_map": gray_in("Lens Aperture Map", 1.0),
        "background_input": color_in("Background input", (0.0, 0.0, 0.0, 1.0)),
        "environment_map": color_in("Environment Map", (0.6, 0.6, 0.6, 1.0)),
    }
    main_input = "base_color"
    outputs = {"beauty": Output(COLOR, "Beauty"), "raw_irradiance": Output(COLOR, "Raw Irradiance"),
               "raw_specular": Output(COLOR, "Raw Specular"), "normal_world_space": Output(COLOR, "Normal World Space"),
               "normal_tangent_space": Output(COLOR, "Normal Tangent Space"), "uvs": Output(COLOR, "UVs")}
    params = {
        "shape": E("Sphere", "Shape", ("Sphere", "Plane", "Cylinder")),
        "displacement_intensity": F(0.1, "Displacement Intensity", 0.0, 0.5),
        "environment_rotation": F(0.0, "Environment Rotation", 0.0, 1.0),
        "background_mode": E("Environment", "Background Mode", ("Color", "Environment", "Ambient", "Background Input")),
        "background_color": C([0.1, 0.1, 0.1], "Background Color"),
        "environment_background_blur": F(0.0, "Environment Background Blur", 0.0, 1.0),
        "scale": F(1.0, "Scale", 0.0, 2.0, group="Shape"),
        "plane_size": F2([1.0, 1.0], "Plane Size", 0.0, 1.0, group="Shape"),
        "cylinder_radius": F(0.5, "Cylinder Radius", 0.0, 1.0, group="Shape"),
        "cylinder_length": F(1.0, "Cylinder Length", 0.0, 1.0, group="Shape"),
        "rotation": F(0.0, "Rotation", 0.0, 1.0, group="Shape"),
        "rotation_direction": F(0.0, "Rotation Direction", 0.0, 1.0, group="Shape"),
        "rotation_around_direction": F(0.0, "Rotation Around Direction", 0.0, 1.0, group="Shape"),
        "shape_position": F3([0.0, 0.0, 0.0], "Shape Position", -1.0, 1.0, group="Shape"),
        "uv_tiling": F(1.0, "UV Tiling", 1.0, 6.0, group="Shape"),
        "sphere_uv_projection": E("Spherical", "Sphere UV Projection", ("Spherical", "Cubic"), group="Shape"),
        "sphere_uv_scale": F2([4.0, 2.0], "Sphere UV Scale", 0.0, 4.0, group="Shape"),
        "plane_uv_scale": F2([1.0, 1.0], "Plane UV Scale", 1.0, 4.0, group="Shape"),
        "cylinder_uv_scale": F2([3.0, 1.0], "Cylinder UV Scale", 1.0, 6.0, group="Shape"),
        "uv_offset": F2([0.0, 0.0], "UV Offset", 0.0, 1.0, group="Shape"),
        "tilt_uvs": B(False, "Tilt UVs", group="Shape"),
        "exposure": F(0.0, "Exposure", -4.0, 4.0, group="Camera"),
        "tone_mapper": E("Linear", "Tone Mapper", ("Linear", "ACES", "Filmic Hejl"), group="Camera"),
        "camera_mode": E("Perspective", "Camera Mode", ("Perspective", "Orthographic"), group="Camera"),
        "field_of_view": F(30.0, "Field of View", 0.01, 100.0, group="Camera"),
        "distance": F(2.0, "Distance", 0.0, 4.0, group="Camera"),
        "vignette_intensity": F(0.5, "Vignette Intensity", 0.0, 1.0, group="Camera"),
        "vignette_radius": F(1.0, "Vignette Radius", 0.0, 1.0, group="Camera"),
        "screen_position": F2([0.5, 0.5], "Screen Position", 0.0, 1.0, group="Camera"),
        "aperture_radius": F(0.0, "Aperture Radius", 0.0, 0.1, group="Depth of Field"),
        "aperture_blades": I(5, "Aperture Blades", 3, 9, group="Depth of Field"),
        "aperture_ring": F(1.0, "Aperture Ring", 0.0, 1.0, group="Depth of Field"),
        "aperture_diffraction": F(1.0, "Aperture Diffraction", 0.0, 2.0, group="Depth of Field"),
        "swirly_bokeh": F(0.0, "Swirly Bokeh", 0.0, 1.0, group="Depth of Field"),
        "focus_mode": E("Auto", "Focus Mode", ("Auto", "Point"), group="Depth of Field"),
        "focus_point": F2([0.6, 0.4], "Focus Point", group="Depth of Field"),
        "focus_offset": F(0.0, "Focus Offset", -0.5, 0.5, group="Depth of Field"),
        "use_custom_aperture_map": B(False, "Use Custom Aperture Map", group="Depth of Field"),
        "enable_post_effects": B(True, "Enable Post Effects", group="Post Effects"),
        "bloom_intensity": F(0.25, "Bloom Intensity", 0.0, 2.0, group="Post Effects"),
        "bloom_threshold": F(1.0, "Bloom Threshold", 0.0, 2.0, group="Post Effects"),
        "bloom_chroma_shift": F(0.1, "Bloom Chroma Shift", 0.0, 1.0, group="Post Effects"),
        "lens_halo_intensity": F(0.5, "Lens Halo Intensity", 0.0, 1.0, group="Post Effects"),
        "lens_flares_intensity": F(0.25, "Lens Flares Intensity", 0.0, 1.0, group="Post Effects"),
        "lens_dirt_intensity": F(0.0, "Lens Dirt Intensity", 0.0, 1.0, group="Post Effects"),
        "diffuse_quality": E("32 Samples", "Diffuse Quality", ("16 Samples", "32 Samples", "64 Samples", "128 Samples"),
                             group="Render Settings"),
        "diffuse_emissive_multiplier": F(1.0, "Diffuse Emissive Multiplier", 0.0, 1.0, group="Render Settings"),
        "diffuse_shadow_intensity": F(1.0, "Diffuse Shadow Intensity", 0.0, 1.0, group="Render Settings"),
        "specular_dithering": F(0.5, "Specular Dithering", 0.0, 1.0, group="Render Settings"),
        "specular_shadow_intensity": F(1.0, "Specular Shadow Intensity", 0.0, 1.0, group="Render Settings"),
        "opacity_mode": E("Dithered Alpha test", "Opacity Mode", ("Dithered Alpha test", "Simple Alpha Blend"),
                          group="Render Settings"),
        "ambient_occlusion_intensity": F(0.0, "Ambient Occlusion Intensity", 0.0, 1.0, group="Render Settings"),
        "recompute_normals": B(False, "Recompute Normals", group="Material Adjustements"),
        "normal_format": E("OpenGL", "Normal Format", NORMAL_FORMATS, group="Material Adjustements"),
        "dielectric_f0_input": E("Constant Value", "Dielectric F0 Input", ("Constant Value", "Specular Level input"),
                                 group="Material Adjustements"),
        "dielectric_f0": F(0.04, "Dielectric F0", 0.0, 0.08, group="Material Adjustements"),
        "enable_clear_coat": B(False, "Enable Clear Coat", group="Clear Coat"),
        "clear_coat_weight": F(1.0, "Clear Coat Weight", 0.0, 1.0, group="Clear Coat"),
        "clear_coat_specular_level": F(0.5, "Clear Coat Specular Level", 0.0, 1.0, group="Clear Coat"),
        "inherit_normal_from_base_layer": B(False, "Inherit Normal from Base Layer", group="Clear Coat"),
        "enable_emissive_lighting": B(True, "Enable Emissive Lighting", group="Emissive"),
        "emissive_intensity": F(2.0, "Emissive Intensity", 0.0, 10.0, group="Emissive"),
        "enable_subsurface_scattering": B(False, "Enable Subsurface Scattering", group="Subsurface Scattering"),
        "scattering_distance": F(1.0, "Scattering Distance", 0.0, 1.0, group="Subsurface Scattering"),
        "red_shift": F(0.0, "Red Shift", 0.0, 1.0, group="Subsurface Scattering"),
        "rayleigh": F(0.0, "Rayleigh", 0.0, 1.0, group="Subsurface Scattering"),
    }

    def encode(self, ctx, job):
        ins = self.inputs
        env, put, render, dof, bright, post = passes_of(self, lambda n: (
            Pass(n, "env", "pbr_render_env.wgsl", {"environment_map": ins["environment_map"]}, {"lvl": Output(COLOR)}),
            Pass(n, "put", "pbr_render_put.wgsl", {"src": Input(COLOR)}, {"atlas": Output(COLOR)},
                 internal={"row": ParamKind.INT}),
            Pass(n, "render", "pbr_render.wgsl",
                 {**{k: v for k, v in ins.items() if k not in ("lens_dirt_map", "lens_aperture_map", "environment_map")},
                  "env": Input(COLOR)},
                 {"hdr": Output(COLOR), **{k: v for k, v in n.outputs.items() if k != "beauty"}}),
            Pass(n, "dof", "pbr_render_dof.wgsl", {"hdr": Input(COLOR), "lens_aperture_map": ins["lens_aperture_map"]},
                 {"out": Output(COLOR)}),
            Pass(n, "bright", "pbr_render_bright.wgsl", {"hdr": Input(COLOR)}, {"out": Output(COLOR)}),
            Pass(n, "post", "pbr_render_post.wgsl",
                 {"img": Input(COLOR), "bloom": Input(COLOR), "bloom2": Input(COLOR), "lens_dirt_map": ins["lens_dirt_map"]},
                 {"beauty": n.outputs["beauty"]})))
        v = job.values
        lvl0 = ctx.scratch(job, False, size=(ENV_W, ENV_H))
        ctx.dispatch(env, job, v, dsts={"lvl": lvl0}, size=(ENV_W, ENV_H))
        atlas = ctx.scratch(job, False, size=(ENV_W, ENV_H * len(ENV_SIGMAS)))
        wrap_h = with_tiling(job, 1)  # a panorama wraps horizontally only
        for row, sigma in enumerate(ENV_SIGMAS):
            t = gaussian(ctx, wrap_h, lvl0, False, sigma) if sigma else lvl0
            ctx.dispatch(put, job, {**v, "row": row}, srcs={"src": (t, False)}, dsts={"atlas": atlas},
                         size=(ENV_W, ENV_H))
        hdr = ctx.scratch(job, False)
        ctx.dispatch(render, job, v, srcs={"env": (atlas, False)}, dsts={"hdr": hdr})
        img = hdr
        if v["aperture_radius"] > 0:
            img = ctx.scratch(job, False)
            ctx.dispatch(dof, job, v, srcs={"hdr": (hdr, False)}, dsts={"out": img})
        black = ctx.ev.constant((0.0, 0.0, 0.0, 1.0))
        b1 = b2 = black
        if v["enable_post_effects"]:
            br = ctx.scratch(job, False)
            ctx.dispatch(bright, job, v, srcs={"hdr": (img, False)}, dsts={"out": br})
            b1 = gaussian(ctx, job, br, False, 3.0 * px256(job))
            b2 = gaussian(ctx, job, br, False, 16.0 * px256(job))
        ctx.dispatch(post, job, v, srcs={"img": (img, False), "bloom": (b1, False), "bloom2": (b2, False)})
