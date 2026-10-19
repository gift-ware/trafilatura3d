from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect
from .texture_3d_surface_render import UNSUPPORTED_RES


class Texture3DVolumeRender(Effect):
    """Renders a volume from 3D textures (16x16 atlas of 256 slices): its signed distance field bounds
    it, its Density (0 transparent, 1 opaque) fills it, faded in from the boundary over Feather
    (density x clamp(-distance / Feather)). Ray marched inside the unit cube (96 steps, start jittered
    by Dithering) with Absorption (per 1/256 of the cube), lit by a directional light
    (from the side opposite the direction Light Position gives, read like the camera's: switched off
    under the ground plane; color from Temperature in Kelvin or RGB, x Light Intensity, shadowed by
    marching towards it) and a hemispherical skylight (Ambient Color x Intensity, attenuated by the
    density above), scattering with Albedo. Camera as in 3D Texture Surface Render; world Z up = the
    slice axis. Background Color is lit by the light (Shaded) or flat (Constant Color); the optional
    ground plane under the cube receives the volume's shadow."""

    type = "3d_texture_volume_render"
    label = "3D Texture Volume Render"
    category = "Filters/Effects"
    catalog = "3d_texture_volume_render"
    libs = ("_volume.wgsl", "_volume_sample.wgsl")
    seamless = False
    inputs = {"sdf_3d": Input(PortType.GRAY, "3D Signed Distance Field", default=(-1.0, -1.0, -1.0, 1.0)),
              "density": Input(PortType.GRAY, "Density", default=(1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "camera_position": Param(ParamKind.FLOAT2, [0.5, 0.5], "Camera Position", 0.0, 1.0),
        "light_position": Param(ParamKind.FLOAT2, [0.9, 0.1], "Light Position", 0.0, 1.0),
        "camera_distance": Param(ParamKind.FLOAT, 4.0, "Camera Distance", 0.0, 8.0),
        "camera_fov": Param(ParamKind.FLOAT, 50.0, "Camera FOV", 0.0, 200.0),
        "absorption": Param(ParamKind.FLOAT, 0.1, "Absorption", 0.0, 1.0),
        "feather": Param(ParamKind.FLOAT, 0.02, "Feather", 0.0, 0.5),
        "light_color_mode": Param(ParamKind.ENUM, "Temperature (Kelvin)", "Light Color Mode",
                                  choices=("Temperature (Kelvin)", "RGB Color")),
        "light_temperature_kelvin": Param(ParamKind.FLOAT, 5500.0, "Light Temperature (Kelvin)", 3000.0, 9000.0),
        "light_color": Param(ParamKind.FLOAT3, [1.0, 0.95, 0.95], "Light Color", 0.0, 1.0),
        "light_intensity": Param(ParamKind.FLOAT, 1.0, "Light Intensity", 0.0, 10.0),
        "ambient_color": Param(ParamKind.FLOAT3, [0.472, 0.6032, 0.8], "Ambient Color", 0.0, 1.0),
        "ambient_intensity": Param(ParamKind.FLOAT, 1.0, "Ambient Intensity", 0.0, 10.0),
        "albedo": Param(ParamKind.FLOAT3, [1.0, 1.0, 1.0], "Albedo", 0.0, 1.0),
        "background_mode": Param(ParamKind.ENUM, "Shaded", "Background Mode", choices=("Shaded", "Constant Color")),
        "background_color": Param(ParamKind.FLOAT4, [0.2, 0.2, 0.2, 1.0], "Background Color", 0.0, 1.0),
        "dithering": Param(ParamKind.FLOAT, 0.5, "Dithering", 0.0, 1.0),
        "enable_ground_plane": Param(ParamKind.BOOL, True, "Enable Ground Plane"),
        "infinite_plane": Param(ParamKind.BOOL, True, "Infinite Plane"),
        "ground_plane_size": Param(ParamKind.FLOAT2, [4.0, 4.0], "Ground Plane Size", 0.0, 8.0),
    }
    unsupported = UNSUPPORTED_RES
