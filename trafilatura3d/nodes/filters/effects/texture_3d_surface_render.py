from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect

UNSUPPORTED_RES = {"Output Resolution": "the node's Output Size base parameter sets the render's resolution, "
                                        "as for every Trafilatura 3D node; a second size control would conflict"}


class Texture3DSurfaceRender(Effect):
    """Renders the surface of a shape from its 3D distance field (16x16 atlas of 256 slices, unit-cube
    units, see 3D Texture SDF) by sphere tracing inside the unit cube, seen by the node's camera (Camera
    Position: X = azimuth over a full turn, Y = elevation from below (0) through level (0.5) to above
    (1); Distance in cube widths; FOV in degrees, the screen spanning -0.5..0.5 x tan(FOV / 2)), world
    Z up = the slice axis. Image-based lighting from the Environment (latitude-longitude, Z up):
    16 cosine-distributed directions per pixel, each shadowed by marching the distance field, on the
    Albedo surface and on the optional ground plane under the cube (infinite or Ground Plane Size).
    Background: Ground Irradiance (what open ground receives), Ambient (the environment blurred by a
    cosine lobe), Uniform Color, or the Environment itself."""

    type = "3d_texture_surface_render"
    label = "3D Texture Surface Render"
    category = "Filters/Effects"
    catalog = "3d_texture_surface_render"
    libs = ("_volume.wgsl", "_volume_sample.wgsl")
    seamless = False
    inputs = {"distance_field_3d": Input(PortType.GRAY, "3D Distance Field", default=(1.0, 1.0, 1.0, 1.0)),
              "environment": Input(PortType.COLOR, "Environment", default=(1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "camera_position": Param(ParamKind.FLOAT2, [0.5, 0.5], "Camera Position", 0.0, 1.0),
        "camera_distance": Param(ParamKind.FLOAT, 4.0, "Camera Distance", 0.0, 8.0),
        "camera_fov": Param(ParamKind.FLOAT, 50.0, "Camera FOV", 0.0, 200.0),
        "albedo": Param(ParamKind.FLOAT3, [1.0, 1.0, 1.0], "Albedo", 0.0, 1.0),
        "background_mode": Param(ParamKind.ENUM, "Ground Irradiance", "Background Mode",
                                 choices=("Ground Irradiance", "Ambient", "Uniform Color", "Environment")),
        "background_color": Param(ParamKind.FLOAT4, [0.5, 0.5, 0.5, 1.0], "Background Color", 0.0, 1.0),
        "enable_ground_plane": Param(ParamKind.BOOL, True, "Enable Ground Plane"),
        "infinite_plane": Param(ParamKind.BOOL, True, "Infinite Plane"),
        "ground_plane_size": Param(ParamKind.FLOAT2, [4.0, 4.0], "Ground Plane Size", 0.0, 8.0),
    }
    unsupported = UNSUPPORTED_RES
