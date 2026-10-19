from .common import B, E, F, F2, Input, Output, Param, PatternNode, PortType

PARAMS = {
    # The position widget: x = horizontal angle (0.5 = none, +-0.5 = half a turn, on the input), y = the
    # view (1 = top, 0.5 / 1.5 = grazing from either side, Ground lets it go below), modulo 2.
    "camera_angle": Param(F2, [0.5, 0.75], "Camera Angle", 0.0, 1.0),
    "camera_scale": Param(F, 0.707, "Camera Scale", 0.001, 3.0),
    "height_scale": Param(F, 1.0, "Height Scale", 0.0, 2.0),
    "vertical_offset": Param(F, 0.0, "Vertical Offset", -1.0, 1.0),
    "ground": Param(B, False, "Ground"),
    "normal_format": Param(E, "OpenGL", "Normal Format", choices=("DirectX", "OpenGL")),
    "normal_intensity": Param(F, 10.0, "Normal Intensity", 0.0, 256.0),
}


class HeightExtrude(PatternNode):
    """Renders the input heightmap (a full-range grayscale map over the unit square) as a 3D
    heightfield seen by an orthographic camera: horizontal angle and Camera Scale act on the input,
    the vertical angle tilts the camera towards the viewer (seen from the other side: the input
    mirrored vertically and Vertical Offset negated). The
    heightfield is ray marched (bounded steps + bisection, independent of the resolution). "depth"
    = distance towards the camera, equal to height x Height Scale in the top view; Ground off =
    black wherever the visible surface is at input 0. "normal" is the depth's normal map (Normal
    Intensity as in the Normal node). Two passes: depth into a 32-bit scratch, then both outputs."""

    type = "height_extrude"
    label = "Height Extrude"
    catalog = "height_extrude"
    seamless = False
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"depth": Output(PortType.GRAY, "Depth"), "normal": Output(PortType.COLOR, "Normal")}
    params = PARAMS
    libs = ("raycast_lib.wgsl",)

    def encode(self, ctx, job):
        depth = ctx.scratch(job, True, "32f")
        ctx.dispatch(DEPTH, job, job.values, dsts={"depth": depth})
        ctx.dispatch(FINAL, job, job.values, srcs={"depth_map": (depth, True)})


class _Depth(PatternNode):
    type = "height_extrude_depth"
    registered = False
    inputs = {"input": Input(PortType.GRAY)}
    outputs = {"depth": Output(PortType.GRAY)}
    params = PARAMS
    libs = ("raycast_lib.wgsl",)
    shader = "height_extrude_depth.wgsl"


class _Final(PatternNode):
    type = "height_extrude_final"
    registered = False
    inputs = {"depth_map": Input(PortType.GRAY)}
    outputs = HeightExtrude.outputs
    params = PARAMS
    shader = "height_extrude_final.wgsl"


DEPTH, FINAL = _Depth(), _Final()
