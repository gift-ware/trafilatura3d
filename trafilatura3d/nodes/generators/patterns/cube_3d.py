from .common import F, F2, F3, Output, Param, PatternNode, PortType, nse

CUBE_PARAMS = {
    # X and Y rotation of the cube, in turns (the dot in the 2D view): Y turns it about the
    # vertical axis, then X tilts it towards the viewer. Default: a three-quarter view.
    "orientation_offset": Param(F2, [0.0625, -0.125], "Orientation Offset", -0.5, 0.5),
    "size": Param(F3, [1.0, 1.0, 1.0], "Size", 0.0, 1.0),
    "scale": Param(F, 0.5, "Scale", 0.0, 1.0),
    **nse(),
}


class Cube3D(PatternNode):
    """A box (Size per axis x Scale, 1 = the frame width) rotated by Orientation Offset (Y then X,
    in turns) and ray cast analytically with an orthographic camera: the output is the screen depth,
    nearer = brighter (1/3 + depth in frame widths, 0 = background), so faces read as shading and edges
    stay exact at every bit depth. A single centred cube: not a tiling pattern."""

    type = "cube_3d"
    label = "Cube 3D"
    catalog = "cube_3d"
    libs = ("raycast_lib.wgsl", "cube_lib.wgsl")
    shader = "cube_3d.wgsl"
    seamless = False
    params = CUBE_PARAMS


class Cube3DGBuffers(Cube3D):
    """Cube 3D with G-buffers: "depth" (the Cube 3D output), "position" (object-space hit point,
    z-up, each axis remapped from the cube's extent to 0-1, white background) and "normal" (the surface
    normal in screen space as an OpenGL normal map, flat (0.5, 0.5, 1) background); a UV output
    is not done."""

    type = "cube_3d_gbuffers"
    label = "Cube 3D GBuffers"
    catalog = "cube_3d_gbuffers"
    outputs = {"depth": Output(PortType.GRAY, "Depth"), "position": Output(PortType.COLOR, "Position"),
               "normal": Output(PortType.COLOR, "Normal")}
    shader = "cube_3d_gbuffers.wgsl"
