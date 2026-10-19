from trafilatura3d.nodes.base import Input, Output, PortType

from ._ao_params import RT_PARAMS
from ._fx import Effect


class AmbientOcclusionRTAO(Effect):
    """Ray traced ambient occlusion of a height map: Samples rays per pixel in a
    cone around the surface's normal (Spread Angle 1 = the whole hemisphere), Cosine-distributed or
    Uniform, marched through the height field up to Maximum Distance (UV); the result is the share of
    rays that escape. Height: Height Scale (white = that fraction of the width) or Physical Size
    (Z / min(X, Y))."""

    type = "ambient_occlusion_rtao"
    label = "Ambient Occlusion (RTAO)"
    category = "Filters/Effects"
    catalog = "ambient_occlusion_rtao"
    libs = ("_rays.wgsl", "_rt_common.wgsl")
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = RT_PARAMS
