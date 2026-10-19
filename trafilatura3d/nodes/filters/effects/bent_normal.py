from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._ao_params import RT_PARAMS
from ._fx import Effect


class BentNormal(Effect):
    """Bent normal map of a height map: the rays of Ambient Occlusion (RTAO) (same parameters), and
    the normal is the average direction of those that escape, so it leans out of cavities towards the
    open sky; on flat ground it is the plain normal (0.5, 0.5, 1), and where no ray escapes too.
    Normal Format flips green for DirectX."""

    type = "bent_normal"
    label = "Bent Normal"
    category = "Filters/Normal Map"
    catalog = "bent_normal"
    libs = ("_rays.wgsl", "_rt_common.wgsl")
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {**RT_PARAMS,
              "normal_format": Param(ParamKind.ENUM, "OpenGL", "Normal Format", choices=("DirectX", "OpenGL"))}
