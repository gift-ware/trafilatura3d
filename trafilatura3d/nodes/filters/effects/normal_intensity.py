from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class NormalIntensity(Effect):
    """Increases or decreases the intensity of a normal map: the angle of each normal from straight up is
    scaled by 1 + Intensity below 0 (-1 flattens it) and moved towards a right angle by Intensity^(1.2 + Intensity)
    above 0 (eased, fitted on the output; 1 lays it flat on the surface), its direction kept; nearly flat normals stay flat, alpha kept."""

    type = "normal_intensity"
    label = "Normal Intensity"
    category = "Filters/Normal Map"
    inputs = {"input": Input(PortType.COLOR, "Input", default=(0.5, 0.5, 1.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR)}
    params = {"intensity": Param(ParamKind.FLOAT, 0.0, "Intensity", min=-1.0, max=1.0, clamp=True)}
