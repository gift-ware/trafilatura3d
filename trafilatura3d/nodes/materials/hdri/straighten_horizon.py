from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, B, F, F2
from trafilatura3d.nodes.materials.hdri._common import HdriNode


class StraightenHorizon(HdriNode):
    """Levels a tilted panorama: the two points mark the horizon, the great circle through them is
    rotated onto the equator; Horizontal Rotation then spins around the vertical axis (turns)."""

    type = "straighten_horizon"
    label = "Straighten Horizon"
    catalog = "straighten_horizon"
    inputs = {"input": Input(COLOR, "Input", (0.0, 0.0, 0.0, 1.0))}
    outputs = {"output": Output(COLOR)}
    params = {
        "enable_horizon_correction": B(False, "Enable Horizon Correction"),
        "horizon_point_1_position": F2([0.25, 0.5], "Horizon Point #1 Position"),
        "horizon_point_2_position": F2([0.75, 0.5], "Horizon Point #2 Position"),
        "horizontal_rotation": F(0.0, "Horizontal Rotation", 0.0, 1.0),
    }
