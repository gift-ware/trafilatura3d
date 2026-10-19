from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, I, MatNode
from trafilatura3d.nodes.materials.scan._common import CATEGORY

N = 8


class MultiAngleToAlbedo(MatNode):
    """Lighting-neutral albedo from photos lit from different angles: per pixel, the samples are
    first brought to the same overall exposure, then ranked by brightness and the extremes (cast
    shadows, highlights) dropped: a trimmed mean, the median for two or three samples."""

    type = "multi_angle_to_albedo"
    label = "Multi-Angle to Albedo"
    category = CATEGORY
    catalog = "multi_angle_to_albedo"
    channels = tuple(f"input_{i}" for i in range(1, N + 1))
    inputs = {f"input_{i}": Input(COLOR, f"Input {i}", (0.0, 0.0, 0.0, 1.0)) for i in range(1, N + 1)}
    outputs = {"output": Output(COLOR)}
    params = {"samples_amount": I(4, "Samples Amount", 2, 8)}
