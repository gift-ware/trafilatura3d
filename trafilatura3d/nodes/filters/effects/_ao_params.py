"""Parameters shared by Ambient Occlusion (RTAO) and Bent Normal (defaults and ranges)."""

from trafilatura3d.nodes.base import Param, ParamKind

DISTRIBUTIONS = ("Cosine", "Uniform")

RT_PARAMS = {
    "use_physical_size": Param(ParamKind.BOOL, False, "Use Physical Size"),
    "physical_size": Param(ParamKind.FLOAT3, [100.0, 100.0, 10.0], "Physical Size", 0.0, 200.0),
    "samples": Param(ParamKind.INT, 128, "Samples", 1, 256),
    "height_scale": Param(ParamKind.FLOAT, 0.1, "Height Scale", 0.0, 1.0),
    "distribution": Param(ParamKind.ENUM, "Cosine", "Distribution", choices=DISTRIBUTIONS),
    "maximum_distance": Param(ParamKind.FLOAT, 0.5, "Maximum Distance", 0.0, 1.0),
    "spread_angle": Param(ParamKind.FLOAT, 1.0, "Spread Angle", 0.001, 1.0),
}
