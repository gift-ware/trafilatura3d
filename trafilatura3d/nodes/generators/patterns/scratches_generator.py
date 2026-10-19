from .common import B, E, F, I, Output, Param, PatternNode, PortType, nse


class ScratchesGenerator(PatternNode):
    """Spline Number random scratches, wrapped so the result tiles. Each spline: centre (uniform at
    Position Random 1, gathered towards the image centre below), direction (Rotation + random),
    length (Scale x random, 1 = the image width), a centreline displaced sideways by smooth 1D noise
    (Distortion amount/random, Frequency 1-21 waves per spline) sampled at Max Segments Per Spline
    points, width (relative: 1 = 1.5 px at 2048 px; or pixels with Set Spline Width in px), luminance
    random, fade towards its start/end over Fade Length. Overlaps keep the brightest. The second
    output "normal" is a Scratches Generator Normal folded in: the normal map of the
    scratches as grooves (Normal Intensity + random, DirectX/OpenGL)."""

    type = "scratches_generator"
    label = "Scratches Generator"
    catalog = "scratches_generator"
    outputs = {"output": Output(PortType.GRAY, "Output"), "normal": Output(PortType.COLOR, "Normal")}
    params = {
        "spline_number": Param(I, 64, "Spline Number", 1, 512),
        "max_segments_per_spline": Param(I, 32, "Max Segments Per Spline", 2, 256),
        "spline_rotation": Param(F, 0.0, "Spline Rotation", 0.0, 1.0),
        "spline_rotation_random": Param(F, 0.5, "Spline Rotation Random", 0.0, 1.0),
        "spline_scale": Param(F, 0.3, "Spline Scale", 0.0, 1.0),
        "spline_scale_random": Param(F, 0.5, "Spline Scale Random", 0.0, 1.0),
        "spline_distortion": Param(F, 0.2, "Spline Distortion", 0.0, 1.0),
        "spline_distortion_random": Param(F, 0.5, "Spline Distortion Random", 0.0, 1.0),
        "spline_distortion_frequency": Param(F, 0.2, "Spline Distortion Frequency", 0.0, 1.0),
        "spline_width": Param(F, 1.0, "Spline Width", 0.0, 2.0),
        "spline_width_random": Param(F, 0.5, "Spline Width Random", 0.0, 1.0),
        "spline_position_random": Param(F, 1.0, "Spline Position Random", 0.0, 1.0),
        "set_spline_width_in_px": Param(B, False, "Set Spline Width in px"),
        "luminance_random": Param(F, 0.5, "Luminance Random", 0.0, 1.0),
        "normal_intensity": Param(F, 0.5, "Normal Intensity", 0.0, 1.0),
        "normal_intensity_random": Param(F, 0.5, "Normal Intensity Random", 0.0, 1.0),
        "normal_format": Param(E, "OpenGL", "Normal Format", choices=("DirectX", "OpenGL")),
        "fade_mode": Param(E, "Start + End", "Fade Mode", choices=("None", "Start", "End", "Start + End")),
        "fade_length": Param(F, 0.2, "Fade Length", 0.0, 1.0),
        **nse(),
    }
