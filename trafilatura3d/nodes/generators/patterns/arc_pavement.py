from .common import E, F, FILTERING, I, Input, Output, Param, PatternNode, PortType, nse, pattern_choices


class ArcPavement(PatternNode):
    """Parisian arc ("fan") pavement: fans of concentric arcs in staggered rows (Scale fans across,
    two rows per fan width, the lower rows overlapping the upper ones so each fan shows its top).
    Each fan has Arcs Amount concentric arcs (the brick height); the outer arc has Pattern Amount
    bricks and inner arcs proportionally fewer, randomised per arc down to Pattern Minimum Amount
    (Pattern Amount Random; this also varies brick sizes). Each brick draws a shape or the Input
    Image, scaled (Pattern Scale, Width, Height + random), with Global Pattern Width Random moving
    the joints instead of shrinking bricks, Pattern Height Decrease squashing the bricks towards the
    ends of each arc, and Color Random per brick. Pattern Variation = the shape's Pattern Specific."""

    type = "arc_pavement"
    label = "Arc Pavement"
    catalog = "arc_pavement"
    pattern_ports = ("input_image",)
    inputs = {"input_image": Input(PortType.ANY, "Input Image")}
    outputs = {"output": Output(PortType.ANY, "Output")}
    params = {
        "scale": Param(I, 2, "Scale", 1, 8),
        "pattern_amount": Param(I, 12, "Pattern Amount", 1, 32),
        "pattern_amount_random": Param(F, 0.0, "Pattern Amount Random", 0.0, 1.0),
        "pattern_minimum_amount": Param(I, 3, "Pattern Minimum Amount", 1, 10),
        "arcs_amount": Param(I, 6, "Arcs Amount", 0, 20),
        "pattern": Param(E, "Square", "Pattern", choices=pattern_choices("Input Image", gradation="Gradations")),
        "input_image_filtering": Param(E, FILTERING[0], "Input Image Filtering", choices=FILTERING),
        "pattern_variation": Param(F, 0.0, "Pattern Variation", 0.0, 1.0),
        "pattern_scale": Param(F, 0.9, "Pattern Scale", 0.0, 1.0),
        "pattern_width": Param(F, 1.0, "Pattern Width", 0.0, 1.0),
        "pattern_height": Param(F, 1.0, "Pattern Height", 0.0, 1.0),
        "pattern_width_random": Param(F, 0.0, "Pattern Width Random", 0.0, 1.0),
        "pattern_height_random": Param(F, 0.0, "Pattern Height Random", 0.0, 1.0),
        "global_pattern_width_random": Param(F, 0.0, "Global Pattern Width Random", 0.0, 1.0),
        "pattern_height_decrease": Param(F, 0.0, "Pattern Height Decrease", 0.0, 1.0),
        "color_random": Param(F, 0.0, "Color Random", 0.0, 1.0),
        **nse(),
    }
