from .common import B, E, F, F2, I, Input, Param, PatternNode, PortType


class ShapeExtrude(PatternNode):
    """Extrudes a 2D shape (built-in Cube / Cylinder, or the binary Extrude Shape Input) along its
    axis, Extrude Height up and Extrude Depth down from the centre (1 = half the frame x Scale), and renders
    the volume as a heightmap seen by an orthographic camera placed by Position (the position
    widget: the centre is the top view, the offset from it tilts the camera towards that direction, by a
    quarter turn at the border).
    The cross-section shrinks towards the top (and the bottom with Mirror Bevel) over Bevel Height
    (fraction of half the length) by Bevel Intensity, along a straight (0), convex (>0) or concave
    (<0) Bevel Curve. Profile Type: Straight; Vertical Gradient = the cross-section is scaled by the
    Profile Gradient read along its u axis (bottom to top: revolution bodies); Mask = sections where
    the Profile Mask, read along u, is below 0.5 are hidden. The volume is ray marched (bounded
    steps + bisection); Downscale Multiplier renders at 1/2^n of the resolution (one ray per 2^n x 2^n
    block: computed smaller for speed). Output =
    height of the visible surface: [0, 1] maps -1..1 to 0..1, [-1, 1] keeps it signed; the
    background is 0 in both."""

    type = "shape_extrude"
    label = "Shape Extrude"
    catalog = "shape_extrude"
    unsupported = {"Downscale Mulitplier": "the documentation's typo of Downscale Multiplier"}
    seamless = False
    libs = ("raycast_lib.wgsl",)
    inputs = {"extrude_shape_input": Input(PortType.GRAY, "Extrude Shape Input"),
              "profile_gradient": Input(PortType.GRAY, "Profile Gradient", default=(1.0, 1.0, 1.0, 1.0)),
              "profile_mask": Input(PortType.GRAY, "Profile Mask", default=(1.0, 1.0, 1.0, 1.0))}
    main_input = "extrude_shape_input"
    params = {
        "extrude_height": Param(F, 1.0, "Extrude Height", 0.0, 1.0),
        "extrude_depth": Param(F, 1.0, "Extrude Depth", 0.0, 1.0),
        "extrude_shape": Param(E, "Cylinder", "Extrude Shape", choices=("Cube", "Cylinder", "Custom Input")),
        "extrude_shape_size": Param(F2, [1.0, 1.0], "Extrude Shape Size", 0.0, 1.0),
        "scale": Param(F, 0.75, "Scale", 0.0, 1.0),
        "profile_type": Param(E, "Straight", "Profile Type", choices=("Straight", "Vertical Gradient", "Mask")),
        "bevel_height": Param(F, 0.1, "Bevel Height", 0.0, 1.0),
        "bevel_intensity": Param(F, 0.25, "Bevel Intensity", 0.0, 1.0),
        "bevel_curve": Param(F, 0.0, "Bevel Curve", -1.0, 1.0),
        "mirror_bevel": Param(B, False, "Mirror Bevel"),
        "downscale_multiplier": Param(I, 0, "Downscale Multiplier", 0, 2),
        "position": Param(F2, [0.75, 0.25], "Position", 0.0, 1.0),
        "output_range": Param(E, "[0, 1]", "Output Range", choices=("[0, 1]", "[-1, 1]")),
    }
