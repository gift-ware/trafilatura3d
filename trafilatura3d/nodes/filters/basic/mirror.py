from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode

CORNERS = ("Top Left", "Top Right", "Bottom Left", "Bottom Right")


class Mirror(BasicNode):
    """Reflect one side onto the other. Axis X: about the vertical line x = Axis X Offset, the left
    side is the source (Invert Axis X: the right). Axis Y: about y = Axis Y Offset, the top is the
    source (Invert: the bottom). Corner: the chosen quadrant mirrored to all four."""

    type = "mirror"
    label = "Mirror (Filter Node)"
    category = "Filters/Transforms"
    catalog = "mirror"
    inputs = {"input": Input(label="Input")}
    params = {
        "mode": Param(ParamKind.ENUM, "Mirror Axis X", "Mode", choices=("Mirror Axis X", "Mirror Axis Y", "Mirror Corner")),
        "axis_x_offset": Param(ParamKind.FLOAT, 0.5, "Axis X Offset", 0.0, 1.0),
        "axis_y_offset": Param(ParamKind.FLOAT, 0.5, "Axis Y Offset", 0.0, 1.0),
        "invert_axis_x": Param(ParamKind.BOOL, False, "Invert Axis X"),
        "invert_axis_y": Param(ParamKind.BOOL, False, "Invert Axis Y"),
        "corner_type": Param(ParamKind.ENUM, "Top Left", "Corner Type", choices=CORNERS),
    }
