from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode


class Skew(BasicNode):
    """Shear: Horizontal shifts each row by Amount x its distance from the Align line (centre, top
    or bottom); Vertical shifts each column likewise (centre, left, right). Off-image areas wrap on
    tiled axes, else are transparent black."""

    type = "skew"
    label = "Skew"
    category = "Filters/Transforms"
    catalog = "skew"
    inputs = {"input": Input(label="Input")}
    params = {
        "axis": Param(ParamKind.ENUM, "Horizontal", "Axis", choices=("Horizontal", "Vertical")),
        "amount": Param(ParamKind.FLOAT, 0.0, "Amount", -1.0, 1.0),
        "align": Param(ParamKind.ENUM, "Top Left", "Align", choices=("Center", "Top Left", "Bottom Right")),
    }
