from trafilatura3d.nodes.base import Input

from ._lib import BasicNode


class CartesianToPolar(BasicNode):
    """Wrap the image around the centre: the input's X becomes the angle (clockwise from the right,
    one full turn across the width) and its Y the radius (top row at the centre, bottom row at the
    edge midpoints), so horizontal lines become circles."""

    type = "cartesian_to_polar"
    label = "Cartesian To Polar"
    category = "Filters/Transforms"
    catalog = "cartesian_to_polar"
    inputs = {"input": Input(label="Input")}
    defines = {"TO_POLAR": "true"}


class PolarToCartesian(BasicNode):
    """The inverse of Cartesian To Polar: unwrap circles around the centre into rows (angle across
    X, radius down Y)."""

    type = "polar_to_cartesian"
    label = "Polar to Cartesian"
    category = "Filters/Transforms"
    catalog = "polar_to_cartesian"
    inputs = {"input": Input(label="Input")}
    defines = {"TO_POLAR": "false"}
