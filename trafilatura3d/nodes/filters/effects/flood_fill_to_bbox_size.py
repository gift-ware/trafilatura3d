from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class FloodFillToBBoxSize(Effect):
    """Each shape's bounding box size as a fraction of the canvas (1 = as wide as the image): its
    width (X), height (Y) or the larger of the two; black outside the shapes."""

    type = "flood_fill_to_bbox_size"
    label = "Flood Fill to BBox Size"
    category = "Filters/Effects"
    catalog = "flood_fill_to_bbox_size"
    libs = ("_ff_read.wgsl",)
    inputs = {"flood_fill": Input(PortType.COLOR, "Flood Fill")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {"output": Param(ParamKind.ENUM, "max(X, Y)", "Output", choices=("max(X, Y)", "X", "Y"))}
