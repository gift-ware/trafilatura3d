from trafilatura3d.nodes.base import Input, Param, ParamKind, PortType

from ._fx import Effect


class VectorMorph(Effect):
    """Flows the input along the Vector Field: 16 steps, each moving the read
    position by (red, green - 0.5) x Amount / 16 UV (image axes, green down) read where the previous
    step landed, so content follows the field's curves. Outside the canvas the edges are clamped and
    repeated instead of tiling."""

    type = "vector_morph"
    label = "Vector Morph"
    category = "Filters/Effects"
    catalog = "vector_morph"
    inputs = {"input": Input(label="Input"),
              "vector_field": Input(PortType.COLOR, "Vector Field", default=(0.5, 0.5, 0.5, 1.0))}
    params = {"amount": Param(ParamKind.FLOAT, 1.0, "Amount", 0.0, 1.0)}
