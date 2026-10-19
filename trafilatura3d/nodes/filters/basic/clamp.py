from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode


class Clamp(BasicNode):
    """Clamp every channel to [Min, Max] (Max of Min, then Min of Max, so Max wins when
    it is below Min); alpha only with Apply to Alpha (colour inputs)."""

    type = "clamp"
    label = "Clamp"
    category = "Filters/Adjustments"
    catalog = "clamp"
    inputs = {"input": Input(label="Input")}
    params = {
        "min": Param(ParamKind.FLOAT, 0.0, "Min", 0.0, 1.0),
        "max": Param(ParamKind.FLOAT, 1.0, "Max", 0.0, 1.0),
        "apply_to_alpha": Param(ParamKind.BOOL, True, "Apply to Alpha"),
    }
