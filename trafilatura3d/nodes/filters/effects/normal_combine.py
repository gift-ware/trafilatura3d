from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect

FLAT = (0.5, 0.5, 1.0, 1.0)


class NormalCombine(Effect):
    """Adds the details of Normal 2 onto the base Normal 1, in encoded values (0.5 =
    flat): Whiteout: XY added (each sum clamped to 0-1), Z multiplied, then normalised. Channel mixer: XY
    added, the lower Z kept, not normalised. Detail oriented: the detail normal rotated onto the base
    normal (reoriented normal mapping). Alpha 1."""

    type = "normal_combine"
    label = "Normal Combine"
    category = "Filters/Normal Map"
    catalog = "normal_combine"
    inputs = {"normal_2": Input(PortType.COLOR, "Normal 2", default=FLAT),
              "normal_1": Input(PortType.COLOR, "Normal 1", default=FLAT)}
    main_input = "normal_1"
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "technique": Param(ParamKind.ENUM, "Whiteout", "Technique",
                           choices=("Whiteout", "Channel mixer", "Detail oriented")),
    }
