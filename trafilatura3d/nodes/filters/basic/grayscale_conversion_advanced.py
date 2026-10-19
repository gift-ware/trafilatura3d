from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode

TYPES = ("Desaturation", "Luma", "Average", "Max", "Min")


class GrayscaleConversionAdvanced(BasicNode):
    """Colour -> grayscale. Desaturation: HSL lightness (max + min) / 2; Luma: Rec. 709 weights;
    Average: (R + G + B) / 3; Max / Min: brightest / darkest channel."""

    type = "grayscale_conversion_advanced"
    label = "Grayscale Conversion Advanced"
    category = "Filters/Adjustments"
    catalog = "grayscale_conversion_advanced"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {"grayscale_type": Param(ParamKind.ENUM, "Desaturation", "Grayscale Type", choices=TYPES)}
