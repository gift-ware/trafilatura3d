from trafilatura3d.nodes.base import Input, PortType
from trafilatura3d.nodes.paint._lib import DOC, B, E, F, Generator, I

TYPES = ("Desaturation", "Luma", "Average", "Max", "Min", "Red channel", "Green channel", "Blue channel")


class GrayscaleConversionGenerator(Generator):
    """The Source Color texture, repeated Tile times and rotated (degrees, around the center; Safe Rotation rounds
    to multiples of 45), made grayscale by Grayscale type, then Balance (0.5 = unchanged), Contrast (0 =
    unchanged), Invert."""
    type = "paint_grayscale_conversion_generator"
    label = "Grayscale Conversion"
    catalog = "painter.generator.grayscale_conversion"
    doc = DOC + "generators/grayscale-conversion"
    includes = ("color/color",)
    inputs = {"source_color": Input(PortType.COLOR, "Source Color", (0.0, 0.0, 0.0, 1.0))}
    params = {"grayscale_type": E("Grayscale type", TYPES, "Luma"), "invert": B("Invert", False),
              "balance": F("Balance", 0.5, 0.0, 1.0), "contrast": F("Contrast", 0.0, 0.0, 1.0),
              "tile": I("Tile", 1, 1, 16), "rotation": F("Rotation", 0.0, 0.0, 360.0),
              "safe_rotation": B("Safe Rotation", False)}
