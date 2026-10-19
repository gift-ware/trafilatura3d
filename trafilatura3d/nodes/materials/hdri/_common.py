from trafilatura3d.nodes.materials._base import MatNode

CATEGORY = "3D View (Library)/HDRI Tools"
HDR = {"output_format": {"mode": "absolute", "value": "32f"}}
GAMMAS = ("sRGB", "Linear")
# Shape Color Mode (Sample Background: the background image behind the light, the sphere also clamped to 1).
COLOR_MODES = ("RGB", "Temperature (Kelvin)", "Image Input", "Sample Background")
SPHERE_COLOR_MODES = COLOR_MODES + ("Sample Background (Clamped)",)
UP_VECTORS = ("Z Up", "Y Up")


class HdriNode(MatNode):
    category = CATEGORY
    seamless = False  # a panorama wraps horizontally only; the poles are not a seam to tile


class HdrGenerator(HdriNode):
    """Writes HDR values: 32F output by default."""
    base_defaults = HDR
