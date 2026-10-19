from trafilatura3d.nodes.base import Input, Param, ParamKind, PortType

from ._fx import Effect


class Mosaic(Effect):
    """Facets a smooth input: Samples Warps in a chain by the Mosaic Map, each of
    Intensity / Samples (Warp's unit: Intensity / 256 x the map's slope in uv), so each pixel climbs the
    map's slope in Samples steps, as long as the slope is steep, and takes the Color input's value where it
    ends. Pixels gather on the map's crests, so slopes break into flat facets; with the same map on both
    inputs the brightest areas grow. Samples 0: the input."""

    type = "mosaic"
    label = "Mosaic"
    category = "Filters/Effects"
    catalog = "mosaic"
    inputs = {"color": Input(label="Color"), "mosaic_map": Input(PortType.GRAY, "Mosaic Map")}
    params = {
        "samples": Param(ParamKind.INT, 8, "Samples", 0, 16),
        "intensity": Param(ParamKind.FLOAT, 0.5, "Intensity", 0.0, 1.0),
    }
