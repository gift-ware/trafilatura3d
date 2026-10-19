from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.paint._lib import DOC, FILTERS, B, F, Filter


class HeightToNormal(Filter):
    """The normal channel from the height channel's slopes (central differences, tangent space, OpenGL like every
    normal of Trafilatura 3D). Use World Units: the texture spans Surface Size (cm) and a height of 1 is Height Depth
    (cm), so the slope is dh/du x depth / size; otherwise slope = dh/du x Normal Intensity / 20 (1 = the slopes of
    a 5 cm relief on a 100 cm surface). The result replaces the normal channel (Overwrite Existing Normal) or is
    laid on it as detail (reoriented normal mapping). The height channel is left as it is."""
    type = "paint_height_to_normal"
    label = "Height To Normal"
    category = FILTERS + "/Advanced"
    catalog = "painter.filter.height_to_normal"
    doc = DOC + "filters/advanced-filters/height-to-normal"
    inputs = {"height": Input(PortType.GRAY, "Height", (0.5, 0.5, 0.5, 1.0)),
              "normal": Input(PortType.COLOR, "Normal", (0.5, 0.5, 1.0, 1.0))}
    outputs = {"normal": Output(PortType.COLOR)}
    params = {"overwrite_existing_normal": B("Overwrite Existing Normal", False),
              "use_world_units": B("Use World Units", False), "normal_intensity": F("Normal Intensity", 1.0, 0.0, 10.0),
              "surface_size_cm": F("Surface Size (cm)", 100.0, 1.0, 1000.0),
              "height_depth_cm": F("Height Depth (cm)", 5.0, 0.0, 100.0)}
