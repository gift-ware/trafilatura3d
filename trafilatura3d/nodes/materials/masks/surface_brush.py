from trafilatura3d.nodes.materials._base import E, F, gray_in
from trafilatura3d.nodes.materials.masks._common import AO, CURV, MASK, WSN, MaskNode


class SurfaceBrush(MaskNode):
    """Brushed-metal scratches over the surface: thin streaks (dots at short Scratches Length, long
    lines at high), kept to faces turned towards the Occlude Axis (World Space Normal), removed in
    occluded areas (AO) and on sharp edges, then sharpened."""

    type = "surface_brush"
    label = "Surface Brush"
    catalog = "surface_brush"
    unsupported = {"Scratches Lenght": "the documentation's typo of Scratches Length"}
    inputs = {"world_space_normal": WSN, "curvature": CURV, "ambient_occlusion": AO,
              "position": gray_in("Position", 0.5), "mask": MASK}
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
        "scratches_length": F(4.0, "Scratches Length", 0.0, 8.0),
        "occlude_axis": E("X", "Occlude Axis", ("X", "Y", "Z", "None")),
        "occlude_axis_intensity": F(0.5, "Occlude Axis Intensity", 0.0, 1.0),
        "occlusion": F(0.5, "Occlusion", 0.0, 1.0),
        "sharpen_intensity": F(0.5, "Sharpen Intensity", 0.0, 1.0),
    }
