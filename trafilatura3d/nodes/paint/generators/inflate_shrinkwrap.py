from trafilatura3d.nodes.paint._lib import DOC, SEED, E, F, mesh_inputs
from trafilatura3d.nodes.paint.generators._common import BorderGenerator


class InflateShrinkwrap(BorderGenerator):
    """A height of a thin film over the surface, from the distance to the UV island borders (computed `uv_islands`
    map: the seams of the film) in centimetres (Surface Size (cm) = the UV square): a profile rising from the seams
    (Preset: Inflated round pillow, Vacuum Pulled steep then flat, Tight short ramp) of Surface Depth (cm), a
    groove on the seam (Seam Intensity), a raised lip beside it (Raised Edge Width, Intensity), and wrinkles running
    out of the seams: ridges across the border (noise at the nearest border point, so constant along the normal),
    Wrinkle Density many, Wrinkle Scale large, closer together at the seam (Wrinkle Tightness), fading at Wrinkle
    Range. Inflate bulges the film; Shrinkwrap keeps it flat with stronger wrinkles that gather in the cavities
    (curvature). Height = Height Position + (height - depth / 2) / Height Range (cm)."""
    type = "paint_inflate_shrinkwrap"
    label = "Inflate Shrinkwrap"
    catalog = "painter.generator.inflate_shrinkwrap"
    doc = DOC + "generators/inflate-shrinkwrap"
    includes = ("generators/common",)
    inputs = mesh_inputs("curvature", "uv_islands")
    params = {"preset": E("Preset", ("Inflated", "Vacuum Pulled", "Tight")),
              "inflate_or_shrinkwrap": E("Inflate or Shrinkwrap", ("inflate", "shrinkwrap")),
              "seam_intensity": F("Seam Intensity", 0.5, 0.0, 1.0),
              "raised_edge_width": F("Raised Edge Width", 0.3, 0.0, 1.0),
              "raised_edge_intensity": F("Raised Edge Intensity", 0.3, 0.0, 1.0),
              "wrinkle_density": F("Wrinkle Density", 0.5, 0.0, 1.0),
              "wrinkle_tightness": F("Wrinkle Tightness", 0.5, 0.0, 1.0),
              "wrinkle_range": F("Wrinkle Range", 0.5, 0.0, 1.0),
              "wrinkle_scale": F("Wrinkle Scale", 0.5, 0.0, 1.0),
              "height_range": F("Height Range", 4.0, 0.01, 20.0),
              "height_position": F("Height Position", 0.5, 0.0, 1.0),
              "surface_size_cm": F("Surface Size (cm)", 100.0, 1.0, 1000.0),
              "surface_depth_cm": F("Surface Depth (cm)", 2.0, 0.0, 50.0)}
    unsupported = {**SEED}
