from trafilatura3d.nodes.paint._lib import DOC, E, F, mesh_inputs
from trafilatura3d.nodes.paint.generators._common import GenNode, custom, group


def axis(a: str) -> dict:
    # The documentation labels every axis' rotation "Rotation X" (Axis X, Axis Y, Axis Z groups).
    return group({f"rotation_{a}": F("Rotation X", 0.0, 0.0, 360.0),
                  f"offset_{a}_x": F(f"Offset {a.upper()} X", 0.0, 0.0, 1.0),
                  f"offset_{a}_y": F(f"Offset {a.upper()} Y", 0.0, 0.0, 1.0)}, f"Axis {a.upper()}")


class TriPlanarAdvanced(GenNode):
    """The mask input projected from the three axes on the position map (Texture Tiling times across the mesh's
    box), each projection with its own rotation (degrees) and offset, blended by the world space normals: Linear,
    the normal raised to a power (Blending Contrast); Advanced, the projections compete by normal plus their own
    value and only those within a band below the maximum (narrower with Blending Contrast) show. Projection: all
    axes, or a single axis everywhere. Unconnected mask: black."""
    type = "paint_tri_planar_advanced"
    label = "Tri-Planar Advanced"
    catalog = "painter.generator.tri_planar_advanced"
    doc = DOC + "generators/tri-planar-advanced"
    inputs = {**mesh_inputs("world_space_normals", "position"), "mask": custom("mask", gray=True)}
    params = {"projection": E("Projection", ("All Axes", "X Axis", "Y Axis", "Z Axis")),
              "blending_mode": E("Blending Mode", ("Linear", "Advanced")),
              "blending_contrast": F("Blending Contrast", 0.5, 0.0, 1.0),
              "texture_tiling": F("Texture Tiling", 1.0, 0.1, 16.0), **axis("x"), **axis("y"), **axis("z")}
