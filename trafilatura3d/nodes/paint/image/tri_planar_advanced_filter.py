from dataclasses import replace

from trafilatura3d.nodes.base import Input
from trafilatura3d.nodes.paint._lib import DOC, FILTERS, E, F, Filter, mesh_inputs

PROJECTIONS = ("X Y Z", "X Y", "X Z", "Y Z", "X", "Y", "Z")


def _axis(axis: str) -> dict:
    """The texture controls of one axis' projection: rotation (turns) and offset (UV) in its plane. The label of
    the rotation of every axis is "Rotation X"."""
    g = f"Axis {axis}"
    a = axis.lower()
    return {f"axis_{a}_rotation": replace(F("Rotation X", 0.0, 0.0, 1.0), group=g),
            f"offset_{a}_x": replace(F(f"Offset {axis} X", 0.0, -1.0, 1.0), group=g),
            f"offset_{a}_y": replace(F(f"Offset {axis} Y", 0.0, -1.0, 1.0), group=g)}


class TriPlanarAdvancedFilter(Filter):
    """Re-projects the layer content (the input, a UV-space texture) from three sides of the mesh: the position
    map (the Texture Set's bounds as 0-1) gives each pixel's 3D point, the projection frame is moved by Offset X /
    Y / Z and rotated by Rotation X / Y / Z (turns, about the world axes), the input is repeated Texture Tiling
    times per unit and sampled in the planes YZ, XZ and XY (each with its own rotation and offsets, groups Axis X
    / Y / Z), and the three are blended by the world space normal: weights |n|^(1 + 15 x Blending Contrast) over
    the axes of Projection, Linear (normalized) or Max (only the dominant axis)."""
    type = "paint_tri_planar_advanced_filter"
    label = "Tri-Planar Advanced"
    category = FILTERS + "/Advanced"
    catalog = "painter.filter.tri_planar_advanced_filter"
    doc = DOC + "filters/advanced-filters/tri-planar-advanced-filter"
    includes = ("image/_image",)
    inputs = {"input": Input(label="Input"), **mesh_inputs("world_space_normals", "position")}
    params = {"projection": E("Projection", PROJECTIONS), "blending_mode": E("Blending Mode", ("Linear", "Max")),
              "blending_contrast": F("Blending Contrast", 0.3, 0.0, 1.0),
              "texture_tiling": F("Texture Tiling", 1.0, 0.1, 16.0),
              "rotation_x": F("Rotation X", 0.0, 0.0, 1.0), "offset_x": F("Offset X", 0.0, -1.0, 1.0),
              "rotation_y": F("Rotation Y", 0.0, 0.0, 1.0), "offset_y": F("Offset Y", 0.0, -1.0, 1.0),
              "rotation_z": F("Rotation Z", 0.0, 0.0, 1.0), "offset_z": F("Offset Z", 0.0, -1.0, 1.0),
              **_axis("X"), **_axis("Y"), **_axis("Z")}
