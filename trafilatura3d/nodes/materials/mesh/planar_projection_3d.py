from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import COLOR, GRAY, C, E, F, F2, MatNode, color_in
from trafilatura3d.nodes.materials.mesh._common import CATEGORY


class PlanarProjection3d(MatNode):
    """Projects a texture (a decal) onto the mesh along a direction, from the baked Position and
    World Space Normal maps, across UV seams. The projector sits at a point picked on the Position
    map (Target UV Position, taking its normal there) or at Target Position / Normal; Rotation turns
    the decal around the projection axis, Scale and Size set its extent. Surfaces deeper than
    Maximum Depth or facing away beyond Normal Threshold are cut, with fades. Output: the decal with
    its mask in alpha, and the mask."""

    type = "planar_projection_3d"
    label = "3D Planar Projection"
    category = CATEGORY
    catalog = "3d_planar_projection"
    inputs = {"position_map": color_in("Position Map", (0.5, 0.5, 0.5, 1.0)),
              "world_space_normal": color_in("World Space Normal", (0.5, 0.5, 1.0, 1.0)),
              "projected_texture": color_in("Projected Texture", (1.0, 1.0, 1.0, 1.0))}
    main_input = "position_map"
    outputs = {"output": Output(COLOR), "mask": Output(GRAY, "Mask")}
    params = {
        "project_input": E("UV Position", "Project Input", ("UV Position", "World Space Position"), group="Positioning"),
        "target_uv_position": F2([0.5, 0.5], "Target UV Position", group="Positioning"),
        "target_position": C([0.5, 0.5, 0.5], "Target Position", group="Positioning"),
        "target_normal": C([0.5, 0.5, 1.0], "Target Normal", group="Positioning"),
        "rotation": F(0.0, "Rotation", 0.0, 1.0, group="Positioning"),
        "scale": F(0.25, "Scale", 0.0, 1.0, group="Positioning"),
        "size": F2([1.0, 1.0], "Size", 0.0, 2.0, group="Positioning"),
        "maximum_depth": F(0.2, "Maximum Depth", 0.0, 1.0, group="Masking"),
        "depth_fade": F(0.2, "Depth Fade", 0.0, 1.0, group="Masking"),
        "normal_threshold": F(0.2, "Normal Threshold", -1.0, 1.0, group="Masking"),
        "normal_fade": F(0.2, "Normal Fade", 0.0, 1.0, group="Masking"),
    }
