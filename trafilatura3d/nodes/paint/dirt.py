from trafilatura3d.nodes.paint._lib import DOC, SEED, B, F, Generator, mesh_inputs
from trafilatura3d.nodes.paint.metal_edge_wear import MICRO


class Dirt(Generator):
    """Dirt where light hardly reaches (occlusion) and in crevices (concave curvature), not on edges, broken by
    a grunge."""
    type = "paint_dirt"
    label = "Dirt"
    catalog = "painter.generator.dirt"
    doc = DOC + "generators/dirt"
    includes = ("grunge",)
    inputs = mesh_inputs("curvature", "baked_ambient_occlusion", "world_space_normals", "position")
    params = {"invert": B("Invert", False), "dirt_level": F("Dirt Level", 0.5, 0.0, 1.0),
              "dirt_contrast": F("Dirt Contrast", 0.3, 0.0, 1.0), "use_triplanar": B("Use Triplanar", True),
              "triplanar_blending_contrast": F("Triplanar Blending Contrast", 0.5, 0.0, 1.0),
              "grunge_amount": F("Grunge Amount", 0.5, 0.0, 1.0), "grunge_scale": F("Grunge Scale", 1.0, 0.1, 8.0),
              "edges_masking": F("Edges Masking", 0.5, 0.0, 1.0)}
    unsupported = {**SEED, **MICRO}
