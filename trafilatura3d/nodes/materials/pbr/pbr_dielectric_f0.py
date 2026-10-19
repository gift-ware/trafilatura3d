from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import COLOR, E, F, MatNode
from trafilatura3d.nodes.materials.pbr._pbr import CATEGORY, DIELECTRICS


class PbrDielectricF0(MatNode):
    """Uniform specular color (F0, sRGB-encoded) of a dielectric family: Specular Range picks within
    the family's measured range (Specular Range - Glass for Glass); Custom IOR uses F0 = ((n - 1) / (n + 1))^2."""

    type = "pbr_dielectric_f0"
    label = "PBR Dielectric F0"
    category = CATEGORY
    catalog = "pbr_dielectric_f0"
    includes = ("_pbr.wgsl",)
    outputs = {"output": Output(COLOR)}
    params = {
        "specular_f0": E("Plastic", "Specular F0", DIELECTRICS + ("Custom IOR",)),
        "specular_range": F(0.5, "Specular Range", 0.01, 1.0),
        "specular_range_glass": F(0.5, "Specular Range - Glass", 0.0, 1.0),
        "ior": F(1.5, "IOR", 1.0, 5.0),
    }
