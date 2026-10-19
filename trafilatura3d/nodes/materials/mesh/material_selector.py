from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import GRAY, B, C, F, I, MatNode, color_in
from trafilatura3d.nodes.materials.mesh._common import CATEGORY

N = 16
H = 0.498
# The default colors: red, green, blue, yellow, cyan, magenta, then the half tones.
COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, H, 0], [0, 1, H], [H, 0, 1],
          [H, 0, 0], [0, H, 0], [0, 0, H], [H, H, 0], [0, H, H], [H, 0, H], [1, 1, H]]


def _params():
    ps = {"materials": I(8, "Materials", 1, 16)}
    for k in range(1, N + 1):
        ps[f"enable_material_{k}"] = B(k == 1, f"Enable Material #{k}")
        ps[f"material_{k}"] = C(COLORS[k - 1], f"Material #{k}")
    ps["fuzziness"] = F(0.1, "Fuzziness", 0.01, 1.0, group="Color Picker Parameters")
    ps["padding"] = F(0.5, "Padding", 0.0, 1.0, group="Color Picker Parameters")
    return ps


class MaterialSelector(MatNode):
    """Black and white mask from a Color ID map: white where it matches any enabled material color
    among the first Materials ones (Fuzziness: tolerance, Padding: transition hardness)."""

    type = "material_selector"
    label = "Material Selector"
    category = CATEGORY
    catalog = "material_selector"
    inputs = {"color_id": color_in("Color ID", (0.0, 0.0, 0.0, 1.0))}
    outputs = {"output": Output(GRAY)}
    params = _params()
    unsupported = {
        "Enable Material #1-16": "one parameter per material: 'Enable Material #1' ... 'Enable Material #16'",
        "Material #1-16": "one parameter per material: 'Material #1' ... 'Material #16'",
    }

    def wgsl(self):
        body = "".join(f"    if ({k} <= p.materials && p.enable_material_{k} != 0) "
                       f"{{ m = max(m, id_mask(idc, p.material_{k}.rgb, p.fuzziness, p.padding)); }}\n"
                       for k in range(1, N + 1))
        return super().wgsl().replace("//$BODY\n", body)
