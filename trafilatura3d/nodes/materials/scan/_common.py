from trafilatura3d.nodes.materials._base import B, C, F2, F4, I2

CATEGORY = "Material filters/Scan Processing"


def crop_params(group: str = "", offset_label: str = "Offset", background: bool = True, area: bool = True,
                scale: float = 2.0) -> dict:
    """The crop parameters and defaults: a 4096 input, a crop of 1/`scale` of its short side."""
    ps = {"show_crop_area": B(True, "Show Crop Area", group=group)} if area else {}
    ps["input_size"] = I2([4096, 4096], "Input Size", 1, 8192, group=group)
    if background:
        ps["background"] = C([0.0, 0.0, 0.0, 0.0], "Background", group=group)
    ps["transform"] = F4([scale, 0.0, 0.0, scale], "Transform", group=group)
    ps["offset"] = F2([0.0, 0.0], offset_label, 0.0, 1.0, group=group)
    return ps
