"""The fifteen grunge maps (Grunge Blotchy ... Grunge Streaked; catalog entries Grunge Map
001-015): complex combined noise maps, each its own recipe of noises, cells, scratches, spots and warps (looks
tuned at the defaults); Brush Pattern masks the edges for brush alphas."""

from ._base import Grunge, f, grunge_params

# Our name -> (the number of its catalog entry, Grunge Map NNN; the look).
_DESIGNS = {
    "blotchy": (1, "bright clouds with dark fractal blotches and crisp grit"),
    "fibrous": (2, "vertical fibrous streaks in dark and light columns"),
    "smeared": (3, "large blotches smeared downwards, streaked"),
    "blobs": (4, "gray and white blobs on black, some merged into clusters"),
    "wavy": (5, "wavy vertical dark lines over glossy folds (Disorder: phase shift)"),
    "bands": (6, "horizontal bands bleeding vertical runs, dark at the bottom"),
    "flakes": (7, "contrasted clouds with crisp light flakes"),
    "network": (8, "light grime crossed by a bright warped cell network"),
    "stains": (9, "mostly clean, with sharp dark grime and stains"),
    "pitted": (10, "dark round spots and fine cracks in gray grime (Cracks: their intensity)"),
    "cracked": (11, "dark grime with a crack network and light streaks (Cracks: their intensity)"),
    "specks": (12, "clean surface with faint thin scratches and specks"),
    "smudges": (13, "dark vertical smudges with light flecks"),
    "blots": (14, "gray grime with large dark vertical stains and light specks"),
    "streaked": (15, "dark streaked top over light clouds"),
}


def _make(name: str):
    n, look = _DESIGNS[name]
    first = {"cracks": f(0.5, "Cracks")} if name in ("pitted", "cracked") else None
    extra = {"disorder": f(0.0, "Disorder")} if name == "wavy" else None
    return type(f"Grunge{name.title()}", (Grunge,), {
        "__doc__": f"Grunge {name.title()}: {look}.", "__module__": __name__,
        "type": f"grunge_{name}", "label": f"Grunge {name.title()}", "catalog": f"grunge_map_{n:03d}",
        "libs": ("grunge_maps_lib.wgsl",), "internal": {"cracks": (0.0, 1.4)} if first else {},
        "params": grunge_params({}, first=first, brush=True, extra_main=extra, contrast=0.5)})


for _name in _DESIGNS:
    globals()[f"Grunge{_name.title()}"] = _make(_name)
