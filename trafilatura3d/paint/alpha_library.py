"""The library of brush alphas made of graphs (`paint/alpha_graphs/<category>.trafilatura3d`, one graph per alpha,
its exposed parameters Hardness, Shape, Invert...), next to the alphas generated in code (paint/alphas.py).

An alpha is referenced by one string everywhere (a tool, a path, a preset): a generated alpha by its name, a
library alpha as "library:<category>/<graph>", its parameter values appended as a query
("library:dirt/grime_blobs?hardness=0.7&invert=1", `seed` the graph's random seed); a value left out keeps the
graph's default. `image` gives the
grayscale stamp of any of them."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

import numpy as np

from trafilatura3d.paint.alphas import ALPHAS, DEFAULT_ALPHA

LIBRARY = "library:"
FOLDER = Path(__file__).parent / "alpha_graphs"


@lru_cache(maxsize=None)
def _package(category: str):
    from trafilatura3d.package import Package
    return Package.load(FOLDER / f"{category}.trafilatura3d")


@lru_cache(maxsize=1)
def listing() -> tuple[tuple[str, str, str], ...]:
    """(reference without values, label, category) of every library alpha, by category then label."""
    out = []
    for f in sorted(FOLDER.glob("*.trafilatura3d")):
        pkg = _package(f.stem)
        out += sorted(((f"{LIBRARY}{f.stem}/{gid}", g.label or gid, f.stem.title()) for gid, g in pkg.graphs.items()),
                      key=lambda e: e[1])
    return tuple(out)


def split(ref: str) -> tuple[str, str, dict]:
    """(category, graph, parameter values given) of a library reference."""
    path, _, query = ref[len(LIBRARY):].partition("?")
    category, _, graph = path.partition("/")
    return category, graph, {k: float(v) for k, v in parse_qsl(query)}


def join(category: str, graph: str, values: dict) -> str:
    query = urlencode({k: f"{float(v):g}" for k, v in values.items()})
    return f"{LIBRARY}{category}/{graph}" + (f"?{query}" if query else "")


def exists(ref: str) -> bool:
    """A generated alpha, or a library alpha whose graph is there."""
    if not ref.startswith(LIBRARY):
        return ref in ALPHAS
    category, graph, _ = split(ref)
    return (FOLDER / f"{category}.trafilatura3d").exists() and graph in _package(category).graphs


def parameters(ref: str) -> list:
    """The exposed parameters (package ExposedParam) of a library alpha's graph, in their order."""
    category, graph, _ = split(ref)
    return list(_package(category).graphs[graph].parameters.values())


def image(ref: str, n: int) -> np.ndarray:
    """The n x n grayscale stamp (0-1) of an alpha; a library alpha is cooked at that size with its values, once
    per process (an evaluator of this module); an unknown one is the default alpha."""
    if not exists(ref):
        ref = DEFAULT_ALPHA
    if not ref.startswith(LIBRARY):
        return np.clip(np.asarray(ALPHAS[ref](n), np.float32), 0, 1)
    return _cooked(ref, int(n))


@lru_cache(maxsize=2048)  # every icon of the Assets, the strokes' stamps: asked again and again
def _cooked(ref: str, n: int) -> np.ndarray:
    from trafilatura3d.compiler import ABSOLUTE, compile_graph
    category, graph, values = split(ref)
    pkg = _package(category)
    g = pkg.graphs[graph]
    base = dict(g.base)
    g.base["output_size"] = {"mode": ABSOLUTE, "value": [int(np.log2(n))] * 2}
    if "seed" in values:
        g.base["seed"] = {"mode": ABSOLUTE, "value": int(values.pop("seed"))}
    try:
        kinds = {pid: p.param.kind.value for pid, p in g.parameters.items()}
        vals = {k: (bool(v) if kinds.get(k) == "bool" else v) for k, v in values.items() if k in kinds}
        job, out = compile_graph(pkg, graph, values=vals).outputs["output"]
        a = np.asarray(_evaluator().read(job, out))
    finally:
        g.base.clear()
        g.base.update(base)
    a = a.astype(np.float32) / np.iinfo(a.dtype).max if a.dtype.kind in "ui" else a.astype(np.float32)
    a = a[..., 0] if a.ndim == 3 else a
    if a.shape[0] != n:  # a graph that does not follow the size asked
        from PIL import Image
        a = np.asarray(Image.fromarray(a).resize((n, n), Image.BILINEAR), np.float32)
    a = np.clip(a, 0, 1)
    a.flags.writeable = False  # shared by the cache
    return a


@lru_cache(maxsize=1)
def _evaluator():
    from trafilatura3d.evaluator import Evaluator
    return Evaluator()
