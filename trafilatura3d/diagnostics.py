"""Warnings of graphs and dependencies, and the dependency list of a package (the Dependency Manager).

A warning never stops the editor: it marks the node (badge), the graph and its package in the
Explorer, and lists in the Graph view."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from trafilatura3d.package import INSTANCE, DocumentError, Package

RESOURCE_NODES = {"bitmap": "pkg_resource_path", "svg": "pkg_resource_path"}


@dataclass
class GraphWarning:
    message: str
    node: str | None = None  # node id inside the graph, when it is about a node
    detail: str = ""


@dataclass
class Dependency:
    kind: str  # "Package", "Bitmap", "SVG", "Font", "3D scene"
    name: str  # resource id or referencing node
    path: str  # as written in the package
    resolved: str  # absolute path
    ok: bool
    users: list = field(default_factory=list)  # (graph id, node id) using it
    resource: str | None = None  # resource id when it is a package resource


def _resolve(pkg: Package, path: str) -> Path:
    from trafilatura3d.project import alias_of, expand
    if alias_of(path):
        try:
            path = expand(path)
        except KeyError:
            return Path("/nonexistent") / path
    p = Path(path)
    if not p.is_absolute() and pkg.path is not None:
        p = pkg.path.parent / p
    return p.resolve()


def dependencies(pkg: Package) -> list[Dependency]:
    """Every file the package depends on: referenced packages (graph instances), linked resources
    (bitmaps, 3D scenes), files named by Bitmap / SVG nodes and Text fonts."""
    deps: dict[tuple, Dependency] = {}

    def add(kind, name, path, users=None, resource=None):
        key = (kind, path)
        if key not in deps:
            r = _resolve(pkg, path)
            deps[key] = Dependency(kind, name, path, str(r), r.exists(), [], resource)
        if users:
            deps[key].users.append(users)
    for rid, res in pkg.resources.items():
        if res.kind == "scene" and res.path.startswith("primitive:"):
            continue
        add({"bitmap": "Bitmap", "scene": "3D scene"}.get(res.kind, res.kind.capitalize()), rid, res.path, resource=rid)
    for gid, g in pkg.graphs.items():
        for n in g.nodes.values():
            if n.type == INSTANCE and "#" in n.ref:
                add("Package", n.ref.rsplit("#", 1)[0], n.ref.rsplit("#", 1)[0], (gid, n.id))
            elif n.type in RESOURCE_NODES and n.params.get(RESOURCE_NODES[n.type]):
                path = n.params[RESOURCE_NODES[n.type]]
                rid = next((k for k, r in pkg.resources.items() if r.path == path), None)
                add("Bitmap" if n.type == "bitmap" else "SVG", rid or n.id, path, (gid, n.id), rid)
            elif n.type == "text" and str(n.params.get("font", "")).lower().endswith((".ttf", ".otf", ".ttc")):
                add("Font", n.id, n.params["font"], (gid, n.id))
    return sorted(deps.values(), key=lambda d: (d.kind, d.name))


def graph_warnings(pkg: Package, gid: str, _seen=None) -> list[GraphWarning]:
    """The warnings of one graph (content warnings plus our structural checks). Every instance of a graph that has
    warnings gets one (the graph is checked once; one being checked up the chain counts as without: a cycle)."""
    from trafilatura3d.functions import FunctionError, FunctionGraph, set_variables
    from trafilatura3d.visibility import check
    seen = _seen if _seen is not None else {}  # (package, graph) -> has warnings (None while being checked)
    seen[(id(pkg), gid)] = None
    g = pkg.graphs[gid]
    out = []
    if not g.output_nodes():
        out.append(GraphWarning("No output node defined"))
    for problem in g.problems():
        out.append(GraphWarning(problem))
    for n in g.nodes.values():
        if n.type == INSTANCE:
            try:
                ref_pkg, ref_graph = pkg.resolve(n.ref)
            except (DocumentError, OSError, ValueError) as e:
                out.append(GraphWarning("Reference resource not found", n.id, str(e)))
                continue
            key = (id(ref_pkg), ref_graph.id)
            if key not in seen:
                seen[key] = bool(graph_warnings(ref_pkg, ref_graph.id, seen))
            if seen[key]:
                out.append(GraphWarning("The referenced data has some warnings", n.id, ref_graph.id))
        elif n.type in RESOURCE_NODES:
            path = n.params.get(RESOURCE_NODES[n.type], "")
            if path and not _resolve(pkg, path).exists():
                out.append(GraphWarning("Reference resource not found", n.id, path))
        elif n.type == "text":
            font = str(n.params.get("font", ""))
            if font.lower().endswith((".ttf", ".otf", ".ttc")) and not _font_ok(_resolve(pkg, font)):
                out.append(GraphWarning("Text node uses invalid font", n.id, font))
        try:
            d = pkg.definition(n)
        except DocumentError:
            continue
        if n.type == "output" and n.params.get("color_space"):
            from trafilatura3d import colormanagement as cm
            if cm.active() and not cm.known(n.params["color_space"]):
                out.append(GraphWarning("Color space not found", n.id, n.params["color_space"]))
        node_sets = set_variables(n.function_graphs())
        for name, function in n.functions.items():
            try:
                FunctionGraph(function, g.function_variables(), node_sets)
            except (FunctionError, KeyError, TypeError) as e:
                label = d.params[name].label if name in d.params else name
                out.append(GraphWarning(f"The {d.label}[{label}] parameter's function has some warnings", n.id, str(e)))
    values = {pid: p.param.default for pid, p in g.parameters.items()}
    for pid, p in g.parameters.items():
        msg = check(p.param.visible_if, values)
        if msg:
            out.append(GraphWarning(f"{p.param.label or pid}: {msg}"))
    out += [GraphWarning(n.message, n.node if n.node in g.nodes else None) for n in pkg.notes if n.graph == gid]
    out += output_size_warnings(pkg, gid)
    return out


def output_size_warnings(pkg: Package, gid: str) -> list[GraphWarning]:
    """The Output nodes with a usage whose size is not the one most of them have (a material's maps are meant to
    match: a metallic at 16x16 because an absolute size upstream was taken for pixels, not log2)."""
    from trafilatura3d.compiler import compile_graph
    g = pkg.graphs[gid]
    outputs = [n for n in g.output_nodes() if pkg.definition(n).values(n)["usage"]]
    if len(outputs) < 2:
        return []
    try:
        jobs = compile_graph(pkg, gid).jobs
    except (DocumentError, KeyError, ValueError):
        return []
    sizes = {n.id: tuple(jobs[n.id].size) for n in outputs if n.id in jobs}
    counts = {}
    for size in sizes.values():
        counts[size] = counts.get(size, 0) + 1
    if len(counts) < 2:
        return []
    common = max(counts, key=lambda s: (counts[s], s))
    return [GraphWarning("Output size differs from the other outputs", node,
                         f"{g.identifier(g.nodes[node])} {w}x{h}, the others {common[0]}x{common[1]}")
            for node, (w, h) in sizes.items() if (w, h) != common]


def _font_ok(path: Path) -> bool:
    try:
        head = path.read_bytes()[:4]
    except OSError:
        return False
    return head in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"ttcf")


def dependency_warnings(pkg: Package) -> list[GraphWarning]:
    """The warnings from dependencies that apply to Trafilatura 3D packages."""
    from trafilatura3d.project import alias_of, configuration
    out = []
    aliases = configuration().aliases()
    for d in dependencies(pkg):
        if d.kind == "Package" and (d.path.startswith("sbs://") or d.path.endswith((".sbs", ".sbsar"))):
            out.append(GraphWarning("Designer dependency with no Trafilatura 3D counterpart", None, d.path))
            continue
        name = alias_of(d.path)
        if name and name not in aliases:
            out.append(GraphWarning(f"Check the alias '{name}' is defined in your project", None, d.path))
            continue
        if d.kind == "Package":
            if not d.ok:
                out.append(GraphWarning("Invalid dependent package", None, d.resolved))
            else:
                error = _package_error(d.resolved)
                if error:
                    out.append(GraphWarning("Invalid dependent package", None, f"{d.resolved}: {error}"))
        elif not d.ok:
            out.append(GraphWarning("Linked file not found", None, f"{d.name}: {d.resolved}"))
    return out


_LOADED: dict[str, tuple[float, str]] = {}  # referenced package -> (mtime, load error or "")


def _package_error(path: str) -> str:
    """Why a referenced package does not load ("" when it does), read again only when it changed."""
    mtime = Path(path).stat().st_mtime
    if path not in _LOADED or _LOADED[path][0] != mtime:
        try:
            Package.load(path)
            _LOADED[path] = (mtime, "")
        except (DocumentError, OSError, ValueError) as e:
            _LOADED[path] = (mtime, str(e))
    return _LOADED[path][1]


def package_warnings(pkg: Package) -> dict[str, list[GraphWarning]]:
    """Graph id -> its warnings, plus "" -> the package's dependency warnings (only non empty lists)."""
    out = {gid: graph_warnings(pkg, gid) for gid in pkg.graphs}
    out[""] = dependency_warnings(pkg)
    return {k: v for k, v in out.items() if v}


def relocate(pkg: Package, old_path: str, new_file: str) -> int:
    """Point every reference to `old_path` (resources, Bitmap/SVG nodes, fonts, package references)
    to `new_file` (written relative to the package when it can). Returns how many were changed."""
    import os
    new = new_file
    if pkg.path is not None:
        try:
            new = os.path.relpath(new_file, pkg.path.parent).replace(os.sep, "/")
        except ValueError:
            pass
    count = 0
    for res in pkg.resources.values():
        if res.path == old_path:
            res.path = new
            count += 1
    for g in pkg.graphs.values():
        for n in g.nodes.values():
            if n.type in RESOURCE_NODES and n.params.get(RESOURCE_NODES[n.type]) == old_path:
                n.params[RESOURCE_NODES[n.type]] = new
                count += 1
            elif n.type == "text" and n.params.get("font") == old_path:
                n.params["font"] = new
                count += 1
            elif n.type == INSTANCE and "#" in n.ref and n.ref.rsplit("#", 1)[0] == old_path:
                n.ref = new + "#" + n.ref.rsplit("#", 1)[1]
                count += 1
    pkg._external.clear()
    return count
