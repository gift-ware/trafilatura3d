"""A Designer package (.sbs) as a Trafilatura 3D package: every graph, its nodes (atomic filters, library graphs mapped to
our nodes, instances of the user's graphs), connections, parameter values and functions, exposed parameters and
their links, image inputs and outputs, base parameters, FX-Maps, frames and comments, positions and bitmaps.

User graphs referenced by the package (of the same file or of other .sbs files) are imported into the same Trafilatura 3D
package, as graphs referenced by id: a package with references to unsaved packages can't be written, and one
self-contained document is what Designer's own "export with dependencies" gives. Designer's library graphs are never
imported: they map to our nodes (mapping.LIBRARY), and one without a counterpart stays a missing reference.

What can't be converted is never faked nor dropped silently. A node we have no counterpart for becomes a graph
instance whose reference can't be resolved (like a Designer node whose dependency is missing): it keeps its
connections, so the graph stays wired, and its values; an unknown parameter, function or feature is left at our
default. All of it is listed in the `Report`, and kept as the package's notes (shown with the graph's warnings)."""

from __future__ import annotations

import dataclasses
import glob
import os
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from trafilatura3d import fonts
from trafilatura3d.compiler import BASE_PARAMS, NODE_DEFAULTS
from trafilatura3d.functions import PARAM_TYPES, FunctionError, FunctionGraph, set_variables
from trafilatura3d.nodes import NODES
from trafilatura3d.nodes.base import Param, ParamKind
from trafilatura3d.package import INSTANCE, DocumentError, Graph, Note, NodeInstance, Package, Resource
from trafilatura3d.sbs import dynamic
from trafilatura3d.sbs.mapping import (ATOMIC, BASE, FORMATS, LIBRARY, LIBRARY_FUNCTIONS, LIBRARY_PACKAGES, PIXEL_RATIOS,
                                 TILINGS, NodeMap, ParamMap, convert)
from trafilatura3d.sbs.reader import (ABSOLUTE, IMAGE_TYPES, RELATIVE_INPUT, RELATIVE_PARENT, TYPES, Value, constant,
                                parameters, text)

LIBRARY_AUTHORS = ("Allegorithmic", "Adobe")
_MODES = {ABSOLUTE: "absolute", RELATIVE_INPUT: "relative_input", RELATIVE_PARENT: "relative_parent"}
_KINDS = {"bool": ParamKind.BOOL, "int": ParamKind.INT, "int2": ParamKind.INT2, "float": ParamKind.FLOAT,
          "float2": ParamKind.FLOAT2, "float3": ParamKind.FLOAT3, "float4": ParamKind.FLOAT4, "string": ParamKind.STRING}
_EDITORS = {"slider": "Slider", "angle": "Angle", "color": "Color", "buttons": "Togglebutton",
            "dropdownlist": "Drop down list", "position": "Position"}


@dataclass
class Report:
    """What the import did: counts of Designer's nodes and the issues, by our graph and node."""
    source: Path
    nodes: int = 0  # compositing nodes (filters and instances) read
    mapped: int = 0  # became one of our nodes or an instance of an imported graph
    missing: int = 0  # kept as missing references
    params_unmapped: int = 0
    graphs: list[str] = field(default_factory=list)  # our ids of the file's own graphs (not its dependencies')
    notes: list[Note] = field(default_factory=list)

    def add(self, graph: str, node: str | None, message: str) -> None:
        self.notes.append(Note(graph, node, message))

    def lines(self) -> list[str]:
        head = (f"{self.source.name}: {self.nodes} nodes, {self.mapped} mapped, {self.missing} missing, "
                f"{self.params_unmapped} parameters not mapped")
        return [head] + [f"  {n.graph}" + (f".{n.node}" if n.node else "") + f": {n.message}" for n in self.notes]


def import_sbs(path: str | Path, base: str | Path | None = None,
               search: tuple[str | Path, ...] = ()) -> tuple[Package, Report]:
    """The package of a .sbs file (not saved: its path is None). File paths are written relative to `base` (the
    folder the .trafilatura3d will be saved in) when given, else absolute. A dependency whose written path doesn't exist
    (an absolute path of another machine) is looked for by file name in the `search` folders and their subfolders."""
    path = Path(path).resolve()
    imp = _Importer(path, Path(base).resolve() if base is not None else None, [Path(d).resolve() for d in search])
    root = imp.file(path)
    for g in root.iter("graph"):
        imp.report.graphs.append(imp.graph(path, text(g, "identifier")))
    imp.pkg.notes = imp.report.notes
    imp.pkg.validate()
    return imp.pkg, imp.report


def _ident(s: str) -> str:
    """A Designer identifier as one of ours (letters, digits and _)."""
    s = re.sub(r"\W", "_", s or "x")
    return s if s.replace("_", "") else "x"


class _Importer:
    def __init__(self, source: Path, base: Path | None, search: list[Path]):
        self.base = base
        self.search = search
        self._found: dict[str, Path | None] = {}  # file name -> where the search folders have it
        self.pkg = Package()
        self.report = Report(source)
        self._files: dict[Path, ET.Element] = {}
        self._graphs: dict[tuple[Path, str], str] = {}  # (file, Designer graph id) -> our graph id
        self._functions: dict[tuple[Path, str], str] = {}  # (file, Designer function id) -> our function name
        self._tables: dict[str, dict] = {}  # our function name -> it and the functions it calls (None: converting)

    # --- files --------------------------------------------------------------------------------

    def file(self, path: Path) -> ET.Element:
        if path not in self._files:
            try:
                self._files[path] = ET.parse(path).getroot()
            except ET.ParseError as e:
                raise DocumentError(f"{path}: not a Designer package ({e})") from None
        return self._files[path]

    def _write_path(self, p: Path) -> str:
        if self.base is None:
            return str(p)
        try:
            return os.path.relpath(p, self.base).replace(os.sep, "/")
        except ValueError:  # another drive
            return str(p)

    def _dependency(self, file: Path, uid: str) -> tuple[str, Path | None]:
        """A dependency of `file` by uid: (its file name as written, its path or None when it is the file itself)."""
        for d in self.file(file).iter("dependency"):
            if text(d, "uid") == uid:
                name = text(d, "filename")
                if name == "?himself":
                    return name, None
                if name.startswith("sbs://"):
                    return name, Path(name[len("sbs://"):])
                return name, self._find(_local(file, name), file.parent)
        return "", Path("")

    def _find(self, p: Path, home: Path | None = None) -> Path:
        """`p`, or when it doesn't exist the file of the same name in a folder between `p` and `home` (a package
        moved up out of its own folder), else in the search folders: in the first folder that has one, the most
        recently modified."""
        if p.is_file():
            return p
        if home is not None and p.is_relative_to(home):
            up = next((d / p.name for d in p.parents[1:] if d.is_relative_to(home) and (d / p.name).is_file()), None)
            if up is not None:
                return up
        if not self.search:
            return p
        name = re.split(r"[/\\]", str(p))[-1]  # also of a Windows path
        if name not in self._found:
            hits = ([f for f in d.rglob(glob.escape(name)) if f.is_file()] for d in self.search)
            self._found[name] = next((max(h, key=lambda f: f.stat().st_mtime) for h in hits if h), None)
        return self._found[name] or p

    def _is_library(self, written: str, path: Path) -> bool:
        if written.startswith("sbs://") or path.suffix == ".sbsar" or path.name in LIBRARY_PACKAGES:
            return True
        if not path.is_file():
            return False
        root = self.file(path)
        return any(text(g, "attributes/author") in LIBRARY_AUTHORS for g in [*root.iter("graph"), *_functions(root)])

    # --- graphs -------------------------------------------------------------------------------

    def graph(self, file: Path, sd_id: str) -> str | None:
        """Our id of a Designer graph, importing it the first time (None when the file has no such graph)."""
        key = (file, sd_id)
        if key in self._graphs:
            return self._graphs[key]
        el = next((g for g in self.file(file).iter("graph") if text(g, "identifier") == sd_id), None)
        if el is None:
            return None
        gid = _ident(sd_id)
        if gid in self.pkg.graphs:  # a graph of another file with the same identifier
            gid = _ident(f"{sd_id}_{file.stem}")
        stem, k = gid, 2
        while gid in self.pkg.graphs:
            gid, k = f"{stem}_{k}", k + 1
        self._graphs[key] = gid
        g = self.pkg.new_graph(gid)
        _GraphImporter(self, file, el, g).run()
        return gid

    # --- function graphs ------------------------------------------------------------------------

    def call(self, file: Path, url: str) -> tuple[str, str, dict]:
        """What a call from `file` of the function graph at `url` (pkg:///Group/name?dependency=uid) is: one of our
        Library nodes (Designer's library functions), or a call of a user function imported once, with the functions
        it calls (dynamic.function's `calls`)."""
        ident = url.split("?")[0].rsplit("/", 1)[-1]
        if "dependency=" in url:
            written, dep = self._dependency(file, url.split("dependency=")[-1])
            dep = dep or file
        else:  # no dependency: a function of the package itself
            written, dep = "?himself", file
        if written != "?himself" and self._is_library(written, dep):
            ours = LIBRARY_FUNCTIONS.get(f"{dep.name}#{ident}")
            if ours is None:
                raise dynamic.Unsupported(f"library function {dep.name}#{ident} has no Trafilatura 3D counterpart")
            return ours, "", {}
        if not dep.is_file():
            raise dynamic.Unsupported(f"function {ident!r}: dependency not found: {dep}")
        name = self.function(dep, ident)
        if self._tables[name] is None:
            raise dynamic.Unsupported(f"function {ident!r} calls itself")
        return "call", name, self._tables[name]

    def function(self, file: Path, sd_id: str) -> str:
        """Our name of a user function graph (a <function> of `file`), converted the first time."""
        key = (file, sd_id)
        if key in self._functions:
            return self._functions[key]
        el = next((f for f in _functions(self.file(file)) if text(f, "identifier") == sd_id), None)
        dyn = el.find("paramValue/dynamicValue") if el is not None else None
        if dyn is None:
            raise dynamic.Unsupported(f"function {sd_id!r} not found in {file.name}")
        name, k = _ident(sd_id), 2
        while name in self._tables:
            name, k = f"{_ident(sd_id)}_{k}", k + 1
        self._functions[key], self._tables[name] = name, None
        try:
            self._tables[name] = self._function_table(file, el, dyn, name)
        except dynamic.Unsupported:
            del self._functions[key], self._tables[name]  # not "calling itself" when another call meets it
            raise
        return name

    def _function_table(self, file: Path, el: ET.Element, dyn: ET.Element, name: str) -> dict:
        """A user function as ours (`name`) with the functions it calls."""
        sd_id = text(el, "identifier")
        inputs, defaults = {}, {}
        for pi in el.findall("paraminputs/paraminput"):
            t = TYPES.get(int(text(pi, "type", "0")))
            if t is None:
                raise dynamic.Unsupported(f"function {sd_id!r}: input {text(pi, 'identifier')!r} of an unknown type")
            inputs[text(pi, "identifier")] = t
            value = constant(pi.find("defaultValue"))
            if value is not None:
                defaults[text(pi, "identifier")] = _typed(value, t)
        body = dynamic.function(dyn, {}, lambda url: self.call(file, url))
        definition = {"inputs": inputs, **({"defaults": defaults} if defaults else {}), "nodes": body["nodes"],
                      "output": body["output"]}
        return {name: definition, **body.get("functions", {})}

    def resource(self, file: Path, url: str, graph: str, node: str, kind: str = "bitmap") -> str:
        """The file path of a pkg:///Group/name?dependency=uid resource (a bitmap, a font), added to the package's
        resources."""
        ident = url.split("?")[0].rsplit("/", 1)[-1]
        uid = url.split("dependency=")[-1] if "dependency=" in url else ""
        written_dep, owner = self._dependency(file, uid) if uid else ("", None)
        if written_dep.startswith("sbs://"):
            self.report.add(graph, node, f"{kind} resource {url} is in the library ({written_dep}), not available")
            return ""
        owner = owner or file
        res = next((r for r in self.file(owner).iter("resource") if text(r, "identifier") == ident), None) \
            if owner.is_file() else None
        if res is None:
            self.report.add(graph, node, f"{kind} resource {url} not found")
            return ""
        p = _local(owner, text(res, "source/externalcopy/filename") or text(res, "filepath"))
        if not p.is_file():
            self.report.add(graph, node, f"{kind} file not found: {p}")
        return self._add_resource(ident, kind, p)

    def _add_resource(self, ident: str, kind: str, p: Path) -> str:
        written = self._write_path(p)
        rid = _ident(ident)
        while rid in self.pkg.resources and self.pkg.resources[rid].path != written:
            rid += "_"
        self.pkg.resources[rid] = Resource(kind, written)
        return written

    def font(self, file: Path, spec: str, graph: str, node: str) -> str:
        """A Text node's font: a font resource (pkg:///...) as its file, a system font ("Family|Style") as the file
        found on this machine, copied into fonts/ next to the package when its folder is known."""
        if spec.startswith("pkg:///"):
            return self.resource(file, spec, graph, node, "font")
        found = fonts.find(*fonts.split_spec(spec)) if spec else None
        if found is None:
            self.report.add(graph, node, f"font {spec!r} not found on this system: the default font is used")
            return spec
        p, face = found
        if face:  # a face of a collection other than the first: kept as the system font's name
            return spec
        if self.base is not None:
            copy = self.base / "fonts" / p.name
            if not copy.is_file():
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, copy)
            p = copy
        return self._add_resource(p.stem, "font", p)


class _GraphImporter:
    """One Designer graph into our graph `g`."""

    def __init__(self, imp: _Importer, file: Path, el: ET.Element, g: Graph):
        self.imp, self.file, self.el, self.g = imp, file, el, g
        self.report = imp.report
        self.names: dict[str, str] = {}  # Designer graph input identifier -> ours
        self.nodes: dict[str, NodeInstance] = {}  # Designer node uid -> our node
        self.maps: dict[str, NodeMap | None] = {}  # Designer node uid -> its map (None: an instance or missing)
        self.outputs: dict[str, dict[str, str]] = {}  # Designer node uid -> output uid -> our output name
        self.scope: frozenset = frozenset()  # what the functions of the node being imported Set

    def note(self, node: str | None, message: str) -> None:
        self.report.add(self.g.id, node, message)

    def calls(self, url: str) -> tuple[str, str, dict]:
        return self.imp.call(self.file, url)

    def run(self) -> None:
        el, g = self.el, self.g
        g.label = text(el, "attributes/label")
        g.description = text(el, "attributes/description")
        for ours, sd in (("category", "category"), ("author", "author"), ("author_url", "authorURL"),
                         ("tags", "tags"), ("user_data", "usertags")):
            if text(el, f"attributes/{sd}"):
                g.attributes[ours] = text(el, f"attributes/{sd}")
        for o in el.findall("options/option"):
            if text(o, "name") == "defaultParentSize":  # "11x11": log2
                w, _, h = text(o, "value").partition("x")
                g.parent_size = [int(w), int(h or w)]
        self._inputs()
        nodes = list(el.findall("compNodes/compNode"))
        for n in nodes:
            self._node(n)
        for n in self._upstream_first(nodes):
            self._connections(n)
        for name, value in parameters(el.find("baseParameters")).items():
            self._base(g.base, name, value, None)
        self._items()

    # --- graph inputs ---------------------------------------------------------------------------

    def _inputs(self) -> None:
        for pi in self.el.findall("paraminputs/paraminput"):
            t = int(text(pi, "type", "0"))
            if t in IMAGE_TYPES:
                continue
            sd_id = text(pi, "identifier")
            ours = _ident(sd_id)
            kind = _KINDS.get(TYPES.get(t, ""))
            if kind is None:
                self.note(None, f"graph input {sd_id!r}: type {TYPES.get(t, t)} not supported")
                continue
            opts = {text(o, "name"): text(o, "value") for o in pi.findall("defaultWidget/options/option")}
            widget = text(pi, "defaultWidget/name")
            num = lambda k: float(opts[k]) if k in opts and _number(opts[k]) else None
            items = ()
            if widget == "dropdownlist" and kind is ParamKind.INT:
                parts = opts.get("parameters", "").split(";")[1:]  # the default, then value;label pairs
                items = tuple((int(parts[i]), parts[i + 1]) for i in range(0, len(parts) - 1, 2))
            default = constant(pi.find("defaultValue"))
            param = Param(kind, None, text(pi, "attributes/label") or sd_id, num("min"), num("max"),
                          group=text(pi, "group"), description=text(pi, "attributes/description"),
                          editor=_EDITORS.get(widget, ""), step=num("step"), clamp=opts.get("clamp") == "1",
                          user_data=text(pi, "attributes/usertag"), visible_if=text(pi, "visibleIf"), items=items)
            try:
                value = convert(default, param, ParamMap(ours)) if default is not None else _ZERO[kind]
                self.g.expose(ours, dataclasses.replace(param, default=value))
                self.names[sd_id] = ours
            except (DocumentError, ValueError) as e:
                self.note(None, f"graph input {sd_id!r}: {e}")

    # --- nodes ----------------------------------------------------------------------------------

    def _add(self, n: ET.Element, node: NodeInstance) -> None:
        base = node.id
        k = 1
        while f"{base}{k}" in self.g.nodes:
            k += 1
        node.id = f"{base}{k}"
        pos = text(n, "GUILayout/gpos", "0 0").split()
        node.position = [float(pos[0]), float(pos[1])]
        self.g.nodes[node.id] = node
        self.nodes[text(n, "uid")] = node

    def _node(self, n: ET.Element) -> None:
        impl = n.find("compImplementation")
        kind = impl[0] if impl is not None and len(impl) else None
        uid = text(n, "uid")
        outs = [text(o, "uid") for o in n.findall("compOutputs/compOutput")]
        if kind is None:
            return
        if kind.tag == "compInputBridge":
            self._input_node(n, kind)
        elif kind.tag == "compOutputBridge":
            self._output_node(n, kind)
        elif kind.tag == "compFilter":
            self.report.nodes += 1
            name = text(kind, "filter")
            nm = ATOMIC.get(name)
            if nm is None:
                # Designer's atomic nodes belong to no package: "sbs://atomic" names them.
                self._missing(n, f"sbs://atomic#{name}", f"atomic node {name!r} has no Trafilatura 3D counterpart",
                              kind, {o: "output" for o in outs})
                return
            self._mapped(n, nm, kind, {o: next(iter(NODES[nm.type].outputs)) for o in outs})
        elif kind.tag == "compInstance":
            self.report.nodes += 1
            self._instance(n, kind)
        else:
            self.note(None, f"node {uid}: {kind.tag} not supported")

    def _input_node(self, n: ET.Element, bridge: ET.Element) -> None:
        entry = text(bridge, "entry")
        pi = next((p for p in self.el.findall("paraminputs/paraminput") if text(p, "uid") == entry), None)
        if pi is None:
            self.note(None, f"input node {text(n, 'uid')}: its graph input is missing")
            return
        t = int(text(pi, "type", "1"))
        node = NodeInstance("input_color" if IMAGE_TYPES.get(t) == "color" else "input_grayscale",
                            "input_color" if IMAGE_TYPES.get(t) == "color" else "input_grayscale")
        node.params = {"identifier": _ident(text(pi, "identifier")), "label": text(pi, "attributes/label"),
                       "description": text(pi, "attributes/description"), "group": text(pi, "group")}
        if text(pi, "visibleIf"):
            node.params["visible_if"] = text(pi, "visibleIf")
        node.params = {k: v for k, v in node.params.items() if v}
        self._add(n, node)
        self.outputs[text(n, "uid")] = {text(o, "uid"): "output" for o in n.findall("compOutputs/compOutput")}

    def _output_node(self, n: ET.Element, bridge: ET.Element) -> None:
        uid = text(bridge, "output")
        go = next((o for o in self.el.findall("graphOutputs/graphoutput") if text(o, "uid") == uid), None)
        if go is None:
            self.note(None, f"output node {text(n, 'uid')}: its graph output is missing")
            return
        node = NodeInstance("output", "output")
        usage = text(go, "usages/usage/name")
        choices = NODES["output"].params["usage"].choices
        node.params = {"identifier": _ident(text(go, "identifier")), "label": text(go, "attributes/label"),
                       "description": text(go, "attributes/description"), "group": text(go, "group"),
                       "usage": usage if usage in choices else "", "visible_if": text(go, "visibleIf")}
        if usage and usage not in choices:
            self.note(None, f"output {text(go, 'identifier')!r}: usage {usage!r} not in ours")
        node.params = {k: v for k, v in node.params.items() if v}
        self._add(n, node)

    def _instance(self, n: ET.Element, inst: ET.Element) -> None:
        path = text(inst, "path")
        sd_gid = path.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
        uid = path.split("dependency=")[-1] if "dependency=" in path else ""
        written, dep = self.imp._dependency(self.file, uid)
        dep = dep or self.file
        bridges = {text(b, "uid"): text(b, "identifier") for b in inst.findall("outputBridgings/outputBridging")}
        try:
            library = written != "?himself" and self.imp._is_library(written, dep)
            gid = self.imp.graph(dep, sd_gid) if not library and dep.is_file() else None
        except DocumentError as e:  # a dependency that is not a readable .sbs
            self._missing(n, f"{written}#{sd_gid}", str(e), inst, bridges)
            return
        if library:
            nm = LIBRARY.get(f"{dep.name}#{sd_gid}")
            if nm is None:
                self._missing(n, f"{written}#{sd_gid}", f"library node {dep.name}#{sd_gid} has no Trafilatura 3D counterpart",
                              inst, bridges)
                return
            unmapped = [b for b in bridges.values() if b not in nm.outputs]
            for b in unmapped:
                self.note(None, f"{sd_gid}: output {b!r} not mapped")
            self._mapped(n, nm, inst, {u: nm.outputs[b] for u, b in bridges.items() if b in nm.outputs},
                         label=f"{dep.name}#{sd_gid}")
            return
        if not dep.is_file():
            self._missing(n, f"{written}#{sd_gid}", f"dependency not found: {dep}", inst, bridges)
            return
        if gid is None:
            self._missing(n, f"{written}#{sd_gid}", f"graph {sd_gid!r} not found in {dep.name}", inst, bridges)
            return
        self.report.mapped += 1
        node = NodeInstance(gid, INSTANCE, ref=gid)
        self._add(n, node)
        self.maps[text(n, "uid")] = None
        self.outputs[text(n, "uid")] = {u: _ident(b) for u, b in bridges.items()}
        sub = self.g.package.graphs[gid]
        # A graph being imported (a cycle through instances) has no parameters yet: its values stay unchecked.
        d = self.g.package.definition(node) if sub.nodes else None
        self.scope = self._sets(inst)
        for name, value in parameters(inst.find("parameters")).items():
            if name in BASE:
                self._base(node.base, name, value, node.id)
            elif d is not None and _ident(name) in d.params:
                self._value(node, d.params[_ident(name)], ParamMap(_ident(name)), value, name)
            else:
                self.report.params_unmapped += 1
                self.note(node.id, f"parameter {name!r} not in graph {gid!r}")

    def _missing(self, n: ET.Element, ref: str, why: str, impl: ET.Element, outputs: dict[str, str]) -> None:
        """A node without a counterpart: a graph instance with an unresolvable reference, its Designer values
        kept by their identifiers, its connections kept by the Designer port names."""
        self.report.missing += 1
        node = NodeInstance(_ident(ref.rsplit("#", 1)[-1]), INSTANCE, ref=ref)
        self._add(n, node)
        self.maps[text(n, "uid")] = None
        self.outputs[text(n, "uid")] = {u: _ident(o) for u, o in outputs.items()}
        # Designer resolves each output's color/grayscale and writes it: what this node gives downstream.
        types = {text(o, "uid"): IMAGE_TYPES.get(int(text(o, "comptype", "0")))
                 for o in n.findall("compOutputs/compOutput")}
        node.outputs = {_ident(o): types[u] for u, o in outputs.items() if types.get(u)}
        for name, value in parameters(impl.find("parameters")).items():
            if name in BASE:
                self._base(node.base, name, value, node.id)
            elif value.dynamic is not None:
                try:
                    node.functions[_ident(name)] = dynamic.function(value.dynamic, self.names, self.calls)
                except dynamic.Unsupported as e:
                    self.note(node.id, f"parameter {name!r}: function not kept: {e}")
            elif value.constant is not None:
                node.params[_ident(name)] = value.constant
        self.note(node.id, f"Missing node: {why}")

    def _mapped(self, n: ET.Element, nm: NodeMap, impl: ET.Element, outputs: dict[str, str], label: str = "") -> None:
        self.report.mapped += 1
        values = parameters(impl.find("parameters"))
        node = NodeInstance(nm.type, nm.type)
        count = self._numbered_count(n, nm)
        if count is not None:
            node.params["input_count"] = count
        self._add(n, node)
        self.maps[text(n, "uid")] = nm
        self.outputs[text(n, "uid")] = outputs
        d = self.g.package.definition(node)
        self.scope = self._sets(impl)
        for key, v in nm.defaults.items():
            node.params[key] = v
        if "output_format" not in getattr(d, "base_defaults", {}):  # not a format our node's data needs
            # The library graph's own format where our node's default differs; one the instance sets wins (below).
            default = d.base_defaults_for(d.values(node), {}).get("output_format")
            if nm.depth != (default["value"] if default else ""):
                node.base["output_format"] = {"mode": _MODES[ABSOLUTE], "value": nm.depth} if nm.depth \
                    else dict(NODE_DEFAULTS["output_format"])
        for name, value in values.items():
            if name in BASE:
                self._base(node.base, name, value, node.id)
            elif name in nm.params:
                pm = nm.params[name]
                self._value(node, d.params[pm.key], pm, value, name)
            else:
                self.report.params_unmapped += 1
                self.note(node.id, f"{label or text(impl, 'filter')}: parameter {name!r} not mapped")
        for arr in impl.findall("paramsArrays/paramsArray"):
            name = text(arr, "name")
            pm = nm.params.get(name)
            kind = d.params[pm.key].kind if pm is not None else None
            if kind not in (ParamKind.GRADIENT, ParamKind.CURVE):
                self.report.params_unmapped += 1
                self.note(node.id, f"parameter array {name!r} not mapped")
                continue
            cells = [parameters(c.find("parameters")) for c in arr.findall("paramsArrayCells/paramsArrayCell")]
            if kind is ParamKind.GRADIENT:  # keys: position, value (midpoint -1: none)
                keys = [{"pos": float(p["position"].constant if "position" in p else 0.0),
                         "color": _rgba(p["value"].constant if "value" in p else 0.0)} for p in cells]
            else:  # Bezier points: position with its left and right handles (absolute); a broken handle is unused, the
                # point itself (measured: the curves then match to 0.0001)
                def handle(p, side):
                    broken = f"is{side.capitalize()}Broken"
                    if side not in p or (broken in p and p[broken].constant):
                        return list(p["position"].constant)
                    return list(p[side].constant)
                keys = [list(p["position"].constant) + (handle(p, "left") + handle(p, "right")
                                                         if "left" in p or "right" in p else [])
                        for p in cells if "position" in p]
            if keys:
                node.params[pm.key] = d.params[pm.key].validate(keys)
        for pg in impl.findall("paramsGraphs/paramsGraph"):
            problems: list[str] = []
            node.params["fxmap"] = dynamic.fxmap(pg, self.names, self.g.function_variables(), problems, self.calls,
                                                 self.scope)
            for p in problems:
                self.note(node.id, p)

    def _sets(self, impl: ET.Element) -> frozenset:
        """What the functions of a node (parameters, base parameters, FX-Map) Set: each of them may Get it."""
        functions = []
        for dyn in impl.iter("dynamicValue"):
            try:
                functions.append(dynamic.function(dyn, self.names, self.calls))
            except dynamic.Unsupported:
                pass
        return set_variables(functions)

    def _numbered_count(self, n: ET.Element, nm: NodeMap) -> int | None:
        """How many numbered inputs a node uses (Pixel processor, FX-Map): the highest connected + 1."""
        sd, _ = nm.numbered
        if not sd:
            return None
        used = [int(p.partition(":")[2] or 0) for p in (text(c, "identifier") for c in n.findall("connections/connection"))
                if p.partition(":")[0] == sd]
        return max(used) + 1 if used else (1 if nm.type == "pixel_processor" else 0)

    # --- values ---------------------------------------------------------------------------------

    def _value(self, node: NodeInstance, param: Param, pm: ParamMap, value: Value, sd_name: str) -> None:
        """Set one of our parameters from a Designer value: a constant, a link to a graph input, or a function."""
        try:
            if value.dynamic is None:
                if param.kind is ParamKind.STRING and pm.key == "pkg_resource_path":
                    node.params[pm.key] = self.imp.resource(self.file, str(value.constant), self.g.id, node.id)
                elif node.type == "text" and pm.key == "font":
                    node.params[pm.key] = self.imp.font(self.file, str(value.constant), self.g.id, node.id)
                else:
                    node.params[pm.key] = convert(value.constant, param, pm)
                return
            f = dynamic.function(value.dynamic, self.names, self.calls)
            if param.kind is ParamKind.FUNCTION:
                FunctionGraph(f, {**self.g.function_variables(), "$pos": "float2"}, self.scope)
                node.params[pm.key] = f
                return
            exposed = dynamic.single_get(f)
            if exposed in self.g.parameters and self.g.parameters[exposed].param.kind is param.kind \
                    and pm.scale == 1.0 and (pm.values is None or all(param.choices.index(c) == a for a, c in pm.values.items())):
                node.links[pm.key] = exposed
                return
            if param.kind not in PARAM_TYPES:
                raise dynamic.Unsupported(f"a function on a {param.kind.value} parameter")
            f = dynamic.adapt(f, PARAM_TYPES[param.kind], self.g.function_variables(), param.kind is ParamKind.COLOR,
                              self.scope)
            if pm.scale != 1.0:
                f = dynamic.scaled(f, pm.scale)
            if pm.values is not None:
                f = dynamic.remap(f, {a: param.choices.index(c) for a, c in pm.values.items()})
            fg = FunctionGraph(f, self.g.function_variables(), self.scope)
            if fg.types[fg.output] != PARAM_TYPES[param.kind]:
                raise dynamic.Unsupported(f"gives a {fg.types[fg.output]}, the parameter takes a "
                                          f"{PARAM_TYPES[param.kind]}")
            node.functions[pm.key] = f
        except (dynamic.Unsupported, FunctionError, KeyError, ValueError, TypeError) as e:
            self.report.params_unmapped += 1
            what = "function" if value.dynamic is not None else f"value {value.constant!r}"
            self.note(node.id, f"parameter {sd_name!r} ({what}) not converted: {e}")

    def _base(self, target: dict, name: str, value: Value, node: str | None) -> None:
        key = BASE[name]
        mode = _MODES.get(value.relative if value.relative is not None else RELATIVE_INPUT, "relative_input")
        if node is None and mode == "relative_input":  # a graph has no input: Designer's graph values are its own
            mode = "relative_parent"
        if value.dynamic is not None:
            kind = BASE_PARAMS[key].kind
            try:
                if node is None or kind is None:
                    raise dynamic.Unsupported("a function on " + ("a graph's base parameter" if node is None
                                                                  else "this base parameter"))
                f = dynamic.function(value.dynamic, self.names, self.calls)
                f = dynamic.adapt(f, PARAM_TYPES[kind], self.g.function_variables(), outer=self.scope)
            except (dynamic.Unsupported, FunctionError) as e:
                self.report.params_unmapped += 1
                self.note(node, f"base parameter {name!r}: function not converted: {e}")
                return
            target[key] = {"mode": mode, "value": NODE_DEFAULTS[key]["value"], "function": f}
            return
        v = value.constant
        try:
            v = {"output_format": lambda x: FORMATS[int(x)], "pixel_ratio": lambda x: PIXEL_RATIOS[int(x)],
                 "tiling": lambda x: TILINGS[int(x)], "seed": int,
                 "output_size": lambda x: [int(c) for c in x], "pixel_size": lambda x: [float(c) for c in x]}[key](v)
        except (IndexError, TypeError, ValueError):
            self.report.params_unmapped += 1
            self.note(node, f"base parameter {name!r}: value {v!r} not converted")
            return
        target[key] = {"mode": mode, "value": v}

    # --- connections ----------------------------------------------------------------------------

    def _upstream_first(self, nodes: list[ET.Element]) -> list[ET.Element]:
        by_uid = {text(n, "uid"): n for n in nodes}
        order, seen = [], set()

        def visit(n):
            uid = text(n, "uid")
            if uid in seen:
                return
            seen.add(uid)
            for c in n.findall("connections/connection"):
                if text(c, "connRef") in by_uid:
                    visit(by_uid[text(c, "connRef")])
            order.append(n)
        for n in nodes:
            visit(n)
        return order

    def _connections(self, n: ET.Element) -> None:
        dst = self.nodes.get(text(n, "uid"))
        if dst is None:
            return
        nm = self.maps.get(text(n, "uid"))
        for c in n.findall("connections/connection"):
            port_sd, src_uid = text(c, "identifier"), text(c, "connRef")
            src = self.nodes.get(src_uid)
            if src is None:
                self.note(dst.id, f"input {port_sd!r}: its source node was not imported")
                continue
            outs = self.outputs.get(src_uid, {})
            out = outs.get(text(c, "connRefOutput")) or (next(iter(outs.values())) if len(outs) == 1 else None)
            if out is None:
                self.note(dst.id, f"input {port_sd!r}: output of {src.id} not mapped, connection dropped")
                continue
            port = self._port(dst, nm, port_sd)
            if port is None:
                self.note(dst.id, f"input {port_sd!r} not mapped, connection dropped")
                continue
            if self._broken(dst) or (self._broken(src) and out not in src.outputs):  # no types to check
                dst.inputs[port] = [src.id, out]
                continue
            try:
                self.g.connect(src.id, out, dst.id, port)
            except DocumentError as e:
                self.note(dst.id, f"connection {src.id}.{out} -> {port} dropped: {e}")

    def _port(self, dst: NodeInstance, nm: NodeMap | None, sd: str) -> str | None:
        if dst.type == "output":
            return "input"
        if nm is None:  # an instance of an imported graph (its inputs keep their identifiers) or a missing node
            return _ident(sd)
        prefix, _, index = sd.partition(":")
        if nm.numbered[0] and prefix == nm.numbered[0]:
            return f"{nm.numbered[1]}{int(index or 0)}"
        return nm.inputs.get(sd)

    def _broken(self, node: NodeInstance) -> bool:
        if node.type != INSTANCE:
            return False
        try:
            self.g.package.definition(node)
            return False
        except (DocumentError, OSError, ValueError):
            return True

    # --- frames, comments, pins -----------------------------------------------------------------

    def _items(self) -> None:
        for o in self.el.findall("GUIObjects/GUIObject"):
            kind = text(o, "type")
            pos = [float(x) for x in text(o, "GUILayout/gpos", "0 0").split()[:2]]
            size = [float(x) for x in text(o, "GUILayout/size", "0 0").split()[:2]]
            body = text(o, "GUIName")
            if kind == "PIN":
                self.g.add_item("pin", pos, title=text(o, "title") or body or "Pin")
            elif kind == "COMMENT" and text(o, "isFrameVisible") == "1":
                color = [float(x) for x in text(o, "frameColor", "0.25 0.45 0.25 0.35").split()]
                self.g.add_item("frame", pos, title=text(o, "title"), color=color, size=size, text=body)
            elif kind == "COMMENT":
                self.g.add_item("comment", pos, text=body, **({"size": size} if any(size) else {}))
            else:
                self.note(None, f"graph object {kind!r} not supported")


_ZERO = {ParamKind.BOOL: False, ParamKind.INT: 0, ParamKind.INT2: [0, 0], ParamKind.FLOAT: 0.0,
         ParamKind.FLOAT2: [0.0, 0.0], ParamKind.FLOAT3: [0.0, 0.0, 0.0], ParamKind.FLOAT4: [0.0, 0.0, 0.0, 0.0],
         ParamKind.STRING: ""}


def _local(file: Path, written: str) -> Path:
    """A path written in `file`: relative to its folder, or absolute (also a Windows one, from another machine)."""
    if re.match(r"[A-Za-z]:[/\\]", written) or Path(written).is_absolute():
        return Path(written)
    return (file.parent / written).resolve()


def _functions(root: ET.Element) -> list[ET.Element]:
    """The function graphs of a package (<function> items, not the function names of function nodes)."""
    return [f for f in root.iter("function") if f.find("identifier") is not None]


def _typed(value, t: str):
    """A Designer constant as a value of function type `t`."""
    cast = {"float": float, "int": int, "bool": bool, "string": str}[t.rstrip("234")]
    if t[-1].isdigit():
        return [cast(x) for x in (value if isinstance(value, list) else [value] * int(t[-1]))]
    return cast(value[0] if isinstance(value, list) else value)


def _rgba(v) -> list[float]:
    return [float(v)] * 3 + [1.0] if not isinstance(v, list) else [float(x) for x in v] + [1.0] * (4 - len(v))


def _number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
