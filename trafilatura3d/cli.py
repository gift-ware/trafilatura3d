"""Command line:
- `trafilatura3d cook package.trafilatura3d --out dir [--graph id] [--res N] [--preset name] [--set input=value]
  [--format png] [--pattern "$(graph)_$(identifier)"] [--output id]` (sbsrender render, with the Export
  outputs dialog's options: a preset, then input values, override the graph's);
- `trafilatura3d info package.trafilatura3d` (sbsrender info: graphs, inputs, outputs, presets);
- `trafilatura3d import file.sbs [--out file.trafilatura3d] [--search dir]` (a .sbs package converted, with the import
  report);
- `trafilatura3d bake low.obj --out dir --baker normal_map_from_mesh --high high.obj ...` (sbsbaker);
- `trafilatura3d new-paint mesh.fbx --out project.trafilatura3dpaint [--resolution 2048] [--template name]
  [--high high.fbx --match "By Mesh Name" --low-suffix _low --high-suffix _high]`: a Paint project on a mesh, as New
  Project makes it (one Texture Set per material), with the baking's high definition meshes already set;
- `trafilatura3d shortcut [--dir folder]`: a Desktop shortcut that starts the editor (editor/shortcut.py).
`--config file.trafilatura3dconfig` loads a project configuration (aliases, color management), as in the editor;
plugins load as in the editor, without its additions (`--no-plugins` skips them, `trafilatura3d plugins` lists them)."""

import argparse
import json
import subprocess
import sys
import time

from trafilatura3d.compiler import application_context
from trafilatura3d.package import SUFFIX, DocumentError, Package


def main(argv=None) -> int:
    from trafilatura3d import bake
    parser = argparse.ArgumentParser(prog="trafilatura3d")
    parser.add_argument("--config", help="project configuration (.trafilatura3dconfig)")
    parser.add_argument("--no-plugins", action="store_true", help="don't load plugins (see `trafilatura3d plugins`)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plugins", help="list the plugins found, their status and the search paths")
    sc = commands.add_parser("shortcut", help="create a Desktop shortcut that starts the editor")
    sc.add_argument("--dir", help="the folder for the shortcut (default: the Desktop)")

    cook = commands.add_parser("cook", help="cook a graph and write its outputs")
    cook.add_argument("package")
    cook.add_argument("--graph", help="graph id (default: the package's only graph)")
    cook.add_argument("--out", required=True, help="folder for the output files")
    cook.add_argument("--res", type=int, help="output size in pixels (default: the graph's)")
    cook.add_argument("--format", help="file format for every output (default: the project's Default image "
                                       "format, else png)")
    cook.add_argument("--pattern", default=None, help="file names: $(graph) $(identifier) $(description) $(label) "
                                                      "$(user_data) $(group) $(colorspace)")
    cook.add_argument("--output", action="append", default=[], metavar="IDENTIFIER",
                      help="export only this output; repeatable (default: all)")
    cook.add_argument("--preset", help="apply one of the graph's presets")
    cook.add_argument("--set", action="append", default=[], metavar="INPUT=VALUE",
                      help="an input parameter's value in JSON (roughness=0.4, tint=[1,0,0,1]); repeatable")
    cook.add_argument("--budget-mb", type=int, default=0,
                      help="GPU memory kept for cached results (default 0: free each result after its last use)")

    info = commands.add_parser("info", help="list a package's graphs, inputs, outputs and presets")
    info.add_argument("package")

    imp = commands.add_parser("import", help="convert a Substance Designer package (.sbs) into a .trafilatura3d package, "
                                             "or a .spp project into a Paint project, and print what could not be "
                                             "converted")
    imp.add_argument("package", help="the .sbs or .spp file")
    imp.add_argument("--out", help="the file to write (default: next to the source)")
    imp.add_argument("--search", action="append", default=[], metavar="DIR",
                     help="a folder (with its subfolders) where a dependency not found at its written path is looked "
                          "for by file name (repeatable)")

    npaint = commands.add_parser("new-paint", help="create a Paint project on a mesh, as New Project does")
    npaint.add_argument("mesh", help="the mesh to paint (FBX, OBJ, glTF, PLY, STL, USD)")
    npaint.add_argument("--out", required=True, help="the .trafilatura3dpaint file to write")
    npaint.add_argument("--resolution", type=int, default=2048, help="the Texture Sets' size")
    npaint.add_argument("--template", help="the template (shader): default the New Project's")
    npaint.add_argument("--high", action="append", default=[], help="high definition mesh for the baking of every "
                                                                    "Texture Set; repeatable")
    npaint.add_argument("--match", choices=bake.MATCH, default="Always")
    npaint.add_argument("--low-suffix", default="_low")
    npaint.add_argument("--high-suffix", default="_high")

    bk = commands.add_parser("bake", help="bake model information from meshes into textures")
    bk.add_argument("low", help="low poly mesh (FBX, OBJ, glTF, PLY, STL, USD)")
    bk.add_argument("--out", required=True, help="folder for the baked maps")
    bk.add_argument("--baker", action="append", required=True, choices=list(bake.BAKERS), help="repeatable")
    bk.add_argument("--high", action="append", default=[], help="high definition mesh; repeatable")
    bk.add_argument("--use-low-as-high", action="store_true")
    bk.add_argument("--size", type=int, default=2048)
    bk.add_argument("--format", choices=bake.FORMATS, default="png")
    bk.add_argument("--antialiasing", choices=list(bake.ANTIALIASING), default="None")
    bk.add_argument("--dilation", type=int, default=16, help="pixels")
    bk.add_argument("--no-diffusion", action="store_true")
    bk.add_argument("--no-average-normals", action="store_true")
    bk.add_argument("--cage", help="cage mesh (same topology as the low poly)")
    bk.add_argument("--frontal", type=float, default=0.01, help="max frontal distance")
    bk.add_argument("--rear", type=float, default=0.01, help="max rear distance")
    bk.add_argument("--absolute-distances", action="store_true",
                    help="distances in scene units, not relative to the bounding box")
    bk.add_argument("--match", choices=bake.MATCH, default="Always")
    bk.add_argument("--low-suffix", default="_low")
    bk.add_argument("--high-suffix", default="_high")
    bk.add_argument("--param", action="append", default=[], metavar="BAKER.NAME=VALUE",
                    help="a baker parameter in JSON (ambient_occlusion_from_mesh.secondary_rays=128); repeatable")
    bk.add_argument("--output-name", default=bake.NAME_TEMPLATE, help="file name template: $(mesh), $(bakername)")
    args = parser.parse_args(argv)

    if args.config:
        from trafilatura3d import project
        project.set_configuration(project.Configuration(args.config))
    if args.command == "shortcut":
        return _shortcut(args)
    from trafilatura3d import plugins
    if args.command == "plugins":
        return _plugins(plugins.manager())
    if not args.no_plugins:
        plugins.manager().load_all()  # without the editor: their host.editor is None
    try:
        if args.command == "info":
            return _info(args)
        if args.command == "import":
            return _import(args)
        if args.command == "bake":
            return _bake(args, bake)
        if args.command == "new-paint":
            return _new_paint(args)
        return _cook(args)
    except (DocumentError, OSError, ValueError) as e:
        source = getattr(args, "package", None) or getattr(args, "low", None) or getattr(args, "mesh", "")
        print(f"trafilatura3d: {source}: {e}", file=sys.stderr)
        return 1


def _plugins(m) -> int:
    for info in m.discover():
        state = "disabled" if not info.enabled else "enabled"
        print(f"{info.name}  {state}  {info.path}" + (f"  - {info.description}" if info.description else ""))
    print("search paths:\n  " + "\n  ".join(m.search_paths()))
    return 0


def _shortcut(args) -> int:
    try:
        from trafilatura3d.editor import shortcut
    except ImportError as e:  # PySide6 draws the icon: the editor's dependencies
        print(f"trafilatura3d: {e}: install the editor (pip install \"trafilatura3d[editor]\")", file=sys.stderr)
        return 1
    try:
        print(shortcut.create(args.dir))
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"trafilatura3d: {e}", file=sys.stderr)
        return 1
    return 0


def _opened(package):
    from trafilatura3d import plugins
    plugins.emit("package_opened", package=package)
    return package


def _value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text  # a plain string


def _cook(args) -> int:
    from trafilatura3d.evaluator import Evaluator
    from trafilatura3d.export import export
    package = _opened(Package.load(args.package))
    graph_id = args.graph or _only_graph(package)
    if graph_id not in package.graphs:
        raise DocumentError(f"no graph {graph_id!r} (graphs: {', '.join(package.graphs)})")
    graph = package.graphs[graph_id]
    values = {}
    if args.preset:
        preset = next((p for p in graph.presets if p["name"] == args.preset), None)
        if preset is None:
            raise DocumentError(f"no preset {args.preset!r} (presets: {', '.join(p['name'] for p in graph.presets)})")
        values.update(preset["values"])
    for item in args.set:
        name, sep, text = item.partition("=")
        if not sep:
            raise ValueError(f"--set {item!r}: expected INPUT=VALUE")
        if name not in graph.parameters:
            raise DocumentError(f"no input {name!r} (inputs: {', '.join(graph.parameters)})")
        values[name] = graph.parameters[name].param.validate(_value(text))
    parent = None
    if args.res:  # the application's Parent Size, as in the Graph view toolbar
        log2 = args.res.bit_length() - 1
        if args.res != 1 << log2:
            raise DocumentError("--res must be a power of two")
        parent = application_context((log2, log2))
    from trafilatura3d.export import PATTERN, output_attributes
    known = {a["identifier"] for a in output_attributes(package, graph_id)}
    missing = [o for o in args.output if o not in known]
    if missing:
        raise DocumentError(f"no output {missing[0]!r} (outputs: {', '.join(sorted(known))})")
    t0 = time.perf_counter()
    paths = export(package, graph_id, args.out, Evaluator(budget_bytes=args.budget_mb << 20), parent, values or None,
                   format=args.format, pattern=args.pattern or PATTERN, outputs=set(args.output) or None)
    for p in paths:
        print(p)
    print(f"{len(paths)} files, {time.perf_counter() - t0:.2f} s", file=sys.stderr)
    return 0


def _info(args) -> int:
    package = _opened(Package.load(args.package))
    for gid, g in package.graphs.items():
        print(f"graph {gid}")
        for pid, p in g.parameters.items():
            print(f"  input  {pid}  {p.param.kind.value}  default={json.dumps(p.param.default)}")
        for n in g.output_nodes():
            print(f"  output {g.identifier(n)}")
        for p in g.presets:
            print(f"  preset {p['name']}")
    return 0


def _import(args) -> int:
    from pathlib import Path

    if args.package.lower().endswith(".spp"):
        return _import_spp(args)
    from trafilatura3d.sbs import import_sbs
    out = Path(args.out) if args.out else Path(args.package).with_suffix(SUFFIX)
    package, report = import_sbs(args.package, search=tuple(args.search))
    package.save(out)
    print("\n".join(report.lines()))
    print(f"wrote {package.path}", file=sys.stderr)
    return 0


def _import_spp(args) -> int:
    from pathlib import Path

    from trafilatura3d.paint.model import PROJECT_SUFFIX
    from trafilatura3d.spp import import_spp
    out = Path(args.out) if args.out else Path(args.package).with_suffix(PROJECT_SUFFIX)
    session = import_spp(args.package, progress=lambda text: print(text, file=sys.stderr))
    out.resolve().parent.mkdir(parents=True, exist_ok=True)
    session.save(out)
    print("\n".join(session.load_warnings))
    print(f"wrote {session.path}", file=sys.stderr)
    return 0


def _new_paint(args) -> int:
    from pathlib import Path

    from trafilatura3d import shaders
    from trafilatura3d.paint.model import PROJECT_SUFFIX
    from trafilatura3d.paint.session import Session
    out = Path(args.out)
    if out.suffix != PROJECT_SUFFIX:
        raise ValueError(f"--out {out}: a {PROJECT_SUFFIX} file")
    if out.exists():
        raise ValueError(f"{out} already exists")
    template = args.template or shaders.DEFAULT_TEMPLATE
    if template not in shaders.TEMPLATES:
        raise ValueError(f"template {template!r} (templates: {', '.join(shaders.TEMPLATES)})")
    session = Session.new(args.mesh, resolution=args.resolution, template=template)
    if args.high:
        from trafilatura3d.paint import baking
        for ts in session.project.texture_sets:              # as Baking mode's High definition meshes stores them
            cfg = baking.config(ts)
            cfg["common"].update(high_meshes=[str(Path(h).resolve()) for h in args.high], match=args.match,
                                 low_suffix=args.low_suffix, high_suffix=args.high_suffix)
            baking.set_config(ts, cfg)
    out.resolve().parent.mkdir(parents=True, exist_ok=True)
    session.save(out)
    print(f"wrote {session.path}", file=sys.stderr)
    return 0


def _bake(args, bake) -> int:
    params = {b: {} for b in args.baker}
    for item in args.param:
        key, sep, text = item.partition("=")
        baker, dot, name = key.partition(".")
        if not (sep and dot) or baker not in params:
            raise ValueError(f"--param {item!r}: expected BAKER.NAME=VALUE for one of the --baker values")
        if name not in bake.BAKER_PARAMS.get(baker, {}):
            raise ValueError(f"--param {item!r}: {baker} parameters: {', '.join(bake.BAKER_PARAMS.get(baker, {}))}")
        params[baker][name] = _value(text)
    settings = bake.Settings(
        size=(args.size, args.size), format=args.format, antialiasing=args.antialiasing, dilation=args.dilation,
        apply_diffusion=not args.no_diffusion, average_normals=not args.no_average_normals, high_meshes=args.high,
        use_low_as_high=args.use_low_as_high, use_cage=bool(args.cage), cage_file=args.cage or "",
        frontal=args.frontal, rear=args.rear, relative_to_bounding_box=not args.absolute_distances, match=args.match,
        low_suffix=args.low_suffix, high_suffix=args.high_suffix)
    t0 = time.perf_counter()
    paths = bake.Baker().bake(args.low, params, settings, args.out, args.output_name)
    for p in paths:
        print(p)
    print(f"{len(paths)} files, {time.perf_counter() - t0:.2f} s", file=sys.stderr)
    return 0


def _only_graph(package: Package) -> str:
    if len(package.graphs) != 1:
        raise DocumentError(f"the package has {len(package.graphs)} graphs: choose one with --graph")
    return next(iter(package.graphs))


if __name__ == "__main__":
    sys.exit(main())
