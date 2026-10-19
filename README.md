# Trafilatura 3D

An editor and engine for node-based procedural textures and for painting textures on 3D meshes, cooked on the
GPU through [wgpu](https://github.com/pygfx/wgpu-py) (Metal, Vulkan, D3D12). It reads `.sbs` files and `.spp`
projects (one way: they are imported into its own format) and follows the workflow familiar to users of node-based
texturing tools. It is a work in progress: what is still missing is listed in [docs/STATUS.md](docs/STATUS.md).
Two workspaces share one engine:

- **Graph**: packages (`.trafilatura3d`, JSON) hold graphs of atomic and library nodes, graph instances, exposed
  parameters, parameter functions, Pixel/Value processors and Stamp Maps; a 3D View with a rasterizer and a GPU
  path tracer, bakers from high poly meshes.
- **Paint**: projects (`.trafilatura3dpaint`) hold Texture Sets with layer stacks (paint, fill, folders, masks,
  effects, the library's filters and generators), brush, path and projection tools, baking and export.

The core is pure Python with no Qt and no project knowledge; the editor (PySide6) is an optional extra and only a
client of the core.

- `trafilatura3d/package.py`: packages, graphs, node instances, exposed parameters, checked edits.
- `trafilatura3d/compiler.py`: package to jobs (instances inlined, parameters and functions resolved, base
  parameters inherited); every job keyed by a hash of its content.
- `trafilatura3d/evaluator.py`: cooks jobs with a content-addressed cache (LRU budget; 0 = streaming), compute
  dispatches and render passes, value ports (1x1 textures), thumbnails.
- `trafilatura3d/functions.py`: function graphs (typing, CPU interpreter, WGSL code generation).
- `trafilatura3d/fxmap.py`: Stamp Map graphs (Quadrant / Iterate / Switch).
- `trafilatura3d/nodes/`: the node library, one class and one `.wgsl` per node (`atomic/`, `generators/`,
  `filters/`, `materials/`, `paint/`); the contract is in `docs/NODE_AUTHORING.md`.
- `trafilatura3d/sbs/`: the `.sbs` importer.
- `trafilatura3d/spp/`: the `.spp` importer (work in progress, see `docs/STATUS.md`).
- `trafilatura3d/paint/`: the Paint core (project model, layer stack compiled into jobs, painting, baking, export).
- `trafilatura3d/editor/`: the Graph and Paint workspaces.

## Install

Requirements: Python 3.10 or newer (tested with 3.12 and 3.14), git, and a GPU with Metal (macOS), Vulkan or
Direct3D 12 (Windows, Linux). There are no installers yet: the program runs from a copy of the repository.

macOS and Linux, in a terminal:

```
git clone https://github.com/gift-ware/trafilatura3d.git
cd trafilatura3d
./install.sh
```

Windows, in a Command Prompt:

```
git clone https://github.com/gift-ware/trafilatura3d.git
cd trafilatura3d
install.bat
```

The script creates a virtual environment in `.venv`, installs the editor in it, puts a shortcut on the Desktop and
makes the editor the application of `.trafilatura3d` packages and `.trafilatura3dpaint` projects (a double click opens
them). To update later: `git pull` in the folder, then the script again. The editor can also be started with
`.venv/bin/trafilatura3d-editor` (Windows: `.venv\Scripts\trafilatura3d-editor`).

The first start, and the first use of each node or tool, compile GPU shaders and build missing thumbnails: it can
take from several seconds to a few minutes on a large project (the splash screen shows what is being done). Later
starts reuse the driver's shader cache and are much faster.

## Commands

```
trafilatura3d cook package.trafilatura3d --out out/ [--res 4096] [--budget-mb 0]
trafilatura3d import material.sbs --out material.trafilatura3d
trafilatura3d import project.spp --out project.trafilatura3dpaint
trafilatura3d-editor [package.trafilatura3d]
trafilatura3d shortcut          # Desktop shortcut, documents open with a double click (install.sh does it)
python -m unittest discover tests
```

## License

MIT (see `LICENSE`). Third-party components and their licenses are listed in `NOTICE`; where the code and the
nodes' functional reference come from is described in `PROVENANCE.md`.

## Trademarks

Adobe, Substance 3D Designer and Substance 3D Painter are trademarks of Adobe Inc. Trafilatura 3D is an independent
project, not affiliated with, endorsed by or sponsored by Adobe; the names appear only to describe file
compatibility and interface conventions.
