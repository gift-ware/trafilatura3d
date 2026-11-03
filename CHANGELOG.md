# Changelog

## 0.6.0 — 3 November 2026

### Brush alphas

- **A library of 416 brush alphas**, each a graph of our own nodes, under our own names
  (`trafilatura3d/paint/alpha_graphs/`, one package per category: basic, brush, celtic, clothing, cracks, dirt,
  fingerprint, footprint, geometric, handprint, leaks, medieval, organic, scratches, splats, stains, symbol, text).
  They sit in Assets > Alphas next to the generated ones; picked for a tool, the Alpha section of the Properties
  shows their parameters, and the values travel with the alpha (tools, paths): `library:<category>/<graph>?hardness=…`.
- **Alpha Shape** node (Paint/Alphas): the Hardness, Shape and Invert every library alpha ends with, plus Order and
  Threshold.
- **Imported `.spp` projects use them**: an alpha a project names is replayed with its library graph and the
  project's values and random seed (`trafilatura3d/spp/alphas.json`, the correspondence table). The alphas respond
  to Hardness, Shape and Invert as the alphas of imported projects do; noise alphas are close, drawn ones (stains,
  brush dabs, hand and finger prints, sign pictograms) are simpler silhouettes. Their other parameters are not
  exposed yet; alphas of older libraries keep their recipes or a generic alpha, named in the import report.

### `.spp` projects of newer versions

- **Document version 1** is read (saves and autosaves of newer releases): classes declared once, values without tags,
  128-bit channel masks; it gives the same tree as version 0, which reads as before.
- **JPEG 2000 textures**: the 16-bit textures of those projects are lossless JPEG 2000 tiles, decoded exactly (new
  dependency `imagecodecs`).

### Fixed

- **Mouse wheel in icon mode** (the Library, the Assets): a step scrolls rows of the grid, as the list mode scrolls
  lines; it moved a single icon, a fraction of a row.
- **Properties**: changing a library alpha's parameter no longer crashes (the panel was rebuilt inside the slider's
  own event).

### Dependencies

- `imagecodecs==2026.10.10`.

## 0.5.0 — 2 November 2026

### Long steps you can follow

- **Baking shows its progress.** The bake runs in a progress window: the maps tile by tile ("Normal Map from Mesh
  tile 37/64"), the map being baked composed on the model in the 3D View as its tiles arrive, Cancel at the next tile.
  Tiles are 1024 supersampled pixels (they were 2048: as fast, measured, and finer to follow). The Paint's bake
  follows its maps tile by tile in its own bar.
- **Exporting shows its progress**: compiling, cooking node by node, writing file by file, with Cancel; the Export
  outputs dialog uses it.
- Plugins: `EditorAccess.bake(...)` and `EditorAccess.export(...)` run a bake or an export in the same window (a
  pipeline's own bake and export commands no longer freeze the editor). The core functions take the callbacks:
  `Baker.bake(each_tile=...)`, `export(progress=...)`.

### Files and packages kept current

- **Linked files written from outside** (a bake, an image editor, a sync) reach the open packages: bitmaps, SVGs,
  fonts, input preview images and resources are watched, and the graphs reading them recompile, without an undo step
  and without marking the package unsaved. Files saved through a temporary file and renamed over are followed too.
- **Referenced packages written from outside** (not open in the editor) are read again, with what they link.
- **An open subgraph is the one its instances use**: a graph instance referring to a package open in the editor uses
  that package, so its edits (and their undo) reach the instances at once. Open reference goes to the open package's
  tab; it opened a second, stale copy of the same file before.

### Graph inputs

- **Levels inputs**: a float input can use the Editor "Levels". Five of them in one group (In low, In mid, In high,
  Out low, Out high) show on the graph's instances as one levels editor over the histogram of the image input named
  in "Histogram of" (what the instance receives there).
- **Swatches**: a color input can offer the colors of an image input of the graph (an ID map's): on its instances a
  picker lists the colors of the image connected there. The colors found leave out the blends along the edges.
- **Input preview images**: an Input color or grayscale with a PKG resource path reads that image when its graph is
  shown on its own (a subgraph previewed with real data); in an instance never. The image is a file of the package.
  Warnings of an instanced graph are computed as its instances compile it, without its preview images.

### MCP

- New tools `new_graph`, `remove_graph`, `remove_resource`.
- New edit ops `set_function` (drive a parameter by a function graph), `set_preset` / `remove_preset` (embedded
  presets of a graph's inputs); `set_graph` takes `parent_size` (in pixels).

### Graph view

- Frames show their properties (title, description, color), comments and pins theirs.
- Moving a selection that holds frames, then duplicating a node, no longer puts the frames back where they were.
- A selection box selects a frame only when the frame lies fully inside it.
- Copy and paste between packages (a graph of a library into another package): the pasted nodes keep their
  references and files.
- Selecting many nodes is faster (the warnings of the newly selected nodes are checked once the selection settles).

### Fixed

- The Baking window stays above the main window (it could fall behind it): non-modal windows of the editor are tool
  windows of the main window (`widgets.dependent`).
- The bakers' normal orientation defaults to the project's normal format.

### Breaking changes

- Plugins: one pair of save events for both workspaces. `before_save(document, path)` and `document_saved(document,
  path)` are heard when a package or a Paint project is saved (`package_saved(package, path)` is gone, and
  `before_save` takes `document` instead of `package`). A Paint project's save can now be refused with `Veto`.
- Plugins: `EditorAccess` has two more methods, `bake` and `export` (an implementation of its own must add them).
- A Paint project saved over a read-only file is refused, as a package is (it was replaced through its temporary
  file).
- A package that uses an alias (`name://`) no loaded project defines does not open: the message names the aliases
  and says to open it with its project's configuration.

## 0.4.0 — 30 October 2026

### New

- Plugins: `show_scene(package, resource_id)` shows a package's 3D scene resource in the 3D View (as a double click on
  it in the Explorer).

### Breaking changes

- A package's dependency folder is `<package name>.dependencies/` next to it (it was `dependencies/<package name>/`):
  one folder per package, beside it. Files left in an old folder stay linked where they are.
- Paint autosaves follow the Graph's convention: `.autosave/<project>_<date>_<time>.trafilatura3dpaint` next to the
  project (they were `<project>_autosave_<n>` beside it), the last ones kept; cleaning a document's autosaves leaves
  those of another document whose name starts the same.

## 0.3.0 — 29 October 2026

### The interface never waits

- **No GPU waits on the interface thread.** Selecting, moving and editing nodes no longer stall while a cook runs
  (stalls of up to 1.4 s before). The evaluator's fences are completion callbacks collected by a non-blocking poll;
  no GPU promise is left pending (one held the device, and every frame, histogram and texture release of the
  interface waited for the whole GPU work).
- **Opening a package** shows the window at once (1 s instead of 42 s on a production graph): the 3D View shows the
  channels already cooked and the others arrive one by one as the refresh computes them.
- **SVG and Text nodes** rasterize in a process of their own (the rasterizer held Python's lock for its whole render:
  5 s for a 4096 mask). Same result, the interface answers meanwhile.
- **Histograms and read-backs as side cooks**: the Levels histogram (Graph and Paint, Auto level included), the 2D
  View's histogram, its pixel readout and Copy image are computed in steps after the refresh and shown when ready;
  the last histograms are remembered.
- **Warnings only where you work**: a package is fully checked when it opens, reloads or after Relocate; after that
  only the nodes being worked on (the selection, the ends of a link made, moved or removed) are checked again.
  Tools > Live warnings (off by default) checks the whole package after every edit. Structure and type problems now
  carry the node they are about.
- **Dragging a value** previews at a lower resolution only what the drag changes; everything else stays at full
  resolution, already cooked (dragging a value of a node feeding nothing shown recooked a whole graph before).
- Longest interface stall on a stress test (selections, moves, edits, undo): 5.5 s → 0.15 s.

### Faster

- **Parameter functions are compiled** once into closures instead of being interpreted node by node: FX-Map walks are
  3.7× faster (the slowest step 106 ms → 27 ms), with identical results (same rules, same random numbers).

### Graph view

- A link released on a Dot's disc connects to it.
- Frames are grabbed by their title bar and border only: a drag inside a frame draws a selection box. Frames and
  comments resize from every side and corner.
- Shift adds to the selection (clicks and selection boxes).
- Dragging nodes, Dots, frames or a selection box near the view's border scrolls the view.
- The 2D View keeps its zoom when only the resolution of the image changes (a dragged value's preview).
- Explorer: Remove on a linked resource (refused while nodes use it).

### Nodes

- **New HCL node** (Filters/Adjustments): hue, chroma and luma adjustments, absolute or proportional.
- **Light**: the light direction and the highlight follow the documented angles, within 0.01 of the reference output.
- **Shadows**: the shadow is how much the heights along each ray stand above the pixel; continuous heightmaps no longer
  come out grey; refitted (mean error 0.018 → 0.005).
- **Normal To Height HQ**: the relief bands weigh half (no clipping); Relief Balance runs from broad forms to details.

### Packages and files

- A saved package keeps the files it uses from elsewhere in its own folder, `dependencies/<package name>/`, with
  relative paths; Save As takes them along. Resources embedded in imported `.sbs` files (SVGs) are extracted.
- A graph's instances follow its inputs when they are renamed or removed.
- Bake configurations store their paths relative to the document.

### Command line and plugins

- `trafilatura3d new-paint mesh --out file` creates a Paint project (with `--high`, `--match`, `--low-suffix`,
  `--high-suffix` for baking).
- Plugins: `edit()` works on the Paint project, `save(document, path)` saves at a new path.
- MCP: `focus(node)` shows a node to the person at the editor.

### Fixed

- Selecting or moving nodes stalled the interface for up to 1.4 s while a cook ran.
- Opening a package froze the window until the whole graph was cooked.
- Selecting a Levels node stalled while its histogram was cooked.
- Dragging a value of a node feeding nothing shown recooked the whole graph at the preview resolution.
- The 2D View fitted the image again (losing the zoom) at every parameter change during a drag.
- A link could not be connected to a Dot's input, only from its output.
- A drag inside a frame moved the frame instead of drawing a selection box; frames resized only from their
  bottom-right corner.
- Shift did not add nodes to the selection.
- The view did not scroll at its border while nodes, Dots or frames were dragged.
- A linked resource (a mesh, a bitmap) could not be removed from a package.
- A graph's instances kept an input renamed or removed in the graph, and reported a parameter it no longer has.
- Shadows came out grey all over a continuous heightmap; Normal To Height HQ clipped at 0 and 1.
- Moving a project or a package broke its baking paths (high meshes, cage, output folder): they are stored relative
  to the document.
- The About box and the status bar showed version 0.1.1.

### Breaking changes

- `import_sbs` has no `base` argument: imported files are referenced by absolute paths, gathered into the package's
  dependency folder when it is saved.
- Saving a package now copies the files it uses from outside its folder into `dependencies/<package name>/` and
  rewrites their paths relative to it.

## 0.2.0 — 24 October 2026

- Project libraries (shaders, export presets, materials, smart materials) declared by the project file and read by
  every workspace; library shaders written in WGSL with a small header.
- Plugin API for pipelines: menus in both workspaces, access to the current documents, export events.

## 0.1.1 — 24 October 2026

- Paint responsiveness: the layer stack updated row by row, panels rebuilt only when what they show changes.
- Baking: per-map log and visualization. Polygon Fill: rectangle, fill modes, visible polygons only.

## 0.1.0 — 19 October 2026

- First public release.
