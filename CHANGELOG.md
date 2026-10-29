# Changelog

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
