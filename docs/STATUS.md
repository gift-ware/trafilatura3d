# Status: work in progress

Trafilatura 3D is usable, but several parts are still being built. This page lists what is known to be missing or
incomplete, so that nobody mistakes a gap for a bug. File formats (`.trafilatura3d`, `.trafilatura3dpaint`) can
still change between versions: there is no backward compatibility yet.

## `.spp` importer (work in progress)

`trafilatura3d import project.spp` (or File > Open in the Paint workspace) reads the data of a `.spp` project and
rebuilds it as a Paint project: mesh, Texture Sets and UV tiles, layers and folders, fill layers, masks, effects
(fill, levels, filters and generators matched by name), embedded images, and the brush strokes, which are replayed
with our own brush (paint, eraser, projection, polygon fill, symmetry, pen pressure). What could not be converted is
listed in the import report (the Log panel). Known gaps:

- **Brush alphas.** Procedural brush alphas stored in compiled form cannot be read. A few common ones are rebuilt
  with our own generators (dirt, cracks, ink and dot alphas), shaped by their hardness, shape and invert parameters;
  every other one is replaced by a generic alpha of the same kind and named in the report. Mapping the complete set
  of procedural brush alphas is still to do, so strokes painted with them will look different from the original.
- **Per-stamp variation.** The original varies the alpha from one stamp to the next (random seeds, cycling stamps);
  we use one shape per stroke. Random roundness and flip of brush presets are not reproduced either.
- **Brush size.** The scale between the file's brush size and ours is estimated, not yet measured.
- **Procedural parameters.** Some wrapped procedurals and the mask editor keep part of their parameters unmapped.
- **Normal map format** of imported normal maps is assumed (OpenGL) when the file does not say.
- Results are close to the original, not identical: compared on the channels cached in the file, colors differ by
  about 5% on average.
- Importing a large project takes minutes, because every stroke is painted again.

## `.sbs` importer

- Library graphs without a counterpart in our library become broken instances, listed in the import report; the
  graph still opens and cooks with them disconnected.
- Our nodes follow the documented parameters and are tuned on the visible output, so imported materials look close
  to, not identical with, the original. Known larger differences: Leaks, Mask Builder, Sun Bleach.

## Graph workspace

Not offered yet: PSD export, font resources, an SVG editor in the 2D View, workspace presets, copy and paste inside
the Stamp Map editors, Dot portals. Opening a graph and selecting a node that is not cached still block the
interface while they cook.

## Paint workspace

Not offered yet: compare mask, color selection, smart masks, particles, tool and brush presets, automatic UV
unwrapping, project templates from other applications, a path-traced viewport, post effects, instancing, layer
flattening.

## Platforms and performance

- Tested on macOS (Apple Silicon, Metal) and Windows (NVIDIA, Vulkan). Linux is untested and has no prebuilt FBX
  library yet. There are no standalone installers yet: `install.sh` / `install.bat` install from a clone of the repository.
- A few generators give visibly different results on different GPUs (about 25 node outputs between Apple and
  NVIDIA); being investigated.
- Some nodes are slower than they should be on large graphs (Edge Detect, Distance, Slope Blur, Normal to Height HQ,
  Non Uniform Blur); a graph with thousands of chained nodes can exceed Python's recursion limit when compiling.
