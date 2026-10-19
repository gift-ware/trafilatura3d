# Driving the editor from an agent (MCP)

`trafilatura3d-mcp` is an MCP server (stdio) that lets an agent such as Claude Code see and edit the graph open
in the running editor, live: edits go through the editor's document, so the Graph view, thumbnails and
Properties follow at once, and every call is one step in Edit > Undo.

```
pip install -e ".[editor,mcp]"
claude mcp add --transport stdio trafilatura3d -- <path>/.venv/bin/trafilatura3d-mcp
```

Open Trafilatura 3D first: the editor listens on a local channel (`trafilatura3d/channel.py`: a unix socket in the
user folder, a named pipe on Windows), one per user; a second editor leaves it to the first and says so in the Log.
`TRAFILATURA3D_CHANNEL` names another channel (both sides read it).

## Tools

`graph` arguments default to the graph shown in the Graph view.

| Tool | |
|---|---|
| `status()` | open packages, the graph shown, undo/redo step names |
| `open_package(path)`, `save(path=None)` | as in the File menu; saving creates the missing folders of the path |
| `new_package(template="Metallic roughness", name="New_graph")` | as the New graph dialog with "New package": "Empty" or a shader template (an Output per channel fed a Uniform color, the 3D View takes the shader) |
| `get_graph(graph=None)` | compact: nodes (type, position, parameters differing from the default), links `src.out -> dst.port`, graph inputs, outputs with usage, `warnings` (the Graph view's) |
| `node_types(query="")` | categories; with a query, matching `[type, label, category]` |
| `node_info(type)` | ports, parameters (kind: `float`, `float2`/`float3`/`float4` vectors, `color`, `int`, `enum`...; default, range, choices) and the base parameters (`set_base`: modes, values); `inherits`: the default mode, the input the size and format come from (`main_input`, else the first image input; `parent` when nothing is connected there or the node has no image input) and the node's own defaults |
| `edit(ops, graph=None)` | `{results, warnings}`: a batch of ops, one undo step, the graph's warnings after it: `add_node`, `remove_node`, `connect`, `disconnect`, `set_param`, `set_base`, `move`, `expose`, `unexpose`, `link`, `set_output`, `set_disabled`, `set_graph`, `set_input`, `insert_dot`, `frame`, `set_item`, `remove_item`; a result per op |
| `layout(graph=None, dots=False)` | layered layout that favours the links (below); returns overlaps, crossings, links over nodes, and with frames `frame_overlaps`, `nodes_in_other_frames` |
| `undo()`, `redo()` | the editor's undo stack |
| `preview(node, output=None, size=256)` | the cooked output as PNG |
| `view3d(size=512, mesh?, viewpoint?, yaw?, pitch?, distance?, environment?, env_rotation?, env_intensity?, key_light?, light?)` | the 3D View (material on the mesh) as PNG; the settings change the editor's 3D View as its menus do (below) |
| `timings(graph=None)` | cook time per node, measured on a cache of its own |

Ops worth knowing:

- `set_param` on a parameter linked to a graph input is refused: the value used (cook, 3D View, instances) is the
  input's default, change it with `set_input(id, default=...)`, or unlink with `link(node, param)`.
- `expose(node, param, id?, label?, group?, range? [min, max], description?, editor?, kind?)` makes a graph input
  from a node parameter and links it; an id already taken is an error (link to it instead). The input has the
  parameter's kind, and the reply says it: `{"id", "kind", "components", "editor"}`. A vector parameter (Levels'
  `level_in_low` is a `float4`, one value per channel) gives a vector input; `editor` picks its control among the
  kind's (`float3`/`float4`: Slider, Color; `float2`: Slider, Position; a wrong one lists them). `kind="float"` on a
  vector makes a one-number input instead, which drives every component through a parameter function (a `float4`
  keeps its fourth, the alpha, as it is); the reply's `drives` says so and `get_graph` lists the parameter under
  the node's `functions`. `link(node, param, input?, component?)` links a parameter to an existing input of the same
  kind, or a `float` input to a vector parameter the same way: every component (a `float4` keeps its alpha) or only
  `component` (0-3 or x/y/z/w, r/g/b/a), the others kept at their values; no `input` unlinks. `set_input(id, ...)`
  changes an input's
  default, label, range (`range` or `min`/`max`), group, description, step, clamp, visible_if: what Properties edits.
- `set_graph(id?, label?, description?)` renames the graph (its instances in the package follow) and sets its label.
- `set_base(node?, name, {"mode": "relative_parent" | "relative_input" | "absolute", "value": ...})`: `node_info`
  lists the base parameters and their values. `output_size` values are **log2** (`[11, 11]` = 2048x2048, `[4, 4]` =
  16x16; relative modes add to the reference's): the reply says the size in pixels.
- Warnings (`get_graph`, `edit`) are the Graph view's, one path (`diagnostics.graph_warnings`): among them Output
  nodes with a usage whose size differs from the others' ("metallic 16x16, the others 2048x2048").
- `insert_dot(to "node.port", position?)`: a Dot node on that link, as Alt+click on a link in the Graph view.
- `frame(nodes, title?, color?, text?)`: a frame (the Graph view's titled box) fitted around those nodes, which it
  keeps as its list (a node in one such frame only); a frame drawn by hand in the editor holds the nodes whose centre
  is in it, as when it is dragged. `get_graph` lists the frames under `items` with their nodes. `set_item(id, title?,
  color?, text?, nodes?, position?, size?)` changes a frame (new `nodes`: fitted around them), comment or pin,
  `remove_item(id)` removes it. Group the graph by what each part makes (e.g. "Rocks", "Moss", "Water"), then
  `layout`: it keeps each frame's nodes together.

## 3D View

`view3d` sets the editor's 3D View through the same settings as its menus, and they stay after the call: `mesh` a
primitive (Cube, Plane, Plane (hi-res), Sphere, Cylinder, Hollow Box, Inner Box, Cloth, Mat Ball, Rounded Cube, Rounded
Cylinder, Sphere 2 tiles, Torus), `viewpoint` one of Camera > Viewpoints (Default, Front, Back, Left, Right, Top,
Bottom), `yaw`/`pitch` (degrees) and `distance` (meters) to orbit, `environment` from the Environment menu (Studio,
Sun & Sky: morning/noon/sunset) or an HDRI file, `env_rotation` (degrees; the key light turns with it),
`env_intensity`, `key_light` (the key light's intensity), and `light` = point light 1: `{azimuth, elevation, distance,
intensity, color, enabled}` (degrees; an elevation of 5-15 is a grazing light that shows the relief: dim the
environment, `env_intensity` ~0.1 and `key_light` 0, or the image-based light hides it). The reply gives the state
the image was taken with.

## Layout

`layout` is a layered layout where the links come first: columns by distance from the Output nodes (on the right),
a waypoint for a link in every column it crosses (a lane of its own), the order in each column that crosses least
(barycentres swept back and forth, then neighbour swaps), and heights that keep the links level using the nodes'
real sizes: never an overlap. With `dots=True` the waypoints of links skipping two columns or more become Dot nodes,
so long links are drawn along their lanes instead of across the nodes. With frames, each frame's nodes stay together in
every column and the frames keep one order in all columns, with a wider gap between frames; each frame is fitted
around its nodes again. A frame spanning many columns can still overlap another one or cover a node that isn't in it:
the reply counts both (`frame_overlaps`, `nodes_in_other_frames`).

The same operations are plain functions for scripts: `trafilatura3d.graphtools` (`apply_ops`, `layout`,
`layout_stats`, `insert_dot`, `describe`, `node_info`, `timings`), and `trafilatura3d.channel.call(cmd, args)` talks to a running editor.
