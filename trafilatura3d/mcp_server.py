"""`trafilatura3d-mcp`: an MCP server (stdio) that lets an agent see and edit the graph open in the running editor.

Each tool sends one command through the editor's local channel (`trafilatura3d/channel.py`); the edits go
through the editor's document, so they show at once, undo as one step each, and the person at the editor sees
them happen. Needs the `mcp` extra; see docs/MCP.md."""

from __future__ import annotations

import base64
import json

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from trafilatura3d import channel

server = MCPServer("trafilatura3d", instructions=(
    "Edits the procedural texture graph open in the running Trafilatura 3D editor, live. Start with status() and "
    "get_graph(). Find node types with node_types(query) and their ports/parameters with node_info(type). Change "
    "the graph with edit(ops) (one undo step), tidy it with layout(), look at results with preview(node) and "
    "view3d(). `graph` arguments default to the graph shown in the editor."))


def _call(cmd: str, **args):
    try:
        return channel.call(cmd, {k: v for k, v in args.items() if v is not None})
    except channel.ChannelError as e:
        raise ToolError(str(e)) from None


def _json(value) -> str:
    return json.dumps(value, separators=(",", ":"))


@server.tool(structured_output=False)
def status() -> str:
    """Open packages (path, graphs, unsaved), the graph shown in the editor, the undo/redo step names."""
    return _json(_call("status"))


@server.tool(structured_output=False)
def open_package(path: str) -> str:
    """Open a .trafilatura3d package (a .sbs is imported) in the editor and show its first graph."""
    return _json(_call("open_package", path=path))


@server.tool(structured_output=False)
def new_package(template: str = "Metallic roughness", name: str = "New_graph") -> str:
    """A new unsaved package with one graph `name` made from a template, shown in the editor: "Empty", or a 3D View
    shader with its blending mode ("Metallic roughness", "OpenPBR - Alpha blend", "Specular glossiness - Alpha test",
    ...: an Output per channel fed a Uniform color, and the 3D View takes the shader)."""
    return _json(_call("new_package", template=template, name=name))


@server.tool(structured_output=False)
def save(path: str | None = None) -> str:
    """Save the shown graph's package (to `path`, required the first time; missing folders are created)."""
    return _json(_call("save", path=path))


@server.tool(structured_output=False)
def get_graph(graph: str | None = None) -> str:
    """The graph, compact: nodes {id: type, pos, params differing from the default, base, linked inputs},
    links "src.output -> dst.port", graph inputs, outputs with identifier and usage, warnings (if any)."""
    return _json(_call("get_graph", graph=graph))


@server.tool(structured_output=False)
def node_types(query: str = "") -> str:
    """Without a query, the node categories and their counts. With one: [type, label, category] of the nodes whose
    label or type matches (fuzzy) or whose category starts with it (at most 60)."""
    return _json(_call("node_types", query=query))


@server.tool(structured_output=False)
def node_info(type: str) -> str:
    """A node type's inputs, outputs (gray/color/any/value), parameters (kind: float, float2/3/4 = vectors of 2-4
    numbers, color, int, bool, enum...; default, range, choices) and the base parameters every node has (set_base:
    output_size, output_format...; modes and values). `inherits`: the default base mode, the input the size and
    format come from ("parent" when nothing is connected there) and the node's own defaults."""
    return _json(_call("node_info", type=type))


@server.tool(structured_output=False)
def edit(ops: list[dict], graph: str | None = None) -> str:
    """Apply a batch of edits as one undo step; each op is whole or reported and skipped. Ops ({"op": name, ...}):
    add_node(type, id?, position?, params?, ref?) -> its id | remove_node(node) |
    connect(from "node[.output]", to "node[.port]"; no port = first output / primary input) | disconnect(to) |
    set_param(node, param, value) or (node, params {name: value}); refused on a parameter linked to a graph input |
    set_base(node?, name, value {"mode", "value"}; no node = the graph's) | move(node, position) |
    expose(node, param, id?, label?, group?, range? [min, max], description?, editor?, kind?) -> graph input linked
    to it, of the parameter's kind (reply: id, kind, components, editor); a float2/3/4 also takes editor "Color" (3/4)
    or "Position" (2), or kind "float": one number driving every component (a float4's fourth kept) |
    set_input(id, default?, label?, range?, group?, description?, ...) changes a graph input | unexpose(id) |
    link(node, param, input?; no input = unlink) | set_output(node, identifier?, usage?, label?) |
    set_disabled(node, disabled) | set_graph(id?, label?, description?) renames the graph |
    insert_dot(to "node.port", position?) -> a Dot on that link |
    frame(nodes [ids], title?, color? [r, g, b, a], text?) -> a frame around those nodes, kept as its list (layout keeps
    them together and fits the frame again) | set_item(id, title?, color?, text?, nodes?, position?, size?) |
    remove_item(id).
    set_base with output_size replies the size in pixels (the values are log2: [11, 11] = 2048x2048).
    Returns {"results": one per op, {"ok": true, "id"?} or {"error"}; "warnings": the graph's warnings after the
    batch (as the Graph view lists them, e.g. outputs of different sizes)}."""
    return _json(_call("edit", ops=ops, graph=graph))


@server.tool(structured_output=False)
def layout(graph: str | None = None, dots: bool = False) -> str:
    """Arrange the graph so its links read well (columns from the outputs on the right, fewest crossings, links
    level, no overlaps), one undo step. `dots`: links that skip two columns or more get Dot nodes along their
    lane. A frame's nodes stay together and the frame is fitted around them again. Returns the overlaps, crossings
    and links passing over nodes left (with frames: frames overlapping, nodes inside a frame not theirs)."""
    return _json(_call("layout", graph=graph, dots=dots))


@server.tool(structured_output=False)
def focus(node: str, graph: str | None = None) -> str:
    """Show a node to the person at the editor: selected and framed in the Graph view, its parameters in
    Properties, its output in the 2D View."""
    return _json(_call("focus", node=node, graph=graph))


@server.tool(structured_output=False)
def undo() -> str:
    """Undo the editor's last step (any edit, also the person's)."""
    return _json(_call("undo"))


@server.tool(structured_output=False)
def redo() -> str:
    """Redo the editor's last undone step."""
    return _json(_call("redo"))


@server.tool(structured_output=False)
def preview(node: str, output: str | None = None, size: int = 256, graph: str | None = None) -> list:
    """A node output's cooked image as PNG (`size` px on the long side; keep it small unless detail matters)."""
    r = _call("preview", node=node, output=output, size=size, graph=graph)
    info = f"{node}: {r['size'][0]}x{r['size'][1]} {'gray' if r['gray'] else 'color'}"
    return [info, Image(data=base64.b64decode(r["png"]), format="png")]


@server.tool(structured_output=False)
def view3d(size: int = 512, mesh: str | None = None, viewpoint: str | None = None, yaw: float | None = None,
           pitch: float | None = None, distance: float | None = None, environment: str | None = None,
           env_rotation: float | None = None, env_intensity: float | None = None, key_light: float | None = None,
           light: dict | None = None) -> list:
    """The 3D View as PNG: the graph's material on the mesh, from the view's camera. The settings change the editor's
    3D View (as its menus do) and stay: mesh (Cube, Plane, Sphere, Rounded Cube, Cylinder, Torus, Mat Ball, Cloth...),
    viewpoint (Default, Front, Back, Left, Right, Top, Bottom), yaw/pitch (degrees) and distance (meters) to orbit,
    environment ("Studio", "Sun & Sky: morning|noon|sunset" or an HDRI file), env_rotation (degrees, turns the key
    light), env_intensity, key_light (key light intensity), light: point light 1 {azimuth, elevation (degrees; 5-15
    = grazing), distance, intensity, color, enabled}. The reply text gives the state the image was taken with."""
    r = _call("view3d", size=size, mesh=mesh, viewpoint=viewpoint, yaw=yaw, pitch=pitch, distance=distance,
              environment=environment, env_rotation=env_rotation, env_intensity=env_intensity, key_light=key_light,
              light=light)
    return [f"3D View of {r['graph']}: {_json(r['view'])}", Image(data=base64.b64decode(r["png"]), format="png")]


@server.tool(structured_output=False)
def timings(graph: str | None = None) -> str:
    """Cook time per node (ms, measured now on a cache of its own at the graph's size) and the total."""
    return _json(_call("timings", graph=graph))


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
