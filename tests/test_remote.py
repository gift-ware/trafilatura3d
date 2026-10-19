"""Remote control of the editor: the core's graph operations (batch edits, layout, description), the editor's
local channel driven by a client like the MCP bridge's (one undo step per batch, the views following), and the
MCP tools themselves."""

import base64
import math
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from trafilatura3d import channel, graphtools
from trafilatura3d.package import DocumentError, Package
from tests.scenes import MATERIAL_GRAPH, material_package


class GraphToolsTest(unittest.TestCase):
    def graph(self):
        g = Package().new_graph("g", 8)
        return g

    def test_ops_each_whole_or_reported(self):
        g = self.graph()
        res = graphtools.apply_ops(g, [
            {"op": "add_node", "type": "uniform_color", "id": "base", "params": {"output_color": [1, 0, 0, 1]}},
            {"op": "add_node", "type": "blur"},
            {"op": "connect", "from": "base", "to": "blur1"},
            {"op": "add_node", "type": "output", "params": {"usage": "baseColor"}},
            {"op": "connect", "from": "blur1.output", "to": "output1.input"},
            {"op": "add_node", "type": "no_such_node"},
            {"op": "set_param", "node": "blur1", "param": "no_such_param", "value": 1},
            {"op": "add_node", "type": "levels", "params": {"no_such_param": 1}},  # rolled back whole
            {"op": "connect", "from": "output1", "to": "base"},  # Output has no output; base no input
        ])
        self.assertEqual([r.get("id") for r in res[:4]], ["base", "blur1", None, "output1"])
        self.assertTrue(all(r.get("ok") for r in res[:5]))
        self.assertTrue(all("error" in r for r in res[5:]), res)
        self.assertEqual(set(g.nodes), {"base", "blur1", "output1"})  # the failed levels left nothing behind
        d = graphtools.describe(g)
        self.assertEqual(d["links"], ["base.output -> blur1.input", "blur1.output -> output1.input"])
        self.assertEqual(d["nodes"]["base"]["params"], {"output_color": [1.0, 0.0, 0.0, 1.0]})
        self.assertNotIn("params", d["nodes"]["blur1"])  # only what differs from the defaults
        self.assertEqual(d["outputs"], [{"node": "output1", "identifier": "output1", "usage": "baseColor"}])
        graphtools.apply_ops(g, [{"op": "expose", "node": "blur1", "param": next(iter(g.definition("blur1").params)),
                                  "id": "amount"}])
        self.assertIn("amount", graphtools.describe(g)["inputs"])

    def test_layout_columns_from_the_outputs(self):
        g = self.graph()
        graphtools.apply_ops(g, [
            {"op": "add_node", "type": "perlin_noise", "id": "noise"},
            {"op": "add_node", "type": "uniform_color", "id": "flat"},
            {"op": "add_node", "type": "blend", "id": "mix"},
            {"op": "add_node", "type": "normal", "id": "nrm"},
            {"op": "add_node", "type": "output", "id": "out_color"},
            {"op": "add_node", "type": "output", "id": "out_normal"},
            {"op": "connect", "from": "noise", "to": "mix.foreground"},
            {"op": "connect", "from": "flat", "to": "mix.background"},
            {"op": "connect", "from": "mix", "to": "out_color"},
            {"op": "connect", "from": "noise", "to": "nrm"},
            {"op": "connect", "from": "nrm", "to": "out_normal"},
        ])
        pos = graphtools.layout(g)
        x = {k: p[0] for k, p in pos.items()}
        self.assertEqual(x["out_color"], x["out_normal"])
        self.assertEqual(max(x.values()), x["out_color"])  # outputs on the right
        for n in g.nodes.values():  # every node left of what it feeds
            for src, _ in n.inputs.values():
                self.assertLess(x[src], x[n.id])
        self.assertEqual(x["noise"], x["mix"] - graphtools.COLUMN)  # noise feeds mix (column 1) and nrm
        for k1, p1 in pos.items():  # no overlaps
            for k2, p2 in pos.items():
                if k1 < k2 and p1[0] == p2[0]:
                    h1 = graphtools.node_size(*map(len, graphtools._ports(g, k1)))[1]
                    h2 = graphtools.node_size(*map(len, graphtools._ports(g, k2)))[1]
                    self.assertTrue(p1[1] + h1 <= p2[1] or p2[1] + h2 <= p1[1], (k1, k2))
        self.assertTrue(all(p[1] % graphtools.GRID == 0 for p in pos.values()))  # on the grid



    def test_layout_generated_graphs(self):
        """Chains, fans, diamonds, long links and random graphs: never an overlap, never more crossings than the
        column layout it replaced (`_column_layout`); with dots, the long links run through Dot nodes."""
        import random
        rng = random.Random(7)
        cases = {"chain": self.chain(8), "fan": self.fan(6), "diamond": self.diamond(4), "long": self.long_links(5)}
        cases.update({f"random{i}": self.random_graph(rng, 14 + 4 * i) for i in range(5)})
        for name, g in cases.items():
            with self.subTest(name):
                links = {(src, out, n.id, p) for n in g.nodes.values() for p, (src, out) in n.inputs.items()}
                _column_layout(g)
                old = graphtools.layout_stats(g)
                graphtools.layout(g)
                new = graphtools.layout_stats(g)
                self.assertEqual(new["overlaps"], 0, name)
                self.assertLessEqual(new["crossings"], old["crossings"], name)
                self.assertEqual(links, {(src, out, n.id, p) for n in g.nodes.values()
                                         for p, (src, out) in n.inputs.items()})  # positions only
                outs = [n for n in g.nodes.values() if n.type == "output"]
                self.assertEqual(len({n.position[0] for n in outs}), 1)
                self.assertEqual(max(n.position[0] for n in g.nodes.values()), outs[0].position[0])
                for n in g.nodes.values():
                    for src, _ in n.inputs.values():
                        self.assertLess(g.nodes[src].position[0], n.position[0])
                    self.assertEqual(n.position[1] % graphtools.GRID, 0)
                graphtools.layout(g, dots=True)
                with_dots = graphtools.layout_stats(g)
                self.assertEqual(with_dots["overlaps"], 0, name)
                self.assertLessEqual(with_dots["links_over_nodes"], new["links_over_nodes"], name)
                self.assertEqual(g.problems(), [])
        g = cases["long"]  # its long links now go through Dots, on their lane
        dots = [n for n in g.nodes.values() if n.type == graphtools.DOT_TYPE]
        self.assertTrue(dots)
        self.assertEqual(graphtools.layout_stats(g)["links_over_nodes"], 0)

    def chain(self, n):
        g = self.graph()
        ops = [{"op": "add_node", "type": "uniform_color", "id": "n0"}]
        ops += [{"op": "add_node", "type": "levels", "id": f"n{i}"} for i in range(1, n)]
        ops += [{"op": "connect", "from": f"n{i - 1}", "to": f"n{i}"} for i in range(1, n)]
        ops += [{"op": "add_node", "type": "output", "id": "out"}, {"op": "connect", "from": f"n{n - 1}", "to": "out"}]
        graphtools.apply_ops(g, ops)
        return g

    def fan(self, n):
        g = self.graph()
        ops = [{"op": "add_node", "type": "perlin_noise", "id": "src"}]
        for i in range(n):
            ops += [{"op": "add_node", "type": "levels", "id": f"l{i}"}, {"op": "connect", "from": "src", "to": f"l{i}"},
                    {"op": "add_node", "type": "output", "id": f"o{i}"}, {"op": "connect", "from": f"l{i}", "to": f"o{i}"}]
        graphtools.apply_ops(g, ops)
        return g

    def diamond(self, n):
        g = self.graph()
        ops = [{"op": "add_node", "type": "perlin_noise", "id": "d0"}]
        for i in range(n):
            ops += [{"op": "add_node", "type": "levels", "id": f"a{i}"}, {"op": "add_node", "type": "invert", "id": f"b{i}"},
                    {"op": "add_node", "type": "blend", "id": f"d{i + 1}"},
                    {"op": "connect", "from": f"d{i}", "to": f"a{i}"}, {"op": "connect", "from": f"d{i}", "to": f"b{i}"},
                    {"op": "connect", "from": f"a{i}", "to": f"d{i + 1}.foreground"},
                    {"op": "connect", "from": f"b{i}", "to": f"d{i + 1}.background"}]
        ops += [{"op": "add_node", "type": "output", "id": "out"}, {"op": "connect", "from": f"d{n}", "to": "out"}]
        graphtools.apply_ops(g, ops)
        return g

    def long_links(self, n):
        """A chain with every node also feeding a blend at the end: links across many columns."""
        g = self.chain(n)
        prev = f"n{n - 1}"
        ops = []
        for i in range(n - 2):
            ops += [{"op": "add_node", "type": "blend", "id": f"m{i}"}, {"op": "connect", "from": prev, "to": f"m{i}.background"},
                    {"op": "connect", "from": f"n{i}", "to": f"m{i}.foreground"}]
            prev = f"m{i}"
        ops += [{"op": "connect", "from": prev, "to": "out"}]
        self.assertTrue(all(r.get("ok") for r in graphtools.apply_ops(g, ops)))
        return g

    def random_graph(self, rng, n):
        """Random DAG of generators, filters and blends, a few outputs; some nodes feed nothing."""
        g = self.graph()
        ids = []
        for i in range(n):
            kind = "perlin_noise" if i < 3 or rng.random() < 0.15 else rng.choice(["levels", "invert", "blend", "blend"])
            nid = f"r{i}"
            graphtools.apply_ops(g, [{"op": "add_node", "type": kind, "id": nid}])
            for port in ([] if kind == "perlin_noise" else ["foreground", "background"] if kind == "blend" else [None]):
                src = rng.choice(ids)
                graphtools.apply_ops(g, [{"op": "connect", "from": src, "to": nid + (f".{port}" if port else "")}])
            ids.append(nid)
        for i, src in enumerate(rng.sample(ids[n // 2:], 3)):
            graphtools.apply_ops(g, [{"op": "add_node", "type": "output", "id": f"out{i}"},
                                     {"op": "connect", "from": src, "to": f"out{i}"}])
        return g

    def test_frames(self):
        """Frames around interleaved branches: the layout keeps each frame's nodes together, the frames apart."""
        g = self.fan(6)
        graphtools.layout(g)
        even = [k for i in range(0, 6, 2) for k in (f"l{i}", f"o{i}")]
        odd = [k for i in range(1, 6, 2) for k in (f"l{i}", f"o{i}")]
        res = graphtools.apply_ops(g, [{"op": "frame", "nodes": even, "title": "Even", "color": [0.2, 0.3, 0.6, 0.4]},
                                       {"op": "frame", "nodes": odd, "title": "Odd"},
                                       {"op": "frame", "nodes": ["l0"]},  # already in a frame
                                       {"op": "frame", "nodes": []},
                                       {"op": "remove_item", "id": "frame9"}])
        self.assertEqual([r.get("id") for r in res[:2]], ["frame1", "frame2"])
        self.assertTrue(all("error" in r for r in res[2:]), res)
        self.assertEqual(len(g.items), 2)
        self.assertGreater(graphtools.layout_stats(g)["frame_overlaps"], 0)  # interleaved branches: frames overlap
        graphtools.layout(g)
        stats = graphtools.layout_stats(g)
        self.assertEqual((stats["overlaps"], stats["frame_overlaps"], stats["nodes_in_other_frames"]), (0, 0, 0), stats)
        frames = {i["title"]: i for i in graphtools.describe(g)["items"]}
        self.assertEqual(sorted(frames["Even"]["nodes"]), sorted(even))
        for f in g.items:  # every member inside its frame, under the title bar
            fx, fy = f["position"]
            for k in f["nodes"]:
                x0, y0, x1, y1 = graphtools._node_rect(g, k)
                self.assertTrue(fx < x0 and x1 < fx + f["size"][0] and fy + graphtools.FRAME_BAR < y0
                                and y1 < fy + f["size"][1], (k, f))
        graphtools.apply_ops(g, [{"op": "set_item", "id": "frame2", "title": "Branches", "nodes": ["l1", "o1"]}])
        self.assertEqual(g.item("frame2")["title"], "Branches")
        self.assertEqual(graphtools.frame_members(g)["frame2"], ["l1", "o1"])

    def test_frames_in_frames(self):
        """A frame in a frame: kept inside it by the layout, fitted first; removing the outer one keeps the inner."""
        g = self.fan(6)
        res = graphtools.apply_ops(g, [
            {"op": "frame", "nodes": ["l0", "o0"], "title": "Inner"},
            {"op": "frame", "nodes": ["l4", "o4"], "frames": ["frame1"], "title": "Outer"},
            {"op": "frame", "nodes": ["l1", "o1", "l3", "o3"], "title": "Odd"},
            {"op": "frame", "nodes": ["l5"], "frames": ["frame1"]},  # already in a frame
            {"op": "set_item", "id": "frame1", "frames": ["frame2"]}])  # Outer holds Inner: a loop
        self.assertEqual([r.get("id") for r in res[:3]], ["frame1", "frame2", "frame3"])
        self.assertTrue(all("error" in r for r in res[3:]), res)
        graphtools.layout(g)
        stats = graphtools.layout_stats(g)
        self.assertEqual((stats["overlaps"], stats["frame_overlaps"], stats["nodes_in_other_frames"]), (0, 0, 0), stats)
        inner, outer = graphtools._frame_box(g.item("frame1")), graphtools._frame_box(g.item("frame2"))
        self.assertTrue(graphtools._inside(inner, outer), (inner, outer))
        items = {i["title"]: i for i in graphtools.describe(g)["items"]}
        self.assertEqual((items["Inner"]["in"], items["Outer"]["frames"]), ("frame2", ["frame1"]))
        graphtools.apply_ops(g, [{"op": "remove_item", "id": "frame2"}])
        self.assertIsNone(graphtools.frame_parents(g)["frame1"])
        self.assertEqual(sorted(graphtools.frame_members(g)["frame1"]), ["l0", "o0"])

    def test_frames_drawn_by_hand(self):
        """Frames without lists hold what lies in them, a frame in a frame included."""
        g = self.chain(2)
        x, y = g.nodes["n0"].position
        g.add_item("frame", [x - 300, y - 300], size=[800.0, 800.0])
        g.add_item("frame", [x - 20, y - 40], size=[200.0, 200.0])
        self.assertEqual(graphtools.frame_parents(g), {"frame1": None, "frame2": "frame1"})
        self.assertIn("n0", graphtools.frame_members(g)["frame2"])  # the smallest frame holding it
        self.assertNotIn("n0", graphtools.frame_members(g)["frame1"])

    def test_dot_ops(self):
        g = self.chain(3)
        res = graphtools.apply_ops(g, [{"op": "insert_dot", "to": "n2"}, {"op": "insert_dot", "to": "n1.nothing"}])
        dot = res[0]["id"]
        self.assertEqual(g.nodes[dot].type, "dot")
        self.assertEqual(g.nodes["n2"].inputs["input"], [dot, "output"])
        self.assertEqual(g.nodes[dot].inputs["input"], ["n1", "output"])
        self.assertIn("error", res[1])
        self.assertEqual(graphtools.node_size(1, 1, dot=True), (graphtools.DOT, graphtools.DOT))

    def test_graph_and_inputs_ops(self):
        pkg = Package()
        g = pkg.new_graph("g", 8)
        user = pkg.new_graph("user", 8)
        user.add_node("graph_instance", ref="g")
        res = graphtools.apply_ops(g, [
            {"op": "add_node", "type": "blur", "id": "lv"},
            {"op": "expose", "node": "lv", "param": "intensity", "id": "bright", "label": "Brightness",
             "range": [0.2, 0.9]},
            {"op": "expose", "node": "lv", "param": "intensity", "id": "bright"},  # the id is taken
            {"op": "set_param", "node": "lv", "param": "intensity", "value": 0.5},  # linked: refused
            {"op": "set_input", "id": "bright", "default": 0.7, "label": "Bright"},
            {"op": "set_input", "id": "bright", "colour": 1},
            {"op": "set_base", "name": "label", "value": "x"},
            {"op": "set_base", "node": "lv", "name": "size", "value": {"mode": "absolute", "value": [8, 8]}},
            {"op": "set_graph", "id": "renamed", "label": "Renamed"},
        ])
        self.assertTrue(all(r.get("ok") for r in (res[0], res[1], res[4], res[8])), res)
        self.assertIn("exists already", res[2]["error"])
        self.assertIn("set_input(id='bright'", res[3]["error"])
        self.assertIn("colour", res[5]["error"])
        self.assertIn("set_graph", res[6]["error"])
        self.assertIn("output_size", res[7]["error"])  # the base parameters are named
        p = g.parameters["bright"].param
        self.assertEqual((p.default, p.label, p.min, p.max), (0.7, "Bright", 0.2, 0.9))
        self.assertEqual((g.id, g.label, list(pkg.graphs)), ("renamed", "Renamed", ["renamed", "user"]))
        self.assertEqual(next(iter(user.nodes.values())).ref, "renamed")  # its instances follow
        info = graphtools.node_info(g.definition("lv"))
        self.assertEqual(info["base"]["output_format"]["choices"], ["8", "16", "16f", "32f"])
        self.assertIn("absolute", info["base"]["output_size"]["modes"])

    def test_node_info_says_where_the_size_comes_from(self):
        """node_info names the input a node inherits its size from, the one the compiler follows (Blend: the
        background, its main input, not the first), and the defaults of Input nodes and of nodes with their own."""
        from trafilatura3d.compiler import compile_graph
        from trafilatura3d.nodes import NODES
        self.assertEqual(graphtools.node_info(NODES["blend"])["inherits"],
                         {"mode": "relative_input", "from": "background", "unconnected": "parent"})
        self.assertEqual(graphtools.inheritance(NODES["input_color"]), {"mode": "relative_parent", "from": "parent"})
        self.assertEqual(graphtools.inheritance(NODES["flood_fill"])["output_format"]["value"], "32f")
        self.assertEqual(graphtools.inheritance(NODES["bitmap"])["output_size"]["mode"], "absolute")
        pkg = Package()
        g = pkg.new_graph("g", 6)
        small, big, b = g.add_node("test_pattern"), g.add_node("test_pattern"), g.add_node("blend")
        g.set_base(small, "output_size", {"mode": "absolute", "value": [4, 4]})
        g.connect(small, "output", b, "foreground")
        g.connect(big, "output", b, "background")
        self.assertEqual(compile_graph(pkg, "g").jobs[b].size, (64, 64))

    def test_expose_vector_parameters(self):
        """A vector parameter is exposed as such (the reply says its kind and size), with a Color editor if asked,
        or as one float driving every component through a function (a fourth component kept)."""
        from trafilatura3d.compiler import compile_graph
        pkg = Package()
        g = pkg.new_graph("g", 6)
        res = graphtools.apply_ops(g, [
            {"op": "add_node", "type": "uniform_color", "id": "c"},
            {"op": "add_node", "type": "levels", "id": "lv", "params": {"level_in_low": [0.1, 0.1, 0.1, 0.25]}},
            {"op": "add_node", "type": "output", "id": "out"},
            {"op": "connect", "from": "c", "to": "lv"}, {"op": "connect", "from": "lv", "to": "out"},
            {"op": "expose", "node": "lv", "param": "level_out_high", "id": "tint", "editor": "Color"},
            {"op": "expose", "node": "lv", "param": "level_in_low", "id": "low", "kind": "float", "range": [0, 0.8]},
            {"op": "expose", "node": "lv", "param": "level_in_high", "id": "high", "editor": "Angle"},
            {"op": "expose", "node": "lv", "param": "level_in_mid", "id": "mid", "kind": "int"},
            {"op": "set_input", "id": "low", "default": 0.3},
        ])
        self.assertEqual(res[5], {"ok": True, "id": "tint", "kind": "float4", "components": 4, "editor": "Color"})
        self.assertEqual({k: res[6][k] for k in ("kind", "components", "editor")},
                         {"kind": "float", "components": 1, "editor": "Slider"})
        self.assertIn("lv.level_in_low (float4)", res[6]["drives"])
        self.assertIn("Slider, Color", res[7]["error"])
        self.assertIn("as float4 or float", res[8]["error"])
        self.assertTrue(res[9]["ok"])
        self.assertEqual((g.parameters["low"].param.min, g.parameters["low"].param.max), (0, 0.8))
        self.assertNotIn("level_in_low", g.nodes["lv"].links)
        self.assertEqual(compile_graph(pkg, "g").jobs["lv"].values["level_in_low"], [0.3, 0.3, 0.3, 0.25])
        d = graphtools.describe(g)
        self.assertEqual(d["inputs"]["tint"]["editor"], "Color")
        self.assertEqual(d["nodes"]["lv"]["functions"], ["level_in_low"])
        self.assertEqual(graphtools.node_info(g.definition("lv"))["params"]["level_in_low"]["kind"], "float4")

    def test_link_float_input_to_vector_components(self):
        """A float input linked to a vector parameter drives it through a function, as expose(kind="float") does: every
        component (a float4 keeps its alpha) or one chosen component; another kind is refused with the reason."""
        from trafilatura3d.compiler import compile_graph
        pkg = Package()
        g = pkg.new_graph("g", 6)
        res = graphtools.apply_ops(g, [
            {"op": "add_node", "type": "uniform_color", "id": "c"},
            {"op": "add_node", "type": "blend", "id": "b"},
            {"op": "add_node", "type": "levels", "id": "lv", "params": {"level_in_low": [0.1, 0.2, 0.3, 0.25]}},
            {"op": "add_node", "type": "levels", "id": "lv2", "params": {"level_in_low": [0.1, 0.2, 0.3, 0.25]}},
            {"op": "connect", "from": "c", "to": "lv"}, {"op": "connect", "from": "c", "to": "lv2"},
            {"op": "expose", "node": "b", "param": "opacity", "id": "p"},
            {"op": "link", "node": "lv", "param": "level_in_low", "input": "p"},
            {"op": "link", "node": "lv2", "param": "level_in_low", "input": "p", "component": "g"},
            {"op": "link", "node": "lv2", "param": "level_in_high", "input": "p", "component": 4},
            {"op": "link", "node": "b", "param": "blending_mode", "input": "p"},
            {"op": "set_input", "id": "p", "default": 0.6},
        ])
        self.assertIn("by a function", res[7]["drives"])
        self.assertTrue(res[8]["ok"])
        self.assertIn("0..3", res[9]["error"])
        self.assertIn("same kind", res[10]["error"])
        jobs = compile_graph(pkg, "g").jobs
        self.assertEqual(jobs["lv"].values["level_in_low"], [0.6, 0.6, 0.6, 0.25])
        np.testing.assert_allclose(jobs["lv2"].values["level_in_low"], [0.1, 0.6, 0.3, 0.25], atol=1e-6)
        self.assertEqual(jobs["b"].values["opacity"], 0.6)

    def test_outputs_of_different_sizes_warn(self):
        """An output smaller than the others (an absolute size taken for pixels: [4, 4] is 16x16) is a graph warning,
        listed by get_graph's description; set_base on output_size replies the size in pixels."""
        from trafilatura3d.package import apply_template
        pkg = Package()
        g = pkg.new_graph("g", 11)
        apply_template(g, "Metallic roughness")
        outs = {g.identifier(n): n.id for n in g.output_nodes()}
        res = graphtools.apply_ops(g, [
            {"op": "add_node", "type": "uniform_color", "id": "c"},
            {"op": "add_node", "type": "uniform_color", "id": "m"},
            {"op": "set_base", "node": "m", "name": "output_size", "value": {"mode": "absolute", "value": [4, 4]}},
            {"op": "set_base", "node": "c", "name": "output_size",
             "value": {"mode": "relative_parent", "value": [-1, -1]}},
            *[{"op": "connect", "from": "c", "to": node} for ident, node in outs.items() if ident != "metallic"],
            {"op": "connect", "from": "m", "to": outs["metallic"]},
        ])
        self.assertEqual(res[2]["output_size"], "16x16 px")
        self.assertIn("x0.5", res[3]["output_size"])
        warnings = graphtools.describe(g)["warnings"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("metallic 16x16, the others 1024x1024", warnings[0])
        self.assertTrue(warnings[0].startswith(outs["metallic"]))

    def test_templates(self):
        """A template per shader and blending mode: an Output per channel of the shader (Opacity only when it sees
        through), each fed a Uniform color at the channel's neutral value; it returns the shader for the 3D View."""
        from trafilatura3d import shaders
        from trafilatura3d.compiler import compile_graph
        from trafilatura3d.package import TEMPLATES, apply_template
        self.assertEqual(TEMPLATES[0], "Empty")
        self.assertIn("Specular glossiness - Alpha blend", TEMPLATES)
        self.assertNotIn("Mesh Info", TEMPLATES)  # no channel to output
        pkg = Package()
        g = pkg.new_graph("g", 4)
        t = apply_template(g, "Metallic roughness")
        self.assertEqual((t.shader, t.blending), ("Metallic roughness", shaders.OPAQUE))
        usages = [o["usage"] for o in graphtools.describe(g)["outputs"]]
        self.assertEqual(usages, list(t.channels))
        self.assertNotIn("opacity", usages)
        from trafilatura3d.evaluator import Evaluator
        ev, c = Evaluator(), compile_graph(pkg, "g")
        usage_of = {o["identifier"]: o["usage"] for o in graphtools.describe(g)["outputs"]}
        for ident, (job, out) in c.outputs.items():  # the material shows the neutral values, not black
            want = shaders.neutral(t.shader, usage_of[ident])
            got = ev.read(job, out)[0, 0].astype(np.float64)
            got = got / 255 if got.max() > 1.0 else got
            n = 1 if shaders.channel_gray(t.shader, usage_of[ident]) else 3  # a grayscale output reads one value
            if n == 1:
                self.assertEqual(len(got), 1, ident)
            np.testing.assert_allclose(got[:n], want[:n], atol=1 / 255, err_msg=ident)
        g2 = pkg.new_graph("g2", 4)
        self.assertIn("opacity", apply_template(g2, "Specular glossiness - Alpha blend").channels)
        self.assertIsNone(apply_template(pkg.new_graph("g3", 4), "Empty"))
        with self.assertRaises(DocumentError):
            apply_template(g, "nothing")


def _column_layout(graph):
    """The layout the layered one replaced (columns by distance from the outputs, each node level with what it feeds,
    pushed below the one above): the baseline of the crossings."""
    import math
    consumers = {k: [] for k in graph.nodes}
    for n in graph.nodes.values():
        for port, (src, out) in n.inputs.items():
            consumers[src].append((n.id, port, out))
    depth = {}
    for k in reversed(graph.upstream(list(graph.nodes))):
        depth[k] = 1 + max((depth[d] for d, _, _ in consumers[k]), default=-1)
    ports = {k: graphtools._ports(graph, k) for k in graph.nodes}
    size = {k: graphtools.node_size(len(i), len(o)) for k, (i, o) in ports.items()}
    pos = {}
    for c in range(max(depth.values()) + 1):
        ids = [k for k in graph.nodes if depth[k] == c]
        target = {}
        for i, k in enumerate(ids):
            ys = [pos[d][1] + graphtools.port_y(ports[d][0].index(p)) - graphtools.port_y(ports[k][1].index(o))
                  for d, p, o in consumers[k] if d in pos and p in ports[d][0]]
            target[k] = (sum(ys) / len(ys), 0, i) if ys else (math.inf, graph.nodes[k].type != "output", i)
        y = -math.inf
        for k in sorted(ids, key=lambda k: target[k]):
            want = target[k][0] if target[k][0] != math.inf else (0.0 if y == -math.inf else y)
            top = math.ceil(max(want, y) / graphtools.GRID) * graphtools.GRID
            pos[k] = [-c * graphtools.COLUMN, float(top)]
            y = top + size[k][1] + graphtools.GAP
    for k, p in pos.items():
        graph.nodes[k].position = p


def pump(app, fn):
    """Run a blocking client call in a thread while the Qt loop serves it (the editor is this process)."""
    out = {}

    def run():
        try:
            out["value"] = fn()
        except Exception as e:  # noqa: BLE001 - raised in the test thread
            out["error"] = e
    t = threading.Thread(target=run)
    t.start()
    while t.is_alive():
        app.processEvents()
        t.join(0.002)
    if "error" in out:
        raise out["error"]
    return out["value"]


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class RemoteEditorTest(unittest.TestCase):
    def test_channel_and_bridge(self):
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication
        import rendercanvas.qt as rq
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        self.addCleanup(setattr, rq.QRenderWidget, "__init__", init)
        from trafilatura3d.editor.mainwindow import MainWindow
        from trafilatura3d.editor.remote import RemoteServer
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        unix = sys.platform != "win32"  # Windows: a named pipe, no file, gone with the editor that made it
        name = str(Path(tmp.name) / "channel") if unix else f"trafilatura3d-test-{os.getpid()}"  # not the user's
        if unix:
            import socket
            with socket.socket(socket.AF_UNIX) as orphan:  # a socket left by an editor that died: taken over
                orphan.bind(channel.address(name))
        w = MainWindow([str(material_package(Path(tmp.name) / "scene.trafilatura3d"))], channel=name)
        self.addCleanup(w.remote.close)
        self.addCleanup(w.graph.scene.selectionChanged.disconnect)  # no slots run while Python tears down
        self.addCleanup(w.doc.undo.indexChanged.disconnect)
        w.show()
        app.processEvents()
        self.assertTrue(w.remote.listening())
        second = RemoteServer(w, name)  # another editor on the same name stays off
        self.assertFalse(second.listening())
        if unix:
            self.assertTrue(Path(channel.address(name)).exists())
        call = lambda cmd, **args: pump(app, lambda: channel.call(cmd, args, channel=name))

        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QDialog
        dlg, got = QDialog(w), {}  # a modal dialog open: the channel still answers
        t = threading.Thread(target=lambda: got.update(r=channel.call("status", channel=name, timeout=20)))
        t.start()
        poll = QTimer(interval=5, timeout=lambda: t.is_alive() or dlg.accept())
        poll.start()
        dlg.exec()
        poll.stop()
        t.join()
        self.assertIn("packages", got["r"])

        st = call("status")
        self.assertEqual(st["current"]["graph"], MATERIAL_GRAPH)
        self.assertEqual(st["packages"][0]["graphs"], [MATERIAL_GRAPH])
        pkg = w.doc.packages[0]
        nodes = lambda: set(pkg.graphs[MATERIAL_GRAPH].nodes)  # undo rebuilds the graphs: look them up again
        before, steps = nodes(), w.doc.undo.count()
        res = call("edit", ops=[{"op": "add_node", "type": "blur", "id": "soft"},
                                {"op": "connect", "from": "perlin_noise1", "to": "soft"},
                                {"op": "connect", "from": "soft", "to": "normal1"},
                                {"op": "add_node", "type": "no_such_node"}])["results"]
        self.assertEqual(res[0], {"ok": True, "id": "soft"})
        self.assertIn("error", res[3])
        self.assertEqual(w.doc.undo.count(), steps + 1)  # the batch is one step
        self.assertEqual(nodes(), before | {"soft"})
        self.assertIn("soft", w.graph.scene.nodes)  # the Graph view follows
        g = call("get_graph")
        self.assertIn("perlin_noise1.output -> soft.input", g["links"])
        self.assertIn("soft.output -> normal1.input", g["links"])
        with self.assertRaises(channel.ChannelError):
            call("edit", ops=[], graph="no_such_graph")
        with self.assertRaises(channel.ChannelError):
            call("no_such_command")
        call("undo")
        self.assertEqual(nodes(), before)
        self.assertNotIn("soft", w.graph.scene.nodes)
        call("redo")
        self.assertEqual(nodes(), before | {"soft"})
        call("layout")
        self.assertEqual(w.doc.undo.count(), steps + 2)
        out_x = {pkg.graphs[MATERIAL_GRAPH].nodes[k].position[0] for k in nodes() if k.startswith("output")}
        self.assertEqual(len(out_x), 1)
        self.assertEqual(w.graph.scene.nodes["soft"].pos().x(), pkg.graphs[MATERIAL_GRAPH].nodes["soft"].position[0])
        p = call("preview", node="soft", size=64)
        img = QImage.fromData(base64.b64decode(p["png"]), "PNG")
        self.assertEqual((img.width(), img.height()), (64, 64))
        self.assertEqual(p["size"], [256, 256])
        v = call("view3d", size=64)
        self.assertEqual(QImage.fromData(base64.b64decode(v["png"]), "PNG").width(), 64)
        lit = call("view3d", size=64, mesh="Plane", viewpoint="Top", distance=3.0, environment="Sun & Sky: noon",
                   env_rotation=90, key_light=0.5, light={"azimuth": 90, "elevation": 10, "intensity": 30})
        self.assertEqual((lit["view"]["mesh"], lit["view"]["environment"]), ("Plane", "Sun & Sky: noon"))
        self.assertEqual((w.view3d.scene.mesh_name, w.view3d.view.look.light_intensity), ("Plane", 0.5))
        self.assertAlmostEqual(lit["view"]["pitch"], math.degrees(1.49), places=0)
        self.assertAlmostEqual(lit["view"]["env_rotation"], 90.0)
        p0 = w.view3d.scene.lights.points[0]
        self.assertTrue(p0.enabled)
        self.assertAlmostEqual(math.degrees(math.asin(p0.position[1] / math.dist(p0.position, (0, 0, 0)))), 10, 4)
        self.assertGreater(QImage.fromData(base64.b64decode(lit["png"]), "PNG").width(), 0)
        with self.assertRaises(channel.ChannelError):
            call("view3d", size=64, mesh="Teapot")
        t = call("timings")
        self.assertIn("soft", t["nodes"])
        self.assertGreaterEqual(t["total_ms"], 0)
        types = call("node_types", query="blur")
        self.assertIn("blur", [row[0] for row in types])
        self.assertIn("params", call("node_info", type="blur"))
        saved = Path(tmp.name) / "new" / "folder" / "saved.trafilatura3d"  # the folders are made
        self.assertEqual(call("save", path=str(saved))["path"], str(saved.resolve()))
        self.assertIn("soft", Package.load(saved).graphs[MATERIAL_GRAPH].nodes)

        # Alt+click on a link puts a Dot there (one step), drawn as a Dot.
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QGraphicsSceneMouseEvent
        from PySide6.QtCore import Qt
        from trafilatura3d.editor.graph.items import DotItem
        scene = w.graph.scene
        link = next(l for l in scene.links if l.dst.node.node_id == "normal1")
        at = link.mapToScene(link.path().pointAtPercent(0.5))
        steps = w.doc.undo.count()
        press = QGraphicsSceneMouseEvent(QEvent.GraphicsSceneMousePress)
        press.setScenePos(at)
        press.setButton(Qt.LeftButton)
        press.setModifiers(Qt.AltModifier)
        scene.mousePressEvent(press)
        app.processEvents()
        g = pkg.graphs[MATERIAL_GRAPH]
        dot = g.nodes["normal1"].inputs["input"][0]
        self.assertEqual(g.nodes[dot].type, "dot")
        self.assertEqual(g.nodes[dot].inputs["input"], ["soft", "output"])
        self.assertEqual(w.doc.undo.count(), steps + 1)
        self.assertIsInstance(scene.nodes[dot], DotItem)
        self.assertTrue(scene.nodes[dot].sceneBoundingRect().contains(at))
        call("undo")

        # A new package from a template, its graph renamed: the Graph view follows.
        made = call("new_package", name="brick")
        self.assertEqual([o["usage"] for o in made["outputs"]][:2], ["baseColor", "normal"])
        self.assertEqual(w.graph.graph_id, "brick")
        res = call("edit", ops=[{"op": "set_graph", "id": "wall", "label": "Wall"}])
        self.assertEqual(res, {"results": [{"ok": True}], "warnings": []})
        self.assertEqual((call("status")["current"]["graph"], w.graph.graph_id), ("wall", "wall"))
        with self.assertRaises(channel.ChannelError):
            call("new_package", template="nothing")
        self.assertEqual(call("layout", dots=True)["overlaps"], 0)
        from trafilatura3d import shaders  # a template's graph: the 3D View takes its shader and blending mode
        call("new_package", template="Specular glossiness - Alpha test", name="spec")
        m = w.view3d.scene.materials[0]
        self.assertEqual((m.shader, m.value("blending")), ("Specular glossiness", shaders.ALPHA_TEST))
        call("new_package", template="Metallic roughness", name="back")
        w.open_graph(pkg, MATERIAL_GRAPH)

        try:
            from trafilatura3d import mcp_server
        except ImportError:  # the mcp extra is not installed
            return
        import asyncio
        import logging
        from mcp.client.client import Client
        logging.getLogger("mcp").setLevel(logging.WARNING)  # the failing call below is expected
        os.environ["TRAFILATURA3D_CHANNEL"] = name
        self.addCleanup(os.environ.pop, "TRAFILATURA3D_CHANNEL", None)

        async def session():
            async with Client(mcp_server.server) as client:
                names = {t.name for t in (await client.list_tools()).tools}
                graph = await client.call_tool("get_graph", {})
                shot = await client.call_tool("preview", {"node": "soft", "size": 32})
                bad = await client.call_tool("preview", {"node": "nothing"})
                return names, graph, shot, bad
        names, graph, shot, bad = pump(app, lambda: asyncio.run(session()))
        self.assertTrue({"status", "edit", "layout", "preview", "view3d", "timings"} <= names)
        self.assertIn("soft", graph.content[0].text)
        self.assertEqual([c.type for c in shot.content], ["text", "image"])
        self.assertTrue(bad.is_error)
        for op in graphtools.OPS:  # the edit tool's description names every op
            self.assertIn(op, mcp_server.edit.__doc__)


class ChannelTest(unittest.TestCase):
    def test_no_editor(self):
        with self.assertRaises(channel.EditorNotRunning) as e:
            channel.call("status", channel=str(Path(tempfile.gettempdir()) / f"trafilatura3d-none-{os.getpid()}"))
        self.assertIn("open Trafilatura 3D first", str(e.exception))


if __name__ == "__main__":
    unittest.main()
