"""Import of .sbs packages: small packages written here (no third-party .sbs file in the repository) cover the
atomic and library mappings, connections, values, exposed parameters and their links, functions, FX-Map, base
parameters, frames, bitmaps, user graphs of other files, missing nodes kept wired and reported, the CLI and the
editor's File > Open."""

import html
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from trafilatura3d.compiler import application_context, compile_graph
from trafilatura3d.diagnostics import dependency_warnings, graph_warnings
from trafilatura3d.nodes.base import PortType
from trafilatura3d.package import INSTANCE, Package
from trafilatura3d.sbs import import_sbs


def _v(kind, value):
    return f'<paramValue><constantValue{kind} v="{value}"/></paramValue>'


def param(name, kind, value, relative=None):
    rel = f'<relativeTo v="{relative}"/>' if relative is not None else ""
    return f'<parameter><name v="{name}"/>{rel}{_v(kind, value)}</parameter>'


def fn_param(name, nodes, root, relative=None):
    rel = f'<relativeTo v="{relative}"/>' if relative is not None else ""
    return (f'<parameter><name v="{name}"/>{rel}<paramValue><dynamicValue><rootnode v="{root}"/><paramNodes>{nodes}'
            f'</paramNodes></dynamicValue></paramValue></parameter>')


def fn_node(uid, function, type, data=None, conns=()):
    datas = (f'<funcDatas><funcData><name v="{function}"/><constantValue><constantValue{data[0]} v="{data[1]}"/>'
             f'</constantValue></funcData></funcDatas>') if data else ""
    cs = "".join(f'<connection><identifier v="{p}"/><connRef v="{src}"/></connection>' for p, src in conns)
    return (f'<paramNode><uid v="{uid}"/><function v="{function}"/><type v="{type}"/>'
            f'{"<connections>" + cs + "</connections>" if cs else ""}{datas}</paramNode>')


def _conns(conns):
    cs = "".join(f'<connection><identifier v="{p}"/><connRef v="{src}"/><connRefOutput v="{out}"/></connection>'
                 for p, (src, out) in conns.items())
    return f"<connections>{cs}</connections>" if cs else ""


def comp_filter(uid, filter, params="", conns=None, pos=(0, 0), extra=""):
    return (f'<compNode><uid v="{uid}"/><GUILayout><gpos v="{pos[0]} {pos[1]} 0"/></GUILayout>'
            f'<compOutputs><compOutput><uid v="{uid}0"/><comptype v="2"/></compOutput></compOutputs>{_conns(conns or {})}'
            f'<compImplementation><compFilter><filter v="{filter}"/><parameters>{params}</parameters>{extra}'
            f'</compFilter></compImplementation></compNode>')


def comp_instance(uid, path, dep, outputs, params="", conns=None, pos=(0, 0), comptype=2):
    outs = "".join(f'<compOutput><uid v="{uid}{i}"/><comptype v="{comptype}"/></compOutput>'
                   for i in range(len(outputs)))
    bridges = "".join(f'<outputBridging><uid v="{uid}{i}"/><identifier v="{o}"/></outputBridging>'
                      for i, o in enumerate(outputs))
    return (f'<compNode><uid v="{uid}"/><GUILayout><gpos v="{pos[0]} {pos[1]} 0"/></GUILayout>'
            f'<compOutputs>{outs}</compOutputs>{_conns(conns or {})}<compImplementation><compInstance>'
            f'<path v="pkg:///{path}?dependency={dep}"/><parameters>{params}</parameters>'
            f'<outputBridgings>{bridges}</outputBridgings></compInstance></compImplementation></compNode>')


def output_node(uid, output_uid, src):
    return (f'<compNode><uid v="{uid}"/><GUILayout><gpos v="500 0 0"/></GUILayout>'
            f'{_conns({"inputNodeOutput": src})}<compImplementation><compOutputBridge><output v="{output_uid}"/>'
            f'</compOutputBridge></compImplementation></compNode>')


def graph(identifier, nodes, outputs=(), inputs="", objects="", base=""):
    outs = "".join(f'<graphoutput><identifier v="{o}"/><uid v="{u}"/><usages><usage><name v="{usage}"/></usage>'
                   f'</usages></graphoutput>' for o, u, usage in outputs)
    return (f'<graph><identifier v="{identifier}"/><uid v="1"/><paraminputs>{inputs}</paraminputs>'
            f'<graphOutputs>{outs}</graphOutputs><compNodes>{nodes}</compNodes><baseParameters>{base}</baseParameters>'
            f'<GUIObjects>{objects}</GUIObjects><options><option><name v="defaultParentSize"/><value v="9x9"/>'
            f'</option></options><root/></graph>')


def package(deps, content):
    ds = "".join(f'<dependency><filename v="{f}"/><uid v="{u}"/><type v="package"/></dependency>' for f, u in deps)
    return f'<?xml version="1.0" encoding="UTF-8"?><package><dependencies>{ds}</dependencies><content>{content}</content></package>'


FLOAT_INPUT = ('<paraminput><identifier v="amount"/><uid v="900"/><attributes><label v="Amount"/></attributes>'
               '<type v="256"/><defaultValue><constantValueFloat1 v="0.25"/></defaultValue><defaultWidget>'
               '<name v="slider"/><options><option><name v="min"/><value v="0"/></option><option><name v="max"/>'
               '<value v="2"/></option></options></defaultWidget><group v="Look"/></paraminput>')


class ImportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def test_library_graphs_keep_their_own_depth(self):
        # Some library graphs give 16 bits in an 8-bit graph whatever their Output Format says (sbs/depths.json):
        # Perlin Noise does, Perlin Noise v2 and Invert follow what they inherit; a format set on the instance wins.
        nodes = (comp_instance("10", "perlin_noise", "50", ["output"], pos=(0, 0)) +
                 comp_instance("11", "perlin_noise", "51", ["output"], pos=(0, 200)) +
                 comp_instance("12", "invert_grayscale", "52", ["Invert_Grayscale"], conns={"Source": ("10", "100")}) +
                 comp_instance("13", "perlin_noise", "50", ["output"], param("format", "Int32", 0, relative=0)))
        path = self.write("depth.sbs", package([("sbs://noise_perlin_noise.sbs", "50"),
                                                ("sbs://noise_perlin_noise_v2.sbs", "51"), ("sbs://invert.sbs", "52")],
                                               graph("Depth", nodes)))
        g = import_sbs(path)[0].graphs["Depth"]
        fmt = {n: node.base.get("output_format") for n, node in g.nodes.items()}
        self.assertEqual(fmt, {"perlin_noise1": {"mode": "absolute", "value": "16"}, "perlin_noise2": None,
                               "invert1": None, "perlin_noise3": {"mode": "absolute", "value": "8"}})

    def test_the_first_color_to_mask_is_the_legacy_node(self):
        nodes = comp_instance("10", "color_to_mask", "50", ["output"],
                              param("mask_range", "Float1", 0.3) + param("keying_type", "Int32", 2))
        path = self.write("ctm.sbs", package([("sbs://color_to_mask.sbs", "50")], graph("Ctm", nodes)))
        pkg, report = import_sbs(path)
        node = next(iter(pkg.graphs["Ctm"].nodes.values()))
        self.assertEqual(report.missing, 0, report.lines())
        self.assertEqual((node.type, node.params["mask_range"], node.params["keying_type"]),
                         ("color_to_mask_legacy", 0.3, "Luminance"))

    def test_a_format_our_node_needs_is_kept(self):
        # Flood Fill measures 16 bits in the original; ours needs 32 bits float: the import leaves its own default.
        nodes = comp_instance("10", "flood_fill", "50", ["output"])
        path = self.write("fill.sbs", package([("sbs://flood_fill_2.sbs", "50")], graph("Fill", nodes)))
        g = import_sbs(path)[0].graphs["Fill"]
        node = next(iter(g.nodes.values()))
        self.assertEqual(node.type, "flood_fill")
        self.assertNotIn("output_format", node.base)

    def test_node_defaults_follow_the_library_they_follow(self):
        # A new Gradient Radial gives 16 bits as the original's does; Perlin Noise inherits (the visible v2 gives
        # 8 bits); Flood Fill keeps the 32 bits float its data needs.
        from trafilatura3d.nodes import NODES
        fmt = {t: NODES[t].base_defaults_for({}, {}).get("output_format") for t in ("gradient_radial", "perlin_noise",
                                                                                     "flood_fill")}
        self.assertEqual(fmt, {"gradient_radial": {"mode": "absolute", "value": "16"}, "perlin_noise": None,
                               "flood_fill": {"mode": "absolute", "value": "32f"}})

    def test_unset_values_take_the_file_formats_defaults(self):
        # A file stores no value left at its default: a Normal without intensity or format is Intensity 1, DirectX
        # (ours default to 10 and OpenGL).
        nodes = (comp_filter("10", "uniform", param("colorswitch", "Bool", 0)) +
                 comp_filter("11", "normal", "", {"input1": ("10", "100")}) +
                 output_node("12", "800", ("11", "110")))
        pkg, _ = import_sbs(self.write("n.sbs", package([], graph("N", nodes, [("normal", "800", "normal")]))))
        n = next(x for x in pkg.graphs["N"].nodes.values() if x.type == "normal")
        self.assertEqual((n.params.get("intensity"), n.params.get("normal_format")), (1.0, "DirectX"))

    def test_curve_points_keep_their_handles(self):
        # Each point's left and right handles are kept (absolute); a broken one is the point itself.
        def cell(pos, left, right, lb, rb):
            ps = (param("position", "Float2", pos) + param("left", "Float2", left) + param("isLeftBroken", "Bool", lb)
                  + param("right", "Float2", right) + param("isRightBroken", "Bool", rb))
            return f"<paramsArrayCell><uid v=\"1\"/><parameters>{ps}</parameters></paramsArrayCell>"
        cells = cell("0 0", "0 0", "0.1 0", 1, 1) + cell("0.5 0.3", "0.3 0.2", "0.7 0.4", 0, 0) + cell("1 1", "0.9 1", "1 1", 1, 1)
        arr = (f'<paramsArrays><paramsArray><name v="curveluminance"/><uid v="2"/><paramsArrayCells>{cells}'
               '</paramsArrayCells></paramsArray></paramsArrays>')
        nodes = (comp_filter("10", "uniform", param("colorswitch", "Bool", 0)) +
                 comp_filter("11", "curve", "", {"input1": ("10", "100")}, extra=arr) +
                 output_node("12", "800", ("11", "110")))
        pkg, _ = import_sbs(self.write("c.sbs", package([], graph("C", nodes, [("height", "800", "height")]))))
        n = next(x for x in pkg.graphs["C"].nodes.values() if x.type == "curve")
        self.assertEqual(n.params["curve"], [[0, 0, 0, 0, 0, 0], [0.5, 0.3, 0.3, 0.2, 0.7, 0.4], [1, 1, 1, 1, 1, 1]])

    def test_atomic_and_library_nodes_values_links_functions(self):
        exposed_opacity = fn_param("opacitymult", fn_node("1", "get_float1", 256, ("String", "amount")), "1")
        size_fn = fn_param("intensity", fn_node("1", "get_float2", 512, ("String", "$size")) +
                           fn_node("2", "swizzle1", 256, ("Int1", "0"), [("vector", "1")]) +
                           fn_node("3", "const_float1", 256, ("Float1", "0.01")) +
                           fn_node("4", "mul", 256, None, [("a", "2"), ("b", "3")]), "4")
        nodes = (
            comp_instance("10", "perlin_noise", "50", ["output"], param("scale", "Int32", 12) +
                          param("randomseed", "Int32", 3, relative=1), pos=(-200, 0)) +
            comp_instance("11", "slope_blur_grayscale_2", "51", ["Slope_Blur"],
                          param("mode", "Int32", 6) + param("Samples", "Int32", 4),
                          {"Source": ("10", "100"), "Effect": ("10", "100")}, pos=(0, 0)) +
            comp_filter("12", "uniform", param("outputcolor", "Float4", "0.2 0.4 0.6 1") +
                        param("colorswitch", "Bool", 1) + param("outputsize", "Int2", "4 4", relative=0), pos=(0, 200)) +
            comp_filter("13", "blend", param("blendingmode", "Int32", 3) + exposed_opacity,
                        {"source": ("11", "110"), "destination": ("12", "120")}, pos=(200, 0)) +
            comp_filter("14", "directionalwarp", size_fn, {"input1": ("13", "130")}, pos=(300, 0)) +
            output_node("15", "800", ("14", "140")))
        frame = ('<GUIObject><type v="COMMENT"/><GUILayout><gpos v="-300 -100 0"/><size v="400 300"/></GUILayout>'
                 '<GUIName v="notes"/><title v="Noise"/><frameColor v="0.1 0.5 1 0.25"/><isFrameVisible v="1"/>'
                 '</GUIObject><GUIObject><type v="COMMENT"/><GUILayout><gpos v="0 400 0"/></GUILayout>'
                 '<GUIName v="a comment"/><isFrameVisible v="0"/></GUIObject>')
        path = self.write("mat.sbs", package([("sbs://noise_perlin_noise.sbs", "50"), ("sbs://slope_blur.sbs", "51")],
                                             graph("Mat", nodes, [("basecolor", "800", "baseColor")], FLOAT_INPUT,
                                                   frame, param("format", "Int32", 1, relative=0))))
        pkg, report = import_sbs(path)
        g = pkg.graphs["Mat"]
        self.assertEqual((report.nodes, report.mapped, report.missing), (5, 5, 0), report.lines())
        self.assertEqual(g.parent_size, [9, 9])
        self.assertEqual(g.base["output_format"], {"mode": "absolute", "value": "16"})
        self.assertEqual(g.parameters["amount"].param.default, 0.25)
        self.assertEqual((g.parameters["amount"].param.max, g.parameters["amount"].param.group), (2.0, "Look"))
        by_type = {n.type: n for n in g.nodes.values()}
        perlin, slope, blend = by_type["perlin_noise"], by_type["slope_blur"], by_type["blend"]
        self.assertEqual(perlin.params["scale"], 12)
        self.assertEqual(perlin.base["seed"], {"mode": "relative_parent", "value": 3})
        self.assertEqual((slope.params["mode"], slope.params["samples"]), ("Min", 4))
        self.assertEqual(slope.inputs, {"input": [perlin.id, "output"], "slope": [perlin.id, "output"]})
        uniform = by_type["uniform_color"]
        self.assertEqual(uniform.params["output_color"], [0.2, 0.4, 0.6, 1.0])
        self.assertEqual(uniform.base["output_size"], {"mode": "absolute", "value": [4, 4]})
        self.assertEqual(blend.params["blending_mode"], "Multiply")
        self.assertEqual(blend.links, {"opacity": "amount"})  # a Get of a graph input: a link
        self.assertEqual(blend.inputs, {"foreground": [slope.id, "output"], "background": [uniform.id, "output"]})
        self.assertIn("intensity", by_type["directional_warp"].functions)  # $size.x * 0.01
        out = by_type["output"]
        self.assertEqual((out.params["identifier"], out.params["usage"]), ("basecolor", "baseColor"))
        self.assertEqual(perlin.position, [-200.0, 0.0])
        kinds = sorted(i["kind"] for i in g.items)
        self.assertEqual(kinds, ["comment", "frame"])
        frame_item = next(i for i in g.items if i["kind"] == "frame")
        self.assertEqual((frame_item["title"], frame_item["size"]), ("Noise", [400.0, 300.0]))
        c = compile_graph(pkg, "Mat", application_context((6, 6)))
        job = c.jobs[by_type["directional_warp"].id]
        self.assertEqual(job.size, (16, 16))  # relative to its input: the blend over the 16 px uniform
        self.assertAlmostEqual(job.values["intensity"], 16 * 0.01, places=5)  # the function, at its $size

    def test_color_graphs_percentages_fixed_modes_and_base_functions(self):
        half = fn_param("outputsize", fn_node("1", "const_int2", 32, ("Int2", "-1 -1")), "1", relative=1)
        nodes = (
            comp_instance("10", "tile_sampler_color", "50", ["output"], pos=(0, 0), comptype=1) +
            comp_instance("11", "messy_fibers_1", "51", ["Messy_Fibers"], param("Disorder", "Float1", 50),
                          pos=(0, 100)) +
            comp_instance("12", "add", "52", ["Add"], conns={"Background": ("11", "110"), "Foreground": ("11", "110")},
                          pos=(100, 100)) +
            comp_filter("13", "uniform", half, pos=(0, 200)) +
            output_node("14", "800", ("10", "100")) +
            comp_instance("15", "cells_1", "53", ["Cells"], param("Disorder", "Float1", 50), pos=(0, 300)))
        path = self.write("v.sbs", package([("sbs://pattern_tile_sampler.sbs", "50"), ("sbs://messy_fibers_1.sbs", "51"),
                                            ("sbs://blend.sbs", "52"), ("sbs://cells_1.sbs", "53")],
                                           graph("V", nodes, [("out", "800", "")])))
        pkg, report = import_sbs(path)
        self.assertEqual((report.missing, report.params_unmapped), (0, 0), report.lines())
        by_type = {n.type: n for n in pkg.graphs["V"].nodes.values()}
        self.assertTrue(by_type["tile_sampler"].params["color_variant"])  # Tile Sampler Color: colour, no input
        # the legacy graph on our current node: its 0-100 percentage converted
        self.assertEqual(by_type["messy_fibers_bright"].params["disorder"], 0.5)
        self.assertEqual(by_type["cells_domes_legacy"].params["disorder"], 50)  # its own legacy node: as it is
        self.assertEqual(by_type["blend_deprecated"].params["blending_mode"], "Add")  # the single-mode graph's mode
        self.assertIn("function", by_type["uniform_color"].base["output_size"])
        c = compile_graph(pkg, "V", application_context((9, 9)))
        self.assertIs(c.jobs[by_type["tile_sampler"].id].output_types["output"], PortType.COLOR)
        self.assertEqual(c.jobs[by_type["uniform_color"].id].size, (256, 256))  # the function: parent size - 1

    def test_missing_nodes_stay_wired_and_are_reported(self):
        nodes = (comp_instance("10", "perlin_noise", "50", ["output"], pos=(0, 0)) +
                 comp_instance("11", "mystery", "52", ["out_a"], param("strength", "Float1", 0.7),
                               {"input_x": ("10", "100")}, pos=(100, 0)) +
                 comp_filter("12", "levels", param("unknownparam", "Float1", 1), {"input1": ("11", "110")}) +
                 comp_filter("13", "svgthing", "", {"input1": ("12", "120")}) +
                 output_node("15", "800", ("12", "120")))
        path = self.write("m.sbs", package([("sbs://noise_perlin_noise.sbs", "50"), ("sbs://mystery_pack.sbs", "52")],
                                           graph("G", nodes, [("height", "800", "height")])))
        pkg, report = import_sbs(path)
        g = pkg.graphs["G"]
        self.assertEqual((report.nodes, report.mapped, report.missing, report.params_unmapped), (4, 2, 2, 1))
        missing = next(n for n in g.nodes.values() if n.ref == "sbs://mystery_pack.sbs#mystery")
        self.assertEqual(missing.type, INSTANCE)
        self.assertEqual(missing.params, {"strength": 0.7})  # its values kept
        perlin = next(n for n in g.nodes.values() if n.type == "perlin_noise")
        self.assertEqual(missing.inputs, {"input_x": [perlin.id, "output"]})  # wired both ways
        levels = next(n for n in g.nodes.values() if n.type == "levels")
        self.assertEqual(levels.inputs, {"input": [missing.id, "out_a"]})
        atomic = next(n for n in g.nodes.values() if n.ref == "sbs://atomic#svgthing")
        self.assertEqual(atomic.inputs, {"input1": [levels.id, "output"]})
        text = "\n".join(report.lines())
        self.assertIn("mystery_pack.sbs#mystery has no Trafilatura 3D counterpart", text)
        self.assertIn("'unknownparam' not mapped", text)
        warnings = graph_warnings(pkg, "G")
        self.assertIn("Reference resource not found", [w.message for w in warnings if w.node == missing.id])
        self.assertTrue(any("unknownparam" in w.message for w in warnings))  # the import report with the warnings
        self.assertTrue(any(w.message.startswith("Designer dependency") for w in dependency_warnings(pkg)))
        c = compile_graph(pkg, "G", application_context((5, 5)))  # the missing node reads as unconnected
        self.assertIsNone(c.jobs[levels.id].inputs["input"])
        out = self.dir / "m.trafilatura3d"
        pkg.save(out)
        again = Package.load(out)  # a saved import opens, the missing node still wired
        self.assertEqual(again.graphs["G"].nodes[levels.id].inputs, {"input": [missing.id, "out_a"]})

    def test_missing_node_keeps_its_declared_output_types(self):
        # A missing node declared color (comptype 1) feeds a Blur (follows its input) feeding a color-only input:
        # downstream inference must see color, so no connection is dropped; also through a user graph whose
        # output comes from a missing node.
        helper = graph("helper", comp_instance("30", "mystery", "52", ["out"], comptype=1) +
                       output_node("31", "801", ("30", "300")), [("normal", "801", "normal")])
        nodes = (comp_instance("10", "mystery", "52", ["out"], comptype=1) +
                 comp_filter("11", "blur", "", {"input1": ("10", "100")}) +
                 comp_instance("12", "helper", "53", ["normal"]) +
                 comp_instance("13", "normal_combine", "51", ["normal"],
                               conns={"Input": ("11", "110"), "Input_1": ("12", "120")}) +
                 output_node("15", "800", ("13", "130")))
        self.write("dep.sbs", package([("sbs://mystery_pack.sbs", "52")], helper))
        path = self.write("m.sbs", package([("sbs://mystery_pack.sbs", "52"), ("sbs://normal_combine.sbs", "51"),
                                            ("dep.sbs", "53")], graph("G", nodes, [("normal", "800", "normal")])))
        pkg, report = import_sbs(path)
        self.assertFalse([n for n in report.lines() if "dropped" in n], report.lines())
        g = pkg.graphs["G"]
        missing = next(n for n in g.nodes.values() if n.ref == "sbs://mystery_pack.sbs#mystery")
        self.assertEqual(missing.outputs, {"out": "color"})
        combine = next(n for n in g.nodes.values() if n.type == "normal_combine")
        blur = next(n for n in g.nodes.values() if n.type == "blur")
        self.assertEqual(combine.inputs["normal_1"], [blur.id, "output"])
        self.assertEqual(g.nodes[combine.inputs["normal_2"][0]].ref, "helper")  # the user graph, imported
        self.assertEqual(g.output_type(blur.id, "output"), PortType.COLOR)
        out = self.dir / "m.trafilatura3d"
        pkg.save(out)
        again = Package.load(out)  # the declared types are saved: the package still validates
        self.assertEqual(again.graphs["G"].nodes[missing.id].outputs, {"out": "color"})
        missing.outputs = {"out": "rgb"}
        self.assertTrue(any("bad declared output types" in p for p in g.problems()))

    def test_user_graphs_of_other_files_are_imported(self):
        helper = graph("helper", comp_filter("20", "levels", fn_param(
            "levelinlow", fn_node("1", "get_float1", 256, ("String", "amount")), "1"), {"input1": ("21", "210")}) +
            '<compNode><uid v="21"/><compOutputs><compOutput><uid v="210"/></compOutput></compOutputs>'
            '<compImplementation><compInputBridge><entry v="901"/></compInputBridge></compImplementation></compNode>' +
            output_node("22", "801", ("20", "200")), [("result", "801", "")],
            FLOAT_INPUT + '<paraminput><identifier v="source"/><uid v="901"/><type v="2"/></paraminput>')
        self.write("deps/helper.sbs", package([], helper))
        main = graph("main", comp_instance("10", "perlin_noise", "50", ["output"]) +
                     comp_instance("11", "helper", "53", ["result"], param("amount", "Float1", 0.75),
                                   {"source": ("10", "100")}) + output_node("12", "800", ("11", "110")),
                     [("out", "800", "")])
        path = self.write("main.sbs", package([("sbs://noise_perlin_noise.sbs", "50"), ("deps/helper.sbs", "53")], main))
        pkg, report = import_sbs(path)
        self.assertEqual(report.graphs, ["main"])
        self.assertEqual(set(pkg.graphs), {"main", "helper"})  # one self-contained package
        inst = next(n for n in pkg.graphs["main"].nodes.values() if n.type == INSTANCE)
        self.assertEqual((inst.ref, inst.params), ("helper", {"amount": 0.75}))
        self.assertEqual(list(inst.inputs), ["source"])
        levels = next(n for n in pkg.graphs["helper"].nodes.values() if n.type == "levels")
        self.assertIn("level_in_low", levels.functions)  # float into a float4: spread by a function
        compile_graph(pkg, "main", application_context((5, 5)))

    def test_dependencies_of_another_machine_are_searched_by_name(self):
        helper = graph("helper", output_node("20", "801", ("21", "210")) +
                       comp_filter("21", "uniform", param("outputcolor", "Float4", "0.5 0.5 0.5 1")),
                       [("result", "801", "")])
        self.write("lib/old/helper.sbs", package([], helper))
        newest = self.write("lib/new/helper.sbs", package([], helper))
        os.utime(newest, (newest.stat().st_atime, newest.stat().st_mtime + 10))
        main = graph("main", comp_instance("10", "helper", "53", ["result"]) + output_node("11", "800", ("10", "100")),
                     [("out", "800", "")])
        path = self.write("work/main.sbs", package([("H:/Users/someone/sbs/helper.sbs", "53")], main))
        _, report = import_sbs(path)
        self.assertEqual(report.missing, 1)  # the written path, not found
        pkg, report = import_sbs(path, search=(self.dir / "lib",))
        self.assertEqual((report.missing, set(pkg.graphs)), (0, {"main", "helper"}))
        from trafilatura3d.cli import main as cli
        with redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(cli(["--no-plugins", "import", str(path), "--out", str(self.dir / "m.trafilatura3d"),
                                  "--search", str(self.dir / "nothing"), "--search", str(self.dir / "lib")]), 0)
        self.assertIn("0 missing", buf.getvalue())
        from trafilatura3d.sbs.importer import _Importer
        found = _Importer(path, [self.dir.resolve() / "lib"])._find(Path("H:\\x\\helper.sbs"))
        self.assertEqual(found, newest.resolve())  # the newest of that name

    def test_text(self):
        """A Text node's parameters; its system font, found by name, is copied into dependencies/ when the package
        is saved (its own folder there); a font the system doesn't have is reported (the default font draws it)."""
        from trafilatura3d import fonts
        found = fonts.find("")
        if found is None or found[1]:
            self.skipTest("no default system font file")
        family, style, _ = fonts.face_info(found[0])

        def text_node(uid, font):
            return comp_filter(uid, "text", param("text", "String", "AB") + param("fontdata", "String", font) +
                               param("fontsize", "Float1", "0.1") + param("alignment", "Int32", "0") +
                               param("position", "Float2", "0.1 -0.2") + param("matrix22", "Float4", "2 0 0 1"))
        main = graph("main", text_node("10", f"{family}|{style}") + text_node("12", "No Such Font|Regular") +
                     output_node("11", "800", ("10", "100")), [("out", "800", "")])
        path = self.write("main.sbs", package([], main))
        out = self.dir / "out"
        pkg, report = import_sbs(path)
        n = pkg.graphs["main"].nodes["text1"]
        self.assertEqual(Path(n.params["font"]), found[0])
        pkg.save(out / "main.trafilatura3d")
        self.assertEqual(n.params["font"], f"dependencies/main/{found[0].name}")
        self.assertTrue((out / "dependencies" / "main" / found[0].name).is_file())
        self.assertEqual((n.params["font_size"], n.params["alignment"], n.params["position"],
                          n.params["transformation"]), (0.1, "Left", [0.1, -0.2], [2.0, 0.0, 0.0, 1.0]))
        self.assertEqual(report.params_unmapped, 0)
        self.assertTrue(any("No Such Font" in note.message for note in report.notes))

    def test_dependency_moved_up_out_of_its_folder(self):
        """A dependency written as deps/helper/helper.sbs but kept as deps/helper.sbs is found next to the
        written path's folders, never above the package's own folder."""
        helper = graph("helper", output_node("20", "801", ("21", "210")) +
                       comp_filter("21", "uniform", param("outputcolor", "Float4", "0.5 0.5 0.5 1")),
                       [("result", "801", "")])
        self.write("pack/deps/helper.sbs", package([], helper))
        main = graph("main", comp_instance("10", "helper", "53", ["result"]) + output_node("11", "800", ("10", "100")),
                     [("out", "800", "")])
        path = self.write("pack/main.sbs", package([("deps/helper/helper.sbs", "53")], main))
        pkg, report = import_sbs(path)
        self.assertEqual((report.missing, set(pkg.graphs)), (0, {"main", "helper"}))
        lost = self.write("pack/sub/main.sbs", package([("deps/helper/helper.sbs", "53")], main))
        self.assertEqual(import_sbs(lost)[1].missing, 1)  # pack/deps is outside pack/sub

    def test_embedded_svg_goes_to_dependencies_on_save(self):
        # An SVG kept inside the .sbs (base64 of its length and zlib data): the SVG node reads it from a temporary
        # file, and saving the package puts it in its dependency folder under the resource's name.
        import base64
        import struct
        import zlib
        svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="4" height="8"/></svg>'
        data = base64.b64encode(struct.pack(">I", len(svg)) + zlib.compress(svg)).decode()
        res = ('<group><identifier v="Resources"/><content><resource><identifier v="Logo"/><type v="svg"/>'
               '<format v="SVG"/><filepath v="dependencies"/><source><binembedded>'
               f'<datalength v="{len(svg)}"/><strdata v="{data}"/></binembedded></source></resource></content></group>')
        nodes = (comp_filter("10", "svg", param("vectorresourcepath", "String", "pkg:///Resources/Logo?dependency=99") +
                             param("colorswitch", "Bool", 0)) + output_node("11", "800", ("10", "100")))
        path = self.write("svg.sbs", package([("?himself", "99")], graph("G", nodes, [("o", "800", "")]) + res))
        pkg, report = import_sbs(path)
        n = next(n for n in pkg.graphs["G"].nodes.values() if n.type == "svg")
        self.assertEqual((report.missing, n.params["color_mode"]), (0, "Grayscale"))
        self.assertEqual(Path(n.params["pkg_resource_path"]).read_bytes(), svg)
        pkg.save(self.dir / "out" / "svg.trafilatura3d")
        self.assertEqual((n.params["pkg_resource_path"], pkg.resources["Logo"].path),
                         ("dependencies/svg/Logo.svg", "dependencies/svg/Logo.svg"))
        self.assertEqual((self.dir / "out" / "dependencies" / "svg" / "Logo.svg").read_bytes(), svg)
        self.assertEqual(pkg.resources["Logo"].kind, "svg")

    def test_bitmap_fxmap_pixel_processor_gradient(self):
        res = ('<group><identifier v="Resources"/><content><resource><identifier v="img"/><type v="bitmap"/>'
               '<filepath v="textures/img.png"/></resource></content></group>')
        fx = ('<paramsGraphs><paramsGraph><rootnode v="7"/><paramsGraphNodes>'
              '<paramsGraphNode><uid v="7"/><type v="paramset"/><data v="70"/><connections><connection>'
              '<identifier v="output1"/><connRef v="8"/></connection></connections></paramsGraphNode>'
              '<paramsGraphNode><uid v="8"/><type v="paramset"/><data v="80"/></paramsGraphNode></paramsGraphNodes>'
              '<paramsGraphDatas><paramsGraphData><uid v="70"/><parameters>' + param("patterntype", "Int32", 3) +
              '</parameters></paramsGraphData><paramsGraphData><uid v="80"/><parameters>' +
              param("patterntype", "Int32", 5) + param("blendingmode", "Int32", 1) + fn_param(
                  "patternrotation", fn_node("1", "get_float1", 256, ("String", "$depth")), "1") +
              '</parameters></paramsGraphData></paramsGraphDatas></paramsGraph></paramsGraphs>')
        grad = ('<paramsArrays><paramsArray><name v="gradientrgba"/><paramsArrayCells>' + "".join(
            f'<paramsArrayCell><parameters>{param("position", "Float1", p)}{param("value", "Float4", c)}'
            f'</parameters></paramsArrayCell>' for p, c in ((0, "0 0 0 1"), (1, "1 0.5 0 1"))) +
            '</paramsArrayCells></paramsArray></paramsArrays>')
        pp = fn_param("perpixel", fn_node("1", "get_float2", 512, ("String", "$pos")) +
                      fn_node("2", "samplelum", 256, ("Int2", "0 0"), [("pos", "1")]), "2")
        nodes = (comp_filter("10", "bitmap", param("bitmapresourcepath", "String",
                                                   "pkg:///Resources/img?dependency=99")) +
                 comp_filter("11", "fxmaps", param("colorswitch", "Bool", 0), {"inputpattern:1": ("10", "100")},
                             extra=fx) +
                 comp_filter("12", "pixelprocessor", pp, {"input": ("11", "110")}) +
                 comp_filter("13", "gradient", "", {"input1": ("12", "120")}, extra=grad) +
                 output_node("14", "800", ("13", "130")))
        path = self.write("fx.sbs", package([("?himself", "99")], graph("G", nodes, [("o", "800", "")]) + res))
        pkg, report = import_sbs(path)
        by = {n.type: n for n in pkg.graphs["G"].nodes.values()}
        img = str(self.dir.resolve() / "textures" / "img.png")
        self.assertEqual(by["bitmap"].params["pkg_resource_path"], img)
        self.assertEqual(pkg.resources["img"].path, img)
        self.assertIn("bitmap file not found", "\n".join(report.lines()))
        fxmap = by["stamp_map"].params["fxmap"]
        q = {n["id"]: n for n in fxmap["nodes"]}
        self.assertEqual(q[fxmap["root"]]["params"]["pattern"], "Disc")
        self.assertEqual(q[fxmap["root"]]["children"], {"q2": "fx8"})
        self.assertEqual((q["fx8"]["params"]["pattern"], q["fx8"]["params"]["blending_mode"]), ("Bell", "Max"))
        self.assertIn("pattern_rotation", q["fx8"]["functions"])
        self.assertEqual((by["stamp_map"].params["input_count"], list(by["stamp_map"].inputs)), (2, ["input_image_1"]))
        self.assertEqual(by["stamp_map"].params["color_mode"], "Grayscale")
        self.assertEqual(by["pixel_processor"].inputs, {"input_image_0": [by["stamp_map"].id, "output"]})
        types = [n["type"] for n in by["pixel_processor"].params["per_pixel_function"]["nodes"]]
        self.assertEqual(types, ["get_float2", "sample_gray"])
        keys = by["gradient_map"].params["gradient"]["keys"]
        self.assertEqual([k["color"] for k in keys], [[0, 0, 0, 1], [1, 0.5, 0, 1]])
        compile_graph(pkg, "G", application_context((5, 5)))

    def test_function_graphs_of_the_package_and_library_functions_are_called(self):
        # scaled(x) = oneminus(x) * 0.5, a function of the package calling a library function; Pi() directly.
        body = (fn_node("1", "get_float1", 256, ("String", "x")) +
                fn_node("2", "instance", 256, ("String", "pkg:///Functions/Math/oneminus?dependency=60"), [("x", "1")]) +
                fn_node("3", "const_float1", 256, ("Float1", "0.5")) + fn_node("4", "mul", 256, None, [("a", "2"), ("b", "3")]))
        function = ('<function><identifier v="scaled"/><uid v="5"/><paraminputs><paraminput><identifier v="x"/>'
                    '<uid v="6"/><type v="256"/><defaultValue><constantValueFloat1 v="2"/></defaultValue></paraminput>'
                    f'</paraminputs><type v="256"/><paramValue><dynamicValue><rootnode v="4"/><paramNodes>{body}'
                    '</paramNodes></dynamicValue></paramValue></function>')
        opacity = fn_param("opacitymult", fn_node("1", "const_float1", 256, ("Float1", "0.2")) +
                           fn_node("2", "instance", 256, ("String", "pkg:///scaled?dependency=99"), [("x", "1")]), "2")
        pi = fn_param("intensity", fn_node("1", "instance", 256, ("String", "pkg:///Functions/Math/Pi?dependency=60")),
                      "1")
        nodes = (comp_filter("10", "uniform", "") +
                 comp_filter("11", "blend", opacity, {"source": ("10", "100"), "destination": ("10", "100")}) +
                 comp_filter("12", "directionalwarp", pi, {"input1": ("11", "110")}) + output_node("13", "800", ("12", "120")))
        path = self.write("calls.sbs", package([("?himself", "99"), ("sbs://functions.sbs", "60")],
                                               graph("G", nodes, [("o", "800", "")]) + function))
        pkg, report = import_sbs(path)
        self.assertEqual(report.params_unmapped, 0, report.lines())
        by = {n.type: n for n in pkg.graphs["G"].nodes.values()}
        f = by["blend"].functions["opacity"]
        self.assertEqual(f["functions"]["scaled"]["inputs"], {"x": "float"})
        self.assertEqual(f["functions"]["scaled"]["defaults"], {"x": 2.0})
        self.assertIn("one_minus", [n["type"] for n in f["functions"]["scaled"]["nodes"]])
        c = compile_graph(pkg, "G", application_context((5, 5)))
        self.assertAlmostEqual(c.jobs[by["blend"].id].values["opacity"], 0.4, places=6)
        self.assertAlmostEqual(c.jobs[by["directional_warp"].id].values["intensity"], 3.14159265, places=6)
        again = Package.from_dict(pkg.to_dict())  # the called functions travel with the function
        self.assertAlmostEqual(compile_graph(again, "G").jobs[by["blend"].id].values["opacity"], 0.4, places=6)

    def test_library_functions_map_to_library_nodes(self):
        from trafilatura3d.functions import FNODES, LIBRARY
        from trafilatura3d.sbs.mapping import LIBRARY_FUNCTIONS, LIBRARY_PACKAGES
        for source, ours in LIBRARY_FUNCTIONS.items():
            self.assertIn(ours, LIBRARY, source)
            self.assertEqual([p for p, _ in FNODES[ours].inputs], list(LIBRARY[ours]["inputs"]))
            self.assertIn(source.split("#")[0], LIBRARY_PACKAGES)

    def test_fxmap_enums_regions_and_transformation_matrix(self):
        from trafilatura3d.functions import FunctionGraph, evaluate
        fx = ('<paramsGraphs><paramsGraph><rootnode v="7"/><paramsGraphNodes>'
              '<paramsGraphNode><uid v="7"/><type v="paramset"/><data v="70"/></paramsGraphNode></paramsGraphNodes>'
              '<paramsGraphDatas><paramsGraphData><uid v="70"/><parameters>' + param("patterntype", "Int32", 18) +
              param("blendingmode", "Int32", 2) + param("randominherited", "Bool", 0) +
              '</parameters></paramsGraphData></paramsGraphDatas></paramsGraph></paramsGraphs>')
        matrix = fn_param("matrix22", fn_node("1", "const_float4", 2048, ("Float4", "2 0 0 2")), "1")
        nodes = (comp_filter("10", "fxmaps", param("renderroi", "Float4", "0 1 0 0.5") + param("hurst", "Float1", 0),
                             extra=fx) +
                 comp_filter("11", "transformation", param("matrix22", "Float4", "2 0 0 4"), {"input1": ("10", "100")}) +
                 comp_filter("12", "transformation", matrix, {"input1": ("11", "110")}) +
                 output_node("13", "800", ("12", "120")))
        pkg, report = import_sbs(self.write("fx2.sbs", package([], graph("G", nodes, [("o", "800", "")]))))
        g = pkg.graphs["G"]
        fxn = next(n for n in g.nodes.values() if n.type == "stamp_map")
        quad = fxn.params["fxmap"]["nodes"][0]["params"]
        self.assertEqual((quad["pattern"], quad["blending_mode"], quad["inherit_random"]),
                         ("Hemisphere", "Alpha blend", False))
        self.assertEqual((fxn.params["render_region"], fxn.params["roughness"]), ([0, 1, 0, 0.5], 0.0))
        # The matrix as the file stores it (our atomic reads its sampling convention): unchanged, constant or function.
        a, b = sorted((n for n in g.nodes.values() if n.type == "transformation_2d"), key=lambda n: bool(n.functions))
        self.assertEqual(a.params["transformation_matrix"], [2, 0, 0, 4])
        f = b.functions["transformation_matrix"]
        self.assertEqual(evaluate(FunctionGraph(f, {}), {}), [2, 0, 0, 2])
        compile_graph(pkg, "G", application_context((5, 5)))

    def test_cli_import(self):
        from trafilatura3d.cli import main
        nodes = comp_instance("10", "mystery", "52", ["o"]) + comp_filter("11", "blur", "", {"input1": ("10", "100")})
        path = self.write("c.sbs", package([("sbs://mystery.sbs", "52")], graph("G", nodes)))
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["--no-plugins", "import", str(path), "--out", str(self.dir / "out" / "c.trafilatura3d")]), 0)
        self.assertIn("2 nodes, 1 mapped, 1 missing", buf.getvalue())
        self.assertIn("G", Package.load(self.dir / "out" / "c.trafilatura3d").graphs)


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class EditorImportTest(unittest.TestCase):
    def test_open_sbs(self):
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication, QPushButton
        import rendercanvas.qt as rq
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        try:
            from trafilatura3d.editor.mainwindow import MainWindow
            with tempfile.TemporaryDirectory() as d:
                nodes = (comp_instance("10", "mystery", "52", ["o"]) +
                         comp_filter("11", "blur", param("nonsense", "Float1", 2), {"input1": ("10", "100")}))
                path = Path(d) / "e.sbs"
                path.write_text(package([("sbs://mystery.sbs", "52")], graph("G", nodes)), encoding="utf-8")
                w = MainWindow([str(path)])
                pkg = w.doc.packages[0]
                self.assertIsNone(pkg.path)  # a new unsaved package
                self.assertIn(id(pkg), w.doc.dirty)
                self.assertEqual(w.graph.graph_id, "G")
                text = html.unescape(w.graph.warnings_label.text())  # rich text: node links
                self.assertIn("Reference resource not found", text)
                self.assertIn("'nonsense' not mapped", text)
                missing = next(k for k, n in pkg.graphs["G"].nodes.items() if n.type == "graph_instance")
                self.assertIn(missing, w.graph.scene.nodes)  # drawn, with the connector its link uses
                self.assertIn("o", w.graph.scene.nodes[missing].outputs)
                self.assertTrue(w.graph.scene.nodes[missing].missing)  # drawn in red
                self.assertFalse(w.graph.scene.nodes[next(k for k in w.graph.scene.nodes if k != missing)].missing)
                w.graph._focus_node(missing)  # its warning's link: selects it, the Properties show why
                app.processEvents()
                self.assertEqual(w.graph.scene.selected_ids(), [missing])
                headers = [b.text() for b in w.properties.widget().findChildren(QPushButton)]
                self.assertTrue(any("Missing dependency" in t for t in headers), headers)
                w.doc.save(pkg, Path(d) / "e.trafilatura3d")
                self.assertTrue((Path(d) / "e.trafilatura3d").exists())
                w.doc.undo.indexChanged.disconnect()
                w.graph.scene.selectionChanged.disconnect()
                app.processEvents()
        finally:
            rq.QRenderWidget.__init__ = init


if __name__ == "__main__":
    unittest.main()
