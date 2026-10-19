"""Function graphs: typing, CPU evaluation, parameter functions, Pixel processor on the GPU."""

import math
import unittest

import numpy as np

from trafilatura3d import DocumentError, Evaluator, Package, compile_graph
from trafilatura3d.functions import FunctionError, FunctionGraph, evaluate
from trafilatura3d.nodes.base import Param, ParamKind

EV = Evaluator()


def fg(nodes, output, variables=None):
    return {"nodes": [{"id": k, "type": t, "params": p, "inputs": i} for k, t, p, i in nodes], "output": output}


def const(k, t, v):
    return (k, f"const_{t}", {"value": v}, {})


class Typing(unittest.TestCase):
    def test_vector_widths_must_add_up(self):
        g = fg([const("a", "float", 1.0), const("b", "float2", [2.0, 3.0]),
                ("v", "vector_float3", {}, {"input_1": "a", "input_2": "b"})], "v")
        self.assertEqual(FunctionGraph(g, {}).types["v"], "float3")
        bad = fg([const("a", "float", 1.0), const("b", "float", 2.0),
                  ("v", "vector_float3", {}, {"input_1": "a", "input_2": "b"})], "v")
        with self.assertRaises(FunctionError):
            FunctionGraph(bad, {})

    def test_get_checks_the_variable_type(self):
        g = fg([("g", "get_float", {"variable": "$size"}, {})], "g")
        with self.assertRaises(FunctionError):
            FunctionGraph(g, {"$size": "float2"})

    def test_no_mixing_int_and_float(self):
        g = fg([const("a", "float", 1.0), const("b", "int", 2), ("s", "add", {}, {"a": "a", "b": "b"})], "s")
        with self.assertRaises(FunctionError):
            FunctionGraph(g, {})


    def test_constants_hold_a_value_of_their_type(self):
        # A scalar in a Float2 constant reached the interpreter as a float times a list (tiles_003's import).
        for t, v in (("float2", 0.0), ("float", [1.0, 2.0]), ("float3", [1.0, 2.0])):
            with self.assertRaises(FunctionError):
                FunctionGraph(fg([const("c", t, v)], "c"), {})


class Evaluation(unittest.TestCase):
    def run_graph(self, nodes, out, variables=None, types=None):
        return evaluate(FunctionGraph(fg(nodes, out), types or {}), variables or {})

    def test_arithmetic_and_swizzle(self):
        r = self.run_graph([const("a", "float3", [1.0, 2.0, 3.0]), const("b", "float3", [4.0, 5.0, 6.0]),
                            ("m", "multiplication", {}, {"a": "a", "b": "b"}),
                            ("s", "swizzle_float2", {"x": "Z", "y": "X"}, {"vector": "m"})], "s")
        self.assertEqual(r, [18.0, 4.0])

    def test_if_else_and_comparison(self):
        nodes = [const("a", "float", 0.3), const("b", "float", 0.5), ("c", "lower", {}, {"a": "a", "b": "b"}),
                 const("x", "float", 1.0), const("y", "float", -1.0),
                 ("i", "if_else", {}, {"condition": "c", "if_": "x", "else_": "y"})]
        self.assertEqual(self.run_graph(nodes, "i"), 1.0)

    def test_while_loop_with_variables(self):
        # n = 0; while not (n >= 5): n = n + 1 -> 5
        nodes = [const("zero", "int", 0), ("init", "set", {"variable": "n"}, {"input": "zero"}),
                 ("get", "get_int", {"variable": "n"}, {}), const("five", "int", 5), const("one", "int", 1),
                 ("cond", "greater_or_equal", {}, {"a": "get", "b": "five"}),
                 ("inc", "add", {}, {"a": "get", "b": "one"}), ("body", "set", {"variable": "n"}, {"input": "inc"}),
                 ("loop", "while_loop", {"max_iterations": 100}, {"init": "init", "exit_cond": "cond", "loop_body": "body"})]
        self.assertEqual(self.run_graph(nodes, "loop"), 5)

    def test_cast_rounds(self):
        self.assertEqual(self.run_graph([const("a", "float", 2.6), ("i", "to_integer", {}, {"input": "a"})], "i"), 3)

    def test_lerp_of_vectors_by_a_float(self):
        nodes = [const("a", "float2", [0.0, 0.0]), const("b", "float2", [0.06, 0.06]),
                 ("x", "get_float", {"variable": "d"}, {}), ("l", "lerp", {}, {"a": "a", "b": "b", "x": "x"})]
        r = self.run_graph(nodes, "l", {"d": 0.5}, {"d": "float"})
        self.assertAlmostEqual(r[0], 0.03)
        self.assertAlmostEqual(r[1], 0.03)

    def test_imported_constants_left_at_zero_are_zero_of_their_type(self):
        # A .sbs file doesn't write a constant's zero value: a Float2 constant without data is (0, 0), not 0.
        import xml.etree.ElementTree as ET

        from trafilatura3d.sbs.dynamic import function
        dyn = ET.fromstring('<dynamicValue><paramNodes><paramNode><uid v="1"/><function v="const_float2"/>'
                            '</paramNode></paramNodes><rootnode v="1"/></dynamicValue>')
        f = function(dyn)
        self.assertEqual(f["nodes"][0]["params"]["value"], [0.0, 0.0])
        self.assertEqual(evaluate(FunctionGraph(f, {}), {}), [0.0, 0.0])


    def test_unary_functions_work_on_float_vectors(self):
        # abs/ceil/pow2 also take float2 and float4 values (non-square output size).
        r = self.run_graph([const("v", "float2", [-1.5, 2.25]), ("a", "abs", {}, {"x": "v"}),
                            ("f", "floor", {}, {"x": "a"})], "f")
        self.assertEqual(r, [1.0, 2.0])

    def test_an_integer_and_a_float_compare(self):
        nodes = [const("a", "float", 2.5), const("b", "int", 2), ("g", "greater", {}, {"a": "a", "b": "b"})]
        self.assertIs(self.run_graph(nodes, "g"), True)

    def test_while_loop_without_init_and_passthrough(self):
        nodes = [("get", "get_int", {"variable": "n"}, {}), const("three", "int", 3), const("one", "int", 1),
                 ("cond", "greater_or_equal", {}, {"a": "get", "b": "three"}), ("inc", "add", {}, {"a": "get", "b": "one"}),
                 ("body", "set", {"variable": "n"}, {"input": "inc"}),
                 ("loop", "while_loop", {"max_iterations": -1}, {"exit_cond": "cond", "loop_body": "body"}),
                 ("p", "passthrough", {}, {"input": "loop"})]
        self.assertEqual(self.run_graph(nodes, "p", {"n": 0}, {"n": "int"}), 3)


HALF = {"inputs": {"x": "float"}, "defaults": {"x": 4.0},
        "nodes": [{"id": "x", "type": "get_float", "params": {"variable": "x"}},
                  {"id": "k", "type": "const_float", "params": {"value": 0.5}},
                  {"id": "m", "type": "multiplication", "inputs": {"a": "x", "b": "k"}}], "output": "m"}


def calling(nodes, out, functions):
    return {**fg(nodes, out), "functions": functions}


class Calls(unittest.TestCase):
    def test_a_call_runs_the_function_on_its_inputs_or_their_defaults(self):
        g = calling([const("a", "float", 3.0), ("c", "call", {"function": "half"}, {"x": "a"}),
                     ("d", "call", {"function": "half"}, {}), ("v", "vector_float2", {}, {"input_1": "c", "input_2": "d"})],
                    "v", {"half": HALF})
        self.assertEqual(evaluate(FunctionGraph(g, {}), {}), [1.5, 2.0])

    def test_inputs_are_typed(self):
        g = calling([const("a", "int", 3), ("c", "call", {"function": "half"}, {"x": "a"})], "c", {"half": HALF})
        with self.assertRaises(FunctionError):
            FunctionGraph(g, {})
        with self.assertRaises(FunctionError):  # no such function
            FunctionGraph(calling([("c", "call", {"function": "nope"}, {})], "c", {}), {})

    def test_variables_are_shared_through_calls(self):
        # The function reads the caller's k and Sets v, which the caller reads after the call.
        setter = {"inputs": {}, "nodes": [{"id": "k", "type": "get_float", "params": {"variable": "k"}},
                                          {"id": "s", "type": "set", "params": {"variable": "v"}, "inputs": {"input": "k"}}],
                  "output": "s"}
        g = calling([("c", "call", {"function": "setter"}, {}), ("v", "get_float", {"variable": "v"}, {}),
                     ("q", "sequence", {}, {"in_": "c", "last": "v"})], "q", {"setter": setter})
        self.assertEqual(evaluate(FunctionGraph(g, {"k": "float"}), {"k": 0.375}), 0.375)

    def test_a_function_calling_itself_is_refused(self):
        loop = {"inputs": {}, "nodes": [{"id": "c", "type": "call", "params": {"function": "loop"}}], "output": "c"}
        with self.assertRaises(FunctionError):
            FunctionGraph(calling([("c", "call", {"function": "loop"}, {})], "c", {"loop": loop}), {})

    def test_two_calls_draw_different_random_numbers(self):
        g = fg([("a", "random_uniform_signed", {}, {}), ("b", "random_uniform_signed", {}, {}),
                ("v", "vector_float2", {}, {"input_1": "a", "input_2": "b"})], "v")
        x, y = evaluate(FunctionGraph(g, {}), {})
        self.assertNotEqual(x, y)
        self.assertTrue(-1 <= x < 1 and -1 <= y < 1)

    def test_library_functions(self):
        g = fg([const("a", "float", 0.25), ("o", "one_minus", {}, {"x": "a"})], "o")
        self.assertEqual(evaluate(FunctionGraph(g, {}), {}), 0.75)
        g = fg([const("t", "float", 0.25), ("r", "rotation_matrix", {}, {"RotationW": "t"})], "r")
        np.testing.assert_allclose(evaluate(FunctionGraph(g, {}), {}), [0, -1, 1, 0], atol=1e-9)

    def test_hash21_and_global_random(self):
        def frac(v): return v - math.floor(v)
        x, y = 3.7, -12.25  # the published hash: p3 = fract(p.xyx * .1031); p3 += dot(p3, p3.yzx + 33.33)
        p = [frac(x * 0.1031), frac(y * 0.1031), frac(x * 0.1031)]
        d = p[0] * (p[1] + 33.33) + p[1] * (p[2] + 33.33) + p[2] * (p[0] + 33.33)
        p = [v + d for v in p]
        g = fg([const("p", "float2", [x, y]), ("h", "hash21", {}, {"input": "p"})], "h")
        self.assertAlmostEqual(evaluate(FunctionGraph(g, {}), {}), frac((p[0] + p[1]) * p[2]), places=6)
        def global_random(seed):
            g = fg([const("x", "float", 2.0), const("s", "int", seed),
                    ("r", "global_random", {}, {"Input": "x", "Seed": "s"})], "r")
            return evaluate(FunctionGraph(g, {}), {})
        values = [global_random(s) for s in range(8)]
        self.assertEqual(values, [global_random(s) for s in range(8)])  # fixed by the seed
        self.assertTrue(all(0.0 <= v < 2.0 for v in values))
        self.assertEqual(len(set(values)), 8)

    def test_random_luminosity_moves_the_three_channels_together(self):
        g = fg([const("c", "float3", [0.5, 0.4, 0.3]), const("k", "float", 0.2),
                ("o", "random_luminosity", {}, {"RGB": "c", "Luminosity_Randomness": "k"})], "o")
        for seed in (0.0, 1.0, 2.0):
            d = np.asarray(evaluate(FunctionGraph(g, {}), {}, seed=seed)) - [0.5, 0.4, 0.3]
            self.assertAlmostEqual(d[0], d[1], places=6)
            self.assertAlmostEqual(d[1], d[2], places=6)
            self.assertLessEqual(abs(d[0]), 0.2)

    def test_parity_disorder_normal_and_temperature(self):
        def one(kind, ports, **inputs):
            nodes = [const(k, t, v) for k, (t, v) in inputs.items()]
            return evaluate(FunctionGraph(fg(nodes + [("o", kind, {}, {p: k for p, k in zip(ports, inputs)})], "o"), {}),
                            {})
        self.assertTrue(one("parity_test", ["Input"], i=("float", 4.0)))
        self.assertFalse(one("parity_test", ["Input"], i=("float", 7.0)))
        self.assertTrue(one("parity_test", ["Input"], i=("float", -2.0)))
        x, y = one("disorder_anim", ["disorder_angle", "speed"], a=("float", 0.0), s=("float", math.tau))
        self.assertTrue(0.25 - 1e-9 <= math.hypot(x, y) <= 1.0)  # a step of 0.25 .. 1 times speed / 2 pi
        x, y = one("normal_distribution", ["radius", "uniform_random"], r=("float", 4.0), u=("float2", [0.5, 0.0]))
        self.assertAlmostEqual(math.hypot(x, y), math.sqrt(2 * math.log(2)), places=6)  # median of a unit Rayleigh
        # Blackbody colours in linear sRGB at luminance 1: warm at 2000 K, about white at 6500 K, blue at 20000 K.
        warm = one("temperature_to_rgb", ["temperature"], t=("float", 2000.0))
        white = one("temperature_to_rgb", ["temperature"], t=("float", 6500.0))
        blue = one("temperature_to_rgb", ["temperature"], t=("float", 20000.0))
        np.testing.assert_allclose(warm, [2.478, 0.660, 0.019], atol=0.012)
        np.testing.assert_allclose(white, [1.042, 0.984, 1.035], atol=0.012)
        self.assertGreater(blue[2], blue[0])
        for c in (warm, white, blue):
            self.assertAlmostEqual(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2], 1.0, delta=0.02)

    def test_non_square_output_size_sets_the_noise_variables(self):
        # A 1024 x 512 image at Scale 4: 8 cells along the long side, the short side's half of them.
        g = fg([const("s", "int", 4), const("size", "int2", [0, 0]), const("nse", "bool", True),
                ("n", "non_square_output_size", {}, {"scale": "s", "output_size": "size", "non_square": "nse"}),
                ("r", "get_float4", {"variable": "render_region"}, {}), ("q", "sequence", {}, {"in_": "n", "last": "r"})],
               "q")
        env = {"$sizelog2": [10.0, 9.0]}
        f = FunctionGraph(g, {"$sizelog2": "float2"})
        self.assertEqual(evaluate(f, env, shared=True), [-3.5, 4.5, -1.5, 2.5])
        self.assertEqual((env["real_scale"], env["xy_ratio"]), (8, [1.0, 0.5]))


class ParameterFunctions(unittest.TestCase):
    def test_function_reads_size_and_graph_parameters(self):
        pkg = Package()
        g = pkg.new_graph("g", 8)  # 256 px
        g.expose("amount", Param(ParamKind.FLOAT, 2.0))
        b = g.add_node("blur")
        # intensity = $size.x / 64 * amount  -> 256 / 64 * 2 = 8
        f = fg([("size", "get_float2", {"variable": "$size"}, {}), ("x", "swizzle_float", {"x": "X"}, {"vector": "size"}),
                const("k", "float", 64.0), ("d", "division", {}, {"a": "x", "b": "k"}),
                ("amt", "get_float", {"variable": "amount"}, {}), ("r", "multiplication", {}, {"a": "d", "b": "amt"})], "r")
        g.set_function(b, "intensity", f)
        self.assertEqual(compile_graph(pkg, "g").jobs[b].values["intensity"], 8.0)

    def test_function_type_must_match_the_parameter(self):
        pkg = Package()
        g = pkg.new_graph("g", 6)
        b = g.add_node("blur")
        with self.assertRaises(DocumentError):
            g.set_function(b, "intensity", fg([const("a", "int", 2)], "a"))

    def test_functions_round_trip(self):
        pkg = Package()
        g = pkg.new_graph("g", 6)
        b = g.add_node("blur")
        g.set_function(b, "intensity", fg([const("a", "float", 3.0)], "a"))
        again = Package.from_dict(pkg.to_dict())
        self.assertEqual(compile_graph(again, "g").jobs[b].values["intensity"], 3.0)


class PixelProcessor(unittest.TestCase):
    def build(self, function, color_mode="Color", inputs=1):
        pkg = Package()
        g = pkg.new_graph("g", 6)
        n = g.add_node("pixel_processor")
        g.set_param(n, "input_count", inputs)
        g.set_param(n, "color_mode", color_mode)
        if function is not None:
            g.set_param(n, "per_pixel_function", function)
        if inputs:
            src = g.add_node("test_pattern_color")
            g.connect(src, "output", n, "input_image_0")
        else:
            src = None
        return pkg, g, n, src

    def read(self, pkg, n):
        a = EV.read(compile_graph(pkg, "g").jobs[n], "output")
        return a.astype(np.float64) / (255.0 if a.dtype == np.uint8 else 1.0)

    def test_default_passes_the_input_through(self):
        pkg, g, n, src = self.build(None)
        np.testing.assert_allclose(self.read(pkg, n), self.read(pkg, src), atol=1 / 255 + 1e-6)

    def test_position_gradient(self):
        f = fg([("pos", "get_float2", {"variable": "$pos"}, {}), ("x", "swizzle_float", {"x": "X"}, {"vector": "pos"})], "x")
        pkg, g, n, _ = self.build(f, "Grayscale", inputs=0)
        a = self.read(pkg, n)[:, :, 0]
        self.assertAlmostEqual(a[10, 0], 0.5 / 64, delta=0.01)
        self.assertAlmostEqual(a[10, 63], 63.5 / 64, delta=0.01)
        self.assertTrue(np.all(np.diff(a[10]) > 0))

    def test_reads_graph_parameters(self):
        f = fg([("v", "get_float", {"variable": "level"}, {})], "v")
        pkg = Package()
        g = pkg.new_graph("g", 5)
        g.expose("level", Param(ParamKind.FLOAT, 0.25))
        n = g.add_node("pixel_processor")
        g.set_param(n, "input_count", 0)
        g.set_param(n, "color_mode", "Grayscale")
        g.set_param(n, "per_pixel_function", f)
        a = EV.read(compile_graph(pkg, "g").jobs[n], "output").astype(np.float64) / 255
        self.assertAlmostEqual(a.mean(), 0.25, delta=0.01)

    def test_calls_run_inline_on_the_gpu(self):
        # out = half(one_minus($pos.x)), half Setting w (its input times 2) for the caller: 0.5 (1 - x) + w * 0
        half = {**HALF, "nodes": HALF["nodes"] + [
            {"id": "two", "type": "const_float", "params": {"value": 2.0}},
            {"id": "w2", "type": "multiplication", "inputs": {"a": "x", "b": "two"}},
            {"id": "s", "type": "set", "params": {"variable": "w"}, "inputs": {"input": "w2"}},
            {"id": "q", "type": "sequence", "inputs": {"in_": "s", "last": "m"}}], "output": "q"}
        f = calling([("pos", "get_float2", {"variable": "$pos"}, {}), ("x", "swizzle_float", {"x": "X"}, {"vector": "pos"}),
                     ("o", "one_minus", {}, {"x": "x"}), ("h", "call", {"function": "half"}, {"x": "o"}),
                     ("w", "get_float", {"variable": "w"}, {}), const("z", "float", 0.0),
                     ("wz", "multiplication", {}, {"a": "w", "b": "z"}), ("r", "add", {}, {"a": "h", "b": "wz"})],
                    "r", {"half": half})
        pkg, g, n, _ = self.build(f, "Grayscale", inputs=0)
        a = self.read(pkg, n)[:, :, 0]
        x = (np.arange(64) + 0.5) / 64
        np.testing.assert_allclose(a[10], 0.5 * (1 - x), atol=1 / 255 + 1e-6)

    def test_reads_boolean_graph_parameters(self):
        # A Boolean parameter is an i32 uniform; an If...Else on it (in a Pixel processor) must compile.
        f = fg([("b", "get_bool", {"variable": "flip"}, {}), const("one", "float", 1.0), const("q", "float", 0.25),
                ("i", "if_else", {}, {"condition": "b", "if_": "q", "else_": "one"})], "i")
        pkg = Package()
        g = pkg.new_graph("g", 5)
        g.expose("flip", Param(ParamKind.BOOL, True))
        n = g.add_node("pixel_processor")
        g.set_param(n, "input_count", 0)
        g.set_param(n, "color_mode", "Grayscale")
        g.set_param(n, "per_pixel_function", f)
        a = EV.read(compile_graph(pkg, "g").jobs[n], "output").astype(np.float64) / 255
        self.assertAlmostEqual(a.mean(), 0.25, delta=0.01)

    def test_input_count_changes_ports_and_drops_links(self):
        pkg, g, n, src = self.build(None)
        self.assertIn("input_image_0", g.nodes[n].inputs)
        g.set_param(n, "input_count", 0)
        self.assertEqual(g.nodes[n].inputs, {})


class LibraryOnTheGpu(unittest.TestCase):
    def test_every_library_function_compiles_and_agrees_with_the_cpu(self):
        # Each Library node in a Value processor, on fixed inputs: the WGSL compiles and gives the CPU's value
        # (random and hash functions only compile: their bits differ between the two by design or by precision, 32 bit
        # floats on the GPU).
        from trafilatura3d.functions import LIBRARY
        rng = np.random.default_rng(3)

        def value(t):
            n = int(t[-1]) if t[-1].isdigit() else 1
            v = {"float": lambda: round(float(rng.uniform(0.1, 0.9)), 3), "int": lambda: int(rng.integers(0, 3)),
                 "bool": lambda: True}[t.rstrip("234")]
            return v() if n == 1 else [v() for _ in range(n)]
        for t, body in LIBRARY.items():
            with self.subTest(t):
                if any(n.get("params", {}).get("variable") == "render_region" for n in body["nodes"]):
                    continue  # needs a caller Setting render_region
                inputs = {p: value(ty) for p, ty in body["inputs"].items()}
                consts = [(f"i_{p}", f"const_{ty}", {"value": " ".join(map(str, v)) if ty in ("int3", "int4") else v}, {})
                          for (p, v), ty in zip(inputs.items(), body["inputs"].values())]
                f = fg(consts + [("c", t, {}, {p: f"i_{p}" for p in inputs})], "c")
                pkg = Package()
                g = pkg.new_graph("g", 4)
                vp = g.add_node("value_processor")
                g.set_param(vp, "value_processor_function", f)
                got = EV.read(compile_graph(pkg, "g").jobs[vp], "output")[0, 0].astype(np.float64)
                if "random" in str(body) or t.startswith("hash") or t == "global_random":
                    continue
                cpu = evaluate(FunctionGraph(f, {"$sizelog2": "float2", "$time": "float", "$tiling": "int"}),
                               {"$sizelog2": [4.0, 4.0], "$time": 0.0, "$tiling": 3})
                cpu = np.atleast_1d(np.asarray(cpu, np.float64))
                np.testing.assert_allclose(got[:len(cpu)], cpu, rtol=1e-3, atol=1e-4)


class Values(unittest.TestCase):
    def test_value_processor_computes_a_value(self):
        pkg = Package()
        g = pkg.new_graph("g", 6)
        g.expose("k", Param(ParamKind.FLOAT, 0.25))
        vp = g.add_node("value_processor")
        f = fg([("k", "get_float", {"variable": "k"}, {}), const("two", "float", 2.0),
                ("m", "multiplication", {}, {"a": "k", "b": "two"})], "m")
        g.set_param(vp, "value_processor_function", f)
        job = compile_graph(pkg, "g").jobs[vp]
        self.assertEqual(job.size, (1, 1))
        self.assertAlmostEqual(float(EV.read(job, "output")[0, 0, 0]), 0.5, places=5)

    def test_values_only_connect_to_values(self):
        pkg = Package()
        g = pkg.new_graph("g", 6)
        vp = g.add_node("value_processor")
        b = g.add_node("blur")
        with self.assertRaises(DocumentError):
            g.connect(vp, "output", b, "input")

    def test_value_into_an_instance_reaches_its_functions(self):
        pkg = Package()
        inner = pkg.new_graph("inner", 5)
        iv = inner.add_node("input_value")
        inner.set_param(iv, "identifier", "amount")
        pp = inner.add_node("pixel_processor")
        inner.set_param(pp, "input_count", 0)
        inner.set_param(pp, "color_mode", "Grayscale")
        inner.set_param(pp, "per_pixel_function", fg([("a", "get_float4", {"variable": "amount"}, {}),
                                                      ("x", "swizzle_float", {"x": "X"}, {"vector": "a"})], "x"))
        out = inner.add_node("output")
        inner.connect(pp, "output", out, "input")
        outer = pkg.new_graph("outer", 5)
        vp = outer.add_node("value_processor")
        outer.set_param(vp, "value_processor_function", fg([const("v", "float4", [0.75, 0, 0, 0])], "v"))
        inst = outer.add_node("graph_instance", ref="inner")
        outer.connect(vp, "output", inst, "amount")
        c = compile_graph(pkg, "outer", evaluator=EV)
        job = c.jobs[f"{inst}/{pp}"]
        self.assertEqual(job.size, (32, 32))  # the value connection doesn't drive inheritance
        a = EV.read(job, "output").astype(np.float64) / 255
        self.assertAlmostEqual(a.mean(), 0.75, delta=0.01)
        # Without an evaluator the Input value's default (0) is used.
        d = compile_graph(pkg, "outer")
        self.assertAlmostEqual(EV.read(d.jobs[f"{inst}/{pp}"], "output").mean(), 0.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
