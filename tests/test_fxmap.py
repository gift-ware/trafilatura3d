"""FX-Map: the quad-tree walk (Quadrant, Iterate, Switch) and the GPU drawing."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph
from trafilatura3d.fxmap import level_weights, levels, walk
from trafilatura3d.nodes import NODES

EV = Evaluator()


def quad(k, pattern="No Pattern", children=None, functions=None, **params):
    return {"id": k, "type": "quadrant", "params": {"pattern": pattern, **params}, "children": children or {},
            "functions": functions or {}}


def number_offset():
    """A function: branch offset = ($number * 0.1, 0)."""
    return {"nodes": [{"id": "n", "type": "get_float", "params": {"variable": "$number"}},
                      {"id": "k", "type": "const_float", "params": {"value": 0.1}},
                      {"id": "x", "type": "multiplication", "inputs": {"a": "n", "b": "k"}},
                      {"id": "z", "type": "const_float", "params": {"value": 0.0}},
                      {"id": "v", "type": "vector_float2", "inputs": {"input_1": "x", "input_2": "z"}}], "output": "v"}


class Walk(unittest.TestCase):
    def test_quadrants_subdivide(self):
        data = {"nodes": [quad("root", "Disc", {s: "leaf" for s in ("q1", "q2", "q3", "q4")}), quad("leaf", "Square")],
                "root": "root"}
        draws = walk(data, {}, {}, (256, 256), 0)
        self.assertEqual(len(draws), 5)
        self.assertEqual(draws[0].center, (0.5, 0.5))
        self.assertEqual(sorted(d.center for d in draws[1:]), [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)])
        self.assertTrue(all(d.size == (0.5, 0.5) for d in draws[1:]))

    def test_iterate_repeats_with_number(self):
        data = {"nodes": [{"id": "it", "type": "iterate", "params": {"iterations": 3}, "children": {"repeated": "q"}},
                          quad("q", "Disc", functions={"branch_offset": number_offset()})], "root": "it"}
        draws = walk(data, {}, {}, (256, 256), 0)
        self.assertEqual([round(d.center[0], 3) for d in draws], [0.5, 0.6, 0.7])

    def test_iterate_default_repeats_once(self):
        """Number Added defaults to 1: an Iterate left at its default merges its two branches."""
        data = {"nodes": [{"id": "it", "type": "iterate", "children": {"pass_through": "a", "repeated": "b"}},
                          quad("a", "Disc"), quad("b", "Square")], "root": "it"}
        self.assertEqual([d.pattern for d in walk(data, {}, {}, (64, 64), 0)], [3, 2])  # Disc, then Square once

    def test_switch_picks_a_branch(self):
        data = {"nodes": [{"id": "s", "type": "switch", "params": {"selector": 1},
                           "children": {"input_1": "a", "input_2": "b"}}, quad("a", "Disc"), quad("b", "Square")],
                "root": "s"}
        self.assertEqual([d.pattern for d in walk(data, {}, {}, (64, 64), 0)], [2])  # Square


class Drawing(unittest.TestCase):
    def cook(self, fx, **params):
        pkg = Package()
        g = pkg.new_graph("g", 6)
        n = g.add_node("stamp_map")
        g.set_param(n, "fxmap", fx)
        for k, v in params.items():
            g.set_param(n, k, v)
        a = EV.read(compile_graph(pkg, "g").jobs[n], "output")
        return a.astype(np.float64)[:, :, 0] / 255

    def test_a_disc(self):
        a = self.cook({"nodes": [quad("r", "Disc", pattern_size=[0.5, 0.5])], "root": "r"})
        self.assertGreater(a[32, 32], 0.95)
        self.assertLess(a[2, 2], 0.05)

    def test_patterns_wrap_across_tiled_edges(self):
        a = self.cook({"nodes": [quad("r", "Disc", pattern_size=[0.5, 0.5], pattern_offset=[0.5, 0.5])], "root": "r"})
        for y, x in ((0, 0), (0, 63), (63, 0), (63, 63)):
            self.assertGreater(a[y, x], 0.9)
        self.assertLess(a[32, 32], 0.05)

    def test_far_and_long_patterns_tile(self):
        # Offsets several images away land modulo the period (Crystal 2's disorder); a pattern longer than the image
        # is drawn on every period it spans: a long, slightly tilted bar comes back as one stripe per period.
        far = self.cook({"nodes": [quad("r", "Disc", pattern_size=[0.5, 0.5], pattern_offset=[-3.0, 4.0])],
                         "root": "r"})
        self.assertGreater(far[32, 32], 0.95)
        long = self.cook({"nodes": [quad("r", "Square", pattern_size=[6.0, 0.03], pattern_rotation=0.01)],
                          "root": "r"})
        column = long[:, 32] > 0.5
        self.assertGreaterEqual(int(np.sum(column[1:] & ~column[:-1])), 6)  # stripes: 7 periods cross the column

    def test_alpha_blend_draws_later_patterns_on_top(self):
        fx = {"nodes": [{"id": "it", "type": "iterate", "params": {"iterations": 1},
                         "children": {"pass_through": "a", "repeated": "b"}},
                        quad("a", "Square", color_luminosity=[1, 1, 1, 1], blending_mode="Alpha blend"),
                        quad("b", "Square", color_luminosity=[0.2, 0.2, 0.2, 1], blending_mode="Alpha blend",
                             pattern_size=[0.5, 0.5])], "root": "it"}
        a = self.cook(fx, background=[0.6, 0.6, 0.6, 1])
        self.assertAlmostEqual(a[32, 32], 0.2, delta=0.02)  # the second square covers the first
        self.assertAlmostEqual(a[4, 4], 1.0, delta=0.02)
        half = self.cook({"nodes": [quad("r", "Square", color_luminosity=[0.2, 0.2, 0.2, 0.5],
                                         blending_mode="Alpha blend")], "root": "r"}, background=[0.6, 0.6, 0.6, 1])
        self.assertAlmostEqual(half[32, 32], 0.4, delta=0.02)  # half way from the background to the colour

    def test_max_keeps_the_brighter_of_background_and_pattern(self):
        fx = {"nodes": [quad("r", "Square", color_luminosity=[0.3, 0.3, 0.3, 1], blending_mode="Max",
                             pattern_size=[0.5, 0.5])], "root": "r"}
        self.assertAlmostEqual(self.cook(fx, background=[0.5, 0.5, 0.5, 1])[32, 32], 0.5, delta=0.02)
        self.assertAlmostEqual(self.cook(fx, background=[0.1, 0.1, 0.1, 1])[32, 32], 0.3, delta=0.02)

    def test_roughness_weighs_the_levels(self):
        # Roughness: r^d normalised over the tree's levels, 1/levels at 1, off at 0.
        two = {"nodes": [quad("r", children={"q1": "l", "q2": "l", "q3": "l", "q4": "l"}), quad("l", "Square")],
               "root": "r"}
        self.assertEqual(levels(two), 2)
        self.assertAlmostEqual(self.cook(two)[16, 16], 0.5, delta=0.02)  # default 1: half for 2 levels
        self.assertAlmostEqual(self.cook(two, roughness=0.0)[16, 16], 1.0, delta=0.02)
        self.assertAlmostEqual(self.cook(two, roughness=0.5)[16, 16], 0.5 * 0.5 / 0.75, delta=0.02)
        self.assertAlmostEqual(self.cook(two, global_opacity=2.0)[16, 16], 1.0, delta=0.02)
        w = level_weights(0.6, np.arange(5), 5)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertAlmostEqual(w[1] / w[0], 0.6)

    def test_grayscale_luminosity_of_a_colour_is_its_rgb_average_bounded_by_alpha(self):
        # The RGB average (Wood fibers 2's (0, 1, 1, rand) is not black), at most the alpha (it keeps the random
        # coverage instead of a constant 2/3); a gray value (v, v, v, 1) or (v, v, v, v) stays v (Clouds 3).
        for color, want in (([0.0, 1.0, 1.0, 0.25], 0.25), ([0.0, 1.0, 1.0, 1.0], 2 / 3), ([0.4, 0.4, 0.4, 1.0], 0.4),
                            ([0.4, 0.4, 0.4, 0.4], 0.4)):
            fx = {"nodes": [quad("r", "Square", color_luminosity=color)], "root": "r"}
            self.assertAlmostEqual(self.cook(fx)[32, 32], want, delta=0.01)

    def test_hemisphere(self):
        a = self.cook({"nodes": [quad("r", "Hemisphere")], "root": "r"})
        self.assertAlmostEqual(a[32, 32], 1.0, delta=0.03)
        self.assertAlmostEqual(a[32, 48], (1 - 0.5 ** 2) ** 0.5, delta=0.04)

    def test_add_sub_is_the_default_and_signed(self):
        from trafilatura3d.fxmap import FX_PARAMS
        self.assertEqual(FX_PARAMS["quadrant"]["blending_mode"].default, "Add Sub")
        fx = {"nodes": [{"id": "it", "type": "iterate", "params": {"iterations": 1},
                         "children": {"pass_through": "a", "repeated": "b"}},
                        quad("a", "Square"),
                        quad("b", "Square", color_luminosity=[-0.5, -0.5, -0.5, 1], pattern_size=[0.5, 0.5])],
               "root": "it"}
        a = self.cook(fx)
        self.assertAlmostEqual(a[32, 32], 0.5, delta=0.02)  # 1 - 0.5: a negative colour subtracts
        self.assertAlmostEqual(a[4, 4], 1.0, delta=0.02)

    def test_global_opacity(self):
        a = self.cook({"nodes": [quad("r", "Square")], "root": "r"}, global_opacity=0.5)
        self.assertAlmostEqual(a[32, 32], 0.5, delta=0.02)

    def test_tiling_copies_are_every_shift_that_reaches_the_image(self):
        """The draw's copies (`Copies`, made by the draw shader) are what testing every shift gives: each pattern,
        then its copies in shift order (x then y: 0, +1, -1, +2... periods), on every tiling, with and without Cull
        outside: the layer drawn from the patterns is the one drawn from those copies listed."""
        import wgpu
        from trafilatura3d.evaluator import Evaluator
        from trafilatura3d.nodes.atomic.stamp_map import MAX_INPUTS, Copies, _instances, _renderer
        ev = Evaluator()
        d = ev.device
        empty, white = ev.constant((0.0, 0.0, 0.0, 0.0)), ev.constant((1.0, 1.0, 1.0, 1.0))

        def draw(data, copies):
            t = d.create_texture(size=(64, 64, 1), format="r32float",
                                 usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
            enc = d.create_command_encoder()
            _renderer(d).draw(enc, t, data, copies, [empty] * MAX_INPUTS, (empty, True), (64, 64), (white, 0.0))
            d.queue.submit([enc.finish()])
            return np.frombuffer(d.queue.read_texture({"texture": t}, {"bytes_per_row": 256}, (64, 64, 1)), np.float32)

        rng = np.random.default_rng(3)
        rows = rng.random((400, 16)).astype(np.float32)
        rows[:, 0:2] = rng.uniform(-2.5, 3.5, (400, 2))
        rows[:, 2:4] *= rng.choice([0.05, 3.0], (400, 1)).astype(np.float32)
        rows[:, 6] = 3  # a disc
        rows[:, 8:12] = 0.01
        rows[:, 14] = 0  # Add Sub
        rows[:, 15] = rng.integers(0, 4, 400)
        v = {"render_region": [0.0, 1.0, 0.0, 1.0], "tiling_region": [0.0, 1.0, 0.0, 1.0], "global_opacity": 0.7,
             "roughness": 0.6}
        plain = _instances(rows, {**v, "cull_outside": False}, 0, 4)[0]  # no tiling: each pattern once, weighted
        for tiling in range(4):
            for cull in (False, True):
                ref = plain.copy()
                for axis in (0, 1):
                    if tiling & (1 << axis):
                        ref[:, axis] = 0.0 + np.mod(ref[:, axis] - 0.0, 1.0)
                reach = 0.5 * np.hypot(ref[:, 2], ref[:, 3])
                k = int(np.ceil(float(reach.max()))) + 1
                shifts = [np.float32(0.0)] + [np.float32(s * j) for j in range(1, k + 1) for s in (1, -1)]
                sx, sy = (shifts if tiling & bit else shifts[:1] for bit in (1, 2))
                expected = []
                for row, r in zip(ref, reach):
                    for ox in sx:
                        for oy in sy:
                            x, y = row[0] + ox, row[1] + oy
                            first = ox == 0 and oy == 0 and not cull
                            if first or (x + r >= 0 and x - r <= 1 and y + r >= 0 and y - r <= 1):
                                expected.append([x, y, *row[2:]])
                with self.subTest(tiling=tiling, cull=cull):
                    data, copies = _instances(rows, {**v, "cull_outside": cull}, tiling, 4)
                    listed = np.array(expected, np.float32).reshape(-1, 16)
                    # the shader adds the shifts: its compiler may fuse that addition (the last digit of an edge)
                    np.testing.assert_allclose(draw(data, copies), draw(listed, Copies(0, 0, 1.0, 1.0, False)),
                                               rtol=0, atol=1e-6)


def engine_generators() -> list[str]:
    """The generators that draw through the FX-Map engine (stamp_map.draw_patterns): cooked once at 32 px with its
    instance builder watched."""
    from unittest import mock

    import trafilatura3d.nodes.atomic.stamp_map as fm
    found = []
    for t, d in sorted(NODES.items()):
        if d.inputs or t == "stamp_map" or not d.category.startswith("Texture generators"):
            continue
        with mock.patch.object(fm, "_instances", wraps=fm._instances) as spy:
            pkg = Package()
            n = pkg.new_graph("g", 5).add_node(t)
            EV.read(compile_graph(pkg, "g").jobs[n], next(iter(d.outputs)))
        if spy.called:
            found.append(t)
    return found


class Resolution(unittest.TestCase):
    def test_engine_generators_look_the_same_at_every_resolution(self):
        """Every generator drawn by the FX-Map engine keeps its mean and spread from 256 to 1024 pixels (a
        rasterisation rule that depends on the pixel size made Waveform 1 swing from black to its preview). Legacy
        nodes that draw at absolute sizes (`absolute_sizes`) change with the resolution."""
        types = engine_generators()
        self.assertGreater(len(types), 15)  # the stamped, legacy and directional noises, Waveform 1...
        for t in types:
            if getattr(NODES[t], "absolute_sizes", False):
                continue
            with self.subTest(node=t):
                stats = []
                for log2 in (8, 9, 10):
                    pkg = Package()
                    g = pkg.new_graph("g", log2)
                    n = g.add_node(t)
                    g.set_base(n, "output_format", {"mode": "absolute", "value": "16f"})
                    a = EV.read(compile_graph(pkg, "g").jobs[n], "output").astype(np.float64)[..., 0]
                    stats.append((a.mean(), a.std()))
                ref = stats[1]
                for m, s in stats:
                    self.assertAlmostEqual(m, ref[0], delta=0.04)
                    self.assertAlmostEqual(s, ref[1], delta=0.04)


if __name__ == "__main__":
    unittest.main()
