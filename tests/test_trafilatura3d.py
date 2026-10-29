"""Package rules, compilation and GPU cooking on small textures: python -m unittest discover tests"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from trafilatura3d import DocumentError, Evaluator, Package, compile_graph, export, imageio
from trafilatura3d.evaluator import BYTES_PER_PIXEL, NORM16, NORM16_FEATURE, NUMPY, _unpacked, default_device
from trafilatura3d.nodes.base import Param, ParamKind, PortType

SMALL = {"output_size": {"mode": "absolute", "value": [6, 6]}}  # 64 px
EV = None


def ev():
    global EV
    if EV is None:
        EV = Evaluator()
    return EV


def soften_package() -> Package:
    """`soften` (gray in -> blur -> levels -> out, blur exposed as `amount`) used by `main`."""
    pkg = Package()
    s = pkg.new_graph("soften")
    i, b, l, o = (s.add_node(t) for t in ("input_grayscale", "blur", "levels", "output"))
    s.connect(i, "output", b, "input")
    s.connect(b, "output", l, "input")
    s.connect(l, "output", o, "input")
    s.expose("amount", Param(ParamKind.FLOAT, 4.0))
    s.link(b, "intensity", "amount")
    g = pkg.new_graph("main")
    g.base = dict(SMALL)
    n = g.add_node("noise")
    inst = g.add_node("graph_instance", ref="soften")
    g.connect(n, "output", inst, "input1")
    gm = g.add_node("gradient_map")
    g.connect(inst, "output1", gm, "input")
    out = g.add_node("output")
    g.connect(gm, "output", out, "input")
    return pkg


def levels_chain(n: int, branches: bool = False) -> tuple[Package, list[str]]:
    """A noise and `n` Levels in 16 bit (one value per texel), each after the one before (`branches`: each after the
    noise), all different."""
    pkg = Package()
    g = pkg.new_graph("g")
    g.base = {**SMALL, "output_format": {"mode": "absolute", "value": "16"}}
    prev = noise = g.add_node("noise")
    nodes = [noise]
    for i in range(n):
        lv = g.add_node("levels")
        g.set_param(lv, "level_in_mid", [0.3 + 0.4 * (i % 2)] * 4)
        g.connect(noise if branches else prev, "output", lv, "input")
        nodes.append(lv)
        prev = lv
    return pkg, nodes


def _size_of(job) -> int:
    return job.size[0] * job.size[1] * BYTES_PER_PIXEL[ev().storage_format(job.depths["output"], True)]


class PackageRules(unittest.TestCase):
    def setUp(self):
        self.pkg = soften_package()
        self.g = self.pkg.graphs["main"]

    def assertRefused(self, fragment, fn, *args):
        before = self.pkg.to_dict()
        with self.assertRaises(DocumentError) as cm:
            fn(*args)
        self.assertIn(fragment, str(cm.exception))
        self.assertEqual(self.pkg.to_dict(), before)

    def test_round_trip(self):
        again = Package.from_dict(self.pkg.to_dict())
        self.assertEqual(again.to_dict(), self.pkg.to_dict())

    def test_save_creates_missing_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new" / "deeper" / "p.trafilatura3d"
            self.pkg.save(path)
            self.assertEqual(self.pkg.path, path.resolve())
            self.assertEqual(Package.load(path).to_dict(), self.pkg.to_dict())
            blocked = Path(tmp) / "file"  # a file where a folder should be: refused, the path stays
            blocked.write_text("", encoding="utf-8")
            with self.assertRaises(OSError):
                self.pkg.save(blocked / "p.trafilatura3d")
            self.assertEqual(self.pkg.path, path.resolve())

    def test_instance_ports_come_from_the_graph(self):
        d = self.g.definition("soften1")
        self.assertEqual(list(d.inputs), ["input1"])
        self.assertEqual(list(d.outputs), ["output1"])
        self.assertEqual(list(d.params), ["amount"])

    def test_exposed_parameter_properties(self):
        """The input parameter properties survive a save; rename follows links and Visible if;
        Clean Inputs finds the unused ones; Clamp makes Min/Max hard limits."""
        g = self.pkg.graphs["soften"]
        g.expose("mode", Param(ParamKind.INT, 1, "Mode", editor="Drop down list", items=((1, "Flat"), (2, "Bumpy"))))
        g.expose("amt", Param(ParamKind.FLOAT, 2.0, "Amount", 0.0, 1.0, group="A/B", description="d", clamp=True,
                              step=0.1, user_data="u", visible_if="input.mode == 2"))
        self.assertEqual(g.parameters["amt"].param.default, 1.0)  # clamped
        again = Package.from_dict(self.pkg.to_dict()).graphs["soften"]
        p = again.parameters["amt"].param
        self.assertEqual((p.group, p.description, p.clamp, p.step, p.user_data, p.visible_if),
                         ("A/B", "d", True, 0.1, "u", "input.mode == 2"))
        self.assertEqual(again.parameters["mode"].param.items, ((1, "Flat"), (2, "Bumpy")))
        g.rename_parameter("mode", "style")
        self.assertEqual(g.parameters["amt"].param.visible_if, "input.style == 2")
        g.rename_parameter("amount", "softness")
        self.assertEqual(g.nodes["blur1"].links["intensity"], "softness")
        self.assertEqual(g.unused_parameters(), ["style", "amt"])

    def test_visible_if_expressions(self):
        from trafilatura3d.visibility import check, visible
        v = {"a": True, "n": 2, "f": 0.5, "v": [1.0, 5.0], "z": 0}
        cases = {"input.a": True, "!input.a": False, "input.n < 3": True, 'input["n"] == 2': True,
                 "input.v.y < 3": False, "input.a || input.z": True, "input.n > 0 && input.f > 1": False,
                 "input.z": False, "input.a == true": True, "": True}
        for expr, want in cases.items():
            self.assertEqual(visible(expr, v), want, expr)
        self.assertIsNotNone(check("input.missing", v))
        self.assertIsNotNone(check("__import__('os')", v))
        self.assertTrue(visible("input.missing", v))  # invalid: visible

    def test_preview_values_override_the_defaults(self):
        a = compile_graph(self.pkg, "soften").jobs["blur1"].values["intensity"]
        b = compile_graph(self.pkg, "soften", values={"amount": 7.0}).jobs["blur1"].values["intensity"]
        self.assertEqual((a, b), (4.0, 7.0))

    def test_input_number_unlocks_numbered_inputs(self):
        """Tile Sampler: Pattern Input Number decides how many Pattern Inputs exist on the node; lowering
        it hides the extra ones but keeps their connections (as Multi Switch does)."""
        ts = self.g.add_node("tile_sampler")
        d = lambda: self.g.definition(ts)
        shown = lambda: [k for k in d().visible_inputs(d().values(self.g.nodes[ts])) if k.startswith("pattern_input_")]
        self.assertEqual(shown(), ["pattern_input_1"])
        self.assertRefused("not available", self.g.connect, "noise1", "output", ts, "pattern_input_3")
        self.g.set_param(ts, "pattern_input_number", 3)
        self.assertEqual(shown(), ["pattern_input_1", "pattern_input_2", "pattern_input_3"])
        self.g.connect("noise1", "output", ts, "pattern_input_3")
        self.g.set_param(ts, "pattern_input_number", 1)
        self.assertEqual(shown(), ["pattern_input_1"])
        self.assertEqual(self.g.nodes[ts].inputs["pattern_input_3"], ["noise1", "output"])

    def test_typed_ports_refuse_the_wrong_type(self):
        self.assertRefused("expects grayscale", self.g.connect, "gradient_map1", "output", "soften1", "input1")

    def test_polymorphic_change_cannot_break_typed_inputs_downstream(self):
        lv = self.g.add_node("levels")
        inst2 = self.g.add_node("graph_instance", ref="soften")
        self.g.connect("noise1", "output", lv, "input")
        self.g.connect(lv, "output", inst2, "input1")
        # levels follows its input: feeding it color would make soften1.input1 receive color
        self.assertRefused("expects grayscale", self.g.connect, "gradient_map1", "output", lv, "input")

    def test_instances_follow_removed_and_renamed_inputs(self):
        soften = self.pkg.graphs["soften"]
        self.g.set_param("soften1", "amount", 2.0)
        soften.rename_parameter("amount", "strength")
        self.assertEqual(self.g.nodes["soften1"].params, {"strength": 2.0})
        soften.unexpose("strength")
        self.assertEqual(self.g.nodes["soften1"].params, {})
        self.assertEqual(self.g.problems(), [])

    def test_cycles_are_refused(self):
        b = self.g.add_node("blur")
        self.g.connect("soften1", "output1", b, "input")
        self.assertRefused("cycle", self.g.connect, b, "output", "soften1", "input1")

    def test_identifiers_unique(self):
        o2 = self.g.add_node("output")
        self.assertNotEqual(self.g.identifier(self.g.nodes[o2]), self.g.identifier(self.g.nodes["output1"]))
        self.assertRefused("identifier", self.g.set_param, o2, "identifier", "output1")

    def test_links_check_kinds(self):
        self.assertRefused("cannot link", self.g.link, "noise1", "scale", "nope")

    def test_broken_documents_do_not_load(self):
        data = self.pkg.to_dict()
        data["graphs"]["main"]["nodes"][0]["params"] = {"scale": "big"}
        with self.assertRaises(DocumentError):
            Package.from_dict(data)

    def test_unknown_reference(self):
        self.assertRefused("unknown graph", self.g.add_node, "graph_instance", (0, 0), "nope")


class Compilation(unittest.TestCase):
    def test_instances_are_inlined_with_their_parameters(self):
        pkg = soften_package()
        pkg.graphs["main"].set_param("soften1", "amount", 9.0)
        c = compile_graph(pkg, "main")
        self.assertIn("soften1/blur1", c.jobs)
        self.assertEqual(c.jobs["soften1/blur1"].values["intensity"], 9.0)
        self.assertIs(c.jobs["soften1/blur1"].inputs["input"][0], c.jobs["noise1"])  # input passes through

    def test_sizes_inherit_from_parent_and_input(self):
        pkg = soften_package()
        g = pkg.graphs["main"]
        g.set_base("noise1", "output_size", {"mode": "relative_parent", "value": [-1, -1]})
        c = compile_graph(pkg, "main")
        self.assertEqual(c.jobs["noise1"].size, (32, 32))
        self.assertEqual(c.jobs["gradient_map1"].size, (32, 32))  # relative to its input
        g.set_base("gradient_map1", "output_size", {"mode": "absolute", "value": [4, 5]})
        self.assertEqual(compile_graph(pkg, "main").jobs["gradient_map1"].size, (16, 32))

    def test_polymorphic_outputs_follow_inputs(self):
        pkg = soften_package()
        g = pkg.graphs["main"]
        lv = g.add_node("levels")
        g.connect("gradient_map1", "output", lv, "input")
        c = compile_graph(pkg, "main")
        self.assertIs(c.jobs[lv].output_types["output"], PortType.COLOR)
        self.assertIs(c.jobs["soften1/levels1"].output_types["output"], PortType.GRAY)

    def test_keys_change_only_downstream_of_an_edit(self):
        pkg = soften_package()
        before = {k: j.key for k, j in compile_graph(pkg, "main").jobs.items()}
        pkg.graphs["main"].set_param("gradient_map1", "gradient", [{"pos": 0, "color": [1, 0, 0]},
                                                                   {"pos": 1, "color": [0, 0, 1]}])
        after = {k: j.key for k, j in compile_graph(pkg, "main").jobs.items()}
        self.assertEqual({k for k in before if before[k] != after[k]}, {"gradient_map1", "output1"})

    def test_shift_computes_every_texture_smaller(self):
        """A dragged value's preview: every texture (relative or absolute size) the same powers of 2 smaller, in the
        evaluator's preview partition, while the sizes functions read stay those of the full resolution."""
        from trafilatura3d.evaluator import PREVIEW
        from trafilatura3d.compiler import system_variables
        pkg = soften_package()
        g = pkg.graphs["main"]
        g.set_base("gradient_map1", "output_size", {"mode": "absolute", "value": [5, 4]})
        full, small = compile_graph(pkg, "main"), compile_graph(pkg, "main", shift=2)
        for path, job in full.jobs.items():
            self.assertEqual(small.jobs[path].size, (job.size[0] // 4, job.size[1] // 4))
            self.assertEqual((job.partition, small.jobs[path].partition), ("", PREVIEW))
            self.assertNotEqual(small.jobs[path].key, job.key)
        self.assertEqual(small.jobs["gradient_map1"].size, (8, 4))
        ctx = small.jobs["noise1"].extra["context"]
        self.assertEqual(system_variables(ctx)["$size"], [64.0, 64.0])


class Cooking(unittest.TestCase):
    def test_edit_recooks_only_what_changed(self):
        pkg = soften_package()
        targets = lambda: [j for j, _ in compile_graph(pkg, "main").outputs.values()]
        e = Evaluator()
        e.cook(targets())
        pkg.graphs["main"].set_param("soften1", "amount", 1.0)
        self.assertEqual(set(e.cook(targets()).cooked), {"soften1/blur1", "soften1/levels1", "gradient_map1",
                                                        "output1"})
        self.assertEqual(e.cook(targets()).cooked, [])

    def test_unconnected_inputs_read_their_default(self):
        pkg = Package()
        g = pkg.new_graph("g")
        g.base = dict(SMALL)
        b = g.add_node("blend")  # nothing connected: background/foreground 0, opacity 1
        g.set_param(b, "blending_mode", "Add")
        job = compile_graph(pkg, "g").jobs[b]
        self.assertTrue(np.all(ev().read(job, "output") == 0))

    def test_histogram_counts_every_texel(self):
        """`Evaluator.histogram` (the Levels editor's) on the GPU: the same counts as numpy over every texel, the
        exact minimum and maximum, a grayscale result read as (v, v, v, 1); values outside 0-1 in the end bins."""
        pkg = Package()
        g = pkg.new_graph("g", 8)
        n = g.add_node("gaussian_noise")
        g.set_base(n, "output_format", {"mode": "absolute", "value": "32f"})
        lv = g.add_node("levels")  # 2v - 0.5: a quarter below 0 and above 1
        g.connect(n, "output", lv, "input")
        g.set_param(lv, "level_out_low", [-0.5] * 4)
        g.set_param(lv, "level_out_high", [1.5] * 4)
        g.set_param(lv, "intermediary_clamp", False)
        g.set_base(lv, "output_format", {"mode": "absolute", "value": "32f"})
        for node in (n, lv):
            job = compile_graph(pkg, "g").jobs[node]
            counts, lo, hi = ev().histogram(job, "output")
            v = ev().read(job, "output")[..., 0].astype(np.float64)
            want = np.histogram(np.clip(v, 0, 1), bins=256, range=(0, 1))[0]
            np.testing.assert_array_equal(counts[:, 0], want)
            np.testing.assert_array_equal(counts[:, 3], [0] * 255 + [v.size])  # alpha 1
            self.assertAlmostEqual(lo[0], v.min(), places=6)
            self.assertAlmostEqual(hi[0], v.max(), places=6)
        self.assertLess(lo[0], 0.0)
        self.assertGreater(hi[0], 1.0)

    def test_depths_quantize(self):
        pkg = soften_package()
        g = pkg.graphs["main"]
        g.set_base("noise1", "output_format", {"mode": "absolute", "value": "8"})
        v = ev().read(compile_graph(pkg, "main").jobs["noise1"], "output")
        self.assertEqual(v.dtype, np.uint8)
        g.set_base("noise1", "output_format", {"mode": "absolute", "value": "16"})
        v = ev().read(compile_graph(pkg, "main").jobs["noise1"], "output").astype(np.float64)
        np.testing.assert_allclose(v * 65535, np.round(v * 65535), atol=1e-2)

    def test_streaming_matches_full_cache(self):
        pkg = soften_package()
        c = compile_graph(pkg, "main")
        job = c.outputs["output1"][0]
        stream = Evaluator(budget_bytes=0)
        np.testing.assert_array_equal(ev().read(job, "output"), stream.read(job, "output"))
        self.assertEqual(set(stream.cache), {job.key})

    def test_cook_steps_a_job_at_a_time(self):
        """`cook_steps` yields after each job (a turn of the event loop between them) and gives what `cook` gives;
        another cook first completes it; closing it at a yield leaves what it computed cached, the next cook
        computes the rest."""
        pkg, nodes = levels_chain(5)
        last = compile_graph(pkg, "g").jobs[nodes[-1]]
        e = Evaluator()
        steps = e.cook_steps([last])
        turns = 0
        while True:
            try:
                next(steps)
                turns += 1
            except StopIteration as done:
                stats = done.value
                break
        self.assertEqual(turns, len(stats.cooked))
        np.testing.assert_array_equal(e.read(last, "output"), Evaluator().read(last, "output"))
        e.clear()
        steps = e.cook_steps([last])
        next(steps)
        e.cook([compile_graph(pkg, "g").jobs[nodes[0]]])  # completes the one in flight first
        self.assertIn(last.key, e.cache)
        with self.assertRaises(StopIteration):
            next(steps)
        e.clear()
        steps = e.cook_steps([last])
        next(steps)
        next(steps)
        steps.close()  # abandoned: two jobs done
        self.assertEqual(len(e.cook([last]).cooked), len(nodes) - 2)
        np.testing.assert_array_equal(e.read(last, "output"), Evaluator().read(last, "output"))

    def test_thumbnails_keep_to_the_budget(self):
        """A graph much larger than the budget (an imported 201 node graph at 4096 held 22 GB for its thumbnails, and
        they came out black): each thumbnail is made as soon as its result is there, the cache stays in budget and the
        images are those of a cache holding everything."""
        pkg, nodes = levels_chain(12)
        c = compile_graph(pkg, "g")
        items = [next(iter(c.nodes[n].values())) for n in nodes]
        one = _size_of(items[0][0])
        small = Evaluator(budget_bytes=3 * one)
        got = small.thumbnails(items, 32)
        self.assertLessEqual(small.cached_bytes(), 3 * one)
        for a, b in zip(got, Evaluator().thumbnails(items, 32)):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(all(a[..., :3].max() > 0 for a in got))

    def test_results_used_together_stay_alive(self):
        """Textures read together (the 3D view's channels) are cooked together: one by one, the second cook gave the
        first back to the pool, destroyed before the renderer copied it ("Texture has been destroyed")."""
        pkg, nodes = levels_chain(2, branches=True)
        c = compile_graph(pkg, "g")
        items = [next(iter(c.nodes[n].values())) for n in nodes]
        small = Evaluator(budget_bytes=_size_of(items[0][0]))
        for (job, out), t in zip(items, small.results(items)):
            dtype, n = NUMPY[t.format]
            data = small.device.queue.read_texture({"texture": t}, {"bytes_per_row": t.width * BYTES_PER_PIXEL[t.format]},
                                                   (t.width, t.height, 1))
            np.testing.assert_array_equal(_unpacked(np.frombuffer(data, dtype).reshape(t.height, t.width, n), t.format),
                                          ev().read(job, out))

    def test_export_files(self):
        """Export outputs: one format for all, the bit depth from each output's format, a name
        Pattern, the outputs chosen, the Batch tab's Output size."""
        pkg = soften_package()
        g = pkg.graphs["main"]
        h = g.add_node("output")
        g.set_param(h, "identifier", "height")
        g.set_param(h, "group", "maps")
        g.set_base(h, "output_format", {"mode": "absolute", "value": "32f"})
        g.connect("soften1", "output1", h, "input")
        with tempfile.TemporaryDirectory() as tmp:
            paths = export(pkg, "main", tmp, ev(), format="exr")
            info = {p.name: (imageio.read(p).shape, imageio.read(p).dtype) for p in paths}
            self.assertEqual(info, {"main_output1.exr": ((64, 64, 3), np.float16),
                                    "main_height.exr": ((64, 64, 1), np.float32)})
            paths = export(pkg, "main", tmp, ev(), format="png", pattern="$(group)-$(identifier)", outputs={"height"})
            self.assertEqual([p.name for p in paths], ["maps-height.png"])
            paths = export(pkg, "main", tmp, ev(), format="jpg", outputs={"output1"})
            self.assertEqual(imageio.read(paths[0]).dtype, np.uint8)
            # The Batch tab's Output size replaces the working (parent) resolution.
            flat = Package()
            fg = flat.new_graph("flat", 6)
            u, o = fg.add_node("uniform_color"), fg.add_node("output")
            fg.connect(u, "output", o, "input")
            paths = export(flat, "flat", tmp, ev(), size=(5, 5))
            self.assertEqual(imageio.read(paths[0]).shape[:2], (32, 32))

    def test_textures_tile(self):
        pkg = soften_package()
        h = ev().read(compile_graph(pkg, "main").jobs["soften1/levels1"], "output")[:, :, 0].astype(np.float64)
        inner = np.abs(np.diff(h, axis=1)).mean()
        self.assertLess(np.abs(h[:, 0] - h[:, -1]).mean(), inner * 2)

    def test_external_package_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = soften_package()
            del lib.graphs["main"]
            lib.save(Path(tmp) / "lib.trafilatura3d")
            pkg = Package(Path(tmp) / "main.trafilatura3d")
            g = pkg.new_graph("main")
            g.base = dict(SMALL)
            n = g.add_node("noise")
            inst = g.add_node("graph_instance", ref="lib.trafilatura3d#soften")
            g.connect(n, "output", inst, "input1")
            pkg.save()
            again = Package.load(Path(tmp) / "main.trafilatura3d")
            self.assertIn(f"{inst}/blur1", compile_graph(again, "main").jobs)


class GraphItems(unittest.TestCase):
    def test_disabled_node_passes_its_input_and_dot_is_transparent(self):
        pkg = Package()
        g = pkg.new_graph("g", 6)
        src, lv, dot, out = g.add_node("test_pattern"), g.add_node("levels"), g.add_node("dot"), g.add_node("output")
        g.set_param(lv, "level_out_high", [0.5] * 4)
        g.connect(src, "output", lv, "input")
        g.connect(lv, "output", dot, "input")
        g.connect(dot, "output", out, "input")
        before = ev().read(compile_graph(pkg, "g").outputs["output1"][0], "output")
        g.set_disabled([lv], True)
        after = ev().read(compile_graph(pkg, "g").outputs["output1"][0], "output")
        np.testing.assert_array_equal(after, ev().read(compile_graph(pkg, "g").jobs[src], "output"))
        self.assertLess(before.max(), after.max())
        self.assertNotIn(dot, compile_graph(pkg, "g").jobs)  # costs nothing

    def test_only_eligible_nodes_disable(self):
        pkg = Package()
        g = pkg.new_graph("g", 6)
        gen = g.add_node("test_pattern")
        with self.assertRaises(DocumentError):
            g.set_disabled([gen], True)  # no input

    def test_items_round_trip(self):
        pkg = soften_package()
        g = pkg.graphs["main"]
        g.add_item("frame", (10, 20), title="Base")
        g.add_item("comment", (0, 0), text="note")
        g.add_item("pin", (5, 5))
        again = Package.from_dict(pkg.to_dict()).graphs["main"]
        self.assertEqual([(i["kind"], i["position"]) for i in again.items],
                         [("frame", [10, 20]), ("comment", [0, 0]), ("pin", [5, 5])])


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless({NORM16_FEATURE, "texture-adapter-specific-format-features"} <= set(default_device().features),
                     "the device has no 16 bit integer storage")
class Norm16(unittest.TestCase):
    """16 bit results stored as 16 bit integers where the device can, else as 32 bit floats: the same values."""

    def test_both_storages_give_the_same_values(self):
        fallback = Evaluator(features=set(ev().device.features) - {NORM16_FEATURE})
        self.assertEqual((ev().storage_format("16", False), ev().storage_format("16", True),
                          ev().storage_format("16", True, alpha=True)), ("rgba16unorm", "r16unorm", "rg16unorm"))
        self.assertEqual((fallback.storage_format("16", False), fallback.storage_format("16", True)),
                         ("rgba32float", "r32float"))
        self.assertEqual(fallback.storage_format("8", True), "r8unorm")  # the other formats stay
        pkg = soften_package()
        g = pkg.graphs["main"]
        g.base = {**SMALL, "output_format": {"mode": "absolute", "value": "16"}}
        c = compile_graph(pkg, "main")
        jobs = [(j, o) for j in c.jobs.values() for o in j.output_types if not j.is_value(o)]
        self.assertTrue(any(j.depths[o] == "16" and j.is_gray(o) for j, o in jobs))
        self.assertTrue(any(j.depths[o] == "16" and not j.is_gray(o) for j, o in jobs))
        for job, out in jobs:
            a, b = ev().read(job, out), fallback.read(job, out)
            self.assertEqual((a.dtype, b.dtype), (b.dtype, b.dtype))
            np.testing.assert_array_equal(a.view(np.uint32) if a.dtype == np.float32 else a,
                                          b.view(np.uint32) if b.dtype == np.float32 else b, err_msg=job.path)
            t = ev().result(job, out)
            if t.format in NORM16:  # what the integers hold: round(v * 65535) of the values read
                dtype, n = NUMPY[t.format]
                raw = np.frombuffer(ev().device.queue.read_texture(
                    {"texture": t}, {"bytes_per_row": t.width * BYTES_PER_PIXEL[t.format]}, (t.width, t.height, 1)),
                    dtype).reshape(t.height, t.width, n)
                np.testing.assert_array_equal(raw, np.round(b[..., :n].astype(np.float64) * 65535).astype(np.uint16))
        with tempfile.TemporaryDirectory() as tmp:  # 16 bit files: the same integers
            for ext in ("png", "exr"):
                one = export(pkg, "main", Path(tmp) / "unorm", ev(), format=ext)
                two = export(pkg, "main", Path(tmp) / "float", fallback, format=ext)
                for p, q in zip(one, two):
                    x, y = imageio.read(p), imageio.read(q)
                    self.assertEqual(x.dtype, np.uint16 if ext == "png" else np.float16)
                    np.testing.assert_array_equal(x, y)

    def test_value_k_over_65535_round_trips(self):
        """Every 16 bit value: stored by one storage and by the other, read back identical, written to PNG as k."""
        fallback = Evaluator(features=set(ev().device.features) - {NORM16_FEATURE})
        pkg = Package()
        g = pkg.new_graph("ramp", 8)
        g.base = {"output_format": {"mode": "absolute", "value": "16"}}
        grad, out = g.add_node("gradient_linear_ramp"), g.add_node("output")
        g.connect(grad, "output", out, "input")
        job, o = next(iter(compile_graph(pkg, "ramp").outputs.values()))
        a, b = ev().read(job, o), fallback.read(job, o)
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
        k = np.arange(65536, dtype=np.uint16)
        v = k.astype(np.float32) / np.float32(65535)
        from trafilatura3d.export import convert
        np.testing.assert_array_equal(convert(v[:, None, None], "16", "png")[:, 0, 0], k)


class Diagnostics(unittest.TestCase):
    """Warnings: a package with missing dependencies still opens, and says what is wrong."""

    def test_broken_dependencies_open_with_warnings_and_relocate(self):
        import json
        from trafilatura3d.diagnostics import dependencies, package_warnings, relocate
        from trafilatura3d.package import Resource
        with tempfile.TemporaryDirectory() as d:
            pkg = Package()
            g = pkg.new_graph("broken", 6)
            b = g.add_node("bitmap")
            g.set_param(b, "pkg_resource_path", "missing.png")
            pkg.resources["rock"] = Resource("bitmap", "missing.png")
            pkg.new_graph("host", 6)
            path = Path(d) / "p.trafilatura3d"
            pkg.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["graphs"]["host"]["nodes"].append({"id": "graph_instance1", "type": "graph_instance",
                                                   "ref": "gone.trafilatura3d#g", "position": [0, 0]})
            path.write_text(json.dumps(data), encoding="utf-8")
            pkg = Package.load(path)  # opens despite the missing package
            compile_graph(pkg, "host")  # and the graph still compiles
            w = {k: [x.message for x in v] for k, v in package_warnings(pkg).items()}
            self.assertIn("No output node defined", w["broken"])
            self.assertIn("Reference resource not found", w["broken"])
            self.assertIn("Reference resource not found", w["host"])
            self.assertEqual(sorted(w[""]), ["Invalid dependent package", "Linked file not found"])
            self.assertEqual({(x.kind, x.ok) for x in dependencies(pkg)}, {("Bitmap", False), ("Package", False)})
            img = Path(d) / "found.png"
            img.write_bytes(b"png")  # relocation only needs the file to exist
            self.assertEqual(relocate(pkg, "missing.png", str(img)), 2)
            self.assertNotIn("Linked file not found", [x.message for x in package_warnings(pkg).get("", [])])

    def test_save_gathers_files_into_dependencies(self):
        """Saving copies the files from elsewhere into <package name>.dependencies/ next to the package (once per
        file, another file of the same name numbered), keeps the ones already beside it, and writes every path
        relative; a missing file and a project alias stay as written. Save As somewhere else, or under another name,
        takes the files along into the new package's own folder."""
        from trafilatura3d.package import Resource
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a").mkdir()
            (d / "b").mkdir()
            (d / "a" / "rock.png").write_bytes(b"a")
            (d / "b" / "rock.png").write_bytes(b"b")
            (d / "pkg" / "maps").mkdir(parents=True)
            (d / "pkg" / "maps" / "baked.png").write_bytes(b"c")
            pkg = Package()
            g = pkg.new_graph("g", 6)
            n1, n2, n3 = (g.add_node("bitmap") for _ in range(3))
            g.set_param(n1, "pkg_resource_path", str(d / "a" / "rock.png"))
            g.set_param(n2, "pkg_resource_path", str(d / "b" / "rock.png"))
            g.set_param(n3, "pkg_resource_path", str(d / "missing.png"))
            pkg.resources["rock"] = Resource("bitmap", str(d / "a" / "rock.png"))
            pkg.resources["baked"] = Resource("bitmap", str(d / "pkg" / "maps" / "baked.png"))
            pkg.resources["shared"] = Resource("bitmap", "alias://lib/stone.png")
            pkg.save(d / "pkg" / "p.trafilatura3d")
            params = [g.nodes[n].params["pkg_resource_path"] for n in (n1, n2, n3)]
            self.assertEqual(params, ["p.dependencies/rock.png", "p.dependencies/rock_2.png", str(d / "missing.png")])
            self.assertEqual({k: r.path for k, r in pkg.resources.items()},
                             {"rock": "p.dependencies/rock.png", "baked": "maps/baked.png",
                              "shared": "alias://lib/stone.png"})
            self.assertEqual((d / "pkg" / "p.dependencies" / "rock_2.png").read_bytes(), b"b")
            pkg.save(d / "pkg" / "q.trafilatura3d")  # another name beside it: its own copies, not p's
            self.assertEqual((pkg.resources["rock"].path, pkg.resources["baked"].path),
                             ("q.dependencies/rock.png", "maps/baked.png"))
            pkg.save(d / "elsewhere" / "q.trafilatura3d")  # Save As: the relative files resolve from the old place
            self.assertEqual(pkg.resources["baked"].path, "q.dependencies/baked.png")
            self.assertEqual(sorted(p.name for p in (d / "elsewhere" / "q.dependencies").iterdir()),
                             ["baked.png", "rock.png", "rock_2.png"])

    def test_every_instance_of_a_graph_with_warnings_says_so(self):
        """Two instances of a graph with warnings: each gets the warning (and its link to that node)."""
        from trafilatura3d.diagnostics import graph_warnings
        pkg = Package()
        pkg.new_graph("helper", 6)  # no output node: a warning
        host = pkg.new_graph("host", 6)
        a = host.add_node("graph_instance", ref="helper")
        b = host.add_node("graph_instance", ref="helper")
        found = [w.node for w in graph_warnings(pkg, "host") if w.message == "The referenced data has some warnings"]
        self.assertEqual(sorted(found), sorted([a, b]))
