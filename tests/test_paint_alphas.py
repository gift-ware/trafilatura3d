"""Alpha Shape: Hardness, Shape and Invert as the brush alphas measured on the originals (spp/alphas.shaped); the
library of graph alphas (paint/alpha_library): every alpha of the reference index but the dynamic ones, references
with their values, stamps cooked with them."""

import unittest

import numpy as np

from trafilatura3d.compiler import ABSOLUTE, compile_graph
from trafilatura3d.evaluator import Evaluator
from trafilatura3d.nodes.paint.alpha_shape import ORDERS
from trafilatura3d.package import Package
from trafilatura3d.spp.alphas import shaped

EV = Evaluator()


def cook(params: dict) -> tuple[np.ndarray, np.ndarray]:
    pkg = Package()
    g = pkg.new_graph("a", 8)
    base = g.add_node("clouds_soft")
    node = g.add_node("paint_alpha_shape")
    g.connect(base, "output", node, "input")
    for k, v in params.items():
        g.set_param(node, k, v)
    for n in (base, node):
        g.set_base(n, "output_format", {"mode": ABSOLUTE, "value": "32f"})
    jobs = compile_graph(pkg, "a").jobs
    return tuple(np.asarray(EV.read(jobs[n], "output"), np.float64)[..., 0] for n in (base, node))


class AlphaShape(unittest.TestCase):
    def test_matches_the_measured_model(self):
        for h, s, inv, order in ((0.5, 0.75, False, 0), (0.2, 0.3, True, 0), (0.85, 1.0, False, 1), (0.6, 0.0, True, 1), (0.0, 0.0, False, 0)):
            with self.subTest(hardness=h, shape=s, invert=inv, order=order):
                base, out = cook({"hardness": h, "shape": s, "invert": inv, "order": ORDERS[order]})
                np.testing.assert_allclose(out, shaped(base, h, s, inv, levels_first=order == 0), atol=2e-3)

    def test_threshold_is_white_above_zero_in_8_bits(self):
        base, out = cook({"hardness": 0.7, "shape": 0.5, "order": ORDERS[1], "threshold": True})
        v = shaped(base, 0.7, 0.5, False, levels_first=False)
        sure = (v == 0) | (v > 2e-3)
        np.testing.assert_array_equal(out[sure], (v[sure] > 0.5 / 255).astype(float))


class AlphaLibrary(unittest.TestCase):
    def test_every_reference_alpha_but_the_dynamic_ones(self):
        """Each alpha of the reference index but the dynamic ones has its library graph through the importer's
        table, one graph each."""
        import json
        from pathlib import Path
        from trafilatura3d.paint import alpha_library as lib
        from trafilatura3d.spp import alphas as spp_alphas
        index = json.loads((Path(__file__).parent.parent / "reference" / "paint" / "alphas.json").read_text())["items"]
        table = json.loads(spp_alphas.LIBRARY_TABLE.read_text())["alphas"]
        want = {i["id"].removeprefix("painter.alpha.") for i in index if i["category"] != "dynamic"}
        self.assertEqual(want - set(table), set())
        targets = [lib.LIBRARY + table[n] for n in want]
        self.assertTrue(all(lib.exists(ref) for ref in targets))
        self.assertEqual(len(set(targets)), len(targets))
        self.assertEqual({ref for ref, _, _ in lib.listing()}, set(targets))

    def test_imported_alpha_names_reach_the_library(self):
        from trafilatura3d.paint import alpha_library as lib
        from trafilatura3d.spp import alphas as spp_alphas
        ref = spp_alphas.library_ref("dirt_spots_hollow", {"hardness": 0.8, "$randomseed": 3, "$outputsize": [9, 9]})
        self.assertEqual(lib.split(ref), ("dirt", "grime_rings", {"hardness": 0.8, "seed": 3.0}))
        self.assertEqual(lib.split(spp_alphas.library_ref("Mold", {}))[:2], ("organic", "mildew_spots"))
        self.assertIsNone(spp_alphas.library_ref("charcoal_brush", {}))
        seeded = lib.image(ref, 64)
        self.assertFalse(np.array_equal(seeded, lib.image(lib.join("dirt", "grime_rings", {"hardness": 0.8}), 64)))

    def test_reference_round_trip_and_values(self):
        from trafilatura3d.paint import alpha_library as lib
        ref = lib.join("dirt", "grime_blobs", {"hardness": 0.9})
        self.assertEqual(lib.split(ref), ("dirt", "grime_blobs", {"hardness": 0.9}))
        soft, hard = lib.image(lib.join("dirt", "grime_blobs", {}), 128), lib.image(ref, 128)
        self.assertEqual(hard.shape, (128, 128))
        self.assertGreater(hard.mean(), soft.mean() + 0.02)  # Hardness reaches the graph
        inv = lib.image(lib.join("dirt", "grime_blobs", {"invert": 1.0}), 128)
        self.assertGreater(inv.mean(), soft.mean() + 0.1)

    def test_unknown_and_generated(self):
        from trafilatura3d.paint import alpha_library as lib
        from trafilatura3d.paint.alphas import DEFAULT_ALPHA
        self.assertFalse(lib.exists("library:dirt/nothing"))
        np.testing.assert_array_equal(lib.image("library:dirt/nothing", 32), lib.image(DEFAULT_ALPHA, 32))
        self.assertEqual(lib.image("Square", 16).min(), 1.0)


if __name__ == "__main__":
    unittest.main()
