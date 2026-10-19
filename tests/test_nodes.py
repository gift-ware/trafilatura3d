"""Every registered node: cooks at 64 px on test inputs without NaN, is deterministic, tiles when it
should, and has the parameter labels its entry of the reference index (reference/) lists."""

import json
import os
import unittest
from pathlib import Path

import numpy as np

from trafilatura3d.evaluator import Evaluator
from trafilatura3d.nodes import NODES
from trafilatura3d.sample import has_any_input, sample_job

REFERENCE = Path(__file__).parent.parent / "reference"
CATALOG = {}  # image nodes by id; function and FX-Map nodes as "function/<id>", "fxmap/<id>"
for _f in (REFERENCE / "graph").glob("*.json"):
    _data = json.loads(_f.read_text(encoding="utf-8"))
    for _key, _prefix in (("nodes", ""), ("function_nodes", "function/"), ("fxmap_nodes", "fxmap/")):
        for _n in _data.get(_key, []):
            CATALOG[_prefix + _n["id"]] = _n
for _f in (REFERENCE / "paint").glob("*.json"):
    for _n in json.loads(_f.read_text(encoding="utf-8"))["items"]:  # the Paint filters and generators: "painter.filter.<id>"
        CATALOG[_n["id"]] = _n

# TRAFILATURA3D_CATEGORY="Filters/Blurs" (a prefix) limits the run to the nodes of a category.
_ONLY = os.environ.get("TRAFILATURA3D_CATEGORY", "")
SELECTED = sorted(t for t, d in NODES.items() if d.category.startswith(_ONLY) or (not _ONLY))

EV = Evaluator()


def as_float(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float64) / (255.0 if a.dtype == np.uint8 else 1.0)


class EveryNode(unittest.TestCase):
    def test_every_module_registers(self):
        from trafilatura3d.nodes import NODE_ERRORS
        self.assertEqual(NODE_ERRORS, {}, "node modules failed to load:\n"
                         + "\n".join(f"{k}: {v}" for k, v in NODE_ERRORS.items()))

    def test_cooks_clean_and_deterministic(self):
        for node_type in SELECTED:
            if getattr(NODES[node_type], "passthrough", False):
                continue  # no job of its own (Dot)
            for color in ([False, True] if has_any_input(node_type) else [False]):
                with self.subTest(node=node_type, color=color):
                    job = sample_job(node_type, color)
                    for out in job.node.outputs:
                        a = as_float(EV.read(job, out))
                        self.assertEqual(a.shape[:2], (1, 1) if job.is_value(out) else (64, 64))
                        self.assertTrue(np.all(np.isfinite(a)), "NaN or inf")
                    first = {o: EV.read(job, o).copy() for o in job.node.outputs}
                    EV.clear()
                    for o in job.node.outputs:
                        np.testing.assert_array_equal(first[o], EV.read(job, o))

    def test_generators_tile(self):
        for node_type, d in ((t, NODES[t]) for t in SELECTED):
            if d.inputs or not d.category or not getattr(d, "seamless", True):
                continue
            with self.subTest(node=node_type):
                job = sample_job(node_type)
                for out in d.outputs:
                    if job.is_value(out):
                        continue
                    a = as_float(EV.read(job, out)).mean(axis=2)
                    # The jump across the edge against the jumps inside: no larger than the largest of them (a
                    # pattern whose cells start at the corner, Checker, has a cell edge on the tile's edge).
                    inner_x = max(np.abs(np.diff(a, axis=1)).mean(0).max(), np.abs(np.diff(a, axis=1)).mean() * 3)
                    inner_y = max(np.abs(np.diff(a, axis=0)).mean(1).max(), np.abs(np.diff(a, axis=0)).mean() * 3)
                    self.assertLess(np.abs(a[:, 0] - a[:, -1]).mean(), inner_x + 1e-3, "seam on the vertical edge")
                    self.assertLess(np.abs(a[0, :] - a[-1, :]).mean(), inner_y + 1e-3, "seam on the horizontal edge")

    def test_catalog_conformance(self):
        for node_type, d in ((t, NODES[t]) for t in SELECTED):
            if not d.catalog:
                continue
            with self.subTest(node=node_type):
                self.assertIn(d.catalog, CATALOG, "catalog id not found")
                entry = CATALOG[d.catalog]
                labels = {p.label for p in d.params.values()}
                missing = [p["name"] for p in entry.get("parameters", [])
                           if p["name"] not in labels and p["name"] not in d.unsupported]
                self.assertEqual(missing, [], "catalog parameters neither implemented nor listed as unsupported")


if __name__ == "__main__":
    unittest.main()
