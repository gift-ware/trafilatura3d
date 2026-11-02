"""Linked files written from outside the editor (a bake, an image editor, a copy) reach the open packages: the
Document watches them and the graphs reading them recompile, without an edit (no undo step, not unsaved)."""

import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from trafilatura3d import imageio


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class LinkedFilesTest(unittest.TestCase):
    def wait_for(self, app, condition, seconds=5.0):
        end = time.monotonic() + seconds
        while not condition() and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.02)
        return condition()

    def test_a_file_written_outside_recooks(self):
        from PySide6.QtWidgets import QApplication
        from trafilatura3d.editor.document import Document
        from trafilatura3d.package import Package
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "maps").mkdir()
            image = d / "maps" / "baked.png"

            def write(value, path=image):
                imageio.write(path, np.full((8, 8, 3), value, np.uint8))
            write(10)
            pkg = Package()
            g = pkg.new_graph("main", 3)
            bitmap = g.add_node("bitmap")
            g.set_param(bitmap, "pkg_resource_path", "maps/baked.png")
            out = g.add_node("output")
            g.connect(bitmap, "output", out, "input")
            pkg.save(d / "p.trafilatura3d")
            doc = Document()
            pkg = doc.open(d / "p.trafilatura3d")

            def value():
                job, port = next(iter(doc.compiled(pkg, "main").outputs.values()))
                doc.ev.cook([job])
                pixels = doc.ev.read(job, port)
                scale = 1 if pixels.dtype == np.uint8 else 255
                return int(round(float(pixels[..., 0].mean()) * scale))
            self.assertEqual(value(), 10)
            changed = []
            doc.changed.connect(lambda p: changed.append(p))
            time.sleep(0.05)  # a new modification time
            write(200)  # written in place, as a bake copies its maps
            self.assertTrue(self.wait_for(app, lambda: changed), "no change heard")
            self.assertEqual(value(), 200)
            self.assertNotIn(id(pkg), doc.dirty)  # not an edit
            self.assertEqual(doc.undo.count(), 0)
            changed.clear()
            aside = d / "maps" / "aside.png"  # written aside and renamed over it, as many programs save
            write(90, aside)
            os.replace(aside, image)
            self.assertTrue(self.wait_for(app, lambda: changed), "no change heard after a replace")
            self.assertEqual(value(), 90)
            doc.undo.indexChanged.disconnect()


    def test_an_open_subgraph_is_the_one_its_instances_use(self):
        """A graph used by another package's instance: written from outside while not open, it is read again; open
        and edited, the instance follows at once (the package using it recompiles, not edited), and its reference
        is the open package, never a second copy."""
        from PySide6.QtWidgets import QApplication
        from trafilatura3d.editor.document import Document
        from trafilatura3d.package import Package
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "lib").mkdir()
            lib = Package()
            sub = lib.new_graph("sub", 3)
            color = sub.add_node("uniform_color")
            sub.set_param(color, "output_color", [0.2, 0.2, 0.2, 1.0])
            out = sub.add_node("output")
            sub.set_param(out, "identifier", "color")
            sub.connect(color, "output", out, "input")
            lib.save(d / "lib" / "lib.trafilatura3d")
            main = Package(d / "main.trafilatura3d")  # where it is saved: the reference is relative to it
            g = main.new_graph("main", 3)
            inst = g.add_node("graph_instance", ref="lib/lib.trafilatura3d#sub")
            out = g.add_node("output")
            g.connect(inst, "color", out, "input")
            main.save(d / "main.trafilatura3d")
            doc = Document()
            main = doc.open(d / "main.trafilatura3d")

            def value():
                job, port = next(iter(doc.compiled(main, "main").outputs.values()))
                doc.ev.cook([job])
                pixels = doc.ev.read(job, port)
                return round(float(pixels[..., 0].mean()) / (255 if pixels.dtype == np.uint8 else 1), 2)
            self.assertEqual(value(), 0.2)  # read through a copy of lib: it is not open
            outside = Package.load(d / "lib" / "lib.trafilatura3d")  # written by another program (a sync)
            outside.graphs["sub"].set_param(color, "output_color", [0.5, 0.5, 0.5, 1.0])
            time.sleep(0.05)  # a new modification time
            outside.save()
            self.assertTrue(self.wait_for(app, lambda: value() == 0.5), "the referenced package was not read again")
            self.assertNotIn(id(main), doc.dirty)
            lib = doc.open(d / "lib" / "lib.trafilatura3d")
            self.assertIs(main.resolve("lib/lib.trafilatura3d#sub")[0], lib)  # Open reference: that package
            changed = []
            doc.changed.connect(changed.append)
            node = next(n for n, v in lib.graphs["sub"].nodes.items() if v.type == "uniform_color")
            doc.edit(lib, "Color", lambda: lib.graphs["sub"].set_param(node, "output_color", [0.8, 0.8, 0.8, 1.0]))
            self.assertIn(main, changed)  # its views refresh
            self.assertEqual(value(), 0.8)
            self.assertNotIn(id(main), doc.dirty)  # not edited
            doc.undo.undo()
            self.assertEqual(value(), 0.5)
            doc.undo.indexChanged.disconnect()


if __name__ == "__main__":
    unittest.main()
