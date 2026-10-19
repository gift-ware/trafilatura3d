"""Display timings: the evaluator times each job it cooks (GPU timestamps, or each job waited for where they don't
work), and the Graph view shows each node's time under its size line, colored, after a warning."""

import os
import tempfile
import unittest
from pathlib import Path

from tests.scenes import material_package
from trafilatura3d.evaluator import Evaluator
from trafilatura3d.sample import sample_job


class EvaluatorTimingsTest(unittest.TestCase):
    def test_each_cooked_job_is_timed(self):
        ev = Evaluator()
        job = sample_job("blur", size_log2=8)
        ev.cook([job])
        self.assertNotIn(job.key, ev.timings)  # off: nothing measured
        ev.timing = True
        ev.forget([job])
        ev.cook([job])
        ev.wait()
        self.assertGreaterEqual(ev.timings[job.key], 0.0)
        before = ev.read(job, "output")
        ev.timing = False
        ev.forget([job])
        ev.cook([job])
        self.assertTrue((ev.read(job, "output") == before).all())  # the same result timed or not


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class GraphTimingsTest(unittest.TestCase):
    def test_display_timings(self):
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication
        import rendercanvas.qt as rq
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        self.addCleanup(setattr, rq.QRenderWidget, "__init__", init)
        from trafilatura3d.editor.mainwindow import MainWindow
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        w = MainWindow([str(material_package(Path(tmp.name) / "scene.trafilatura3d"))])

        def close():
            w.doc.undo.indexChanged.disconnect()
            w.graph.scene.selectionChanged.disconnect()
            w.close()
            w.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.addCleanup(close)
        w.show()
        app.processEvents()
        panel = w.graph
        panel._confirm_timings = lambda: False  # the warning answered Cancel: stays off
        panel.act_timings.setChecked(True)
        self.assertFalse(panel.act_timings.isChecked())
        self.assertFalse(w.doc.ev.timing)
        panel._confirm_timings = lambda: True
        panel.act_timings.setChecked(True)
        app.processEvents()
        self.assertTrue(w.doc.ev.timing)
        timed = [i for i in panel.scene.nodes.values() if i.timing_ms is not None]
        self.assertTrue(timed)
        self.assertTrue(all(i.show_timing for i in panel.scene.nodes.values()))
        panel.act_timings.setChecked(False)
        self.assertFalse(w.doc.ev.timing)
        self.assertFalse(any(i.show_timing for i in panel.scene.nodes.values()))


if __name__ == "__main__":
    unittest.main()
