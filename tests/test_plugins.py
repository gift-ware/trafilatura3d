"""Plugins (trafilatura3d/plugins.py): self-contained packages and single files found in the search paths, their
events (and Veto on the "before" ones), enable / disable / reload, failures that never stop Trafilatura 3D, the
command line, and the editor's additions removed on unload."""

import contextlib
import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from trafilatura3d import plugins, project
from trafilatura3d.package import Package

SAMPLE_INIT = '''
"""Records what Trafilatura 3D does (a sample pipeline integration)."""
from .recorder import Recorder  # a self-contained package: its own modules, relative imports

recorder = Recorder()


def initialize_plugin(host):
    for event in ("package_opened", "package_saved", "outputs_exported", "bake_finished"):
        host.events.subscribe(event, lambda event=event, **data: recorder.seen.append((event, sorted(data))))
    recorder.seen.append(("initialized", host.editor is not None))
    if host.editor is not None:
        host.editor.add_menu_action("Studio", "Validate", lambda: recorder.seen.append(("menu", True)))
        from PySide6.QtWidgets import QLabel
        host.editor.add_dock("Studio", QLabel("pipeline"))


def uninitialize_plugin(host):
    recorder.seen.append(("uninitialized", True))
'''
RECORDER = "class Recorder:\n    def __init__(self):\n        self.seen = []\n"
VETOER = '''
"""Refuses to export the graph called blocked."""
from trafilatura3d.plugins import Veto


def initialize_plugin(host):
    def check(package, graph_id, out_dir):
        if graph_id == "blocked":
            raise Veto("naming: blocked is not a valid graph name")
    host.events.subscribe("before_export", check)
'''
BROKEN = '''
def initialize_plugin(host):
    raise RuntimeError("cannot reach the server")
'''


def flat_package(graph_id):
    pkg = Package()
    g = pkg.new_graph(graph_id, 4)
    u, o = g.add_node("uniform_color"), g.add_node("output")
    g.connect(u, "output", o, "input")
    return pkg


class PluginTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.env = os.environ.get("TRAFILATURA3D_USER_DIR")
        os.environ["TRAFILATURA3D_USER_DIR"] = str(self.dir / "user")
        folder = self.dir / "plugins"
        (folder / "sample").mkdir(parents=True)
        (folder / "sample" / "__init__.py").write_text(textwrap.dedent(SAMPLE_INIT), encoding="utf-8")
        (folder / "sample" / "recorder.py").write_text(RECORDER, encoding="utf-8")
        (folder / "vetoer.py").write_text(textwrap.dedent(VETOER), encoding="utf-8")
        (folder / "broken.py").write_text(textwrap.dedent(BROKEN), encoding="utf-8")
        self.m = plugins.manager()
        self.m.plugins, self.m.editor_factory = {}, None
        self.messages = []
        self.m.reporter = self.messages.append
        self.m.save_settings({"paths": [str(folder)], "disabled": []})

    def tearDown(self):
        self.m.unload_all()
        self.m.plugins, self.m.reporter, self.m.editor_factory = {}, None, None
        if self.env is None:
            os.environ.pop("TRAFILATURA3D_USER_DIR", None)
        else:
            os.environ["TRAFILATURA3D_USER_DIR"] = self.env
        project.set_configuration(project.Configuration())
        self.tmp.cleanup()

    def recorder(self):
        return sys.modules["trafilatura3d_plugin_sample"].recorder

    def test_load_events_veto_and_failures(self):
        from trafilatura3d.export import export
        self.m.load_all()
        self.assertEqual({n for n, i in self.m.plugins.items() if i.loaded}, {"sample", "vetoer"})
        self.assertIn("cannot reach the server", self.m.plugins["broken"].error)  # reported, not fatal
        self.assertTrue(any("broken failed to load" in msg for msg in self.messages))
        self.assertEqual(self.m.plugins["sample"].description, "Records what Trafilatura 3D does (a sample pipeline integration).")
        self.assertIn(("initialized", False), self.recorder().seen)  # no editor here
        pkg = flat_package("ok")
        pkg.save(self.dir / "ok.trafilatura3d")
        export(pkg, "ok", self.dir / "out")
        events = [e for e, _ in self.recorder().seen]
        self.assertIn("package_saved", events)
        self.assertIn("outputs_exported", events)
        with self.assertRaises(plugins.Veto) as refused:
            export(flat_package("blocked"), "blocked", self.dir / "out2")
        self.assertIn("vetoer: naming", str(refused.exception))
        self.assertFalse((self.dir / "out2").exists())  # refused before anything was written

    def test_disable_reload_and_project_paths(self):
        self.m.load_all()
        recorder = self.recorder()
        self.m.set_enabled("sample", False)
        self.assertIn(("uninitialized", True), recorder.seen)
        self.assertNotIn("trafilatura3d_plugin_sample", sys.modules)  # its modules are gone
        self.assertNotIn("trafilatura3d_plugin_sample.recorder", sys.modules)
        self.assertIn("sample", json.loads(self.m.settings_path().read_text(encoding="utf-8"))["disabled"])
        self.m.set_enabled("sample", True)
        self.assertTrue(self.m.reload("sample"))
        self.assertIs(self.m.plugins["sample"].loaded, True)
        # A project's plugin paths come first (a pipeline declares its plugin folder there).
        proj = self.dir / "studio.trafilatura3dproject"
        proj.write_text(json.dumps({"trafilatura3d_project": 1, "plugins": {"paths": ["pipeline_plugins"]}}), encoding="utf-8")
        cfg = self.dir / "studio.trafilatura3dconfig"
        cfg.write_text(json.dumps({"projects": [proj.name]}), encoding="utf-8")
        project.set_configuration(project.Configuration(cfg))
        self.assertEqual(Path(self.m.search_paths()[0]), (self.dir / "pipeline_plugins").resolve())

    def test_command_line(self):
        from trafilatura3d.cli import main
        pkg = flat_package("ok")
        pkg.save(self.dir / "ok.trafilatura3d")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["plugins"]), 0)
        self.assertIn("sample  enabled", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["cook", str(self.dir / "ok.trafilatura3d"), "--out", str(self.dir / "cli")]), 0)
        self.assertIn("package_opened", [e for e, _ in self.recorder().seen])
        self.m.unload_all()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            main(["--no-plugins", "info", str(self.dir / "ok.trafilatura3d")])
        self.assertFalse(self.m.plugins["sample"].loaded)

    @unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
    def test_editor_additions(self):
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication, QDockWidget
        import rendercanvas.qt as rq
        QCoreApplication.setOrganizationName("Trafilatura3DTest")
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        try:
            from trafilatura3d.editor.mainwindow import MainWindow
            from trafilatura3d.editor.plugin_host import PluginManager
            w = MainWindow([])
            self.assertNotIn("Studio", [a.text() for a in w.menuBar().actions()])  # the Home's menus are generic
            w._show(w.workspace)  # plugins add to the graph workspace's menus
            menus = {a.text(): a.menu() for a in w.menuBar().actions()}
            self.assertIn("Studio", menus)
            action = next(a for a in menus["Studio"].actions() if a.text() == "Validate")
            action.trigger()
            self.assertIn(("menu", True), self.recorder().seen)
            self.assertTrue(any(d.windowTitle() == "Studio" for d in w.workspace.findChildren(QDockWidget)))
            dlg = PluginManager(w)
            rows = {dlg.tree.topLevelItem(i).text(0): dlg.tree.topLevelItem(i).text(1)
                    for i in range(dlg.tree.topLevelItemCount())}
            self.assertEqual(rows, {"broken": "Failed", "sample": "Loaded", "vetoer": "Loaded"})
            self.m.unload("sample")
            app.processEvents()
            self.assertNotIn("Studio", [a.text() for a in w.menuBar().actions()])  # removed with it
            w.doc.undo.indexChanged.disconnect()
            w.graph.scene.selectionChanged.disconnect()
        finally:
            rq.QRenderWidget.__init__ = init


if __name__ == "__main__":
    unittest.main()
