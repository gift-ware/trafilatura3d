"""Project configuration (SBSPRJ / SBSCFG) and color management (OCIO)."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trafilatura3d import colormanagement as cm
from trafilatura3d import project
from trafilatura3d.package import Package


class ProjectStack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.env = {k: os.environ.get(k) for k in ("TRAFILATURA3D_USER_DIR", "TRAFILATURA3D_CONFIG", "OCIO")}
        os.environ["TRAFILATURA3D_USER_DIR"] = str(self.dir / "user")
        os.environ.pop("TRAFILATURA3D_CONFIG", None)
        os.environ.pop("OCIO", None)

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        project.set_configuration(project.Configuration())
        cm.reset_cache()
        self.tmp.cleanup()

    def write_project(self, name, data):
        p = self.dir / f"{name}{project.PROJECT_SUFFIX}"
        p.write_text(json.dumps({"trafilatura3d_project": 1, **data}), encoding="utf-8")
        return p

    def configure(self, *projects):
        cfg = self.dir / f"studio{project.CONFIG_SUFFIX}"
        cfg.write_text(json.dumps({"projects": [p.name for p in projects]}), encoding="utf-8")
        c = project.Configuration(cfg)
        project.set_configuration(c)
        return c

    def test_override_and_expand(self):
        (self.dir / "assets").mkdir()
        a = self.write_project("base", {"aliases": {"lib": "assets", "x": "one"}, "general": {"normal_format": "DirectX"},
                                        "library": {"paths": ["assets"]}})
        b = self.write_project("show", {"aliases": {"x": "two"}, "general": {"normal_format": "OpenGL"},
                                        "library": {"paths": ["more"]}})
        c = self.configure(a, b)
        self.assertEqual(c.setting("general", "normal_format"), "OpenGL")  # the later project overrides
        aliases = c.aliases()
        self.assertEqual(Path(aliases["lib"]), (self.dir / "assets").resolve())  # relative to its project
        self.assertEqual(Path(aliases["x"]).name, "two")  # an alias defined again overrides
        self.assertEqual([Path(p).name for p in c.library_paths()], ["assets", "more"])  # library paths expand
        self.assertEqual(Path(project.expand("lib://rock.png")), (self.dir / "assets" / "rock.png").resolve())

    def test_alias_in_package_references(self):
        from trafilatura3d.diagnostics import package_warnings
        (self.dir / "shared").mkdir()
        lib = Package()
        lib.new_graph("soft", 6)
        lib.save(self.dir / "shared" / "lib.trafilatura3d")
        self.configure(self.write_project("p", {"aliases": {"shared": "shared"}}))
        pkg = Package()
        g = pkg.new_graph("main", 6)
        g.add_node("graph_instance", ref="shared://lib.trafilatura3d#soft")
        self.assertEqual(pkg.resolve("shared://lib.trafilatura3d#soft")[1].id, "soft")
        bad = Package()
        g2 = bad.new_graph("main", 6)
        g2.nodes  # an unknown alias: the package still works and says what to define
        bad.path = self.dir / "bad.trafilatura3d"
        data = bad.to_dict()
        data["graphs"]["main"]["nodes"].append({"id": "graph_instance1", "type": "graph_instance",
                                               "ref": "nowhere://lib.trafilatura3d#soft", "position": [0, 0]})
        bad = Package.from_dict(data, self.dir / "bad.trafilatura3d")
        messages = [w.message for w in package_warnings(bad).get("", [])]
        self.assertIn("Check the alias 'nowhere' is defined in your project", messages)

    def test_ocio_variable_overrides_the_project(self):
        c = self.configure(self.write_project("p", {"color_management": {"mode": "OCIO", "ocio_config": "a.ocio"}}))
        self.assertTrue(c.setting("color_management", "ocio_config").endswith("a.ocio"))
        os.environ["OCIO"] = "/studio/config.ocio"
        self.assertEqual(c.setting("color_management", "ocio_config"), "/studio/config.ocio")

    def test_ocio_views_and_export_conversion(self):
        self.configure(self.write_project("p", {"color_management": {"mode": "OCIO"}}))
        cm.reset_cache()
        self.assertTrue(cm.active())
        d, v = cm.default_display_view()
        self.assertIn(d, cm.displays())
        lut = cm.display_lut(cm.texture_space(), d, v)
        self.assertEqual(lut.shape, (cm.LUT_SIZE, cm.LUT_SIZE, cm.LUT_SIZE, 4))
        mid = cm.convert(np.full((1, 1, 3), 0.5, np.float32), cm.texture_space(), cm.working_space())
        self.assertAlmostEqual(float(mid[0, 0, 0]), 0.214, delta=0.01)  # sRGB 0.5 is linear 0.214


if __name__ == "__main__":
    unittest.main()
