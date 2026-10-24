"""Project configuration (pipeline and project configuration):

- a project file (`.trafilatura3dproject`, JSON) holds settings in sections: aliases, library, general,
  color management, 3D view, baking;
- projects stack: the Default project (built in, read only), the User project (always present, where
  edits go by default), then the custom projects of the configuration list, in order;
- a later project overrides most settings of the earlier ones, but aliases and library paths expand
  (an alias defined again overrides the earlier one);
- relative paths in a project are relative to its file; an alias `name://rest` stands for a path prefix;
- the configuration list (`.trafilatura3dconfig`, JSON: {"projects": [paths]}) is chosen with the
  TRAFILATURA3D_CONFIG environment variable or the CLI's --config; its relative paths are relative to it;
- the OCIO environment variable overrides the project's OpenColorIO config path.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

PROJECT_SUFFIX = ".trafilatura3dproject"
CONFIG_SUFFIX = ".trafilatura3dconfig"
COLOR_MODES = ("Legacy", "OCIO")
NORMAL_FORMATS = ("DirectX", "OpenGL")
IMAGE_FORMATS = ("png", "tga", "tif", "exr", "jpg", "bmp")  # the export's formats (export.py)

# The default project.
DEFAULTS = {
    "aliases": {},  # name -> path (relative to the project file)
    "library": {"paths": []},  # watched folders whose packages' graphs appear in the Library
    "general": {"normal_format": "DirectX", "image_format": "png", "templates": []},
    "color_management": {"mode": "Legacy", "ocio_config": "", "working_space": "", "display": "", "view": "",
                         "texture_space": "", "data_space": ""},
    "view3d": {"environment": "Studio", "hdri": ""},
    "baking": {"high_suffix": "_high", "low_suffix": "_low"},
    "plugins": {"paths": []},  # folders holding plugins (trafilatura3d/plugins.py), e.g. a pipeline's integration
    # libraries of resources with a folder per kind (LIBRARY_KINDS), as the Paint's libraries: a pipeline's shaders,
    # export presets, materials… read by every workspace while the project is loaded
    "libraries": {"paths": []},
}
EXPANDING = {("aliases",), ("library", "paths"), ("general", "templates"), ("plugins", "paths"), ("libraries", "paths")}
# The kinds of a library, one folder each: `shaders` (.wgsl, shaders.py), `export_presets` (the Paint's output
# templates, .json), `materials` (packages) and `smart_materials` (the Paint's Assets).
LIBRARY_KINDS = ("shaders", "export_presets", "materials", "smart_materials")
_ALIAS = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)://(.*)$")


def user_dir() -> Path:
    base = os.environ.get("TRAFILATURA3D_USER_DIR")
    if base:
        return Path(base)
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home())) / "Trafilatura 3D"
    if os.uname().sysname == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Trafilatura 3D"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "trafilatura3d"


class Project:
    """One project file."""

    def __init__(self, path: Path | None, data: dict | None = None, name: str = "", read_only: bool = False):
        self.path = Path(path).resolve() if path else None
        self.data = data if data is not None else {}
        self.name = name or (self.path.stem if self.path else "Project")
        self.read_only = read_only

    @classmethod
    def load(cls, path) -> Project:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if data and data.get("trafilatura3d_project") != 1:
            raise ValueError(f"{path}: not a Trafilatura 3D project file")
        data.pop("trafilatura3d_project", None)
        return cls(path, data)

    def save(self) -> None:
        if self.read_only or self.path is None:
            raise ValueError(f"{self.name}: read only")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"trafilatura3d_project": 1, **self.data}, indent=1), encoding="utf-8")

    def get(self, section: str, key: str | None = None, default=None):
        sec = self.data.get(section, {})
        return sec if key is None else sec.get(key, default)

    def set(self, section: str, key: str, value) -> None:
        self.data.setdefault(section, {})[key] = value

    def resolve_path(self, value: str) -> str:
        """A path written in this project, made absolute (relative to the project file)."""
        if not value or _ALIAS.match(value) or self.path is None:
            return value
        p = Path(value)
        return str(p if p.is_absolute() else (self.path.parent / p).resolve())


class Configuration:
    """The project stack and the settings it results in."""

    def __init__(self, config_path: str | Path | None = None):
        config_path = config_path or os.environ.get("TRAFILATURA3D_CONFIG")
        self.config_path = Path(config_path).resolve() if config_path else None
        self.default = Project(None, copy.deepcopy(DEFAULTS), "Default project", read_only=True)
        self.user = Project.load(user_dir() / f"user{PROJECT_SUFFIX}")
        self.user.name = "User project"
        self.custom: list[Project] = []
        if self.config_path and self.config_path.exists():
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            for p in data.get("projects", []):
                path = Path(p)
                path = path if path.is_absolute() else self.config_path.parent / path
                self.custom.append(Project.load(path))

    @property
    def projects(self) -> list[Project]:
        return [self.default, self.user, *self.custom]

    def save_config(self) -> None:
        if self.config_path is None:
            self.config_path = user_dir() / f"default{CONFIG_SUFFIX}"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        rel = []
        for p in self.custom:
            try:
                rel.append(os.path.relpath(p.path, self.config_path.parent).replace(os.sep, "/"))
            except ValueError:
                rel.append(str(p.path))
        self.config_path.write_text(json.dumps({"projects": rel}, indent=1), encoding="utf-8")

    def add_project(self, path) -> Project:
        p = Project.load(path)
        if not Path(path).exists():
            p.data = {}
            p.save()
        self.custom.append(p)
        self.save_config()
        return p

    def remove_project(self, project: Project) -> None:
        self.custom.remove(project)
        self.save_config()

    # --- resolved settings ---------------------------------------------------------------------

    def setting(self, section: str, key: str):
        """The value of the last project that sets it (most settings override)."""
        value = DEFAULTS.get(section, {}).get(key)
        for p in self.projects:
            if key in p.get(section):
                v = p.get(section, key)
                value = p.resolve_path(v) if key in ("ocio_config", "hdri") else v
        if (section, key) == ("color_management", "ocio_config") and os.environ.get("OCIO"):
            value = os.environ["OCIO"]  # the OCIO variable overrides the project
        return value

    def aliases(self) -> dict[str, str]:
        """Aliases of every project (expanding; a name defined again overrides)."""
        out = {}
        for p in self.projects:
            for name, path in p.get("aliases").items():
                out[name] = p.resolve_path(path)
        return out

    def library_paths(self) -> list[str]:
        out = []
        for p in self.projects:
            out += [p.resolve_path(x) for x in p.get("library", "paths", [])]
        return out

    def library_folders(self, kind: str) -> list[Path]:
        """A kind's folder (LIBRARY_KINDS) in every library the projects declare, in project order."""
        if kind not in LIBRARY_KINDS:
            raise ValueError(f"library kind {kind!r} (kinds: {', '.join(LIBRARY_KINDS)})")
        out = []
        for p in self.projects:
            out += [Path(p.resolve_path(x)) / kind for x in p.get("libraries", "paths", [])]
        return out

    def plugin_paths(self) -> list[str]:
        out = []
        for p in self.projects:
            out += [p.resolve_path(x) for x in p.get("plugins", "paths", [])]
        return out

    def templates(self) -> list[str]:
        out = []
        for p in self.projects:
            out += [p.resolve_path(x) for x in p.get("general", "templates", [])]
        return out

    def expand(self, path: str) -> str:
        """`alias://rest` as a real path; other paths unchanged."""
        m = _ALIAS.match(path or "")
        if not m:
            return path
        aliases = self.aliases()
        if m.group(1) not in aliases:
            raise KeyError(m.group(1))
        return str(Path(aliases[m.group(1)]) / m.group(2))


_CURRENT: Configuration | None = None


def configuration() -> Configuration:
    """The process' configuration (created on first use from TRAFILATURA3D_CONFIG)."""
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = Configuration()
    return _CURRENT


def set_configuration(config: Configuration) -> None:
    global _CURRENT
    _CURRENT = config


def expand(path: str) -> str:
    return configuration().expand(path)


def alias_of(path: str) -> str | None:
    m = _ALIAS.match(path or "")
    return m.group(1) if m else None
