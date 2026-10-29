"""Plugins: a plugin is a Python file or a self-contained package (a folder
with its __init__.py and everything it needs) defining `initialize_plugin(host)`, called when it is loaded,
and optionally `uninitialize_plugin(host)`, called when it is unloaded. The host is the plugin's typed access
to Trafilatura 3D:

- `host.events`: subscribe to what happens: package opened / saved, outputs exported, bake finished, and the
  "before" events (before save, before export, before bake) whose handlers may refuse the operation by
  raising `Veto` (a pipeline's validation);
- `host.editor`: the editor's additions (menu actions, docks), None from the command line.

Everything a plugin registers through its host is removed when it is unloaded; its errors are reported and
never stop Trafilatura 3D. Plugins are searched in the project's plugin paths (Project settings > plugins), the
user's own paths and the user folder's `plugins`; disabled ones are listed but not loaded. The integration
with a pipeline is a plugin living in that pipeline: Trafilatura 3D knows nothing of it."""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

EVENTS = (
    "package_opened",     # package
    "before_save",        # package, path: raise Veto to refuse
    "package_saved",      # package, path
    "before_export",      # package, graph_id, out_dir, paths (the files it will write): raise Veto to refuse
    "outputs_exported",   # package, graph_id, paths
    "before_export_textures",  # session, out_dir, paths (the files it will write): raise Veto to refuse
    "textures_exported",  # session, paths
    "before_bake",        # low_path, bakers, settings, out_dir: raise Veto to refuse
    "bake_finished",      # low_path, bakers, settings, paths
)
MODULE_PREFIX = "trafilatura3d_plugin_"


class Veto(ValueError):
    """Raised by a "before" event handler: the operation does not happen; the message says why (shown
    where Trafilatura 3D shows its own errors: a ValueError)."""


class EditorAccess(ABC):
    """What a plugin may add to the editor (trafilatura3d.editor.plugin_host implements it)."""

    @abstractmethod
    def add_menu_action(self, menu: str, text: str, callback: Callable[[], None], shortcut: str = "",
                        workspaces: tuple = ("graph", "paint")):
        """An action in a menu of the menu bar of each workspace in `workspaces` ("graph", "paint"; the menu created
        when missing, e.g. "Studio")."""

    @abstractmethod
    def add_menu_separator(self, menu: str, workspaces: tuple = ("graph", "paint")) -> None:
        """A separator in that menu."""

    @abstractmethod
    def workspace(self) -> str:
        """What the window shows: "home", "graph" or "paint"."""

    @abstractmethod
    def current_paint(self):
        """The Paint project open (its session), or None."""

    @abstractmethod
    def edit(self, document, label: str, fn: Callable[[], object]) -> None:
        """Change an open package or the Paint project (its session) with `fn()` as one undoable step (marked
        unsaved, the views refreshed)."""

    @abstractmethod
    def open(self, path) -> None:
        """Open a package or a Paint project in the editor, as File > Open does."""

    @abstractmethod
    def add_dock(self, title: str, widget):
        """A dock with the plugin's widget."""

    @abstractmethod
    def main_window(self):
        """The main window (for dialogs' parent)."""

    @abstractmethod
    def current(self) -> tuple:
        """(package, graph id) shown in the Graph view, or (None, "")."""

    @abstractmethod
    def save(self, document, path=None) -> None:
        """Save an open package or the Paint project (its session) as the editor's Save does (at `path`: as Save As
        does, without asking), its unsaved marks cleared; raises OSError when it could not be saved (the editor has
        shown why)."""

    @abstractmethod
    def remove_all(self) -> None:
        """Remove what this plugin added (called on unload)."""


class Events:
    """One plugin's subscriptions (the manager dispatches to every plugin's)."""

    def __init__(self):
        self.handlers: dict[str, list[Callable]] = {name: [] for name in EVENTS}

    def subscribe(self, event: str, handler: Callable) -> None:
        if event not in self.handlers:
            raise ValueError(f"unknown event {event!r} (events: {', '.join(EVENTS)})")
        self.handlers[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable) -> None:
        if handler in self.handlers.get(event, []):
            self.handlers[event].remove(handler)


class PluginHost:
    """A plugin's access to Trafilatura 3D: its event subscriptions, the editor's additions (or None)."""

    def __init__(self, name: str, editor: EditorAccess | None):
        self.name = name
        self.events = Events()
        self.editor = editor

    def log(self, message: str) -> None:
        manager().report(f"{self.name}: {message}")


@dataclass
class PluginInfo:
    name: str
    path: Path
    description: str = ""
    enabled: bool = True
    loaded: bool = False
    error: str = ""
    module: object = None
    host: PluginHost | None = None
    search_path: str = ""


@dataclass
class Manager:
    """Discovers, loads and unloads plugins, and dispatches events to them."""
    plugins: dict[str, PluginInfo] = field(default_factory=dict)
    editor_factory: Callable[[str], EditorAccess] | None = None  # set by the editor: an EditorAccess per plugin
    reporter: Callable[[str], None] | None = None  # where messages go (the editor's status bar; else stderr)

    # --- settings: the user's plugin paths and disabled plugins -------------------------------------

    @staticmethod
    def settings_path() -> Path:
        from trafilatura3d.project import user_dir
        return user_dir() / "plugins.json"

    def settings(self) -> dict:
        p = self.settings_path()
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except (OSError, ValueError):
            data = {}
        return {"paths": list(data.get("paths", [])), "disabled": list(data.get("disabled", []))}

    def save_settings(self, data: dict) -> None:
        p = self.settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def search_paths(self) -> list[str]:
        """The project's plugin paths, the user's, then the user folder's plugins."""
        from trafilatura3d.project import configuration, user_dir
        out = []
        for p in [*configuration().plugin_paths(), *self.settings()["paths"], str(user_dir() / "plugins")]:
            if p not in out:
                out.append(p)
        return out

    # --- discovery and loading ---------------------------------------------------------------------

    def discover(self) -> list[PluginInfo]:
        """Plugins found in the search paths (a package folder or a .py file; the first of a name wins)."""
        disabled = set(self.settings()["disabled"])
        found = {}
        for folder in self.search_paths():
            base = Path(folder)
            if not base.is_dir():
                continue
            for entry in sorted(base.iterdir()):
                if entry.name.startswith((".", "_")):
                    continue
                if entry.is_dir() and (entry / "__init__.py").exists():
                    name = entry.name
                elif entry.is_file() and entry.suffix == ".py":
                    name = entry.stem
                else:
                    continue
                if name in found:
                    continue
                info = self.plugins.get(name)
                if info is None or info.path != entry:
                    info = PluginInfo(name, entry, search_path=folder)
                info.enabled = name not in disabled
                info.description = info.description or _description(entry)
                found[name] = info
        for name, info in list(self.plugins.items()):  # loaded plugins whose files went away stay listed
            if name not in found and info.loaded:
                found[name] = info
        self.plugins = found
        return list(found.values())

    def load_all(self) -> None:
        self.discover()
        for info in self.plugins.values():
            if info.enabled and not info.loaded:
                self.load(info.name)

    def load(self, name: str) -> bool:
        info = self.plugins[name]
        if info.loaded:
            return True
        module_name = MODULE_PREFIX + name
        try:
            if info.path.is_dir():  # a self-contained package: its relative imports resolve inside it
                spec = importlib.util.spec_from_file_location(module_name, info.path / "__init__.py",
                                                              submodule_search_locations=[str(info.path)])
            else:
                spec = importlib.util.spec_from_file_location(module_name, info.path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            init = getattr(module, "initialize_plugin", None)
            if init is None:
                raise AttributeError("defines no initialize_plugin(host)")
            host = PluginHost(name, self.editor_factory(name) if self.editor_factory else None)
            init(host)
        except Exception:  # noqa: BLE001 - a plugin's failure is reported, never fatal
            info.error = traceback.format_exc(limit=6)
            self._forget(module_name)
            if "host" in locals() and host.editor is not None:
                host.editor.remove_all()
            self.report(f"plugin {name} failed to load: {info.error.strip().splitlines()[-1]}")
            return False
        info.module, info.host, info.loaded, info.error = module, host, True, ""
        return True

    def unload(self, name: str) -> None:
        info = self.plugins.get(name)
        if info is None or not info.loaded:
            return
        try:
            done = getattr(info.module, "uninitialize_plugin", None)
            if done is not None:
                done(info.host)
        except Exception:  # noqa: BLE001
            self.report(f"plugin {name} failed to unload: {traceback.format_exc(limit=3).strip().splitlines()[-1]}")
        if info.host is not None and info.host.editor is not None:
            info.host.editor.remove_all()
        self._forget(MODULE_PREFIX + name)
        info.module, info.host, info.loaded = None, None, False

    def reload(self, name: str) -> bool:
        self.unload(name)
        self.discover()
        return self.load(name) if name in self.plugins else False

    def unload_all(self) -> None:
        for name in list(self.plugins):
            self.unload(name)

    def set_enabled(self, name: str, enabled: bool) -> None:
        data = self.settings()
        disabled = set(data["disabled"])
        disabled.discard(name) if enabled else disabled.add(name)
        data["disabled"] = sorted(disabled)
        self.save_settings(data)
        info = self.plugins.get(name)
        if info is not None:
            info.enabled = enabled
            self.load(name) if enabled else self.unload(name)

    @staticmethod
    def _forget(module_name: str) -> None:
        for key in [k for k in sys.modules if k == module_name or k.startswith(module_name + ".")]:
            del sys.modules[key]

    # --- events ------------------------------------------------------------------------------------

    def emit(self, event: str, **data) -> None:
        """Tell every loaded plugin; a handler's error is reported. For a "before" event, a handler's Veto
        is raised to the caller (the operation must not happen)."""
        for info in list(self.plugins.values()):
            if not info.loaded or info.host is None:
                continue
            for handler in list(info.host.events.handlers.get(event, [])):
                try:
                    handler(**data)
                except Veto as v:
                    if event.startswith("before_"):
                        raise Veto(f"{info.name}: {v}") from None
                    self.report(f"plugin {info.name}: Veto is only for before_ events ({event})")
                except Exception:  # noqa: BLE001
                    self.report(f"plugin {info.name} failed on {event}: "
                                f"{traceback.format_exc(limit=3).strip().splitlines()[-1]}")

    def report(self, message: str) -> None:
        if self.reporter is not None:
            self.reporter(message)
        else:
            print(f"trafilatura3d: {message}", file=sys.stderr)


def _description(path: Path) -> str:
    """The first line of the plugin's docstring, read without importing it."""
    import ast
    source = path / "__init__.py" if path.is_dir() else path
    try:
        doc = ast.get_docstring(ast.parse(source.read_text(encoding="utf-8")))
    except (OSError, SyntaxError, ValueError):
        return ""
    return (doc or "").strip().splitlines()[0] if doc else ""


_manager = Manager()


def manager() -> Manager:
    return _manager


def emit(event: str, **data) -> None:
    """For Trafilatura 3D's own code at the places the events name."""
    _manager.emit(event, **data)
