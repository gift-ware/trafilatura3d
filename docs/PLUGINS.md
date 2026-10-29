# Writing a Trafilatura 3D plugin

A plugin adds what a pipeline needs (naming checks, publishing, asset ids…) without Trafilatura 3D knowing that
pipeline. It is a **self-contained** Python package — a folder with an `__init__.py` and everything it
needs — or a single `.py` file, with an initialize and an optional uninitialize function.

```
studio_trafilatura3d/            # the folder's name is the plugin's name
├── __init__.py            # first docstring line: the description shown in the Plugin Manager
└── checks.py              # its own modules, imported relatively: from .checks import ...
```

```python
"""Studio pipeline: naming checks and publishing."""
from trafilatura3d.plugins import Veto
from .checks import valid_name


def initialize_plugin(host):
    def before_export(package, graph_id, out_dir, paths):
        if not valid_name(graph_id):
            raise Veto(f"{graph_id}: not a valid asset name")   # the export does not happen
    host.events.subscribe("before_export", before_export)
    if host.editor is not None:                                  # None from the command line
        host.editor.add_menu_action("Studio", "Publish", publish)


def uninitialize_plugin(host):   # optional; what the host registered is removed anyway
    pass
```

A pipeline brings its resources (shaders, export presets, materials) as a library its project file declares, next to
its plugin: see `docs/SHADERS.md`.

## Where Trafilatura 3D looks

In order (the first plugin of a name wins): the project's plugin paths (`"plugins": {"paths": [...]}` in a
`.trafilatura3dproject`, relative to it; a pipeline declares its folder there), the paths added in Tools > Plugin
Manager, then `plugins` in the user folder (`~/Library/Application Support/Trafilatura 3D` on macOS, or
`TRAFILATURA3D_USER_DIR`).

## Events (`host.events.subscribe(name, handler)`)

| Event | Arguments | |
|---|---|---|
| `package_opened` | package | |
| `before_save` | package, path | raise `Veto` to refuse |
| `package_saved` | package, path | |
| `before_export` | package, graph_id, out_dir, paths | raise `Veto` to refuse; `paths`: the files it will write |
| `outputs_exported` | package, graph_id, paths | |
| `before_export_textures` | session, out_dir, paths | Paint export; raise `Veto` to refuse; `paths`: the files it will write |
| `textures_exported` | session, paths | Paint export; `session.path` is None for a project never saved |
| `before_bake` | low_path, bakers, settings, out_dir | raise `Veto` to refuse |
| `bake_finished` | low_path, bakers, settings, paths | |

A `Veto` message is shown where Trafilatura 3D shows its errors (dialogs, the command line's stderr). Any other
error in a handler is reported and ignored: a plugin never stops Trafilatura 3D.

## Editor (`host.editor`, None from the command line)

- `add_menu_action(menu, text, callback, shortcut="", workspaces=("graph", "paint"))`: in that menu of each
  workspace's menu bar, created when missing; `add_menu_separator(menu, workspaces=…)`.
- `workspace()` → "home", "graph" or "paint"; `current_paint()` → the open Paint project's session, or None.
- `edit(document, label, fn)`: change an open package or the Paint project (`current_paint()`) with `fn()` as one
  undoable step; `open(path)`: open a package or a Paint project as File > Open does.
- `add_dock(title, widget)`: a dock in the workspace.
- `main_window()`, `current()` → (package, graph id) in the Graph view.
- `save(document, path=None)`: save an open package or the Paint project (`session`) as Save does, at `path` as
  Save As does without asking (the unsaved marks cleared);
  raises `OSError` when it could not be saved. From the command line there is no editor: `package.save()`.

Everything added is removed when the plugin is unloaded (disabled or reloaded in the Plugin Manager).
The command line loads plugins too (`--no-plugins` skips them, `trafilatura3d plugins` lists them).
