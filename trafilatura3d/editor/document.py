"""The editor's model: the open packages, undo/redo, and compiled graphs.

Every change goes through `edit()`: it snapshots the package before and after, so undo/redo restore
whole states (simple and always consistent). Consecutive edits with the same merge key (a slider
being dragged) collapse into one undo step. Graphs are recompiled on demand after a change; the
evaluator's content-addressed cache recooks only what really changed.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from trafilatura3d.compiler import Compiled, compile_graph
from trafilatura3d.editor.live import PREVIEW_SIZE, LiveCook
from trafilatura3d.evaluator import Evaluator
from trafilatura3d.package import DocumentError, Package


class _Snapshot(QUndoCommand):
    def __init__(self, doc: Document, package: Package, label: str, before: str, after: str, merge: str | None,
                 layout: bool):
        super().__init__(label)
        self.doc, self.package, self.before, self.after, self.merge = doc, package, before, after, merge
        self.layout = layout
        self._first = True

    def redo(self):
        if self._first:  # the edit is already applied
            self._first = False
            return
        self.doc._restore(self.package, self.after, self.layout)

    def undo(self):
        self.doc._restore(self.package, self.before, self.layout)

    def id(self):
        return 1 if self.merge else -1

    def mergeWith(self, other):
        if other.merge != self.merge or other.package is not self.package or other.layout != self.layout:
            return False
        self.after = other.after
        return True


class Document(QObject):
    changed = Signal(object)  # a package whose content changed
    layout_changed = Signal(object)  # a package whose arrangement only changed: nothing to recook
    packages_changed = Signal()  # a package opened, closed, saved or renamed
    dirty_changed = Signal()  # an edit, undo or redo: the unsaved marks may change
    warnings_changed = Signal(object)  # a package whose warnings were checked again (`check`, `check_nodes`)
    error = Signal(str)

    def __init__(self, evaluator: Evaluator | None = None):
        super().__init__()
        self.ev = evaluator or Evaluator()
        self.packages: list[Package] = []
        self.undo = QUndoStack(self)
        self.dirty: set[int] = set()  # id() of packages with unsaved changes
        self._compiled: dict[tuple, Compiled] = {}
        self._revision: dict[int, int] = {}
        # Preview mode of a graph's input parameters: (id(package), graph id) -> {input id: value}.
        # Not saved; "Apply" makes them the defaults.
        self.preview: dict[tuple, dict] = {}
        self.dragging = False  # the last edit is a value being dragged (`edit` drag): its release comes next
        self.live: LiveCook | None = None  # the Graph workspace's refresh (set by the main window)
        # The warnings found (`check`, `check_nodes`): id(package) -> graph id ("" its dependencies) -> warnings
        self.warnings: dict[int, dict[str, list]] = {}
        self.live_warnings = False  # Tools > Live warnings: every edit checks its whole package again (slow)
        self.undo.indexChanged.connect(lambda _: self.dirty_changed.emit())

    # --- packages -----------------------------------------------------------------------------

    def new_package(self) -> Package:
        """A new, empty package (its graphs come from the New graph dialog)."""
        pkg = Package()
        self.packages.append(pkg)
        self.dirty.add(id(pkg))
        self.packages_changed.emit()
        return pkg

    def open(self, path: str | Path) -> Package:
        path = Path(path).resolve()
        for pkg in self.packages:
            if pkg.path == path:
                return pkg
        pkg = Package.load(path)
        self.packages.append(pkg)
        self.check(pkg)  # opened: what is wrong in it, shown at once
        self.packages_changed.emit()
        from trafilatura3d import plugins
        plugins.emit("package_opened", package=pkg)
        return pkg

    def import_sbs(self, path: str | Path) -> Package:
        """A .sbs package converted into a new unsaved package (Save writes a .trafilatura3d); what the import could
        not convert is in the package's notes, shown with the graphs' warnings."""
        from trafilatura3d.sbs import import_sbs
        pkg, _ = import_sbs(path)
        self.packages.append(pkg)
        self.dirty.add(id(pkg))
        self.check(pkg)  # what the import could not convert, shown at once
        self.packages_changed.emit()
        from trafilatura3d import plugins
        plugins.emit("package_opened", package=pkg)
        return pkg

    def save(self, pkg: Package, path: str | Path | None = None) -> None:
        pkg.save(path)
        self.dirty.discard(id(pkg))
        self.packages_changed.emit()

    def close(self, pkg: Package) -> None:
        self.undo.clear()  # snapshots refer to the closed package (first: its signals reach views still showing it)
        self.packages.remove(pkg)
        self.dirty.discard(id(pkg))
        self.warnings.pop(id(pkg), None)
        self.packages_changed.emit()

    def set_preview(self, pkg: Package, graph_id: str, values: dict | None) -> None:
        """Turn the Preview mode of a graph's inputs on (values) or off (None); views recook."""
        if values is None:
            self.preview.pop((id(pkg), graph_id), None)
        else:
            self.preview[(id(pkg), graph_id)] = dict(values)
        self.changed.emit(pkg)

    # --- warnings -------------------------------------------------------------------------------

    def check(self, pkg: Package) -> None:
        """Every warning of a package (`diagnostics.package_warnings`: every graph, every dependency, slow): when it
        opens, and after each edit with Tools > Live warnings."""
        from trafilatura3d.diagnostics import package_warnings
        try:
            self.warnings[id(pkg)] = package_warnings(pkg)
        except Exception:  # noqa: BLE001 - diagnostics must never break the editor
            self.warnings[id(pkg)] = {}
        self.warnings_changed.emit(pkg)

    def check_nodes(self, pkg: Package, graph_id: str, node_ids) -> None:
        """The warnings of the nodes being worked on (selected, just connected) checked again
        (`diagnostics.node_warnings`), the others kept as last found: the work's own nodes, not the chain."""
        from trafilatura3d.diagnostics import node_warnings
        if graph_id not in pkg.graphs:
            return
        ids = set(node_ids)
        try:
            fresh = node_warnings(pkg, graph_id, ids)
        except Exception:  # noqa: BLE001 - diagnostics must never break the editor
            fresh = []
        found = self.warnings.setdefault(id(pkg), {})
        before = found.get(graph_id, [])
        now = [w for w in before if w.node not in ids] + fresh
        if now == before:
            return
        if now:
            found[graph_id] = now
        else:
            found.pop(graph_id, None)
        self.warnings_changed.emit(pkg)

    def graph_warnings(self, pkg: Package, graph_id: str) -> list:
        """A graph's warnings as last found, without the nodes deleted since ("" the package's dependencies')."""
        g = pkg.graphs.get(graph_id)
        return [w for w in self.warnings.get(id(pkg), {}).get(graph_id, [])
                if w.node is None or g is None or w.node in g.nodes]

    def set_live_warnings(self, on: bool) -> None:
        """Tools > Live warnings: every edit checks its whole package again (slow); on, every package now."""
        self.live_warnings = on
        if on:
            for pkg in self.packages:
                self.check(pkg)

    def name(self, pkg: Package) -> str:
        return (pkg.path.stem if pkg.path else "Unsaved Package") + ("*" if id(pkg) in self.dirty else "")

    # --- edits --------------------------------------------------------------------------------

    def edit(self, pkg: Package, label: str, fn, merge: str | None = None, layout: bool = False, drag: bool = False):
        """Apply `fn()` to the package as one undoable step; a DocumentError leaves it untouched. A `layout` edit only
        arranges (positions of nodes, frames, comments, pins): no new revision, `layout_changed` instead of
        `changed`, so nothing recompiles or recooks. A `drag` edit is a value being dragged (`dragging` while
        `changed` is emitted: the views show it at the preview resolution); the next edit without it is the
        release, `changed` again even when the value did not move."""
        before = json.dumps(pkg.to_dict())
        try:
            result = fn()
        except DocumentError as e:
            pkg.restore(json.loads(before))
            self.error.emit(str(e))
            return None
        after = json.dumps(pkg.to_dict())
        released = self.dragging and not drag
        self.dragging = drag
        if after != before:
            self.undo.push(_Snapshot(self, pkg, label, before, after, merge, layout))
            self._touched(pkg, layout)
        elif released:
            self.changed.emit(pkg)
        return result

    def _restore(self, pkg: Package, state: str, layout: bool) -> None:
        self.dragging = False
        if layout:
            pkg.restore_layout(json.loads(state))  # in place: the panels that skip layout changes hold these objects
        else:
            pkg.restore(json.loads(state))
        self._touched(pkg, layout)

    def _touched(self, pkg: Package, layout: bool = False) -> None:
        self.dirty.add(id(pkg))
        if layout:
            self.layout_changed.emit(pkg)
            return
        self._revision[id(pkg)] = self._revision.get(id(pkg), 0) + 1
        if self.live_warnings and not self.dragging:  # at a dragged value's release, not at each step
            self.check(pkg)
        self.changed.emit(pkg)

    # --- compilation --------------------------------------------------------------------------

    def compiled(self, pkg: Package, graph_id: str, shift: int = 0) -> Compiled | None:
        """A graph compiled at its resolution, or `shift` powers of 2 below it (`compile_graph`)."""
        preview = self.preview.get((id(pkg), graph_id))
        key = (id(pkg), graph_id, self._revision.get(id(pkg), 0), json.dumps(preview, sort_keys=True), shift)
        if key not in self._compiled:
            if graph_id not in pkg.graphs:
                return None
            try:  # one revision kept per graph, at every resolution asked
                self._compiled = {k: v for k, v in self._compiled.items() if k[:2] != key[:2] or k[2:4] == key[2:4]}
                self._compiled[key] = compile_graph(pkg, graph_id, evaluator=self.ev, values=preview, shift=shift)
            except (DocumentError, OSError, ValueError) as e:
                self.error.emit(str(e))
                return None
        return self._compiled[key]

    def shown(self, pkg: Package, graph_id: str) -> Compiled | None:
        """What the viewports show of a graph: at its resolution, or while a value is dragged (`live` previewing)
        compiled at most PREVIEW_SIZE on a side where the drag changes it. A node output whose full resolution result
        is cached is not changed by the drag (a job's key is its content): it stays at full resolution (a node linked
        to nothing, dragged, recooked the whole graph at the preview resolution)."""
        full = self.compiled(pkg, graph_id)
        if full is None or self.live is None or not self.live.previewing:
            return full
        largest = max((max(j.size) for j in full.jobs.values()), default=1)
        shift = max(0, largest.bit_length() - PREVIEW_SIZE.bit_length())
        preview = self.compiled(pkg, graph_id, shift) if shift else full
        if preview is None or preview is full:
            return full
        cache = self.ev.cache

        def pick(src, low):
            return src if src[0].key in cache or low is None else low
        return Compiled(preview.jobs,
                        {nid: {out: pick(src, preview.nodes.get(nid, {}).get(out)) for out, src in outs.items()}
                         for nid, outs in full.nodes.items()},
                        {ident: pick(src, preview.outputs.get(ident)) for ident, src in full.outputs.items()},
                        preview.graph)
