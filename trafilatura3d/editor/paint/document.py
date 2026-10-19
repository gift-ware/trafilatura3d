"""The open Paint project in the editor: the session, the History (undo stack), the current Texture Set and
the selected layer (or its mask).

Every change to the document goes through `edit()` (a snapshot of the project before and after); a stroke
goes through `stroke_done()` (the painted entries before and after, on the GPU). The History panel shows the
stack; as the documentation says it is global to the project and lives only for the session.
"""

from __future__ import annotations

import json

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from trafilatura3d.paint.model import Project, ProjectError
from trafilatura3d.paint.session import Session


class _Snapshot(QUndoCommand):
    def __init__(self, doc, label, before: str, after: str, merge: str | None):
        super().__init__(label)
        self.doc, self.before, self.after, self.merge = doc, before, after, merge
        self._first = True

    def redo(self):
        if self._first:
            self._first = False
            return
        self.doc._restore(self.after)

    def undo(self):
        self.doc._restore(self.before)

    def id(self):
        return 1 if self.merge else -1

    def mergeWith(self, other):
        if not isinstance(other, _Snapshot) or other.merge != self.merge:
            return False
        self.after = other.after
        return True


class _Stroke(QUndoCommand):
    """A stroke: the region it reached in each painted entry, before and after, on the GPU
    (paint.PaintStore.stroke_undo)."""

    def __init__(self, doc, label, snaps: dict):
        super().__init__(label)
        self.doc, self.snaps = doc, snaps
        self._first = True

    def _apply(self, after: bool):
        for key, snap in self.snaps.items():
            self.doc.session.store.apply_undo(key, snap, after)
        self.doc.content_changed.emit()

    def redo(self):
        if self._first:
            self._first = False
            return
        self._apply(True)

    def undo(self):
        self._apply(False)


class PaintDocument(QObject):
    content_changed = Signal()  # pixels of the stack changed: viewports and thumbnails update
    # a value being dragged changed (an edit merged into the previous one): the viewports follow it at a reduced
    # resolution, the full one comes back once it settles (content_changed)
    content_previewed = Signal()
    structure_changed = Signal()  # layers, Texture Sets or channels changed: panels rebuild
    selection_changed = Signal()
    error = Signal(str)

    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self.undo = QUndoStack(self)
        self.undo.setUndoLimit(200)
        self.dirty = False
        self._previewed = False  # a dragged value shown since the last content_changed of an edit
        self.undo.indexChanged.connect(self._mark_dirty)
        sets = session.project.texture_sets
        self.ts_name = sets[0].name if sets else ""
        self.selected: str | None = None  # a layer's or an effect's uid
        self.mask_selected = False  # the selected layer's mask (or an effect of the mask) is selected
        self.multi: list[str] = []  # the layers selected together (Ctrl/Cmd and Shift click), `selected` among them
        self.viewmode = "basecolor"  # the Layer stack's channel dropdown
        self._select_default()

    @property
    def project(self) -> Project:
        return self.session.project

    @property
    def ts(self):
        try:
            return self.project.texture_set(self.ts_name)
        except StopIteration:
            return self.project.texture_sets[0] if self.project.texture_sets else None

    def layer(self):
        """The selected layer (the owner of the selected effect)."""
        ts = self.ts
        return ts.owner_of(self.selected)[0] if ts is not None and self.selected else None

    def effect(self):
        """The selected effect, or None when a layer (or its mask) is selected."""
        ts = self.ts
        return ts.owner_of(self.selected)[1] if ts is not None and self.selected else None

    def _mark_dirty(self, *_):
        self.dirty = True

    def _select_default(self):
        ts = self.ts
        if ts is None:
            return
        layer, _e, in_mask = ts.owner_of(self.selected or "")
        self.multi = [u for u in self.multi if ts.find(u) is not None]
        if layer is None:
            self.selected = ts.layers[-1].uid if ts.layers else None
            self.mask_selected = False
            self.multi = [self.selected] if self.selected else []
        elif self.mask_selected and layer.mask is None:
            self.mask_selected = False
        if self.viewmode not in ts.channel_ids() and ts.channels:
            self.viewmode = ts.channels[0].id

    # --- selection ------------------------------------------------------------------------------------

    def select_texture_set(self, name: str):
        if name == self.ts_name:
            return
        self.ts_name = name
        self.selected = None
        self._select_default()
        self.structure_changed.emit()
        self.selection_changed.emit()

    def select(self, uid: str | None, mask: bool = False, multi: list | None = None):
        """Select a layer (`mask`: its mask) or an effect (in a mask: the mask stays selected); `multi`: the
        layers selected together (default: just this one)."""
        layer, effect, in_mask = self.ts.owner_of(uid) if uid else (None, None, False)
        self.selected = uid
        self.multi = list(multi) if multi else ([uid] if uid and effect is None else [])
        self.mask_selected = bool(layer is not None and layer.mask is not None and (mask or in_mask))
        self.selection_changed.emit()

    # --- edits ----------------------------------------------------------------------------------------

    def edit(self, label: str, fn, merge: str | None = None, structure: bool = True):
        """Apply `fn()` to the project as one History step; an invalid result is undone and reported."""
        before = json.dumps(self.project.to_dict())
        try:
            result = fn()
            self.project.validate()
        except (ProjectError, ValueError) as e:
            self._restore(before)
            self.error.emit(str(e))
            return None
        after = json.dumps(self.project.to_dict())
        if after != before:
            self.undo.push(_Snapshot(self, label, before, after, merge))
            self._select_default()
            if structure:
                self.structure_changed.emit()
        if merge is not None:
            if after != before:
                self._previewed = True
                self.content_previewed.emit()
        elif after != before or self._previewed:  # the drag's release, even when the value did not move again
            self._previewed = False
            self.content_changed.emit()
        return result

    def _restore(self, state: str):
        restored = Project.from_dict(json.loads(state))
        self.project.__dict__.update(restored.__dict__)
        self._select_default()
        self.structure_changed.emit()
        self.content_changed.emit()
        self.selection_changed.emit()

    def stroke_done(self, label: str, saved):
        """A finished stroke: `saved` what its entries held before it (brush.Saved, destroyed here)."""
        store = self.session.store
        snaps = {key: store.stroke_undo(key, saved) for key in saved.keys()}
        saved.destroy()
        snaps = {k: v for k, v in snaps.items() if v is not None}
        if snaps:
            self.undo.push(_Stroke(self, label, snaps))

    def save(self, path=None):
        self.session.save(path)
        self.dirty = False
        self.undo.setClean()
