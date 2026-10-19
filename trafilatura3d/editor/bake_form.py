"""The bakers' parameter rows, shared by the Graph workspace's Baking window (`bakers.py`) and the Paint Mesh map
settings (`paint/baking_mode.py`): one editor per kind of parameter, and the rows of a baker
(`bake_params`: labels, options, defaults, the "if X is ..." conditions). The two views keep their own layouts."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QPushButton, QSpinBox,
                               QWidget)

from trafilatura3d import bake
from trafilatura3d.bake_params import P


def spec_of(p: P):
    """The editor of a baker parameter: options (a list), "bool", "file", "vec3", or a (min, max) range."""
    if p.options:
        return list(p.options)
    if p.kind:
        return p.kind
    if isinstance(p.default, bool):
        return "bool"
    if isinstance(p.default, int):
        return (1, 4096)
    return (-1000.0, 1000.0)


def editor_widget(value, editor, set_, pick=None) -> QWidget:
    """A widget editing `value` with `editor` (see `spec_of`; also "text" and "files"); `set_(value)` on every
    change. `pick(line_edit, many)` browses for the file editors."""
    if editor == "bool":
        w = QCheckBox()
        w.setChecked(bool(value))
        w.toggled.connect(set_)
    elif isinstance(editor, list):
        w = QComboBox()
        w.addItems([str(o) for o in editor])
        if value in editor:
            w.setCurrentIndex(editor.index(value))
        w.currentIndexChanged.connect(lambda i: set_(editor[i]))
    elif editor == "text":
        w = QLineEdit(str(value))
        w.textChanged.connect(set_)
    elif editor in ("file", "files"):
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        line = QLineEdit("; ".join(value) if editor == "files" else str(value))
        line.textChanged.connect(lambda t: set_([p.strip() for p in t.split(";") if p.strip()]
                                                if editor == "files" else t))
        row.addWidget(line, 1)
        if pick is not None:
            b = QPushButton("…")
            b.clicked.connect(lambda: pick(line, editor == "files"))
            row.addWidget(b)
    elif editor == "vec3":
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        vec = list(value)
        for i in range(3):
            s = QDoubleSpinBox()
            s.setRange(-1.0, 1.0)
            s.setSingleStep(0.1)
            s.setValue(float(vec[i]))
            s.valueChanged.connect(lambda v, i=i: (vec.__setitem__(i, v), set_(list(vec))))
            row.addWidget(s)
    elif isinstance(editor[0], int):
        w = QSpinBox()
        w.setRange(*editor)
        w.setValue(int(value))
        w.valueChanged.connect(set_)
    else:
        w = QDoubleSpinBox()
        w.setDecimals(5)
        w.setRange(*editor)
        w.setSingleStep(0.01)
        w.setValue(float(value))
        w.valueChanged.connect(set_)
    return w


def baker_rows(form, baker: str, values: dict, names, store, refresh, pick=None) -> int:
    """Add to a QFormLayout the rows of a baker's parameters `names` that apply given `values` (the others are
    hidden, as the pages say "if X is ..."); `store(name, value)` on every change, then `refresh()` (deferred)
    when a row depends on that parameter. Returns the rows added."""
    spec = bake.BAKER_PARAMS.get(baker, {})
    depends = {other for p in spec.values() for other, _ in p.when}
    added = 0
    for name in names:
        p = spec[name]
        if not bake.applies(baker, name, values):
            continue

        def set_(v, name=name):
            store(name, v)
            if name in depends:
                QTimer.singleShot(0, refresh)
        form.addRow(p.label, editor_widget(values.get(name, p.default), spec_of(p), set_, pick))
        added += 1
    return added
