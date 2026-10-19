"""Shortcuts on macOS: Qt's "Ctrl" is the Command key there, so the physical Control key would do nothing;
both work (the shortcuts say Ctrl, Mac users reach for Command)."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut


def variants(*seqs) -> list[QKeySequence]:
    """The shortcuts (every platform binding of a standard key), and on macOS the same with the physical
    Control key; no duplicates."""
    keys = []
    for seq in seqs:
        keys += QKeySequence.keyBindings(seq) if isinstance(seq, QKeySequence.StandardKey) else [QKeySequence(seq)]
    out, seen = [], set()
    for key in keys:
        text = key.toString(QKeySequence.PortableText)
        forms = [text] + ([text.replace("Ctrl+", "Meta+")] if sys.platform == "darwin" and "Ctrl+" in text else [])
        for t in forms:
            if t and t not in seen:
                seen.add(t)
                out.append(QKeySequence(t))
    return out


def shortcut(seq, parent, slot, context=Qt.WindowShortcut) -> list[QShortcut]:
    """QShortcuts for every variant of `seq` on `parent`."""
    return [QShortcut(k, parent, activated=slot, context=context) for k in variants(seq)]
