"""The editable shortcuts of the Paint workspace (Shortcuts page): keyboard shortcuts of actions and mouse
bindings of the viewports, with the page's Windows and macOS defaults, the user's changes in the application's
QSettings ("paint_shortcuts", JSON name -> binding, "" = none).

Notation is Qt's: "Ctrl" is Cmd on macOS and "Meta" its Control key (Qt swaps them), so the page's "⌘+Z" is
"Ctrl+Z" and its macOS "Ctrl+Shift+Mouse left" is "Meta+Shift+Mouse left". A mouse binding is modifiers, an
optional key held down (a letter: "S+Mouse left") and a button.

Not offered, the features are missing: Rendering (Iray), Pause engine computation, the "+ Particles" tool
variants, Toggle animation, Auto-Focus (Depth of Field), Resume previous stroke."""

from __future__ import annotations

import json
import sys

from PySide6.QtCore import QSettings

MAC = sys.platform == "darwin"

# name -> (Windows, macOS); None = no shortcut. Groups as the page's table.
KEYBOARD = {
    "Hide UI": ("Tab", "Tab"),
    "Undo": ("Ctrl+Z", "Ctrl+Z"),
    "Redo": ("Ctrl+Y", "Ctrl+Y"),
    "Baking": ("F8", "F8"),
    "Painting": ("F9", "F9"),
    "Open project": ("Ctrl+O", "Ctrl+O"),
    "Close project": ("Ctrl+F4", "Ctrl+W"),
    "Save project": ("Ctrl+S", "Ctrl+S"),
    "Save as project": (None, None),
    "New project": ("Ctrl+N", "Ctrl+N"),
    "Quit application": ("Alt+F4", "Ctrl+Q"),
    "Show/Hide manipulator": ("Q", "Q"),
    "Increase the manipulator size": ("+", "+"),
    "Decrease the manipulator size": ("-", "-"),
    "Cycle through manipulation spaces": ("T", "T"),
    "Toggle warp edition mode": ("Shift+V", "Shift+V"),
    "Translate tool": ("W", "W"),
    "Rotate tool": ("E", "E"),
    "Scale tool": ("R", "R"),
    "Surface tool": ("Shift+W", "Shift+W"),
    "Symmetry": ("L", "L"),
    "Select Paint tool": ("1", "1"),
    "Select Eraser tool": ("2", "2"),
    "Select Projection tool": ("3", "3"),
    "Select Polygon Fill": ("4", "4"),
    "Select Smudge tool": ("5", "5"),
    "Select Clone tool (relative source)": ("6", "6"),
    "Select Clone tool (absolute source)": ("Ctrl+6", "Ctrl+6"),
    "Bake Mesh Maps": ("Ctrl+Shift+B", "Ctrl+Shift+B"),
    "Increase tool size": ("]", "]"),
    "Decrease tool size": ("[", "["),
    "Invert grayscale tool": ("X", "X"),
    "Pick stroke material": ("P", "P"),
    "Lazy Mouse": ("D", "D"),
    "Hide/ignore excluded geometry": ("Alt+H", "Alt+H"),
    "Camera Perspective": ("F5", "F5"),
    "Camera Orthographic": ("F6", "F6"),
    "Display next channel": ("C", "C"),
    "Display previous channel": ("Shift+C", "Shift+C"),
    "Display material": ("M", "M"),
    "Display next mesh map": ("B", "B"),
    "Display previous mesh map": ("Shift+B", "Shift+B"),
    "Export textures": ("Ctrl+Shift+E", "Ctrl+Shift+E"),
    "Center the whole mesh": ("F", "F"),
    "Toggle quick mask edition": ("U", "U"),
    "Clear quick mask": ("Y", "Y"),
    "Invert quick mask": ("I", "I"),
    "Viewport layout 3D/2D": ("F1", "F1"),
    "Viewport layout 3D only": ("F2", "F2"),
    "Viewport layout 2D only": ("F3", "F3"),
    "Swap 2D / 3D view": ("F4", "F4"),
    "Texture Set isolate": ("Alt+Q", "Alt+Q"),
    "Ignore stencil mask": ("N", "N"),  # held down
}

# The page leaves the macOS column empty for Camera translate (alternative) and Stencil translation: the
# Windows binding there is ours.
MOUSE = {
    "Use tool / paint": ("Mouse left", "Mouse left"),
    "Draw straight lines": ("Shift+Mouse left", "Shift+Mouse left"),
    "Draw straight lines with snapping": ("Ctrl+Shift+Mouse left", "Meta+Shift+Mouse left"),
    "Camera rotate": ("Alt+Mouse left", "Alt+Mouse left"),
    "Camera snap rotate": ("Alt+Shift+Mouse left", "Ctrl+Shift+Mouse left"),
    "Camera translate": ("Alt+Mouse middle", "Ctrl+Alt+Mouse left"),
    "Camera translate (alternative)": ("Ctrl+Alt+Mouse left", "Alt+Mouse middle"),
    "Camera zoom": ("Alt+Mouse right", "Mouse middle"),
    "Stencil rotate": ("S+Mouse left", "S+Mouse left"),
    "Stencil snap rotate": ("Shift+S+Mouse left", "Shift+S+Mouse left"),
    "Stencil translation": ("S+Mouse middle", "S+Mouse middle"),
    "Stencil zoom": ("S+Mouse right", "S+Mouse right"),
    "Change tool Size / Hardness": ("Ctrl+Mouse right", "Ctrl+Mouse right"),
    "Change tool Flow / Rotation": ("Ctrl+Mouse left", "Ctrl+Mouse left"),
    "Rotate environment": ("Shift+Mouse right", "Shift+Mouse right"),
    "Texture set selection": ("Ctrl+Alt+Mouse right", "Ctrl+Alt+Mouse right"),
    "Context menu": ("Mouse right", "Mouse right"),
    "Set Clone tool source location": ("V+Mouse left", "V+Mouse left"),
}

BUTTONS = {"Mouse left": 1, "Mouse right": 2, "Mouse middle": 3}  # rendercanvas button numbers
MODIFIERS = {"Ctrl": "Control", "Alt": "Alt", "Shift": "Shift", "Meta": "Meta"}  # notation -> event names
KEY = "paint_shortcuts"


def _overrides() -> dict:
    try:
        return dict(json.loads(QSettings().value(KEY, "{}")))
    except (TypeError, ValueError):
        return {}


def default(name: str) -> str:
    pair = KEYBOARD.get(name) or MOUSE[name]
    return pair[1 if MAC else 0] or ""


def binding(name: str) -> str:
    """The current binding of a shortcut ("" = none)."""
    return _overrides().get(name, default(name))


def set_binding(name: str, value: str | None) -> None:
    """A new binding (None: back to the default)."""
    o = _overrides()
    if value is None or value == default(name):
        o.pop(name, None)
    else:
        o[name] = value
    QSettings().setValue(KEY, json.dumps(o))


def parse_mouse(text: str):
    """(modifier event names, held key or None, button) of a mouse binding, None when empty or unreadable."""
    if not text:
        return None
    parts = text.split("+")
    button = BUTTONS.get(parts[-1])
    if button is None:
        return None
    mods, held = set(), None
    for p in parts[:-1]:
        if p in MODIFIERS:
            mods.add(MODIFIERS[p])
        elif len(p) == 1:
            held = p.lower()
        else:
            return None
    return frozenset(mods), held, button


def mouse_text(mods, held, button) -> str:
    names = {v: k for k, v in MODIFIERS.items()}
    parts = [names[m] for m in ("Control", "Meta", "Alt", "Shift") if m in mods]
    if held:
        parts.append(held.upper())
    return "+".join(parts + [next(k for k, v in BUTTONS.items() if v == button)])


def match_mouse(button: int, mods, held: set) -> list[str]:
    """The mouse bindings a press matches, the ones with a held key first (they are the more specific). On macOS
    a binding with Ctrl (Cmd) also takes the Control key when no binding is exactly that press, as the keyboard
    shortcuts do (`keys.variants`)."""
    def matching(pressed):
        out = []
        for name in MOUSE:
            b = parse_mouse(binding(name))
            if b is None or b[2] != button or b[0] != pressed:
                continue
            if b[1] is not None and b[1] not in held:
                continue
            out.append((b[1] is None, name))
        return [name for _, name in sorted(out)]
    pressed = frozenset(mods)
    found = matching(pressed)
    if not found and MAC and "Meta" in pressed and "Control" not in pressed:
        found = matching(pressed - {"Meta"} | {"Control"})
    return found


def label(text: str) -> str:
    """A binding as the platform writes it (macOS: ⌘ Option Ctrl)."""
    if not MAC or not text:
        return text
    return "+".join({"Ctrl": "⌘", "Alt": "Option", "Meta": "Ctrl"}.get(p, p) for p in text.split("+"))
