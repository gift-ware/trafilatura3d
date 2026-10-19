"""Layer blending modes, from the official documentation's Blending modes page: names, order, the
user-data identifiers (blendingmode=...) and the formulas it gives. Where it describes a mode without a
formula (Exclusion, Overlay, Soft/Hard/Vivid/Linear/Pin Light, Color Burn's ambiguous wording) the
usual compositing definitions are used; Signed Addition's factor is ours (2: white adds 1, black
subtracts 1). Every mode works on linear values (the documentation: "performed in linear gamma space")."""

from __future__ import annotations

# (identifier, label, short label of the layer row). Linear Light has no user-data identifier in the
# documentation: "linearlight" is ours.
MODES = [
    ("normal", "Normal", "Norm"),
    ("passthrough", "Passthrough", "Pthr"),
    ("disable", "Disable", "Dis"),
    ("replace", "Replace", "Repl"),
    ("multiply", "Multiply", "Mul"),
    ("divide", "Divide", "Div"),
    ("inversedivide", "Inverse Divide", "IDiv"),
    ("darken", "Darken (Min)", "Dark"),
    ("lighten", "Lighten (Max)", "Light"),
    ("lineardodge", "Linear Dodge (Add)", "LDdg"),
    ("subtract", "Subtract", "Sub"),
    ("inversesubtract", "Inverse Subtract", "ISub"),
    ("difference", "Difference", "Diff"),
    ("exclusion", "Exclusion", "Excl"),
    ("signedaddition", "Signed Addition (AddSub)", "AdSb"),
    ("overlay", "Overlay", "Ovrl"),
    ("screen", "Screen", "Scrn"),
    ("linearburn", "Linear Burn", "LBrn"),
    ("colorburn", "Color Burn", "CBrn"),
    ("colordodge", "Color Dodge", "CDdg"),
    ("softlight", "Soft Light", "SftL"),
    ("hardlight", "Hard Light", "HrdL"),
    ("vividlight", "Vivid Light", "VvdL"),
    ("linearlight", "Linear Light", "LinL"),
    ("pinlight", "Pin Light", "PinL"),
    ("tint", "Tint", "Tint"),
    ("saturation", "Saturation", "Sat"),
    ("color", "Color", "Col"),
    ("value", "Value", "Val"),
    ("normalcombine", "Normal Map Combine", "NCmb"),
    ("normaldetail", "Normal Map Detail", "NDtl"),
    ("normalinversedetail", "Normal Map Inverse Detail", "NIDt"),
]
IDS = [m[0] for m in MODES]
LABEL = {m[0]: m[1] for m in MODES}
SHORT = {m[0]: m[2] for m in MODES}
INDEX = {m[0]: i for i, m in enumerate(MODES)}
NORMAL_ONLY = ("normalcombine", "normaldetail", "normalinversedetail")
