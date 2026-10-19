"""The Paint filters and generators as nodes of the library: usable in a graph and, through `paint/graph_nodes.py`, as effects of the
Paint workspace (decision 23: one implementation for both).

Shared machinery in `_lib.py` (base class, parameter shorthands, helper passes) and the WGSL helpers `lib.wgsl`
(prepended to every shader of this area) and `grunge.wgsl` (for the generators that read position and world
space normal maps)."""
