"""Node library: every module in this package and its sub-packages defines node types; adding a node
= adding a module and its `.wgsl` (see docs/NODE_AUTHORING.md)."""

import importlib
import pkgutil

from trafilatura3d.nodes.base import Node

# A module that fails to import is skipped (one broken node must not take the library down) and
# recorded here; tests/test_nodes.py requires this to be empty.
NODE_ERRORS: dict[str, str] = {}


def _on_error(name):
    import traceback
    NODE_ERRORS[name] = traceback.format_exc(limit=3)


for _module in pkgutil.walk_packages(__path__, prefix=f"{__name__}.", onerror=_on_error):
    if _module.name != f"{__name__}.base":
        try:
            importlib.import_module(_module.name)
        except Exception:  # noqa: BLE001 - reported through NODE_ERRORS
            _on_error(_module.name)


def _concrete(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _concrete(sub)


# Names a shader can't use for parameters or ports (WGSL keywords and reserved words).
WGSL_RESERVED = set("""
alias break case const const_assert continue continuing default diagnostic discard else enable
false fn for if let loop override requires return struct switch true var while
NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto await become
cast catch class co_await co_return co_yield coherent column_major common compile compile_fragment
concept const_cast consteval constexpr constinit crate debugger decltype delete demote
demote_to_helper do dynamic_cast enum explicit export extends extern external fallthrough filter
final finally friend from fxgroup get goto groupshared highp impl implements import inline
instanceof interface layout lowp macro macro_rules match mediump meta mod module move mut mutable
namespace new nil noexcept noinline nointerpolation non_coherent noncoherent noperspective null
nullptr of operator package packoffset partition pass patch pixelfragment precise precision
premerge priv protected pub public readonly ref regardless register reinterpret_cast require
resource restrict self set shared sizeof smooth snorm static static_assert static_cast std
subroutine super target template this thread_local throw trait try type typedef typeid typename
typeof union unless unorm unsafe unsized use using varying virtual volatile wgsl where with
writeonly yield
""".split())

NODES: dict[str, Node] = {}
for _cls in _concrete(Node):
    if "type" in vars(_cls) and getattr(_cls, "registered", True):
        if _cls.type in NODES:
            NODE_ERRORS[_cls.__module__] = f"two nodes with type {_cls.type!r}: {type(NODES[_cls.type])} and {_cls}"
            continue
        try:
            _instance = _cls()
        except Exception:  # noqa: BLE001
            _on_error(_cls.__module__)
            continue
        _bad = WGSL_RESERVED & {name for name, _ in _instance.uniform_fields()}  # ports always get a prefix
        if _bad:
            NODE_ERRORS[_cls.__module__] = f"{_cls.type}: {sorted(_bad)} are reserved in WGSL, rename them"
            continue
        NODES[_cls.type] = _instance
