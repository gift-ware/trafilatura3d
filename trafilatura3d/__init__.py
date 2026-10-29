"""Trafilatura 3D: node-based procedural texture graphs cooked on the GPU."""

__version__ = "0.3.0"  # the release (CHANGELOG.md); pyproject.toml reads it

from trafilatura3d.compiler import compile_graph
from trafilatura3d.evaluator import Evaluator
from trafilatura3d.export import export
from trafilatura3d.package import DocumentError, Package

__all__ = ["DocumentError", "Evaluator", "Package", "compile_graph", "export"]
