"""Make the Library's thumbnails (trafilatura3d/editor/thumbnails/<node type>.png, 96 px): the nodes on the test
inputs, the filters and generators Paint offers as effects on the shader ball (Qt offscreen). Run it after adding
or changing nodes; the editor ships them so browsing the Library never waits for the GPU.

    python tools/library_thumbnails.py [--only TYPE ...]
"""

import argparse
import os
import sys

from trafilatura3d.thumbnails import OUT, rebuild


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="node types to (re)make (default: all)")
    args = parser.parse_args(argv)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # the shader ball renders in a hidden window
    from PySide6.QtWidgets import QApplication
    from trafilatura3d.editor.paint import shaderball
    app = QApplication.instance() or QApplication([])  # noqa: F841 - Qt needs it alive
    paint = set(shaderball.effect_nodes())
    only = args.only
    made, failed = 0, []
    core = [t for t in only if t not in paint] if only else None
    if core is None or core:  # an empty list would mean every node to the core's rebuild
        made, failed = rebuild(only=core)
    if not only or any(t in paint for t in only):
        n, f = shaderball.rebuild(only=[t for t in only if t in paint] if only else None)
        made, failed = made + n, failed + f
    print(f"{made} thumbnails in {OUT}")
    for f in failed:
        print("failed:", f, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
