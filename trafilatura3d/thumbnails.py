"""The Library's thumbnails (trafilatura3d/editor/thumbnails/<node type>.png, 96 px): the generators cooked (on the
standard test inputs if they take any), but the Paint workspace's own filters and generators, which the editor
draws on the shader ball (trafilatura3d/editor/paint/shaderball.py). Preferences >
Library > Rebuild thumbnails and tools/library_thumbnails.py make both. Other filters show their initials (`cooked`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "editor" / "thumbnails"
SIZE = 96


def cooked(node_type: str) -> bool:
    """Whether the Library and the Paint Assets show the node's image: generators cooked (their own output says what
    they make) and the Paint nodes on the shader ball. A filter cooked on the test pattern says nothing of what it
    does: it shows its initials (editor) instead."""
    from trafilatura3d.nodes import NODES
    d = NODES[node_type]
    return not d.inputs or d.category.startswith(("Texture generators", "Paint/"))


def rebuild(ev=None, only: list[str] | None = None, out: Path = OUT) -> tuple[int, list[str]]:
    """(thumbnails made, failures "type: error")."""
    import OpenImageIO as oiio
    from trafilatura3d.evaluator import Evaluator
    from trafilatura3d.nodes import NODES
    from trafilatura3d.sample import sample_job
    ev = ev or Evaluator()
    out.mkdir(parents=True, exist_ok=True)
    types = [t for t in (only or sorted(t for t, d in NODES.items() if d.category
                                        and not getattr(d, "passthrough", False)))
             if cooked(t) and not NODES[t].category.startswith("Paint/")]  # Paint's: shaderball.py
    failed = []
    for t in types:
        try:
            job = sample_job(t, False, 7)
            img = ev.thumbnails([(job, next(iter(job.node.outputs)))], SIZE)[0]
        except Exception as e:  # noqa: BLE001 - report and go on
            failed.append(f"{t}: {e}")
            continue
        h, w = img.shape[:2]
        buf = oiio.ImageBuf(oiio.ImageSpec(w, h, 4, oiio.UINT8))
        buf.set_pixels(oiio.ROI(), np.ascontiguousarray(img))
        buf.write(str(out / f"{t}.png"))
    return len(types) - len(failed), failed
