"""Render every node of a category on the standard test inputs, one tile per node and output (and
per grayscale/color variant), into a labelled contact sheet.

    python tools/node_sheet.py "Atomic nodes" sheet.png [--size 256]
    python tools/node_sheet.py all sheet.png
"""

import argparse
import sys

import numpy as np
import OpenImageIO as oiio

from trafilatura3d.evaluator import Evaluator
from trafilatura3d.nodes import NODES
from trafilatura3d.sample import has_any_input, sample_job

TILE_LABEL = 22


def tile(ev, job, out, size):
    a = ev.read(job, out).astype(np.float32)
    if a.dtype != np.float32 or ev.read(job, out).dtype == np.uint8:
        a = ev.read(job, out).astype(np.float32) / (255.0 if ev.read(job, out).dtype == np.uint8 else 1.0)
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    a = np.clip(a[:, :, :3], 0, 1)
    if a.shape[0] != size:
        buf = oiio.ImageBuf(oiio.ImageSpec(a.shape[1], a.shape[0], 3, oiio.FLOAT))
        buf.set_pixels(oiio.ROI(), np.ascontiguousarray(a))
        a = oiio.ImageBufAlgo.resize(buf, roi=oiio.ROI(0, size, 0, size, 0, 1, 0, 3)).get_pixels(oiio.FLOAT)
    return a


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("category")
    parser.add_argument("out")
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--cols", type=int, default=6)
    args = parser.parse_args(argv)
    types = sorted(t for t, d in NODES.items() if d.category and (args.category == "all" or d.category == args.category))
    if not types:
        print(f"no nodes in {args.category!r}", file=sys.stderr)
        return 1
    ev = Evaluator()
    log2 = max(6, int(np.log2(args.size)))
    tiles = []
    for t in types:
        for color in ([False, True] if has_any_input(t) else [False]):
            job = sample_job(t, color, log2)
            for out in job.node.outputs:
                name = NODES[t].label + (f" [{out}]" if len(job.node.outputs) > 1 else "") + (" (color)" if color else "")
                try:
                    tiles.append((name, tile(ev, job, out, args.size)))
                except Exception as e:  # show the failure instead of stopping the sheet
                    print(f"{t}: {e}", file=sys.stderr)
                    tiles.append((name + " FAILED", np.full((args.size, args.size, 3), [0.6, 0, 0], np.float32)))
    cols = min(args.cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    th = args.size + TILE_LABEL
    sheet = oiio.ImageBuf(oiio.ImageSpec(cols * args.size, rows * th, 3, oiio.FLOAT))
    oiio.ImageBufAlgo.fill(sheet, (0.12, 0.12, 0.12))
    for i, (name, img) in enumerate(tiles):
        x, y = (i % cols) * args.size, (i // cols) * th
        buf = oiio.ImageBuf(oiio.ImageSpec(args.size, args.size, 3, oiio.FLOAT))
        buf.set_pixels(oiio.ROI(), np.ascontiguousarray(img))
        oiio.ImageBufAlgo.paste(sheet, x, y + TILE_LABEL, 0, 0, buf)
        oiio.ImageBufAlgo.render_text(sheet, x + 4, y + 16, name[:30], 13, "", (0.85, 0.85, 0.85))
    sheet.write(args.out, "uint8")
    print(f"{len(tiles)} tiles -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
