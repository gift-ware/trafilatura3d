"""Image files through OpenImageIO: PNG, EXR and the other formats it knows, by extension."""

from pathlib import Path

import numpy as np
import OpenImageIO as oiio

_TYPE = {np.dtype(np.uint8): oiio.UINT8, np.dtype(np.uint16): oiio.UINT16,
         np.dtype(np.float16): oiio.HALF, np.dtype(np.float32): oiio.FLOAT}


def write(path: Path, pixels: np.ndarray) -> None:
    """Write a (height, width, channels) array; the pixel type follows the array's dtype."""
    height, width, channels = pixels.shape
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise OSError(f"{path}: {oiio.geterror()}")
    spec = oiio.ImageSpec(width, height, channels, _TYPE[pixels.dtype])
    if not (out.open(str(path), spec) and out.write_image(np.ascontiguousarray(pixels)) and out.close()):
        raise OSError(f"{path}: {out.geterror()}")


def read_config() -> oiio.ImageSpec:
    """How files are opened: colours as stored, not multiplied by their alpha (OpenImageIO associates the alpha of a
    PNG by default; the materials and projects imported expect it straight)."""
    config = oiio.ImageSpec()
    config.attribute("oiio:UnassociatedAlpha", 1)
    return config


def read(path: Path) -> np.ndarray:
    """Read a file as a (height, width, channels) array in its stored pixel type."""
    inp = oiio.ImageInput.open(str(path), read_config())
    if inp is None:
        raise OSError(f"{path}: {oiio.geterror()}")
    try:
        return inp.read_image(inp.spec().format)
    finally:
        inp.close()
