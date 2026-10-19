"""Messy fibers 1, 2, 3: straight fibers scattered on a periodic lattice, oriented by Angle. The recipes
are ours, tuned, and read Scale as a 4 x Scale lattice, Angle
random x 2 and Lines number scaled to their own thread counts (`internal`)."""

from ._base import Generator, b, disorder_params, f, f2, i


class _Fibers(Generator):
    libs = ("fibers_lib.wgsl",)
    internal = {"scale": (0.0, 4.0), "angle_random": (0.0, 2.0)}


def _params(last, scale_max=8):
    return {"scale": i(1, "Scale", 1, scale_max), **disorder_params(),
            "angle": f(0.0, "Angle"), "angle_random": f(0.5, "Angle random"),
            **last, "tile_offset": f2((0.0, 0.0), "Tile offset"),
            "non_square_expansion": b(True, "Non-square expansion")}


class MessyFibers1(_Fibers):
    """Long ribbons of crisp fine parallel threads (Lines number / 2), screen-blended: a bright mesh."""
    type = "messy_fibers_bright"
    label = "Messy Fibers Bright"
    catalog = "messy_fibers_1"
    params = _params({"lines_number": f(16.0, "Lines number", 3.0, 32.0)})
    internal = {**_Fibers.internal, "lines_number": (0.0, 0.25)}


class MessyFibers2(_Fibers):
    """Narrow, dimmer streaks of soft threads (Lines number x 9 / 16): blurred bundles with a fine moire."""
    type = "messy_fibers_soft"
    label = "Messy Fibers Soft"
    catalog = "messy_fibers_2"
    params = _params({"lines_number": f(16.0, "Lines number", 3.0, 16.0)})
    internal = {**_Fibers.internal, "lines_number": (0.0, 3 / 16)}


class MessyFibers3(_Fibers):
    """Single thin tubes, each darkened by a random amount (Luminance random, from 0.3 at 0 to 1), stacked
    in random depth order with a dark rim where one crosses another."""
    type = "messy_fibers_thin"
    label = "Messy Fibers Thin"
    catalog = "messy_fibers_3"
    params = _params({"luminance_random": f(0.0, "Luminance random")}, scale_max=16)
    internal = {**_Fibers.internal, "luminance_random": (0.3, 0.7)}
