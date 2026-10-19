"""Common Python of the shape effects: which channel holds the shape, jump flooding on it."""

from ._fx import nearest_seeds, source


def shape_seeds(ctx, job, port="input"):
    """Seeds of the shape (grayscale: the value; color: the alpha), cut at 0.5."""
    src = source(ctx, job, port)
    return nearest_seeds(ctx, job, src, 0.5, channel=0 if src[1] else 1)
