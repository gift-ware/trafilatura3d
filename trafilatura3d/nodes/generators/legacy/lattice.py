"""Lattice noises (ours) shared by the legacy nodes: one site per cell of an n x n lattice (lattice_sites.wgsl) and,
on the sites, the Max of paraboloids (domes.wgsl: Cells 1) or the sum of gaussian bells (blobs.wgsl: Perlin noise
zoom)."""

from __future__ import annotations

from pathlib import Path

from trafilatura3d.nodes.base import Input, Node, Output, ParamKind, PortType

from ._lib import target


class _Pass(Node):
    """A lattice pass (its shader `<type>.wgsl` in this folder)."""
    registered = False
    outputs = {"output": Output(PortType.GRAY)}

    def wgsl(self) -> str:
        return (Path(__file__).parent / f"{self.type}.wgsl").read_text(encoding="utf-8")


class _Sites(_Pass):
    type = "lattice_sites"
    outputs = {"output": Output(PortType.COLOR)}
    internal_uniforms = {"spread": ParamKind.FLOAT, "edge": ParamKind.FLOAT, "group": ParamKind.FLOAT,
                         "patchiness": ParamKind.FLOAT, "disorder": ParamKind.FLOAT}


class _Domes(_Pass):
    type = "domes"
    inputs = {"sites": Input(PortType.COLOR)}
    internal_uniforms = {"radius": ParamKind.FLOAT}


class _Blobs(_Pass):
    type = "blobs"
    inputs = {"sites": Input(PortType.COLOR)}
    internal_uniforms = {"sigma": ParamKind.FLOAT, "gain": ParamKind.FLOAT, "lift": ParamKind.FLOAT,
                         "contrast": ParamKind.FLOAT, "random_height": ParamKind.INT, "flip": ParamKind.INT,
                         "reach": ParamKind.INT}


SITES, DOMES, BLOBS = _Sites(), _Domes(), _Blobs()


def sites(ctx, job, n: int, spread: float, edge: float, disorder: float, group: float = 0.0, patch: float = 0.0):
    """The n x n lattice's sites (offsets from the cells' centres, in cells; `group`: a shared offset per 2 x 2 cells)
    and their random heights (`patch`: modulated by a smooth field over n / 8 cells), at the job's seed."""
    out = ctx.scratch(job, False, "32f", size=(n, n))
    ctx.dispatch(SITES, job, {"spread": spread, "edge": edge, "group": group, "patchiness": patch,
                              "disorder": disorder},
                 dsts={"output": out}, size=(n, n))
    return out


def domes(ctx, job, n: int, spread: float, edge: float, radius: float, disorder: float, dst=None):
    """Max of paraboloids `radius` cells wide on the lattice's sites, into `target(dst)`; returns the image."""
    out = target(ctx, job, dst)
    ctx.dispatch(DOMES, job, {"radius": radius}, srcs={"sites": (sites(ctx, job, n, spread, edge, disorder), False)},
                 dsts={"output": out})
    return out, True


def blobs(ctx, job, n: int, spread: float, edge: float, sigma: float, gain: float, lift: float, disorder: float,
          contrast: float = 1.0, flip: bool = False, group: float = 0.0, random_height: bool = False,
          patch: float = 0.0, dst=None):
    """lift + gain x the sum of gaussian bells (sigma in cells; heights 1 or random) on the lattice's sites, its
    contrast around the mean times `contrast`; returns the image. The search covers 5 x 5 cells when the sites stay
    within 2.5 cells with their bells (to 3 sigma), else 7 x 7."""
    out = target(ctx, job, dst)
    reach = 2 if spread + group + 3.0 * sigma <= 2.5 else 3
    ctx.dispatch(BLOBS, job, {"sigma": sigma, "gain": gain, "lift": lift, "contrast": contrast,
                              "random_height": int(random_height), "flip": int(flip), "reach": reach},
                 srcs={"sites": (sites(ctx, job, n, spread, edge, disorder, group, patch), False)}, dsts={"output": out})
    return out, True
