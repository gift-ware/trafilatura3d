"""The rough fractal noises: Dirt 1-5 and Dirt gradient (Clouds 1-3 and BnW spots 1-3 are made of patterns:
stamped.py; the dirts stay here: their look is the pixel grain cut on black, the same either way).

All are sums of tileable lattice octaves from the Scale grid down to ~2 pixels (the dirts down to the
pixel: the rough noises gain detail with the resolution); each octave layer is a Disorder
"ingredient", shifted on its own direction. The sum is standardized (`fbm_z`), so the contrast does
not depend on the resolution, then skewed and mapped to the variant's mean and contrast. The
variants differ in the octave kind (value, gradient), the roughness and that tone, all tuned at the
defaults (mean, spread, skew and energy per octave)."""

from .common import CLASSIC, DISORDER, ANISO, NSE, NoiseNode, scale


class Rough(NoiseNode):
    shader = "rough"
    params = {**scale(1, 1, 4), **CLASSIC}
    scale_cells = 4


class Dirt1(Rough):
    """Bright pixel grains on black, denser in soft patches."""
    type = "dirt_patchy"
    label = "Dirt Patchy"
    catalog = "dirt_1"
    variant = 5


class Dirt2(Rough):
    """Sparse bright specks on black, gathered in patches."""
    type = "dirt_sparse"
    label = "Dirt Sparse"
    catalog = "dirt_2"
    variant = 6


class Dirt3(Rough):
    """Uniform fine grain on black, slightly brighter than Dirt 5."""
    type = "dirt_fine"
    label = "Dirt Fine"
    catalog = "dirt_3"
    variant = 7


class Dirt4(Rough):
    """Dense coarse grain (gradient octaves down to the pixel, roughness just above 1)."""
    type = "dirt_coarse"
    label = "Dirt Coarse"
    catalog = "dirt_4"
    variant = 8


class Dirt5(Rough):
    """Uniform fine grain on black, a little clumpier than Dirt 3."""
    type = "dirt_clumpy"
    label = "Dirt Clumpy"
    catalog = "dirt_5"
    variant = 9


class DirtGradient(NoiseNode):
    """Grainy clouds fading along a vertical falloff: almost black at the top, bright at the bottom.
    The falloff is a gradient across the image, so the result does not tile vertically
    (seamless = False)."""
    type = "dirt_gradient"
    label = "Dirt gradient"
    catalog = "dirt_gradient"
    shader = "dirt_gradient"
    seamless = False
    params = {**DISORDER, **ANISO, **NSE}
