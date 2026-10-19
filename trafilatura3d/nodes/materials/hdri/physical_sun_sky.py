from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import COLOR, C, F, F2
from trafilatura3d.nodes.materials.hdri._common import HdrGenerator


class PhysicalSunSky(HdrGenerator):
    """Analytic daylight panorama: sky radiance from the Preetham model (turbidity driven, the
    published analytic fit), a sun disc tinted by atmospheric extinction, a ground plane lit by sun and sky, and
    ground albedo brightening the horizon. Linear HDR, the zenith of a clear day near 1 at 0 EV."""

    type = "physical_sun_sky"
    label = "Physical Sun/Sky"
    catalog = "physical_sun_sky"
    outputs = {"output": Output(COLOR)}
    params = {
        "sun_position": F2([0.5, 0.25], "Sun Position"),  # longitude, latitude (0 top, 0.5 horizon)
        "turbidity": F(5.0, "Turbidity", 1.0, 10.0),
        "albedo": F(0.1, "Albedo", 0.0, 1.0),
        "ground_color": C([0.1, 0.1, 0.1], "Ground Color"),
        "exposure": F(0.0, "Exposure (EV)", -1.0, 4.0),
        "sun_size": F(1.0, "Sun Size", 0.0, 4.0),
        "sun_intensity": F(1.0, "Sun Intensity", 0.0, 1.0),
        "sky_intensity": F(1.0, "Sky Intensity", 0.0, 1.0),
    }
