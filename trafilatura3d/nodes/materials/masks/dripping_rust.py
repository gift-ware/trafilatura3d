from trafilatura3d.nodes.materials._base import F, I, gray_in, px256
from trafilatura3d.nodes.materials.masks._common import AO, CURV, MASK, BlurredMaskNode


class DrippingRust(BlurredMaskNode):
    """Rust flakes and specks placed by curvature (edges) and AO (crevices), smeared by Spreading
    Smoothness, with drips running down from them: downhill on the baked (grayscale) Position,
    traced over Drips Samples Amount steps, their length set by Drips Intensity."""

    type = "dripping_rust"
    label = "Dripping Rust"
    catalog = "dripping_rust"
    inputs = {"curvature": CURV, "ambient_occlusion": AO, "position": gray_in("Position", 0.5), "mask": MASK}
    pre_inputs = ("curvature", "ambient_occlusion")
    params = {
        "rust_spreading": F(0.5, "Rust Spreading", 0.0, 1.0),
        "rust_contrast": F(0.5, "Rust Contrast", 0.0, 1.0),
        "spreading_smoothness": F(0.0, "Spreading Smoothness", 0.0, 1.0),
        "drips_intensity": F(0.1, "Drips Intensity", 0.0, 1.0),
        "drips_smoothness": F(0.05, "Drips Smoothness", 0.0, 1.0),
        "drips_samples_amount": I(32, "Drips Samples Amount", 0, 32),
    }

    def blur_sigma(self, job):
        return job.values["spreading_smoothness"] * 6.0 * px256(job)
