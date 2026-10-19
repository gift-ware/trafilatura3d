from trafilatura3d.nodes.materials._base import B, E, F, I, gray_in, px256
from trafilatura3d.nodes.materials.masks._common import AO, CURV, MASK, NOISE_TYPES, POS, WSN, BlurredMaskNode


class MaskBuilder(BlurredMaskNode):
    """All-in-one mask from bakes, the sum of signed layers then levelled: AO (range, contrast,
    noise), curvature (convex and concave ranges, smoothed by Smoothness, noise), a top/down
    gradient on Position, six world-space directions, then multiplied by a grunge (built-in,
    triplanar or the custom Grunge Inputs) and enriched with scratches and a scattered custom stamp
    (Scatter Input) inside the white areas. Positive layer weights add, negative ones mask away."""

    type = "mask_builder"
    label = "Mask Builder"
    catalog = "mask_builder"
    inputs = {"ambient_occlusion": AO, "curvature": CURV, "world_space_normal": WSN,
              "grunge_input": gray_in("Grunge Input", 0.5), "grunge_input_2": gray_in("Grunge Input 2", 0.5),
              "scatter_input": gray_in("Scatter Input", 0.0), "mask": MASK, "position": POS}
    pre_inputs = ("curvature",)
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.6, "Contrast", 0.0, 1.0),
        "invert": B(False, "Invert"),
        "use_triplanar": B(False, "Use Triplanar"),
        "triplanar_blending_contrast": F(0.5, "Triplanar Blending Contrast", 0.0, 1.0),
        "grunge": F(0.25, "Grunge", 0.0, 1.0),
        "grunge_scale": I(2, "Scale", 0, 10, group="Grunge"),
        "use_custom_grunge": B(False, "Use Custom Grunge", group="Grunge"),
        "secondary_custom_grunge": F(0.0, "Secondary Custom Grunge", 0.0, 1.0, group="Grunge"),
        "grunge_invert": B(False, "Invert", group="Grunge"),
        "ao": F(1.0, "AO", -1.0, 1.0),
        "ao_range": F(0.5, "Range", 0.0, 1.0, group="AO"),
        "ao_contrast": F(0.0, "Contrast", 0.0, 1.0, group="AO"),
        "ao_noisiness": F(0.25, "Noisiness", 0.0, 1.0, group="AO"),
        "ao_noise_scale": I(2, "Noise Scale", 0, 10, group="AO"),
        "ao_noise_type": E("Spots", "Noise Type", NOISE_TYPES, group="AO"),
        "ao_invert": B(False, "Invert", group="AO"),
        "curvature": F(1.0, "Curvature", 0.0, 1.0),
        "convex_range": F(0.5, "Convex Range", -1.0, 1.0, group="Curvature"),
        "convex_contrast": F(0.0, "Convex Contrast", 0.0, 1.0, group="Curvature"),
        "convex_invert": B(False, "Convex Invert", group="Curvature"),
        "concave_range": F(0.0, "Concave Range", -1.0, 1.0, group="Curvature"),
        "concave_contrast": F(0.0, "Concave Contrast", 0.0, 1.0, group="Curvature"),
        "concave_invert": B(False, "Concave Invert", group="Curvature"),
        "smoothness": F(0.0, "Smoothness", 0.0, 16.0, group="Curvature"),
        "level_boost": F(0.25, "Level Boost", 0.0, 1.0, group="Curvature"),
        "curvature_noisiness": F(0.25, "Noisiness", 0.0, 1.0, group="Curvature"),
        "curvature_noise_scale": I(2, "Noise Scale", 0, 10, group="Curvature"),
        "curvature_noise_type": E("White Noise", "Noise Type", NOISE_TYPES, group="Curvature"),
        "top_down_gradient": F(0.0, "Top/Down Gradient", -1.0, 1.0),
        "position_range": F(0.5, "Range", 0.0, 1.0, group="Gradient"),
        "position_contrast": F(0.5, "Contrast", 0.0, 1.0, group="Gradient"),
        "position_invert": B(False, "Invert", group="Gradient"),
        "world_space_normal": F(0.0, "World Space Normal", 0.0, 1.0),
        "top_intensity": F(0.0, "Top Intensity", -1.0, 1.0, group="World Space Normal"),
        "bottom_intensity": F(0.0, "Bottom Intensity", -1.0, 1.0, group="World Space Normal"),
        "front_intensity": F(0.0, "Front Intensity", -1.0, 1.0, group="World Space Normal"),
        "back_intensity": F(0.0, "Back Intensity", -1.0, 1.0, group="World Space Normal"),
        "right_intensity": F(0.0, "Right Intensity", -1.0, 1.0, group="World Space Normal"),
        "left_intensity": F(0.0, "Left Intensity", -1.0, 1.0, group="World Space Normal"),
        "scratches": F(0.0, "Scratches", -1.0, 1.0),
        "amount": I(2000, "Amount", 0, 4096, group="Scratches"),
        "scratches_scale": F(0.5, "Scale", 0.0, 1.0, group="Scratches"),
        "scatter": F(0.0, "Scatter", -1.0, 1.0),
        "scatter_scale": I(20, "Scale", 0, 50, group="Scatter"),
        "density": F(1.0, "Density", 0.0, 1.0, group="Scatter"),
        "scatter_size": F(1.0, "Size", 0.0, 4.0, group="Scatter"),
        "size_variation": F(0.5, "Size Variation", 0.0, 1.0, group="Scatter"),
        "opacity_variation": F(0.0, "Opacity Variation", 0.0, 1.0, group="Scatter"),
    }

    def blur_sigma(self, job):
        return job.values["smoothness"] * px256(job)
