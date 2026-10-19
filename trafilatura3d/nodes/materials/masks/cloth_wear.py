from trafilatura3d.nodes.materials._base import F, gray_in, px256
from trafilatura3d.nodes.materials.masks._common import CURV, MASK, BlurredMaskNode


class ClothWear(BlurredMaskNode):
    """Frazzled cloth edges: raised edges of the curvature (sharpened by Hard Edges Amount),
    softened by Wear Softness, worn first where the tiling Cloth Height pattern is high (the
    threads), so the cloth detail map drives most of the look."""

    type = "cloth_wear"
    label = "Cloth Wear"
    catalog = "cloth_wear"
    inputs = {"cloth_height": gray_in("Cloth Height", 0.5), "mask": MASK, "curvature": CURV}
    pre_inputs = ("curvature",)
    params = {
        "hard_edges_amount": F(0.5, "Hard Edges Amount", 0.0, 1.0),
        "wear_softness": F(1.0, "Wear Softness", 0.0, 5.0),
    }

    def blur_sigma(self, job):
        return job.values["wear_softness"] * 2.0 * px256(job)
