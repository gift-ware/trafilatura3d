"""Mesh based generators/Mask Generators: grayscale masks from baked mesh maps.

Conventions: curvature 0.5 = flat, brighter convex, darker concave; ambient occlusion 1 = open;
world space normals encoded 0..1 with Y up; position maps normalized 0..1 with Y (green) up, or a
grayscale height. Level reveals the effect (histogram scan), Contrast sharpens it; the optional
Mask multiplies the result."""
