"""The named grunge maps (Concrete, Damas, Galvanic, Leaks, Rust, Scratches, Spots...): layered
procedural textures with an Advanced group of layer controls. The recipes are ours, tuned, and read the Advanced values
through `internal` (the parameter's default -> the tuned value)."""

from ._base import Grunge, b, f, grunge_params, i


class GrungeConcrete(Grunge):
    """Concrete height: mid-gray base with large variations and noisiness, pores and dirt specks,
    faint scratches."""
    type = "grunge_concrete"
    label = "Grunge Concrete"
    catalog = "grunge_concrete"
    sharpen = "sharpen"
    params = grunge_params({
        "base_noisiness": f(1.0, "Base Noisiness"), "dirt_specks_opacity": f(0.5, "Dirt Specks Opacity"),
        "invert_dirt": b(False, "Invert Dirt"), "scratches_opacity": f(0.215, "Scratches Opacity"),
        "sharpen": f(0.5, "Sharpen"), "large_variation_intensity": f(0.5, "Large Variation Intensity")})
    internal = {"base_noisiness": (0.0, 0.5), "dirt_specks_opacity": (0.0, 1.2),
                "scratches_opacity": (0.0, 0.3 / 0.215)}


class GrungeDamas(Grunge):
    """Damascus steel: Divisions gradient bands bent by Waves (each wave randomly turned), Distortion
    and fine Details."""
    type = "grunge_damas"
    label = "Grunge Damas"
    catalog = "grunge_damas"
    params = grunge_params({
        "distortion": f(1.0, "Distortion"), "divisions": i(16, "Divisions", 0, 32), "waves": i(1, "Waves", 1, 4),
        "details": f(0.5, "Details"), "rotation_random": f(0.05, "Rotation Random")})
    internal = {"distortion": (0.0, 0.4), "divisions": (0.0, 0.75), "waves": (0.0, 4.0), "details": (0.0, 0.6),
                "rotation_random": (0.0, 6.0)}


class GrungeGalvanicLarge(Grunge):
    """Galvanized spangles: large warped crystals of flat random grays with dendritic streaks, bright
    ridges on top."""
    type = "grunge_galvanic_large"
    label = "Grunge Galvanic Large"
    catalog = "grunge_galvanic_large"
    sharpen = "sharpen_intensity"
    params = grunge_params({
        "warp_intensity": f(64.0, "Warp Intensity", 32.0, 128.0), "ridge_detail_opacity": f(1.0, "Ridge Detail Opacity"),
        "sharpen_intensity": f(0.8, "Sharpen Intensity")})
    internal = {"warp_intensity": (0.0, 0.4 / 64), "ridge_detail_opacity": (0.0, 0.4),
                "sharpen_intensity": (0.0, 0.5 / 0.8)}


class GrungeGalvanicSmall(Grunge):
    """Dense galvanized spangles: small crystals (Crispness), high-frequency warp, dirt overlay."""
    type = "grunge_galvanic_small"
    label = "Grunge Galvanic Small"
    catalog = "grunge_galvanic_small"
    params = grunge_params({
        "crispness": f(0.5, "Crispness"), "dirt": f(0.5, "Dirt"), "micro_distortion": f(0.5, "Micro Distortion")})
    internal = {"crispness": (0.0, 1.2), "dirt": (0.0, 0.8), "micro_distortion": (0.0, 0.6)}


class GrungeLeaks(Grunge):
    """Drips running down a greasy blotchy surface; Shape Contrast moves them from dark to bright."""
    type = "grunge_leaks"
    label = "Grunge Leaks"
    catalog = "grunge_leaks"
    sharpen = "sharpen_intensity"
    params = grunge_params({
        "drip_length": f(64.0, "Drip Length", 0.0, 128.0), "shape_contrast": f(0.5, "Shape Contrast"),
        "drip_crispness": f(0.5, "Drip Crispness"), "sharpen_intensity": f(1.0, "Sharpen Intensity", 0.0, 2.0)})
    internal = {"drip_length": (0.0, 0.6 / 64), "shape_contrast": (0.0, 0.6), "sharpen_intensity": (0.0, 0.5)}


class GrungeLeakyPaint(Grunge):
    """Paint drips from leaks, on a lattice of Leak Scale columns, slanted up to Leak Angle Random."""
    type = "grunge_leaky_paint"
    label = "Grunge Leaky Paint"
    catalog = "grunge_leaky_paint"
    params = grunge_params({
        "leak_intensity": f(0.5, "Leak Intensity"), "leak_scale": i(1, "Leak Scale", 1, 8),
        "leak_angle_random": f(0.0, "Leak Angle Random"), "leak_crispness": f(0.9, "Leak Crispness")})
    internal = {"leak_scale": (0.0, 8.0), "leak_angle_random": (0.05, 0.2), "leak_crispness": (0.0, 0.6 / 0.9)}


class GrungeRoughDirty(Grunge):
    """Rough dirty surface: warped blotches, fine parallel scratches, grain."""
    type = "grunge_rough_dirty"
    label = "Grunge Rough Dirty"
    catalog = "grunge_rough_dirty"
    params = grunge_params({
        "main_grunge_intensity": f(0.5, "Main Grunge Intensity"), "invert_scratches": b(True, "Invert Scratches"),
        "scratches_intensity": f(0.2, "Scratches Intensity"), "grain_intensity": f(1.0, "Grain Intensity")})
    internal = {"main_grunge_intensity": (0.0, 1.4), "invert_scratches": (1.0, -1.0),
                "scratches_intensity": (0.0, 2.5), "grain_intensity": (0.0, 0.3)}


class GrungeRustFine(Grunge):
    """Fine rust: warped contrasted grunge base, brighter vertical streaks and spots, noise. Base Warp Intensity:
    the slider spans the default."""
    type = "grunge_rust_fine"
    label = "Grunge Rust Fine"
    catalog = "grunge_rust_fine"
    sharpen = "sharpen_intensity"
    params = grunge_params({
        "base_grunge_contrast": f(0.0, "Base Grunge Contrast", -1.0, 1.0),
        "base_warp_intensity": f(128.0, "Base Warp Intensity", 0.0, 256.0),
        "streaks_intensity": f(0.5, "Streaks Intensity"), "noise_intensity": f(0.5, "Noise Intensity"),
        "sharpen_intensity": f(0.5, "Sharpen Intensity", 0.0, 3.0)})
    internal = {"base_grunge_contrast": (0.5, 0.5), "base_warp_intensity": (0.0, 0.5 / 128),
                "streaks_intensity": (0.0, 0.8), "noise_intensity": (0.0, 0.8)}


class GrungeScratchesDirty(Grunge):
    """Finely scratched dirty surface: dirt blotches and scratches in every direction."""
    type = "grunge_scratches_dirty"
    label = "Grunge Scratches Dirty"
    catalog = "grunge_scratches_dirty"
    params = grunge_params({
        "base_grunge_intensity": f(0.25, "Base Grunge Intensity"), "scratches_intensity": f(0.5, "Scratches Intensity")})
    internal = {"base_grunge_intensity": (0.0, 2.4), "scratches_intensity": (0.0, 1.4)}


class GrungeScratchesFine(Grunge):
    """Clean dark surface with fine bright scratches of varied brightness."""
    type = "grunge_scratches_fine"
    label = "Grunge Scratches Fine"
    catalog = "grunge_scratches_fine"
    sharpen = "sharpen_intensity"
    params = grunge_params({
        "scratches_amount": f(0.5, "Scratches Amount"), "sharpen_intensity": f(0.25, "Sharpen Intensity", 0.0, 3.0),
        "scratch_value_bias": f(0.5, "Scratch Value Bias")})
    internal = {"sharpen_intensity": (0.0, 1.2)}


class GrungeScratchesRough(Grunge):
    """Rough scratches (width, length, blur, masking, dirtiness, doubled), damaged spots and dust."""
    type = "grunge_scratches_rough"
    label = "Grunge Scratches Rough"
    catalog = "grunge_scratches_rough"
    sharpen = "sharpen_intensity"
    params = grunge_params({
        "scratch_quantity": f(1.0, "Scratch Quantity"), "scratch_tiling": i(1, "Scratch Tiling", 1, 6),
        "scratch_blur": f(0.0, "Scratch Blur"), "scratch_width": f(0.5, "Scratch Width"),
        "scratch_length": f(1.0, "Scratch Length"), "scratch_masking": f(0.25, "Scratch Masking"),
        "scratch_dirtiness": f(0.5, "Scratch Dirtiness"), "double_scratch": f(0.5, "Double Scratch"),
        "scratch_spots_intensity": f(1.0, "Scratch Spots Intensity"),
        "scratch_spots_tiling": i(1, "Scratch Spots Tiling", 1, 6),
        "dust_intensity": f(0.5, "Dust Intensity"), "dust_tiling": i(1, "Dust Tiling", 1, 16),
        "sharpen_intensity": f(0.25, "Sharpen Intensity", 0.0, 3.0)})
    internal = {"scratch_quantity": (0.0, 0.5), "scratch_tiling": (0.0, 4.0), "scratch_blur": (0.2, 0.8),
                "scratch_width": (0.0, 0.6), "scratch_length": (0.0, 0.6), "scratch_masking": (0.0, 1.6),
                "scratch_dirtiness": (0.0, 0.8), "scratch_spots_intensity": (0.0, 0.4),
                "scratch_spots_tiling": (0.0, 8.0), "dust_intensity": (0.0, 0.8), "dust_tiling": (0.0, 8.0),
                "sharpen_intensity": (0.0, 1.2)}


class GrungeShavings(Grunge):
    """Curled shavings gathered in scratched spots, over dust."""
    type = "grunge_shavings"
    label = "Grunge Shavings"
    catalog = "grunge_shavings"
    sharpen = "sharpen_intensity"
    params = grunge_params({
        "scratch_spots_amount": f(0.5, "Scratch Spots Amount"),
        "scratch_spots_tiling": i(1, "Scratch Spots Tiling", 1, 6),
        "dust_intensity": f(0.5, "Dust Intensity"), "sharpen_intensity": f(0.25, "Sharpen Intensity")})
    internal = {"scratch_spots_tiling": (0.0, 4.0), "dust_intensity": (0.0, 0.8), "sharpen_intensity": (0.0, 1.2)}


class GrungeSplashesDusty(Grunge):
    """Liquid splashes (warped blobs with satellite droplets) on spreading dust."""
    type = "grunge_splashes_dusty"
    label = "Grunge Splashes Dusty"
    catalog = "grunge_splashes_dusty"
    params = grunge_params({
        "splashes_quantity": f(1.0, "Splashes Quantity"), "splashes_distortion": f(0.75, "Splashes Distortion"),
        "splash_dirt_ratio": f(0.5, "Splash/Dirt Ratio"), "dirt_spread": f(0.5, "Dirt Spread")})
    internal = {"splashes_quantity": (0.0, 0.5), "splashes_distortion": (0.0, 0.5 / 0.75),
                "splash_dirt_ratio": (0.0, 1.2)}


class GrungeSpots(Grunge):
    """Fine splattered spots on several scales, broken up by Details, gathered by a coverage mask."""
    type = "grunge_spots"
    label = "Grunge Spots"
    catalog = "grunge_spots"
    params = grunge_params({
        "details": f(0.0, "Details"), "coverage": f(0.65, "Coverage"), "coverage_contrast": f(1.0, "Coverage Contrast")})
    internal = {"details": (0.5, 0.5), "coverage": (0.0, 0.6 / 0.65), "coverage_contrast": (0.0, 0.5)}


class GrungeSpotsDirty(Grunge):
    """Soft spots of dirt with ragged edges, finer with Scale."""
    type = "grunge_spots_dirty"
    label = "Grunge Spots Dirty"
    catalog = "grunge_spots_dirty"
    params = grunge_params({"coverage": f(0.5, "Coverage"), "scale": i(1, "Scale", 1, 8)})
    internal = {"scale": (0.0, 4.0)}
