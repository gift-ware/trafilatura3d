// A variant: standardized fractal sum (`fbm_z`) from the Scale grid down, skewed (`skew_z`) and
// mapped to the variant's mean and contrast. The dirts go one octave further, down to the pixel,
// with a roughness above 1 (the grain dominates) and a mean low enough that the clamp leaves
// sparse grains; Dirt 1 and 2 add a coarse term gathering the grains in patches.
// Scale (BnW spots, dirts) is the number of repetitions of the whole pattern in the tile:
// the octaves are those of the default Scale (down to the pixel or ~2 px), all shrunk
// together, the finest dropped once they would fall below the pixel. Below the default Scale the
// grain grows with the pattern instead of staying at the pixel.
struct Recipe {
    kind: i32,       // octave kind of `fbm`
    rough: f32,      // amplitude ratio between octaves
    grain: i32,      // extra octaves (1: down to the pixel)
    patches: f32,    // weight of the coarse density term
    skew: f32,       // `skew_z`
    mean: f32,
    contrast: f32,
    whole: bool,     // Scale repeats the whole pattern: the octave count of DEFAULT_SCALE (false: down to
                     // ~2 px at any Scale)
    sharp: f32,      // lattice offset from the pixel centres, in pixels: 0 the pixel octave is a 2x2 blur,
                     // 0.5 one raw grain per pixel (and a grain two pixels across visibly softer)
};

fn recipe() -> Recipe {
    switch VARIANT {
        case 5: { return Recipe(0, 2.4, 1, 0.3, -0.4, 0.14, 0.21, true, 0.25); }   // Dirt 1
        case 6: { return Recipe(0, 2.4, 1, 0.3, -0.3, -1.035, 0.496, true, 0.5); }   // Dirt 2
        case 7: { return Recipe(0, 2.4, 1, 0.0, -0.4, 0.132, 0.138, true, 0.25); }   // Dirt 3
        case 8: { return Recipe(1, 1.2, 1, 0.0, -0.05, 0.219, 0.247, true, 0.0); }  // Dirt 4
        default: { return Recipe(0, 1.8, 1, 0.0, -0.25, 0.079, 0.167, true, 0.25); } // Dirt 5
    }
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let r = recipe();
    let asp = aspect(p.non_square_expansion);
    // Tile units: fbm multiplies by the periods, which already carry the aspect.
    // The lattice shifted by `sharp` pixels (see Recipe).
    let x = ((uv_of(c) - r.sharp / p.size_) * asp + p.tile_offset) / asp;
    let sc = f32(max(p.scale, 1));
    let per0 = vec2i(round(sc * asp));
    let dis = p.disorder * p.disorder_speed;
    let an = p.disorder_anisotropy;
    let ang = p.disorder_anisotropy_angle;
    let s = p.seed_ + f32(VARIANT) * 101.0;
    // Octaves counted on the short side (auto_octaves counts on the long one: one octave too many
    // on non-square images, below the pixel with the dirts' extra octave).
    var oct = auto_octaves(sc * max(asp.x, asp.y), 12);
    if (r.whole) { oct = min(oct, auto_octaves(DEFAULT_SCALE * max(asp.x, asp.y), 12)); }
    var u = fbm_z(x, per0, oct + r.grain, r.rough, s, r.kind, dis, an, ang);
    if (r.patches > 0.0) {
        let coarse = fbm_z(x, per0, oct, 0.5, s + 7.0, 1, dis, an, ang);
        u = (u + r.patches * coarse) / sqrt(1.0 + r.patches * r.patches);
    }
    let v = clamp(r.mean + r.contrast * skew_z(u, r.skew), 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
