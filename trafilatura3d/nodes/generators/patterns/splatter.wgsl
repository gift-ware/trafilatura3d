// Splatter (ours): one Input image per cell of a 2^Grid Number grid, the image's tile repeated in every direction.
// Sizes are in percent of a cell, Disorder in thousandths of the image, Zoom magnifies the image about its centre,
// Output Level Min/Max are the input levels of the result. Shared by the passes (splatter_draw.wgsl,
// splatter_layer.wgsl); each sets p.gray_, whether the node's output is grayscale.
fn sp_n() -> i32 { return 1 << u32(clamp(p.grid_number, 0, 8)); }
fn sp_gray() -> bool { return p.gray_ == 1; }

// Size Variation's spread: a uniform +-SP_SIZE_SPREAD x Size Variation percent (fitted on the output's spread of sizes).
const SP_SIZE_SPREAD = 0.71;
// Zoom: magnification 1 at Zoom 100 growing to SP_ZOOM_MAX at Zoom 10000, as 1 / (1 - k t), t = (Zoom - 100) / 9900
// (fitted on the output).
const SP_ZOOM_MAX = 10.0;

fn sp_magnify() -> f32 {
    let t = clamp((p.zoom - 100.0) / 9900.0, 0.0, 1.0);
    return 1.0 / (1.0 - (1.0 - 1.0 / SP_ZOOM_MAX) * t);
}
// The pattern's uv at output pixel c (Zoom about the centre, then Pan).
fn sp_uv(c: vec2i) -> vec2f {
    return 0.5 + ((vec2f(c) + 0.5) / p.size_ - 0.5) / sp_magnify() - vec2f(p.pan_x, p.pan_y) / 100.0;
}

// The tile of a cell: its centre's offset from the cell's (`center`), its local frame, its tint.
fn sp_tile(cell: vec2i) -> Tile {
    var t: Tile;
    let n = f32(sp_n());
    let sv = p.size_variation * SP_SIZE_SPREAD * (2.0 * rnd(cell, 3u) - 1.0);
    let half = max((vec2f(p.pattern_size_width, p.pattern_size_height) + sv) / 100.0 * 0.5 / n, vec2f(1e-6));
    // Rotation (degrees, clockwise as the FX-Map) plus up to Rotation Variation more.
    let a = (p.rotation + p.rotation_variation * rnd(cell, 23u)) / 360.0;
    t.inv = mat2x2f(1.0 / half.x, 0.0, 0.0, 1.0 / half.y) * rot(-a);
    // Disorder: up to Disorder/1000 of the image along Disorder Angle; Disorder Random picks the angle in
    // [0, Disorder Angle).
    var ang = radians(p.disorder_angle);
    if (p.disorder_random == 1) { ang *= rnd(cell, 42u); }
    t.center = p.disorder / 1000.0 * rnd(cell, 41u) * vec2f(cos(ang), sin(ang));
    var tint = vec3f(1.0 - p.luminance_variation * rnd(cell, 51u));
    if (!sp_gray()) {
        tint = mix(vec3f(1.0), vec3f(rnd(cell, 52u), rnd(cell, 53u), rnd(cell, 54u)), p.color_variation);
    }
    t.color = vec4f(tint * p.gain, 1.0);
    return t;
}
