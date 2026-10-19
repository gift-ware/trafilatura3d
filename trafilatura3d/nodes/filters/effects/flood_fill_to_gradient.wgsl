@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = size_flood_fill();
    let q = ff_pixel(c, n);
    let cell = ff_cell(texel_flood_fill(q), q, n);
    if (!cell.ok) { store(c, fx_gray(0.0)); return; }
    var turns = p.angle + ff_random(cell, 5.0) * p.angle_variation;
    if (ANGLE_INPUT_CONNECTED) { turns += sample_angle_input(cell.centre).r * p.angle_image_input_multiplier; }
    let dir = fx_dir(turns);
    // Projection on the direction, scaled so the box's corners land on 0 and 1.
    let rel = (cell.local - 0.5) * cell.size;
    let reach = 0.5 * (abs(dir.x) * cell.size.x + abs(dir.y) * cell.size.y);
    let g = clamp(0.5 + 0.5 * dot(rel, dir) / max(reach, 1e-6), 0.0, 1.0);
    let slope = mix(1.0, sample_slope_input(cell.centre).r, p.slope_image_input_multiplier);
    var v = mix(p.flat_slope_color, g, clamp(slope, 0.0, 1.0));
    v *= mix(1.0, max(cell.size.x, cell.size.y), p.multiply_by_bounding_box_size);
    v *= mix(1.0, slope, p.multiply_by_slope_intensity);
    store(c, fx_gray(v));
}
