// A channel pushed towards white by a > 0 (as Screen with a) or towards black by -a (as Multiply with 1 + a).
fn push(v: f32, a: f32) -> f32 { return select(mix(v, 0.0, -a), mix(v, 1.0, a), a >= 0.0); }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = size_flood_fill();
    let q = ff_pixel(c, n);
    let cell = ff_cell(texel_flood_fill(q), q, n);
    if (!cell.ok) { store(c, vec4f(0.0, 0.0, 0.0, 1.0)); return; }
    let m = size_grayscale_color_input();
    var base = sample_grayscale_color_input(cell.centre);
    if (p.filtering_mode == 0) { base = texel_grayscale_color_input(min(vec2i(cell.centre * vec2f(m)), m - 1)); }
    if (OUT_OUTPUT_GRAY) {
        let a = clamp(p.luminance_color_adjustment + ff_random(cell, 7.0) * p.luminance_color_random, -1.0, 1.0);
        store(c, vec4f(vec3f(push(base.r, a)), base.a));
        return;
    }
    let r = vec3f(ff_random(cell, 7.0), ff_random(cell, 8.0), ff_random(cell, 9.0));
    let a = clamp(p.hsl_adjustment + r * p.hsl_random, vec3f(-1.0), vec3f(1.0));
    let hsl = rgb_to_hsl(base.rgb);
    let rgb = hsl_to_rgb(vec3f(fract(hsl.x + a.x), push(hsl.y, a.y), push(hsl.z, a.z)));
    store(c, vec4f(rgb, base.a));
}
