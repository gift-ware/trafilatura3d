// Final pass: color + sheen x Highlight Color - shade x (1 - Shadow Color). The sheen: the plain light fading
// with Gloss, and chrome bands read through the smoothed light's direction (gradient) and value, only where it
// is lit, solid where it is strong. Constants fitted on the output.
const BANDS = 1.5;      // stripes per unit of the band coordinate at Gloss 1
const GROW = 1.1;       // the bands grow as Gloss^GROW
const VALUE = 3.2;      // weight of the light's value in the band coordinate (its gradient weighs 1)
const EDGE = 3.0;       // the stripes' sharpness at Gloss 0 (they widen as Gloss grows)

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let base = load_color(c);
    if (p.intensity == 0.0) { store(c, base); return; }
    let s = ewg_slope(c);
    let lit = clamp(s, 0.0, 1.0);
    let sm = texel_smooth(c).r;
    let gx = texel_smooth(c + vec2i(1, 0)).r - texel_smooth(c - vec2i(1, 0)).r;
    let gy = texel_smooth(c + vec2i(0, 1)).r - texel_smooth(c - vec2i(0, 1)).r;
    let g = length(vec2f(gx, gy)) * f32(d.x) / 16.0;  // the light's change (resolution independent)
    let u = pow(p.gloss, GROW) * BANDS * (g + VALUE * sm);
    let stripes = pow(0.5 + 0.5 * cos(2.0 * PI * u), EDGE * (1.0 - p.gloss));
    let where_lit = clamp((sm - 0.05) * 6.0, 0.0, 1.0) * min(1.0, p.gloss * 10.0);  // no bands without Gloss
    let solid = clamp((sm - 0.6) * 3.0, 0.0, 1.0) * min(1.0, p.gloss * 4.0);
    let plain = clamp(2.0 * s, 0.0, 1.0);  // the plain sheen saturates twice as fast (fitted)
    let sheen = plain * clamp(1.0 - 4.0 * p.gloss, 0.0, 1.0) + max(stripes * where_lit, solid);
    let shade = clamp(-s, 0.0, 1.0) * (1.0 - p.shadow_color.rgb);
    store(c, vec4f(clamp(base.rgb + sheen * p.highlight_color.rgb - shade, vec3f(0.0), vec3f(1.0)), base.a));
}
