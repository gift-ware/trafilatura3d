// The X mask: 1 in the top and bottom quarters (the documentation's Offset1), 0 in the left and right ones, the
// boundary the diagonals through the centre, softened over NU_SOFT of the image on each side.
const NU_SOFT = 0.03;

fn x_mask(uv: vec2f) -> f32 {
    let q = abs(uv - 0.5);
    return smoothstep(-NU_SOFT, NU_SOFT, q.y - q.x);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_at(c);
    let a = sample_input(fract(2.0 * (uv - 0.5 + vec2f(p.offset1x, p.offset1y)) + 0.5)).r;
    let b = sample_input(fract(-2.0 * (uv - 0.5 + vec2f(p.offset2x, p.offset2y)) + 0.5)).r;
    var o = mix(b, a, x_mask(uv));
    if (MODE == 2) {
        // The two scales mixed: the input stretched to the output and the upscaled copy (weights fitted).
        o = 0.47 * sample_input(uv).r + 0.53 * o;
    }
    store(c, gray4(o));
}
