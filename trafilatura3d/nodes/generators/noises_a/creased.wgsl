// Creased, last pass (ours): the blades read through a domain warp, then 0.5 + CR_GAIN x blades, clamped. The warp
// bends every row sideways by Warp Intensity x (1 - (2y - 1)^2) of the tile (the creases lean one way in the top half,
// the other in the bottom half) and curls them by a coarse gradient noise (Scale x (4, 1) cells) of CR_CURL x Warp
// Intensity / Scale. Constants fitted on the output.

const CR_GAIN = 0.235;
const CR_CURL = 0.3;

fn cr_grad(q: vec2i, per: vec2i, s: f32) -> vec2f {
    let a = hw(q, per, s) * TAU;
    return vec2f(cos(a), sin(a));
}

fn cr_noise(x: vec2f, per: vec2i, s: f32) -> f32 {
    let i = vec2i(floor(x));
    let f = fract(x);
    let u = quintic(f);
    let a = dot(cr_grad(i, per, s), f);
    let b = dot(cr_grad(i + vec2i(1, 0), per, s), f - vec2f(1.0, 0.0));
    let c = dot(cr_grad(i + vec2i(0, 1), per, s), f - vec2f(0.0, 1.0));
    let d = dot(cr_grad(i + vec2i(1, 1), per, s), f - vec2f(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = max(p.scale, 1);
    let uv = uv_of(c);
    let asp = aspect(p.non_square_expansion);
    let t = 2.0 * uv.y - 1.0;
    let bend = (1.0 - t * t) * p.warp_intensity;
    let per = vec2i(n * 4, n);
    let curl = cr_noise((uv * asp + vec2f(bend, 0.0)) * vec2f(per), per, p.seed_ + 9.7);
    let shift = vec2f(bend + CR_CURL * p.warp_intensity * curl / f32(n), 0.0) / asp;
    let o = clamp(0.5 + CR_GAIN * sample_blades(uv + shift).r, 0.0, 1.0);
    store(c, vec4f(o, o, o, 1.0));
}
