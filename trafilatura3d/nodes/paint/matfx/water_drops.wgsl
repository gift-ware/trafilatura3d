// Drops: domes in the cells of two grids, oriented and trailed along gravity; then the material under them.

// Gravity in the image: down the world direction given by the Influence of the position map's axes.
fn down_dir(uv: vec2f) -> vec2f {
    let infl = vec3f(p.influence_x, p.influence_y, p.influence_z);
    if (!POSITION_CONNECTED || length(infl) < 1e-4) { return vec2f(0.0, 1.0); }
    let o = 2.0 / p.size_;
    let y0 = dot(sample_position(uv).rgb, infl);
    let g = vec2f(dot(sample_position(uv + vec2f(o.x, 0.0)).rgb, infl) - y0, dot(sample_position(uv + vec2f(0.0, o.y)).rgb, infl) - y0);
    if (length(g) < 1e-6) { return vec2f(0.0, 1.0); }
    return -normalize(g);
}

// One grid of `n` cells: max dome height (x) of the drops around uv.
fn grid(uv: vec2f, n: i32, size: f32, down: vec2f, seed: f32) -> f32 {
    let x = uv * f32(n);
    let i = vec2i(floor(x));
    let dn = normalize(mix(vec2f(0.0, 1.0), down, p.drops_rotation_vector_map_multiplier) + vec2f(0.0, 1e-4));
    let side = vec2f(-dn.y, dn.x);
    var h = 0.0;
    for (var j = -2; j <= 2; j++) {
        for (var k = -2; k <= 2; k++) {
            let q = wrap(i + vec2i(k, j), vec2i(n));
            if (hash21(q, seed + 1.0) > p.drops_amount) { continue; }
            let ctr = vec2f(i + vec2i(k, j)) + 0.2 + 0.6 * hash22(q, seed);
            let s = size * (1.0 - p.drops_scale_random * hash21(q, seed + 2.0));
            let rel = x - ctr;
            var a = dot(rel, dn) / (s * p.drops_scale_y);
            let b = dot(rel, side) / (s * p.drops_scale_x);
            var w = 1.0;
            if (a < 0.0) {  // the trail left above the drop, thinner
                let len = 1.0 + p.drops_direction_intensity * 8.0 * hash21(q, seed + 3.0);
                a = a / len;
                w = mix(0.45, 1.0, mx_sat(1.0 + a * 2.0));
            }
            let d2 = a * a + (b * b) / (w * w);
            if (d2 < 1.0) { h = max(h, sqrt(1.0 - d2) * mix(0.35, 1.0, w * w)); }
        }
    }
    return h;
}

fn drops(uv: vec2f, down: vec2f) -> f32 {
    return max(grid(uv, 14, 0.42, down, p.seed_), grid(uv, 34, 0.38, down, p.seed_ + 31.0) * 0.8);
}

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let o = 1.0 / p.size_;
    let down = down_dir(uv);
    let n = mx_ndec(load_world_space_normals(c).rgb);
    let dir_w = vec3f(n.x * n.x, n.y * n.y, n.z * n.z);
    let facing = dir_w.x * select(p.left_intensity, p.right_intensity, n.x > 0.0)
               + dir_w.y * select(p.bottom_intensity, p.top_intensity, n.y > 0.0)
               + dir_w.z * select(p.back_intensity, p.front_intensity, n.z > 0.0);
    let mask = mix(1.0, facing, p.masking_intensity);
    let h = drops(uv, down) * mask;
    let hx = drops(uv + vec2f(o.x, 0.0), down) * mask - h;
    let hy = drops(uv - vec2f(0.0, o.y), down) * mask - h;
    let cover = smoothstep(0.0, 0.15, h);
    let slope = mx_nslope(hx, hy, 6.0 * mx_px() * p.normal_intensity);
    let wet = sample_water(uv).r * p.intensity;
    // Base color: refracted under the drops, darker where wet.
    let bco = load_basecolor(c);
    let bw = sample_basecolor(uv - slope.xy * vec2f(1.0, -1.0) * p.drops_base_color_warp_intensity * 0.03).rgb;
    let bc = mix(bco.rgb, bw * 0.85, cover) * (1.0 - wet * 0.45);
    store_basecolor(c, vec4f(bc, bco.a));
    let r = load_roughness(c);
    let rr = mix(r.r, p.drops_roughness, max(cover * p.drops_roughness_blending, wet));
    store_roughness(c, vec4f(vec3f(rr), r.a));
    let m = load_metallic(c);
    store_metallic(c, vec4f(vec3f(mix(m.r, p.drops_metallic, cover * p.drops_metallic_blending)), m.a));
    let nn = load_normal(c);
    store_normal(c, vec4f(mx_nenc(mx_ndetail(mx_ndec(nn.rgb), slope)), nn.a));
    let hh = load_height(c);
    store_height(c, vec4f(vec3f(hh.r + h * p.drops_height_intensity * 0.05), hh.a));
}
