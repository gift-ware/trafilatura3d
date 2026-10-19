// Height box-averaged over 2^k x 2^k pixels (k = 1..11), bilinear.
fn mip(k: i32, uv: vec2f) -> f32 {
    switch k {
        case 1: { return sample_m1(uv).r; }
        case 2: { return sample_m2(uv).r; }
        case 3: { return sample_m3(uv).r; }
        case 4: { return sample_m4(uv).r; }
        case 5: { return sample_m5(uv).r; }
        case 6: { return sample_m6(uv).r; }
        case 7: { return sample_m7(uv).r; }
        case 8: { return sample_m8(uv).r; }
        case 9: { return sample_m9(uv).r; }
        case 10: { return sample_m10(uv).r; }
        default: { return sample_m11(uv).r; }
    }
}

const FULL = 0.4;  // the points count fully up to this fraction of the radius (fitted on the output)

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c);
    let side = min(p.size_.x, p.size_.y);
    var depth = p.height_depth;
    if (p.use_world_units == 1) { depth = p.height_scale_cm / max(p.surface_size, 1e-6); }
    let factor = depth * side;  // height in pixels for a height of 1
    let reach = max(side * p.radius, 1e-6);
    let fall0 = 1.0 / (1.0 - FULL);  // falloff = fall0 - dist x fall1
    let fall1 = fall0 / reach;
    let full = clamp(i32(floor(log2(max(FULL * reach, 1.0)))), 0, p.levels);  // points 2^k <= FULL x reach
    let dirs = select(select(16, 8, p.quality == 1), 4, p.quality == 0);
    let h0 = load_input(c).r;
    var total = 0.0;
    let turn = vec2f(cos(2.0 * PI / f32(dirs)), sin(2.0 * PI / f32(dirs)));  // the angle between directions
    let texel = 1.0 / select(vec2f(max(p.size_.x, p.size_.y)), p.size_, p.non_square == 1);
    var rot = vec2f(1.0, 0.0);
    for (var i = 0; i < dirs; i++) {
        let dir = rot * texel;
        // Marching out, every rise of the horizon adds its sine gain weighted by the distance falloff (1 up to
        // FULL x the radius, then linear to 0 at the radius): with no falloff the sum is the sine of the highest
        // horizon, the first point counts from 0 so slopes going down are negative and a tilted plane stays white.
        // Where the falloff is 1 the gains add up to the sine of the highest point: only the top is tracked there.
        var top = -1e30;
        var step = dir;          // dir x 2^k
        var gain = factor;       // factor / 2^k
        // Past `full` there are at most two points (the falloff spans 1.3 octaves): kept and added after the march.
        var tail = vec2f(-1e30);
        for (var k = 1; k <= p.levels; k++) {
            step *= 2.0;
            gain *= 0.5;
            let s = (mip(k, uv + step) - h0) * gain;
            top = select(top, max(top, s), k <= full);
            tail = select(tail, vec2f(s, tail.y), k == full + 1);
            tail = select(tail, vec2f(tail.x, s), k == full + 2);
        }
        var top_sin = select(top * inverseSqrt(1.0 + top * top), 0.0, full == 0);
        var occ = top_sin;
        for (var j = 0; j < min(p.levels - full, 2); j++) {
            let s = tail[j];
            if (s > top) {
                let sn = s * inverseSqrt(1.0 + s * s);
                occ += (fall0 - exp2(f32(full + 1 + j)) * fall1) * (sn - top_sin);  // the falloff is in 0..1 here
                top_sin = sn;
                top = s;
            }
        }
        total += occ;
        rot = vec2f(rot.x * turn.x - rot.y * turn.y, rot.x * turn.y + rot.y * turn.x);
    }
    store(c, fx_gray(clamp(1.0 - total / f32(dirs), 0.0, 1.0)));
}
