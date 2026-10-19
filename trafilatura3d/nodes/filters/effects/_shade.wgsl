// Fake lighting of a `height` input: normal, light direction, cast shadows by marching the height field.

// Relief: height 1 rises one unit per 12.8 px at 256 (twice Normal Sobel at intensity 1).
const FX_RELIEF = 0.078125;

fn fx_height_normal(c: vec2i) -> vec3f {
    let uv = fx_uv(c);
    let e = 1.0 / p.size_;
    var h: array<f32, 9>;
    for (var j = 0; j < 3; j++) {
        for (var i = 0; i < 3; i++) {
            h[j * 3 + i] = sample_height(uv + vec2f(f32(i - 1), f32(j - 1)) * e).r;
        }
    }
    return fx_slope_normal(fx_slopes3(h, e, true) * FX_RELIEF);
}

// Light from Light Angle (turns, in the image plane, like Directional warp) at 45 degrees elevation,
// in normal space (+Y up).
fn fx_light(turns: f32) -> vec3f {
    let d = fx_dir(turns);
    return normalize(vec3f(d.x, -d.y, 1.0));
}

// 1 when the height field hides the light from uv (24 steps over a tenth of the image), else 0..1 soft.
fn fx_cast_shadow(uv: vec2f, l: vec3f) -> f32 {
    let h0 = sample_height(uv).r * FX_RELIEF;
    let dir = vec2f(l.x, -l.y);
    let len = length(dir);
    if (len < 1e-4) { return 0.0; }
    let rise = l.z / len;  // height gained per UV travelled towards the light
    var occ = 0.0;
    for (var i = 1; i <= 24; i++) {
        let t = f32(i) / 24.0 * 0.1;
        let hs = sample_height(uv + dir / len * t).r * FX_RELIEF;
        occ = max(occ, clamp((hs - (h0 + rise * t)) * 400.0, 0.0, 1.0));
    }
    return occ;
}
