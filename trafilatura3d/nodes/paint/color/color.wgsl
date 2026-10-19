// Helpers of the color filters (included by the nodes of this folder that need them).

// Weights of shadows, midtones, highlights for a value in [0, 1] (Bernstein basis: they sum to 1).
fn tone_weights(l: f32) -> vec3f {
    let x = clamp(l, 0.0, 1.0);
    return vec3f((1.0 - x) * (1.0 - x), 2.0 * x * (1.0 - x), x * x);
}

// Grayscale of a color: 0 Desaturation (halfway between the strongest and weakest channel), 1 Luma (perceived
// brightness, favoring green), 2 Average, 3 Max, 4 Min, 5 Red, 6 Green, 7 Blue.
fn gray_of(v: vec3f, mode: i32) -> f32 {
    let mx = max(v.r, max(v.g, v.b));
    let mn = min(v.r, min(v.g, v.b));
    switch mode {
        case 0: { return (mx + mn) * 0.5; }
        case 1: { return luminance(v); }
        case 2: { return (v.r + v.g + v.b) / 3.0; }
        case 3: { return mx; }
        case 4: { return mn; }
        case 5: { return v.r; }
        case 6: { return v.g; }
        default: { return v.b; }
    }
}

// Balance (0.5 = unchanged, shifts the midpoint toward black or white) then Contrast (0 = unchanged) then Invert.
fn balance_contrast(x: f32, balance: f32, contrast: f32, invert: i32) -> f32 {
    let y = contrasted(level_in(x, balance), contrast);
    return select(y, 1.0 - y, invert != 0);
}

// OKLab of a linear color, and back.
fn oklab(c: vec3f) -> vec3f {
    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    let q = pow(max(vec3f(l, m, s), vec3f(0.0)), vec3f(1.0 / 3.0));
    return vec3f(0.2104542553 * q.x + 0.7936177850 * q.y - 0.0040720468 * q.z,
                 1.9779984951 * q.x - 2.4285922050 * q.y + 0.4505937099 * q.z,
                 0.0259040371 * q.x + 0.7827717662 * q.y - 0.8086757660 * q.z);
}
fn oklab_inv(c: vec3f) -> vec3f {
    let q = vec3f(c.x + 0.3963377774 * c.y + 0.2158037573 * c.z, c.x - 0.1055613458 * c.y - 0.0638541728 * c.z,
                  c.x - 0.0894841775 * c.y - 1.2914855480 * c.z);
    let t = q * q * q;
    return vec3f(4.0767416621 * t.x - 3.3077115913 * t.y + 0.2309699292 * t.z,
                 -1.2684380046 * t.x + 2.6097574011 * t.y - 0.3413193965 * t.z,
                 -0.0041960863 * t.x - 0.7034186147 * t.y + 1.7076147010 * t.z);
}
// Stored (sRGB) color <-> OKLab.
fn srgb_to_oklab(c: vec3f) -> vec3f { return oklab(srgb_to_linear(clamp(c, vec3f(0.0), vec3f(1.0)))); }
fn oklab_to_srgb(c: vec3f) -> vec3f { return clamp(linear_to_srgb(oklab_inv(c)), vec3f(0.0), vec3f(1.0)); }
// Quantize's distance space: 0 Lab (OKLab of the stored color), 1 RGB (the stored values).
fn to_space(v: vec3f, space: i32) -> vec3f { if (space == 0) { return srgb_to_oklab(v); } return v; }
fn from_space(x: vec3f, space: i32) -> vec3f { if (space == 0) { return oklab_to_srgb(x); } return x; }
