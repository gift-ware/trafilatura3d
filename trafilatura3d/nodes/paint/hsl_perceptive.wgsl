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
fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    if (p.mode == 1) {  // Regular: HSL of the stored values
        var h = rgb_to_hsl(clamp(v.rgb, vec3f(0.0), vec3f(1.0)));
        let dl = (p.lightness - 0.5) * 2.0 * select(h.z, 1.0 - h.z, p.lightness > 0.5);
        h = vec3f(fract(h.x + p.hue - 0.5), clamp(h.y * p.saturation * 2.0, 0.0, 1.0), clamp(h.z + dl, 0.0, 1.0));
        return vec4f(hsl_to_rgb(h), v.a);
    }
    // Perceptive: OKLCh of the linear color
    let lab = oklab(srgb_to_linear(clamp(v.rgb, vec3f(0.0), vec3f(1.0))));
    let ch = length(lab.yz) * p.saturation * 2.0;
    let ang = atan2(lab.z, lab.y) + (p.hue - 0.5) * 6.2831853;
    let l = clamp(lab.x + (p.lightness - 0.5), 0.0, 1.0);
    return vec4f(clamp(linear_to_srgb(oklab_inv(vec3f(l, ch * cos(ang), ch * sin(ang)))), vec3f(0.0), vec3f(1.0)), v.a);
}
