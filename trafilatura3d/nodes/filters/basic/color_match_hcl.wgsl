// The luma / chroma / hue colour model (HCY): hexagonal hue in turns, chroma = max - min, Rec. 601 luma.
fn luma601(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }

fn rgb_to_hcl(c: vec3f) -> vec3f {
    return vec3f(rgb_to_hsl(c).x, max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b)), luma601(c));
}

// The fully saturated colour of the hue scaled by the chroma, then moved along grey to the luma.
fn hcl_to_rgb(hcl: vec3f) -> vec3f {
    let hue = clamp(abs(fract(hcl.x + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
    let b = hue * hcl.y;
    return b + (hcl.z - luma601(b));
}
