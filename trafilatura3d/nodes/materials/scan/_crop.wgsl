// ---- scan/_crop.wgsl: the crop rectangle: T(x) = 0.5 + Transform (x - 0.5) + Offset
// maps the input (in units of its short side, Input Size aspect) to the output; the output samples
// T^-1. At Transform 1 the crop is a square of the input's short side; 2 (the default) halves it. ----
fn crop_ext() -> vec2f {
    let sz = vec2f(max(p.input_size, vec2i(1)));
    let a = sz.x / sz.y;
    return select(vec2f(1.0, 1.0 / a), vec2f(a, 1.0), a >= 1.0);  // input extent in short-side units
}
// Output uv -> input uv.
fn crop_uv(uv: vec2f) -> vec2f {
    let ext = crop_ext();
    return (0.5 * ext + inv2(p.transform) * (uv - 0.5 - p.offset)) / ext;
}
fn crop_inside(q: vec2f) -> bool { return all(q >= vec2f(0.0)) && all(q <= vec2f(1.0)); }
// Input uv -> is it in the crop (the Area output marks it).
fn crop_area(x: vec2f) -> bool {
    let ext = crop_ext();
    let m = p.transform;
    return crop_inside(0.5 + mat2x2f(m.x, m.y, m.z, m.w) * (x * ext - 0.5 * ext) + p.offset);
}
// The rotation of the crop applied to a normal map (xy turned with the content).
fn crop_normal(c: vec3f, directx: bool) -> vec3f {
    var n = n_dec(c, directx);
    let m = p.transform;
    let r = rot2(vec2f(n.x, -n.y), atan2(m.y, m.x));
    n = vec3f(r.x, -r.y, n.z);
    return n_enc(n, directx);
}
// The Area output: the input with the crop lightened (`show`: Show Crop Area), else the input.
fn crop_overlay(v: vec4f, x: vec2f, show: bool) -> vec4f {
    if (!show || !crop_area(x)) { return v; }
    return vec4f(mix(v.rgb, vec3f(1.0), 0.5), v.a);
}
