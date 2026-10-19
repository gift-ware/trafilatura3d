// The input bilinearly sampled, always repeating (the projected texture tiles whatever the node's tiling).
fn wrapped(uv: vec2f) -> vec4f {
    let s = size_input();
    let x = uv * vec2f(s) - 0.5;
    let i = vec2i(floor(x));
    let f = fract(x);
    return mix(mix(texel_input(wrap(i, s)), texel_input(wrap(i + vec2i(1, 0), s)), f.x),
               mix(texel_input(wrap(i + vec2i(0, 1), s)), texel_input(wrap(i + vec2i(1, 1), s)), f.x), f.y);
}

// Rotation of the projection frame: about X, then Y, then Z (turns).
fn frame(v: vec3f) -> vec3f {
    let yz = rot2(v.yz, p.rotation_x);
    var r = vec3f(v.x, yz.x, yz.y);
    let zx = rot2(vec2f(r.z, r.x), p.rotation_y);
    r = vec3f(zx.y, r.y, zx.x);
    let xy = rot2(r.xy, p.rotation_z);
    return vec3f(xy.x, xy.y, r.z);
}

const AXES = array<vec3f, 7>(vec3f(1.0, 1.0, 1.0), vec3f(1.0, 1.0, 0.0), vec3f(1.0, 0.0, 1.0), vec3f(0.0, 1.0, 1.0),
                             vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));

fn effect(c: vec2i) -> vec4f {
    let n = frame(normalize(load_world_space_normals(c).rgb * 2.0 - 1.0));
    let q = frame(load_position(c).rgb - 0.5 + vec3f(p.offset_x, p.offset_y, p.offset_z));
    let on = AXES[p.projection];
    var w = pow(abs(n), vec3f(1.0 + 15.0 * p.blending_contrast)) * on;
    if (w.x + w.y + w.z < 1e-6) { w = on; }
    if (p.blending_mode == 1) {  // Max: the dominant axis only
        let m = max(w.x, max(w.y, w.z));
        w = select(vec3f(0.0), vec3f(1.0), w >= vec3f(m));
    }
    w /= (w.x + w.y + w.z);
    let t = p.texture_tiling;
    let ux = rot2(vec2f(q.z, -q.y) * t, p.axis_x_rotation) + vec2f(p.offset_x_x, p.offset_x_y) + 0.5;
    let uy = rot2(vec2f(q.x, q.z) * t, p.axis_y_rotation) + vec2f(p.offset_y_x, p.offset_y_y) + 0.5;
    let uz = rot2(vec2f(q.x, -q.y) * t, p.axis_z_rotation) + vec2f(p.offset_z_x, p.offset_z_y) + 0.5;
    return w.x * wrapped(ux) + w.y * wrapped(uy) + w.z * wrapped(uz);
}
