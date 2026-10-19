// Direction on the unit sphere for a panorama point (u = longitude, v = 0 top / 1 bottom), y up.
fn pano_dir(uv: vec2f) -> vec3f {
    let lon = (uv.x - 0.5) * TAU;
    let lat = (0.5 - uv.y) * PI;
    return vec3f(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / p.size_;
    // Matrix rows (x0, x1), (y0, y1): tangent plane -> shape space; its inverse moves the centre by Offset.
    let mat = mat2x2f(p.matrix.x, p.matrix.z, p.matrix.y, p.matrix.w);
    let det = determinant(mat);
    var inv = mat2x2f(1.0, 0.0, 0.0, 1.0);
    if (abs(det) > 1e-6) { inv = mat2x2f(mat[1][1], -mat[0][1], -mat[1][0], mat[0][0]) * (1.0 / det); }
    let dir = pano_dir(uv);
    let ctr = pano_dir(vec2f(0.5, 0.25) + inv * p.offset);
    // Tangent frame at the centre: east (along longitude) and north.
    var east = cross(vec3f(0.0, 1.0, 0.0), ctr);
    if (length(east) < 1e-4) { east = vec3f(1.0, 0.0, 0.0); }
    east = normalize(east);
    let north = cross(ctr, east);
    let z = dot(dir, ctr);
    var cover = 0.0;
    var light = vec3f(0.0);
    if (z > 1e-3) {
        // Gnomonic projection onto the tangent plane (x right, y down), then the shape space ([-1, 1] inside).
        let l = mat * (vec2f(dot(dir, east), -dot(dir, north)) / z);
        var r: f32;
        if (p.shape == 0) { r = max(abs(l.x), abs(l.y)); } else { r = length(l); }
        // Soft border: from 1 - soft (full) to 1 (none); antialiased at least over one pixel.
        let px = PI / p.size_.y * max(length(mat[0]), length(mat[1])) * 2.0;
        let soft = max(p.shape_soft_border, px);
        cover = clamp((1.0 - r) / soft, 0.0, 1.0);
        cover = cover * cover * (3.0 - 2.0 * cover);
        var col = p.shape_color.rgb;
        if (p.enable_texture_input == 1) {
            let t = sample_input(clamp(l * 0.5 + 0.5, vec2f(0.0), vec2f(1.0)));
            col *= t.rgb;
            if (!INPUT_GRAY) { cover *= t.a; }
        }
        // Hotspot: a soft disc of radius Hotspot Size at Hotspot Position (shape space 0-1).
        let hc = (p.hotspot_position - 0.5) * 2.0;
        let hr = max(p.hotspot_size, 1e-4);
        let hd = length(l - hc) / hr;
        let hot = 1.0 - smoothstep(1.0 - max(p.hotspot_falloff, 1e-3), 1.0, hd);
        light = col * cover * (p.shape_intensity + p.hotspot_intensity * hot);
    }
    var base = load_background(c);
    if (!BACKGROUND_CONNECTED) { base = vec4f(0.0, 0.0, 0.0, 1.0); }
    if (p.enable_background == 1) { base = p.background_color; }
    store(c, vec4f(base.rgb * (1.0 - cover) + light, max(base.a, cover)));
}
