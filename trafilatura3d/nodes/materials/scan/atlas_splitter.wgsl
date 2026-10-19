struct Px {
    bc: vec3f,
    n: vec3f,
    m: f32,
    r: f32,
    h: f32,
    o: f32,
    u1: f32,
    u2: f32,
}

fn read(auv: vec2f) -> Px {
    return Px(sample_basecolor(auv).rgb, sample_normal(auv).rgb, sample_metallic(auv).r, sample_roughness(auv).r,
              sample_height(auv).r, sample_opacity(auv).r, sample_user_1(auv).r, sample_user_2(auv).r);
}

fn bg_px() -> Px {
    if (p.custom_background_color != 0) {
        return Px(p.base_color_bg_color, p.normal_bg_color, p.metallic_bg_color, p.roughness_bg_color,
                  p.height_bg_color, 0.0, p.user_1_bg_color, p.user_2_bg_color);
    }
    return Px(vec3f(0.0), vec3f(0.5, 0.5, 1.0), 0.0, 0.5, 0.0, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let n = shape_count();
    var out = bg_px();
    let sel = p.shape_selection - 1;  // 1-based
    var j = clamp(sel, 0, max(n - 1, 0));
    var local = uv;
    var g = 1.0;
    var line = 0.0;
    if (p.grid_view != 0) {
        g = ceil(sqrt(f32(max(n, 1))));
        let cell = floor(uv * g);
        j = i32(cell.y * g + cell.x);
        local = fract(uv * g);
        let lw = 0.015 * g;
        line = select(0.0, 1.0, min(min(local.x, 1.0 - local.x), min(local.y, 1.0 - local.y)) < lw);
    }
    var rot = p.rotation * TAU;
    if (j < n) {
        let k = shape_entry(j);
        let bb = shape_bbox(k);
        let root = shape_root(k);
        var s = bb.zw - bb.xy;
        if (p.auto_rotation != 0 && s.y > s.x) { rot += PI * 0.5; }
        // Cell -> element: centred, margin unless Auto Crop, fitted (Auto Scale) or at atlas scale.
        let margin = select(1.15, 1.0, p.auto_crop != 0);
        var q = rot2(local - 0.5, -rot);
        var ext = vec2f(max(s.x, s.y));
        if (p.auto_crop != 0 && p.auto_crop_mode == 0) {  // Largest shape (Square): every element at one scale
            for (var i = 0; i < n; i++) {
                let bi = shape_bbox(shape_entry(i));
                ext = max(ext, vec2f(max(bi.z - bi.x, bi.w - bi.y)));
            }
        }
        var span = select(vec2f(1.0) / g, ext, p.auto_scale != 0) * margin;
        if (p.auto_crop != 0 && p.auto_crop_mode == 2 && p.auto_scale != 0) { span = s; }  // Individual (Rectangle)
        let ctr = (bb.xy + bb.zw) * 0.5;
        let auv = ctr + q * span;
        let inside = in_shape(auv, root);
        if (inside) {
            out = read(auv);
        } else if (p.dilation_width > 0.0 && p.custom_background_color == 0) {
            // Dilate colors from the nearest inside point found on two rings.
            var best = 1e9;
            for (var ring = 1; ring <= 3; ring++) {
                let r = f32(ring) / 3.0 * p.dilation_width / 16.0 * 0.08 * max(span.x, span.y);
                for (var a = 0; a < 12; a++) {
                    let an = f32(a) * TAU / 12.0;
                    let pa = auv + vec2f(cos(an), sin(an)) * r;
                    if (r < best && in_shape(pa, root)) {
                        best = r;
                        let v = read(pa);
                        out = Px(v.bc, out.n, v.m, v.r, out.h, 0.0, v.u1, v.u2);
                    }
                }
            }
        }
        if (inside) {
            // Normals turn with the element.
            let dx = p.input_normal_format == 0;
            var nn = n_dec(out.n, dx);
            let t = rot2(vec2f(nn.x, -nn.y), rot);
            nn = vec3f(t.x, -t.y, nn.z);
            out.n = n_enc(nn, dx);
        }
    }
    if (p.enable_additional_inputs == 0) { out.u1 = bg_px().u1; out.u2 = bg_px().u2; }
    var bc = out.bc;
    if (p.grid_view != 0) {
        if (j == sel) { bc = mix(bc, vec3f(1.0, 0.6, 0.0), p.grid_selection_opacity * 0.5); }
        bc = mix(bc, vec3f(1.0), line * p.grid_opacity);
    }
    store_basecolor(c, vec4f(bc, 1.0));
    store_normal(c, vec4f(out.n, 1.0));
    store_metallic(c, g4(out.m));
    store_roughness(c, g4(out.r));
    store_height(c, g4(out.h));
    store_opacity(c, g4(out.o));
    store_user_1(c, g4(out.u1));
    store_user_2(c, g4(out.u2));
}
