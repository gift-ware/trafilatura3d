fn hr(q: vec2i, k: f32) -> f32 { return hash21(q, p.seed_ + k * 17.0); }

fn bg_h(uv: vec2f) -> f32 { return sample_background_height(uv).r; }

fn bg_slope(uv: vec2f) -> vec2f {
    let e = 0.004 + p.background_slope_smoothness * 0.02;
    return vec2f(bg_h(uv + vec2f(e, 0.0)) - bg_h(uv - vec2f(e, 0.0)),
                 bg_h(uv + vec2f(0.0, e)) - bg_h(uv - vec2f(0.0, e))) / (2.0 * e) * 0.05;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let grid = vec2i(max(p.x_amount, 1), max(p.y_amount, 1));
    let cs = 1.0 / vec2f(grid);
    let n = shape_count();
    let dxf = p.normal_format == 0;
    // Background.
    var bc = sample_background_basecolor(uv).rgb;
    var nrm = n_dec(sample_background_normal(uv).rgb, dxf);
    var h = bg_h(uv);
    var rough = sample_background_roughness(uv).r;
    var cov = 0.0;
    var best = -1e9;
    // Pattern Range: the first and last element used (0 0 = all).
    let r0 = clamp(p.pattern_range.x, 0, max(n - 1, 0));
    let nr = select(n - r0, min(p.pattern_range.y, n - 1) - r0 + 1, p.pattern_range.y > 0);
    let ci = vec2i(floor(uv * vec2f(grid)));
    if (n > 0) {
        for (var dy = -2; dy <= 2; dy++) {
            for (var dx = -2; dx <= 2; dx++) {
                let cell = ci + vec2i(dx, dy);
                let cw = wrap(cell, grid);
                let ctr0 = (vec2f(cell) + 0.5) * cs;
                let cuv = fract(ctr0);
                let mapv = sample_map_input(cuv).r;
                let vec = sample_vector_map(cuv).rg * 2.0 - 1.0;
                let slope = bg_slope(cuv);
                // Masking.
                if (hr(cw, 1.0) < p.mask_random + mapv * p.mask_random_map_multiplier * (1.0 - p.mask_random)) { continue; }
                let sl = sat(length(slope) * 2.0);
                if (p.mask_from_bg_slope > 0.0 && sl > 1.0 - p.mask_from_bg_slope) { continue; }
                if (p.mask_from_bg_slope < 0.0 && sl < -p.mask_from_bg_slope) { continue; }
                // Position.
                var rnd = vec2f(hr(cw, 2.0), hr(cw, 3.0)) - 0.5;
                if (p.random_distribution == 0) {
                    let u1 = max(hr(cw, 2.0), 1e-4);
                    rnd = sqrt(-2.0 * log(u1)) * vec2f(cos(TAU * hr(cw, 3.0)), sin(TAU * hr(cw, 3.0))) * 0.25;
                }
                var pos = ctr0 + rnd * p.position_random * cs + vec2f(p.offset_horizontal, p.offset_vertical) * cs
                    + vec * p.position_vector_map_multiplier * cs;
                // Element choice.
                var k = i32(hr(cw, 4.0) * f32(nr));
                if (p.pattern_distribution_mode == 1) { k = cw.x + cw.y * grid.x; }
                if (p.pattern_distribution_mode == 2) { k = cw.y; }
                if (p.pattern_distribution_mode == 3) { k = cw.x; }
                k = imod(k + i32(mapv * p.pattern_distribution_map_multiplier * f32(nr)), max(nr, 1));
                let entry = shape_entry(k + r0);
                let bb = shape_bbox(entry);
                let root = shape_root(entry);
                // Rotation.
                var ang = f32(p.pattern_rotation) * PI * 0.5 + p.rotation * TAU + (hr(cw, 5.0) - 0.5) * p.rotation_random * TAU;
                if (hr(cw, 6.0) < p.pattern_rotation_random) { ang += floor(hr(cw, 7.0) * 4.0) * PI * 0.5; }
                if (length(slope) > 1e-5) { ang += atan2(slope.y, slope.x) * p.rotation_from_bg_slope; }
                ang += mapv * p.rotation_map_multiplier * TAU;
                if (length(vec) > 1e-3) { ang += atan2(vec.y, vec.x) * p.rotation_vector_map_multiplier; }
                // Size: cell-sized, the element's own proportions kept by Preserve Aspect Ratio.
                let sc = p.scale * (1.0 - hr(cw, 8.0) * p.scale_random) * mix(1.0, mapv * 2.0, p.scale_map_multiplier)
                    * (1.0 - p.scale_no_overlap * 0.4);
                let es = max(bb.zw - bb.xy, vec2f(1e-4));
                let a = es.x / es.y;
                var ext = cs * 0.5 * sc * p.size;
                let fit = select(vec2f(ext.x, ext.x / a), vec2f(ext.y * a, ext.y), ext.x / ext.y > a);
                ext = mix(ext, fit, p.preserve_aspect_ratio);
                ext.x *= 1.0 + sl * p.size_ratio_from_bg_slope;
                // Two cells at most from the cell centre.
                let lim = cs * 2.0;
                let reach = length(pos - ctr0) + max(ext.x, ext.y);
                if (reach > min(lim.x, lim.y)) {
                    if (p.out_of_bounds_option == 0) { ext *= max(min(lim.x, lim.y) - length(pos - ctr0), 0.0) / max(ext.x, ext.y); }
                    else { pos = ctr0 + normalize(pos - ctr0 + 1e-6) * max(min(lim.x, lim.y) - max(ext.x, ext.y), 0.0); }
                }
                let q = rot2(uv - pos, -ang) / max(ext, vec2f(1e-5));
                if (abs(q.x) > 1.0 || abs(q.y) > 1.0) { continue; }
                let auv = mix(bb.xy, bb.zw, q * 0.5 + 0.5);
                if (!in_shape(auv, root) || sample_atlas_opacity(auv).r < 0.5) { continue; }
                let ah = sample_atlas_height(auv).r;
                if (p.cutout_black_pixels != 0 && ah < 0.02) { continue; }
                // Height: scaled, offset, conformed, skewed.
                var hs = ah * p.height_scale * (1.0 - hr(cw, 9.0) * p.height_scale_random) * mix(1.0, mapv * 2.0, p.height_scale_map_multiplier);
                if (p.height_scale_auto_adjust != 0) { hs *= sc; }
                hs += p.height_offset + (hr(cw, 10.0) - 0.5) * p.height_offset_random + (mapv - 0.5) * p.height_offset_map_multiplier;
                let base = mix(bg_h(uv), bg_h(cuv), sat(p.smooth_conformed_background * 0.5));
                hs += mix(bg_h(cuv), base, p.conform_to_background);
                hs += dot(slope, (uv - pos) / cs) * p.height_skew * 0.5;
                let wins = select(hs > best, true, p.blend_mode == 1);
                if (!wins) { continue; }
                best = hs;
                cov = 1.0;
                // Color, normal, roughness.
                var col = sample_atlas_basecolor(auv).rgb + p.color_adjustment * 0.5;
                var hsv = rgb2hsv(sat3(col));
                // Color Random / from Background per channel (R hue, G saturation, B value).
                hsv.x = fract(hsv.x + (hr(cw, 11.0) - 0.5) * p.color_random.x);
                hsv.y = sat(hsv.y * (1.0 + (hr(cw, 16.0) - 0.5) * 2.0 * p.color_random.y));
                hsv.z *= 1.0 + (hr(cw, 12.0) - 0.5) * p.color_random.z;
                col = mix(hsv2rgb(hsv), sample_background_basecolor(cuv).rgb, p.color_from_background);
                var sn = n_dec(sample_atlas_normal(auv).rgb, dxf);
                let t = rot2(vec2f(sn.x, -sn.y), ang);
                sn = vec3f(t.x, -t.y, sn.z);
                let tilt = (vec2f(hr(cw, 13.0), hr(cw, 14.0)) - 0.5) * p.normal_random;
                sn = n_combine(sn, normalize(vec3f(tilt, 1.0)));
                sn = n_combine(n_lerp(vec3f(0.0, 0.0, 1.0), n_dec(sample_background_normal(cuv).rgb, dxf), p.normal_skew), sn);
                var ro = sample_atlas_roughness(auv).r + p.roughness_adjustment * 0.5 + (hr(cw, 15.0) - 0.5) * p.roughness_random;
                ro = mix(ro, sample_background_roughness(cuv).r, p.roughness_from_background);
                bc = col;
                nrm = sn;
                rough = sat(ro);
                var under = bg_h(uv);
                if (p.flatten_pattern_base != 0) { under = bg_h(cuv); }
                h = select(max(hs, under), hs, p.blend_mode == 1);
            }
        }
    }
    store_basecolor(c, vec4f(bc, 1.0));
    store_normal(c, vec4f(n_enc(nrm, dxf), 1.0));
    store_height(c, g4(h));
    store_roughness(c, g4(rough));
    store_opacity(c, g4(cov));
}
