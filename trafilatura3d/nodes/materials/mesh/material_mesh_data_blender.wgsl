// One channel per pass ($CHID: index in the full material: diffuse, basecolor, normal, specular,
// emissive, glossiness, roughness, metallic, specular_level, ambient_occlusion, height, opacity).
const CH = $CHID;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_$ch(c);
    // A baked map switched off reads its neutral value.
    let ao = select(1.0, load_baked_ambient_occlusion(c).r, p.use_baked_ambient_occlusion != 0);
    let curv = select(0.5, load_baked_curvature(c).r, p.use_baked_curvature != 0);
    let edge = sat((curv - 0.5) * 2.0);
    let outline = smoothstep(0.3, 0.6, abs(curv - 0.5) * 2.0);
    let vc = select(v.rgb, load_baked_vertex_color(c).rgb, p.use_baked_vertex_color != 0);
    let wn = normalize(select(vec3f(0.5, 1.0, 0.5), load_baked_world_space_normal(c).rgb, p.use_baked_world_space_normal != 0) * 2.0 - 1.0 + vec3f(0.0, 1e-5, 0.0));
    let bn = n_dec(select(vec3f(0.5, 0.5, 1.0), load_baked_normal(c).rgb, p.use_baked_normal != 0), false);
    let bh = select(0.5, load_baked_height(c).r, p.use_baked_height != 0);
    var o = v;
    switch CH {
        case 0: {
            var col = v.rgb * mix(1.0, ao, p.diffuse_ao) * (1.0 + (curv - 0.5) * p.diffuse_sharp_edges * 1.5);
            col = mix(col, vc, p.diffuse_color_from_vertex_color);
            let ndl = sat(dot(wn, normalize(vec3f(0.4, 0.8, 0.45))));
            let layers = f32(max(p.diffuse_cartoon_pre_lighting_layers, 1));
            let toon = floor(ndl * layers + 0.5) / layers;
            let lit = mix(ndl, toon, p.diffuse_cartoon_lighting_balance);
            col *= mix(1.0, 0.35 + 0.9 * lit, p.diffuse_pre_lighting);
            col *= 1.0 - outline * p.diffuse_cartoon_outlines;
            o = vec4f(col, v.a);
        }
        case 1: {
            var col = v.rgb * mix(1.0, ao, p.base_color_ao) * (1.0 + (curv - 0.5) * p.base_color_sharp_edges * 1.5);
            o = vec4f(mix(col, vc, p.base_color_from_vertex_color), v.a);
        }
        case 2: {
            let mn = n_dec(v.rgb, false);
            o = vec4f(n_enc(n_combine(bn, n_lerp(vec3f(0.0, 0.0, 1.0), mn, p.normal_material_intensity)), false), 1.0);
        }
        case 3: {
            var col = v.rgb * mix(1.0, ao, p.specularao) + edge * p.specular_bright_sharp_edges * 0.3;
            o = vec4f(col + outline * p.specular_cartoon_outlines * 0.3, v.a);
        }
        case 5: { o = g4(v.r - edge * p.glossiness_dark_sharp_edges * 0.5); }
        case 6: { o = g4(v.r + edge * p.roughness_bright_sharp_edges * 0.5 + outline * p.roughness_cartoon_outlines * 0.5); }
        case 7: { o = g4(v.r + edge * p.metallic_bright_sharp_edges + outline * p.metallic_cartoon_outlines); }
        case 9: { o = g4(mix(v.r, v.r * ao, p.ao_materiel_intensity)); }
        case 10: {
            if (p.height_material_blending_type == 0) { o = g4(v.r + (bh - 0.5) * p.height_material_intensity); }
            else { o = g4(mix(v.r, bh, p.height_material_intensity)); }
        }
        default: {}
    }
    if ($K == 2) { o = g4(o.r); }
    store_$ch(c, out_$ch(c, o));
}
