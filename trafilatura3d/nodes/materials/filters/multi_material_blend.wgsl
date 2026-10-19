const KIND = $K;

fn put(o: vec4f, v: vec4f, m: f32) -> vec4f {
    if (KIND == 1) { return vec4f(n_enc(n_lerp(n_dec(o.rgb, false), n_dec(v.rgb, false), m), false), 1.0); }
    return mix(o, v, m);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = p.materials + 2;
    let idc = load_color_id(c).rgb;
    var o = $START;
$BODY
    if (KIND == 2) { o = g4(o.r); }
    store_$ch(c, out_$ch(c, o));
}
