@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let idc = load_color_id(c).rgb;
    var m = 0.0;
//$BODY
    store(c, g4(m));
}
