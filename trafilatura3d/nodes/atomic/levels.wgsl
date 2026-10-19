@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var v = load_input(c);
    var lo = p.level_in_low;
    var hi = p.level_in_high;
    var mid = p.level_in_mid;
    var olo = p.level_out_low;
    var ohi = p.level_out_high;
    if (INPUT_GRAY) { lo = vec4f(lo.x); hi = vec4f(hi.x); mid = vec4f(mid.x); olo = vec4f(olo.x); ohi = vec4f(ohi.x); }
    let span = hi - lo;
    let safe = select(span, vec4f(1e-5), abs(span) < vec4f(1e-5));
    var t = (v - lo) / safe;
    if (p.intermediary_clamp != 0) { t = clamp(t, vec4f(0.0), vec4f(1.0)); }
    // The midtone as a gamma: t^g, g = 1 + 16 (mid - 0.5)
    // above 0.5 (darker), 1 / (1 + 16 (0.5 - mid)) below (lighter); 0.5 is linear.
    let g = select(1.0 / (1.0 + 16.0 * (0.5 - mid)), 1.0 + 16.0 * (mid - 0.5), mid >= vec4f(0.5));
    t = sign(t) * pow(abs(t), g);
    var o = mix(olo, ohi, t);
    if (INPUT_GRAY) { o = vec4f(o.x, o.x, o.x, 1.0); }
    store(c, o);
}
