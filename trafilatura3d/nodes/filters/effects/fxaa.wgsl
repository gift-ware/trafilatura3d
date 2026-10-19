// The channel the edges are found on: ch 0 = the luma (0.299, 0.587, 0.114) made
// linear (sRGB curve), the gray value for grayscale; ch 1 = the alpha.
var<private> ch: i32;

fn lum(q: vec2f) -> f32 {
    let t = sample_input(q);
    if (ch == 1) { return t.a; }
    let v = select(dot(t.rgb, vec3f(0.299, 0.587, 0.114)), t.r, INPUT_GRAY);
    return select(pow((v + 0.055) / 1.055, 2.4), v / 12.92, v <= 0.04045);
}

// Where FXAA reads the pixel at uv from (uv itself away from edges).
fn fxaa_uv(uv: vec2f) -> vec2f {
    let px = 1.0 / p.size_;
    let lM = lum(uv);
    let lN = lum(uv + vec2f(0.0, -px.y));
    let lS = lum(uv + vec2f(0.0, px.y));
    let lE = lum(uv + vec2f(px.x, 0.0));
    let lW = lum(uv + vec2f(-px.x, 0.0));
    let hi = max(lM, max(max(lN, lS), max(lE, lW)));
    let lo = min(lM, min(min(lN, lS), min(lE, lW)));
    let range = hi - lo;
    if (range < max(0.0312, hi * 0.125)) { return uv; }
    let lNW = lum(uv + vec2f(-px.x, -px.y));
    let lNE = lum(uv + vec2f(px.x, -px.y));
    let lSW = lum(uv + vec2f(-px.x, px.y));
    let lSE = lum(uv + vec2f(px.x, px.y));
    // Sub-pixel aliasing amount.
    let avg = (2.0 * (lN + lS + lE + lW) + lNW + lNE + lSW + lSE) / 12.0;
    let sub = clamp(abs(avg - lM) / range, 0.0, 1.0);
    let subBlend = pow(smoothstep(0.0, 1.0, sub), 2.0) * 0.75;
    // Edge orientation.
    let edgeH = abs(lNW + lNE - 2.0 * lN) + 2.0 * abs(lW + lE - 2.0 * lM) + abs(lSW + lSE - 2.0 * lS);
    let edgeV = abs(lNW + lSW - 2.0 * lW) + 2.0 * abs(lN + lS - 2.0 * lM) + abs(lNE + lSE - 2.0 * lE);
    let horizontal = edgeH >= edgeV;
    var stepLen = select(px.x, px.y, horizontal);
    let lP = select(lE, lS, horizontal);
    let lNg = select(lW, lN, horizontal);
    let gP = abs(lP - lM);
    let gN = abs(lNg - lM);
    var lEdge = 0.5 * (lP + lM);
    var grad = gP;
    if (gN > gP) { stepLen = -stepLen; lEdge = 0.5 * (lNg + lM); grad = gN; }
    var e0 = uv;
    if (horizontal) { e0.y += 0.5 * stepLen; } else { e0.x += 0.5 * stepLen; }
    let along = select(vec2f(0.0, px.y), vec2f(px.x, 0.0), horizontal);
    // Search both ends of the edge.
    let steps = array<f32, 12>(1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0);
    var pP = e0;
    var pN = e0;
    var dP = 0.0;
    var dN = 0.0;
    var doneP = false;
    var doneN = false;
    let thresh = grad * 0.25;
    for (var i = 0; i < 12; i++) {
        if (!doneP) { pP += along * steps[i]; dP = lum(pP) - lEdge; doneP = abs(dP) >= thresh; }
        if (!doneN) { pN -= along * steps[i]; dN = lum(pN) - lEdge; doneN = abs(dN) >= thresh; }
        if (doneP && doneN) { break; }
    }
    let distP = select(pP.y - uv.y, pP.x - uv.x, horizontal);
    let distN = select(uv.y - pN.y, uv.x - pN.x, horizontal);
    let closerN = distN < distP;
    let dEnd = select(dP, dN, closerN);
    let goodSpan = (dEnd < 0.0) != (lM - lEdge < 0.0);
    var edgeBlend = 0.0;
    if (goodSpan) { edgeBlend = 0.5 - min(distN, distP) / (distN + distP); }
    let blend = max(edgeBlend, subBlend);
    var q = uv;
    if (horizontal) { q.y += blend * stepLen; } else { q.x += blend * stepLen; }
    return q;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c);
    ch = 0;
    var o = sample_input(fxaa_uv(uv));
    if (!INPUT_GRAY) {
        o.a = load_input(c).a;
        if (p.affect_alpha_channel == 1) { ch = 1; o.a = sample_input(fxaa_uv(uv)).a; }
    }
    store(c, o);
}
