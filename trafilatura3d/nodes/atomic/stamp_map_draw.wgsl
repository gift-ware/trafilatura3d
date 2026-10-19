// FX-Map patterns: one instanced quad per pattern (4 vec4 of data each, see stamp_map.py _instances).
@group(0) @binding(0) var<storage, read> pats: array<vec4f>;
@group(0) @binding(1) var img0: texture_2d<f32>;
@group(0) @binding(2) var img1: texture_2d<f32>;
@group(0) @binding(3) var img2: texture_2d<f32>;
@group(0) @binding(4) var img3: texture_2d<f32>;
@group(0) @binding(5) var img4: texture_2d<f32>;
@group(0) @binding(6) var img5: texture_2d<f32>;
@group(0) @binding(7) var img6: texture_2d<f32>;
@group(0) @binding(8) var img7: texture_2d<f32>;
@group(0) @binding(9) var background: texture_2d<f32>;
// u[0] x: the background is grayscale; yz: the layer's size; w: the cull threshold. The layer holds the result minus
// the background. u[1]: the tiling periods (xy) and the copies either way along each (zw); u[2] x: Cull outside.
@group(0) @binding(10) var<uniform> u: array<vec4f, 3>;
// A pattern whose centre reads less than u[0].w here is not drawn (white and 0 when nothing culls).
@group(0) @binding(11) var cull: texture_2d<f32>;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) local: vec2f,  // -1..1 across the pattern
    @location(1) @interpolate(flat) id: u32,
};

// The shift of copy j along an axis of period `period`: 0, +1, -1, +2, -2... periods.
fn copy_shift(j: u32, period: f32) -> f32 {
    if (j == 0u) { return 0.0; }
    let k = f32((j + 1u) / 2u);
    return select(-period * k, period * k, (j & 1u) == 1u);
}

@vertex
fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) ii: u32) -> VOut {
    var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                                  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
    // The instance: a pattern and one of its copies (x shift major, y minor), as stamp_map.py's `Copies` says.
    let nx = u32(2.0 * u[1].z + 1.0);
    let ny = u32(2.0 * u[1].w + 1.0);
    let i = ii / (nx * ny);
    let copy = ii % (nx * ny);
    let raw = pats[i * 4u];
    let a = vec4f(raw.xy + vec2f(copy_shift(copy / ny, u[1].x), copy_shift(copy % ny, u[1].y)), raw.zw);
    let b = pats[i * 4u + 1u];
    var o: VOut;
    // A copy missing the image is not drawn; with Cull outside off the pattern itself always is.
    let reach = 0.5 * length(a.zw);
    let hits = all(a.xy + reach >= vec2f(0.0)) && all(a.xy - reach <= vec2f(1.0));
    let cd = vec2i(textureDimensions(cull));
    if ((!hits && !(copy == 0u && u[2].x < 0.5))
            || (u[0].w > 0.0 && textureLoad(cull, clamp(vec2i(fract(a.xy) * vec2f(cd)), vec2i(0), cd - 1), 0).r < u[0].w)) {
        o.pos = vec4f(2.0, 2.0, 2.0, 1.0);  // outside the clip volume: not drawn
        o.local = vec2f(0.0);
        o.id = i;
        return o;
    }
    // A pattern under a pixel is drawn a pixel wide (`covered`): its quad grows to one pixel along that axis.
    let half = max(abs(a.zw) * 0.5 * u[0].yz, vec2f(1e-9));  // in pixels
    let k = corners[v] * max(vec2f(1.0), 0.5 / half);
    let ang = b.x * 6.2831853;  // turns, clockwise on screen (y down)
    let r = vec2f(k.x * a.z, k.y * a.w) * 0.5;
    let p = a.xy + vec2f(r.x * cos(ang) - r.y * sin(ang), r.x * sin(ang) + r.y * cos(ang));
    o.pos = vec4f(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    o.local = k;
    o.id = i;
    return o;
}

fn bilinear(t: texture_2d<f32>, uv: vec2f, nearest: bool) -> vec4f {
    let d = vec2i(textureDimensions(t));
    if (nearest) { return textureLoad(t, clamp(vec2i(uv * vec2f(d)), vec2i(0), d - 1), 0); }
    let x = uv * vec2f(d) - 0.5;
    let i = vec2i(floor(x));
    let f = fract(x);
    let a = textureLoad(t, clamp(i, vec2i(0), d - 1), 0);
    let b = textureLoad(t, clamp(i + vec2i(1, 0), vec2i(0), d - 1), 0);
    let c = textureLoad(t, clamp(i + vec2i(0, 1), vec2i(0), d - 1), 0);
    let e = textureLoad(t, clamp(i + vec2i(1, 1), vec2i(0), d - 1), 0);
    return mix(mix(a, b, f.x), mix(c, e, f.x), f.y);
}

fn image(k: i32, uv: vec2f, nearest: bool) -> vec4f {
    switch k {
        case 1: { return bilinear(img1, uv, nearest); }
        case 2: { return bilinear(img2, uv, nearest); }
        case 3: { return bilinear(img3, uv, nearest); }
        case 4: { return bilinear(img4, uv, nearest); }
        case 5: { return bilinear(img5, uv, nearest); }
        case 6: { return bilinear(img6, uv, nearest); }
        case 7: { return bilinear(img7, uv, nearest); }
        default: { return bilinear(img0, uv, nearest); }
    }
}

// The shape `kind` at the pixel centre, local q, `e` the pixel's extent in local units per axis. Along an axis under
// a pixel the pattern is drawn one pixel wide at full intensity (q squeezed to it), as the previews show: Waveform 1's
// 1 px Gaussians are full white bars in its preview, Directional Scratches' 1 px Cones keep its preview's mean (area
// weighting dimmed both). The antialiasing width is the finer axis's (a thin pattern does not smear along its length).
fn covered(kind: i32, q: vec2f, s: f32, e: vec2f) -> f32 {
    let k = max(vec2f(1.0), 0.5 * e);
    let w = e / k;
    return shape(kind, q / k, s, min(w.x, w.y));
}

@fragment
fn fs(i: VOut) -> @location(0) vec4f {
    // The pixel's extent in pattern units, per axis.
    let e = vec2f(abs(dpdx(i.local.x)) + abs(dpdy(i.local.x)), abs(dpdx(i.local.y)) + abs(dpdy(i.local.y)));
    let b = pats[i.id * 4u + 1u];
    let color = pats[i.id * 4u + 2u];
    let c = pats[i.id * 4u + 3u];
    let kind = i32(b.z);
    let weight = c.w;  // the pattern's opacity weight: Global opacity x Roughness for its level
    var pattern: vec4f;
    var offset = 0.0;
    if (kind == 1) {  // Input image, stretched over the pattern
        pattern = image(i32(b.w), i.local * 0.5 + 0.5, c.x > 0.5);
        if (c.y > 0.5) { pattern.a = 1.0; }
        pattern *= f32(all(abs(i.local) <= vec2f(1.0)));
    } else if (kind == 17) {  // Gradation w. offset: Variation added to the gradation x colour (Cross Section)
        let g = covered(15, i.local, 0.0, e);
        pattern = vec4f(g);
        offset = b.y * covered(0, i.local, 0.0, e);
    } else {
        pattern = vec4f(covered(kind - 2, i.local, b.y, e));  // the patterns after No Pattern and Input image
    }
    let blend = i32(c.z);
    if (blend == 0) { return (pattern * color + offset) * weight; }  // Add Sub (signed: a negative colour subtracts)
    let d = vec2i(textureDimensions(background));
    var bg = textureLoad(background, clamp(vec2i(i.pos.xy * vec2f(d) / u[0].yz), vec2i(0), d - 1), 0);
    if (u[0].x > 0.5) { bg = vec4f(bg.rrr, 1.0); }
    if (blend == 2) {  // Alpha blend: the colour over what is below, the pattern as opacity (premultiplied)
        let o = pattern.a * color.a * weight;
        let rgb = select(color.rgb, pattern.rgb * color.rgb, kind == 1);  // a shape is only coverage
        return vec4f((rgb - bg.rgb) * o, o);
    }
    return (pattern * color + offset) * weight - bg;  // Max
}
