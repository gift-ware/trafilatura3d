// Ribbon path: per texel the value of the ribbon laid along the curve (alpha = how much, 0 outside). Each texel
// takes the nearest point of the curve (its length s along it and its side distance), within half the width and
// on a face turned toward the curve's normal; s and the side give the image's coordinates through the
// Stretching & Tiling settings (Ribbon path page; the mapping is ours, the documentation gives none).

struct Uniforms {
    lo: vec4f,     // bounding box min, width (world)
    hi: vec4f,     // bounding box max, length of the curve
    p: vec4f,      // tile length (one image along the path), tiling amount, start offset, end offset
    color: vec4f,  // the uniform value (no image)
    mode: vec4i,   // stretching mode, tiling mode, ratio | alignment << 2 | flags << 4, sample count
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> curve: array<vec4f>;  // per sample: (position, s), (normal, 0)
@group(0) @binding(2) var pos_map: texture_2d<f32>;
@group(0) @binding(3) var nrm_map: texture_2d<f32>;
@group(0) @binding(4) var image: texture_2d<f32>;
@group(0) @binding(5) var dst: texture_storage_2d<rgba32float, write>;

fn flag(bit: i32) -> bool { return ((u.mode.z >> u32(4 + bit)) & 1) != 0; }

fn bilinear_wrap(t: texture_2d<f32>, uv: vec2f) -> vec4f {
    let sz = vec2i(textureDimensions(t));
    let x = uv * vec2f(sz) - 0.5;
    let i = vec2i(floor(x));
    let f = fract(x);
    let a = textureLoad(t, wrap(i, sz), 0);  // wrap: nodes/int_math.wgsl, prepended by the loader
    let b = textureLoad(t, wrap(i + vec2i(1, 0), sz), 0);
    let c = textureLoad(t, wrap(i + vec2i(0, 1), sz), 0);
    let d = textureLoad(t, wrap(i + vec2i(1, 1), sz), 0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// The image's coordinate along the path (tile index in the integer part) for length s, or -1 where nothing is laid.
fn along(s: f32) -> f32 {
    let len = max(u.hi.w, 1e-9);
    let tile = max(u.p.x, 1e-9);
    let tiling = u.mode.y;
    let ratio = u.mode.z & 3;
    let align = (u.mode.z >> 2u) & 3;
    if (u.mode.x == 0) {  // Stretch along path
        var n = 1.0;
        if (tiling == 1) { n = max(1.0, round(len / tile)); }
        if (tiling == 2) { n = max(1.0, u.p.y); }
        if (flag(4) && !flag(5)) {  // stretch between offsets: the ends intact, the middle stretched (and tiled)
            let a = u.p.z * tile;
            let b = u.p.w * tile;
            if (s < a) { return s / tile; }
            if (s > len - b) { return (n - 1.0) + 1.0 - (len - s) / tile; }
            let m = (s - a) / max(len - a - b, 1e-9) * n;
            return floor(m) + u.p.z + fract(m) * (1.0 - u.p.z - u.p.w);
        }
        return s / len * n;
    }
    // Keep aspect ratio
    if (ratio == 1) {  // Fit to path length: an exact number of tiles, the ratio about kept
        return s / len * max(1.0, round(len / tile));
    }
    let count = ceil(len / tile);
    var x = s;
    if (align == 1) { x = s + (count * tile - len) * 0.5; }
    if (align == 2) { x = s + (count * tile - len); }
    let k = floor(x / tile);
    if (flag(3)) {  // Remove clipped tiles: only whole tiles inside the path
        var start = 0.0;
        if (align == 1) { start = (count * tile - len) * 0.5; }
        if (align == 2) { start = count * tile - len; }
        if (k * tile < start - 1e-6 || (k + 1.0) * tile > start + len + 1e-6) { return -1.0; }
    }
    return x / tile;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= vec2i(textureDimensions(pos_map)))) { return; }
    var out = vec4f(0.0);
    let pw = textureLoad(pos_map, c, 0);
    let p = pw.xyz;
    if (pw.a > 0.5 && all(p >= u.lo.xyz) && all(p <= u.hi.xyz)) {
        var best = 1e30;
        var s = 0.0;
        var side = 0.0;
        var nn = vec3f(0.0, 1.0, 0.0);
        var beyond = false;  // nearest to an open curve's end from outside: the ribbon ends flat there
        let count = u.mode.w;
        for (var i = 0; i < count - 1; i++) {
            let a = curve[2 * i];
            let b = curve[2 * i + 2];
            let ab = b.xyz - a.xyz;
            let tr = dot(p - a.xyz, ab) / max(dot(ab, ab), 1e-20);
            let t = clamp(tr, 0.0, 1.0);
            let q = a.xyz + ab * t;
            let dd = dot(p - q, p - q);
            if (dd < best) {
                best = dd;
                beyond = (i == 0 && tr < 0.0) || (i == count - 2 && tr > 1.0);
                s = mix(a.w, b.w, t);
                nn = normalize(mix(curve[2 * i + 1].xyz, curve[2 * i + 3].xyz, t));
                side = dot(p - q, normalize(cross(nn, ab)));
            }
        }
        let n = textureLoad(nrm_map, c, 0).xyz;
        if (sqrt(best) <= u.lo.w * 0.5 && dot(n, nn) > 0.0 && !beyond) {
            let img = along(s);
            if (img >= 0.0) {
                var fu = fract(img);
                if (flag(2) && (i32(floor(img)) & 1) == 1) { fu = 1.0 - fu; }  // mirror every 2nd tile
                var fv = side / u.lo.w + 0.5;
                if (flag(0)) { fv = 1.0 - fv; }  // flip image
                var uv = vec2f(fu, fv);
                if (flag(1)) { uv = vec2f(fv, fu); }  // vertical image orientation
                if (flag(7)) {
                    var t = bilinear_wrap(image, uv);
                    if (flag(6)) { t = vec4f(t.rrr, 1.0); }  // a single channel image
                    out = t;
                } else {
                    out = vec4f(u.color.rgb, 1.0);
                }
            }
        }
    }
    if (out.a > 0.0) { textureStore(dst, c, out); }  // only where this ribbon is (a mirrored one adds to it)
}
