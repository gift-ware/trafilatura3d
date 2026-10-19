// Creased, first pass (ours): a sparse convolution of blades, drawn at half resolution. Two octaves of `cols` = 8 and
// 16 x Scale blade units across the tile (the second weighted CR_ROUGH); the tile is cut in cells 1 unit wide and 4
// tall, each holding CR_PER_CELL blades at random points (Disorder moves each around a loop of a quarter unit). A
// blade is an ellipse 2-4 units half-long and 0.5-1 half-wide, tilted up to CR_TILT / 2 radians off vertical, with
// the profile (1 - r^2)^3 bright on one side of its axis and dimmed to CR_DIM on the other (a crisp edge along it),
// and a signed random amplitude. Constants fitted on the output (spread, spectrum, look).

const CR_PER_CELL = 4;
const CR_ROUGH = 0.18;
const CR_DIM = 0.2;
const CR_SOFT = 0.15;  // half-width of the edge along the blade's axis, in half-widths
const CR_TILT = 0.84;

fn cr_rnd(st: ptr<function, u32>) -> f32 {
    *st = pcg(*st);
    return f32(*st) / 4294967295.0;
}

fn cr_blades(x: vec2f, cols: i32, s: f32, dis: f32) -> f32 {
    let per = vec2i(cols, cols / 4);
    let P = x * f32(cols);
    let ci = i32(floor(P.x));
    let cj = i32(floor(P.y / 4.0));
    var sum = 0.0;
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -2; di <= 2; di++) {
            let q = vec2i(ci + di, cj + dj);
            let w = wrap(q, per);
            var st = pcg(u32(w.x) ^ pcg(u32(w.y) ^ pcg(bitcast<u32>(s))));
            for (var k = 0; k < CR_PER_CELL; k++) {
                let h0 = cr_rnd(&st);
                let h1 = cr_rnd(&st);
                let h2 = cr_rnd(&st);
                let h3 = cr_rnd(&st);
                let h4 = cr_rnd(&st);
                let h5 = cr_rnd(&st);
                let h6 = cr_rnd(&st);
                let a = TAU * (dis + h6);
                let centre = vec2f(f32(q.x) + h0, 4.0 * (f32(q.y) + h1)) + 0.25 * vec2f(cos(a), sin(a));
                let half_len = 2.0 + 2.0 * h2;
                let half_w = 0.5 + 0.5 * h3;
                let tilt = (h4 - 0.5) * CR_TILT;
                let d = P - centre;
                let cs = cos(tilt);
                let sn = sin(tilt);
                let u = (cs * d.x + sn * d.y) / half_w;
                let v = (-sn * d.x + cs * d.y) / half_len;
                let r = u * u + v * v;
                if (r < 1.0) {
                    let t = 1.0 - r;
                    let side = mix(1.0, CR_DIM, smoothstep(-CR_SOFT, CR_SOFT, u));
                    sum += (2.0 * h5 - 1.0) * t * t * t * side;
                }
            }
        }
    }
    return sum;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = max(p.scale, 1);
    let x = uv_of(c) * aspect(p.non_square_expansion);
    let v = cr_blades(x, 8 * n, p.seed_, p.disorder) + CR_ROUGH * cr_blades(x, 16 * n, p.seed_ + 41.0, p.disorder);
    store(c, vec4f(v, v, v, 1.0));
}
