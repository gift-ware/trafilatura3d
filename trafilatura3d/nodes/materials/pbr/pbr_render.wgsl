// ---- PBR Render: ray cast of the shape, image based lighting ----
const LEVELS = 6.0;
const UNIT = 0.45;  // object size of Scale 1 (a sphere filling ~85% of the frame at Distance 2, FOV 30)

fn env_rot(dir: vec3f) -> vec3f { return rot_axis(vec3f(0.0, 1.0, 0.0), -p.environment_rotation * TAU) * dir; }

// Prefiltered environment at a blur level 0..5 (atlas of 6 panoramas stacked vertically).
fn env(dir: vec3f, level: f32) -> vec3f {
    let uv = dir_ll(env_rot(dir));
    let l = clamp(level, 0.0, LEVELS - 1.0);
    let l0 = floor(l);
    let v = clamp(uv.y, 0.5 / 128.0, 1.0 - 0.5 / 128.0);
    let a = sample_env(vec2f(uv.x, (v + l0) / LEVELS)).rgb;
    let b = sample_env(vec2f(uv.x, (v + min(l0 + 1.0, LEVELS - 1.0)) / LEVELS)).rgb;
    return mix(a, b, l - l0);
}

struct Hit {
    t: f32,
    pos: vec3f,   // object space
    n: vec3f,     // object space geometric normal
    tg: vec3f,    // tangent (u direction)
    uv: vec2f,
}

// Ray against the shape in object space, the surface pushed along its normal by `off`.
fn intersect(ro: vec3f, rd: vec3f, off: f32) -> Hit {
    var h = Hit(-1.0, vec3f(0.0), vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), vec2f(0.0));
    if (p.shape == 0) {
        let r = max(p.scale * UNIT, 0.005) + off;
        let b = dot(ro, rd);
        let disc = b * b - (dot(ro, ro) - r * r);
        if (disc < 0.0) { return h; }
        let t = -b - sqrt(disc);
        if (t <= 0.0) { return h; }
        h.t = t;
        h.pos = ro + rd * t;
        h.n = normalize(h.pos);
        let th = atan2(h.n.x, h.n.z);
        h.tg = vec3f(cos(th), 0.0, -sin(th));
        h.uv = vec2f(th / TAU + 0.5, acos(clamp(h.n.y, -1.0, 1.0)) / PI) * p.sphere_uv_scale;
        if (p.sphere_uv_projection == 1) {  // Cubic: the face of the dominant axis, its own tangent
            let a = abs(h.n);
            var f = vec2f(h.n.x, -h.n.y) / max(a.z, 1e-4) * sign(h.n.z);
            h.tg = vec3f(sign(h.n.z), 0.0, 0.0);
            if (a.x >= a.y && a.x >= a.z) { f = vec2f(-h.n.z, -h.n.y) / a.x * sign(h.n.x); h.tg = vec3f(0.0, 0.0, -sign(h.n.x)); }
            else if (a.y >= a.z) { f = vec2f(h.n.x, h.n.z) / a.y * sign(h.n.y); h.tg = vec3f(sign(h.n.y), 0.0, 0.0); }
            h.uv = (f * 0.5 + 0.5) * p.sphere_uv_scale * 0.5;
        }
        return h;
    }
    if (p.shape == 1) {
        let half = max(p.plane_size * UNIT, vec2f(1e-3));
        if (abs(rd.z) < 1e-6) { return h; }
        let t = (off - ro.z) / rd.z;
        let q = ro + rd * t;
        if (t <= 0.0 || abs(q.x) > half.x || abs(q.y) > half.y) { return h; }
        h.t = t;
        h.pos = q;
        h.uv = vec2f(q.x / half.x * 0.5 + 0.5, 0.5 - q.y / half.y * 0.5) * p.plane_uv_scale;
        return h;
    }
    // Cylinder along Y with caps.
    let r = max(p.cylinder_radius * 1.5 * UNIT, 0.005) + off;
    let hl = max(p.cylinder_length * UNIT, 0.005);
    var best = 1e9;
    let a = rd.x * rd.x + rd.z * rd.z;
    if (a > 1e-8) {
        let b = ro.x * rd.x + ro.z * rd.z;
        let cc = ro.x * ro.x + ro.z * ro.z - r * r;
        let disc = b * b - a * cc;
        if (disc >= 0.0) {
            let t = (-b - sqrt(disc)) / a;
            let q = ro + rd * t;
            if (t > 0.0 && abs(q.y) <= hl) {
                best = t;
                h.t = t;
                h.pos = q;
                h.n = normalize(vec3f(q.x, 0.0, q.z));
                let th = atan2(q.x, q.z);
                h.tg = vec3f(cos(th), 0.0, -sin(th));
                h.uv = vec2f(th / TAU + 0.5, (hl - q.y) / (2.0 * hl)) * p.cylinder_uv_scale;
            }
        }
    }
    if (abs(rd.y) > 1e-6) {
        for (var s = -1.0; s <= 1.0; s += 2.0) {
            let t = (s * hl - ro.y) / rd.y;
            let q = ro + rd * t;
            if (t > 0.0 && t < best && q.x * q.x + q.z * q.z <= r * r) {
                best = t;
                h.t = t;
                h.pos = q;
                h.n = vec3f(0.0, s, 0.0);
                h.tg = vec3f(1.0, 0.0, 0.0);
                h.uv = (vec2f(q.x, q.z * s) / r * 0.5 + 0.5) * p.cylinder_uv_scale;
            }
        }
    }
    return h;
}

fn tex_uv(h: Hit) -> vec2f {
    var uv = h.uv;
    if (p.tilt_uvs != 0 && p.shape == 0) { uv = rot2(uv, PI * 0.25); }
    return uv * p.uv_tiling + p.uv_offset;
}

// Split-sum environment BRDF, analytic fit (Karis): (scale, bias) of F0.
fn env_brdf(rough: f32, nov: f32) -> vec2f {
    let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
    return vec2f(-1.04, 1.04) * a004 + r.zw;
}

fn background(c: vec2i, uv: vec2f, dir: vec3f) -> vec3f {
    switch p.background_mode {
        case 1: { return env(dir, p.environment_background_blur * (LEVELS - 1.0)); }
        case 2: { return env(dir, LEVELS - 1.0); }
        case 3: { return srgb2lin(sample_background_input(uv).rgb); }
        default: { return srgb2lin(p.background_color.rgb); }
    }
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let suv = uv_of(c, d);
    // Camera orbiting the origin (Screen Position = yaw, pitch; 0.5 0.5 = in front).
    let dist = max(p.distance, 0.05);
    let sp = (p.screen_position - 0.5) * 2.0;
    let rot = rot_axis(vec3f(0.0, 1.0, 0.0), sp.x * PI) * rot_axis(vec3f(1.0, 0.0, 0.0), -sp.y * PI * 0.5);
    let cam = rot * vec3f(0.0, 0.0, dist);
    let fwd = normalize(-cam);
    let right = normalize(cross(fwd, rot * vec3f(0.0, 1.0, 0.0)));
    let up = cross(right, fwd);
    let aspect = f32(d.x) / f32(d.y);
    let ndc = vec2f((suv.x - 0.5) * 2.0 * aspect, (0.5 - suv.y) * 2.0);
    let tf = tan(clamp(p.field_of_view, 0.01, 170.0) * PI / 360.0);
    var ro = cam;
    var rd = normalize(fwd + (ndc.x * right + ndc.y * up) * tf);
    if (p.camera_mode == 1) { ro = cam + (ndc.x * right + ndc.y * up) * tf * dist; rd = fwd; }
    // Object transform: tilt around an axis in the screen plane, spin around its own Y, offset.
    let ra = p.rotation_direction * TAU;
    let m = rot_axis(vec3f(cos(ra), sin(ra), 0.0), p.rotation * TAU) * rot_axis(vec3f(0.0, 1.0, 0.0), p.rotation_around_direction * TAU);
    let mt = transpose(m);
    let shift = p.shape_position * UNIT;
    let oro = mt * (ro - shift);
    let ord = mt * rd;
    // Displacement: iterate the hit with the height found at the previous one.
    var h = intersect(oro, ord, 0.0);
    let disp = p.displacement_intensity * UNIT;
    if (h.t > 0.0 && disp > 0.0) {
        for (var k = 0; k < 3; k++) {
            let hh = sample_height(fract(tex_uv(h))).r;
            let h2 = intersect(oro, ord, disp * (hh - 0.5));
            if (h2.t <= 0.0) { break; }
            h = h2;
        }
    }
    let bg = background(c, suv, rd);
    if (h.t <= 0.0) {
        store_hdr(c, vec4f(bg, 1e4));
        store_raw_irradiance(c, vec4f(0.0, 0.0, 0.0, 1.0));
        store_raw_specular(c, vec4f(0.0, 0.0, 0.0, 1.0));
        store_normal_world_space(c, vec4f(0.0, 0.0, 0.0, 1.0));
        store_normal_tangent_space(c, vec4f(0.0, 0.0, 0.0, 1.0));
        store_uvs(c, vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }
    var uv = tex_uv(h);
    var tg = h.tg;
    if (p.tilt_uvs != 0 && p.shape == 0) { tg = rot_axis(h.n, -PI * 0.25) * tg; }
    let fu = fract(uv);
    // Tangent frame in world space (B = texture up).
    let N = normalize(m * h.n);
    let T = normalize(m * tg);
    let Bu = cross(N, T);
    let dxf = p.normal_format == 0;
    var nts = n_dec(sample_normal(fu).rgb, dxf);
    if (p.recompute_normals != 0) {
        let e = 1.0 / 512.0;
        let k = disp / UNIT * 30.0;
        let hn = n_from4(sample_height(fract(uv - vec2f(e, 0.0))).r, sample_height(fract(uv + vec2f(e, 0.0))).r,
                         sample_height(fract(uv - vec2f(0.0, e))).r, sample_height(fract(uv + vec2f(0.0, e))).r, k);
        nts = n_combine(nts, hn);
    }
    let n = normalize(T * nts.x + Bu * nts.y + N * nts.z);
    let V = -rd;
    let nov = max(dot(n, V), 1e-3);
    // Material.
    let base = srgb2lin(sample_base_color(fu).rgb);
    let metal = sat(sample_metallic(fu).r);
    let rough = clamp(sample_roughness(fu).r, 0.02, 1.0);
    var f0d = p.dielectric_f0;
    if (p.dielectric_f0_input == 1) { f0d = 0.08 * sample_specular_level(fu).r; }
    let f0 = mix(vec3f(f0d), base, metal);
    let albedo = base * (1.0 - metal);
    let ao = mix(1.0, sample_ambient_occlusion(fu).r, p.ambient_occlusion_intensity);
    let emis = srgb2lin(sample_emissive(fu).rgb) * p.emissive_intensity;
    // Diffuse: cosine-weighted samples of a blurred level.
    let ns = 16u << u32(p.diffuse_quality);
    let fr = frame(n);
    var irr = vec3f(0.0);
    for (var i = 0u; i < ns; i++) {
        let r = sqrt((f32(i) + 0.5) / f32(ns));
        let a = f32(i) * 2.39996323;
        let l = fr * vec3f(r * cos(a), r * sin(a), sqrt(max(1.0 - r * r, 0.0)));
        irr += env(l, 3.0);
    }
    irr /= f32(ns);
    // Self shadowing from the height towards the sky (texture-space horizon test).
    let lts = vec2f(dot(vec3f(0.0, 1.0, 0.0), T), -dot(vec3f(0.0, 1.0, 0.0), Bu));
    let h0 = sample_height(fu).r;
    var occ = 0.0;
    if (length(lts) > 1e-3) {
        let dir2 = normalize(lts);
        for (var k = 1; k <= 8; k++) {
            let hk = sample_height(fract(uv + dir2 * f32(k) * 0.006)).r;
            occ = max(occ, (hk - h0 - f32(k) * 0.015) * 8.0);
        }
    }
    irr *= mix(1.0, 1.0 - sat(occ), p.diffuse_shadow_intensity);
    if (p.enable_emissive_lighting != 0) { irr += emis * p.diffuse_emissive_multiplier; }
    var diffuse = albedo * irr * ao;
    // Subsurface: light wrapping through, shifted to red and to blue at grazing angles.
    if (p.enable_subsurface_scattering != 0) {
        let tr = sat(sample_translucency(fu).r);
        let sd = sample_scattering_distance_scale(fu).rgb * p.scattering_distance;
        let back = env(-n, LEVELS - 1.0) * sd + env(n, LEVELS - 1.0) * (1.0 - sd * 0.5);
        var sss = albedo * back * (vec3f(1.0) + p.red_shift * vec3f(0.6, 0.0, -0.3));
        sss += vec3f(0.2, 0.4, 1.0) * p.rayleigh * pow(1.0 - nov, 3.0) * 0.3 * luma(back);
        diffuse = mix(diffuse, sss, tr);
    }
    // Specular: bent normal for anisotropy, dithered lookup, environment BRDF.
    let aa = sample_anisotropy_angle(fu).r * TAU;
    let at = normalize(cos(aa) * T + sin(aa) * Bu);
    let anb = cross(cross(at, V), at);
    let alv = sat(sample_anisotropy_level(fu).r);
    let ns_ = normalize(mix(n, select(n, normalize(anb), length(anb) > 1e-4), alv * 0.8));
    var R = reflect(-V, ns_);
    let jit = (hash22(c, p.seed_ + 3.0) - 0.5) * p.specular_dithering * rough * 0.3;
    R = normalize(R + T * jit.x + Bu * jit.y);
    let ab = env_brdf(rough, nov);
    var spec = env(R, sqrt(rough) * (LEVELS - 1.0)) * (f0 * ab.x + ab.y);
    spec *= mix(1.0, ao, p.specular_shadow_intensity);
    var col = diffuse + spec + emis;
    // Clear coat.
    if (p.enable_clear_coat != 0) {
        let cn = select(N, n, p.inherit_normal_from_base_layer != 0);
        let cnov = max(dot(cn, V), 1e-3);
        let cf0 = 0.08 * p.clear_coat_specular_level;
        let cab = env_brdf(0.05, cnov);
        let fc = (cf0 * cab.x + cab.y) * p.clear_coat_weight;
        col = col * (1.0 - fc) + env(reflect(-V, cn), 0.3) * fc;
    }
    // Opacity.
    let op = sat(sample_opacity_mask(fu).r);
    var cov = 1.0;
    if (p.opacity_mode == 0) {
        if (hash21(c, p.seed_ + 7.0) > op) { col = bg; cov = 0.0; }
    } else {
        col = mix(bg, col, op);
    }
    store_hdr(c, vec4f(col, h.t));
    store_raw_irradiance(c, vec4f(irr, 1.0));
    store_raw_specular(c, vec4f(spec, 1.0));
    store_normal_world_space(c, vec4f(n * 0.5 + 0.5, 1.0));
    store_normal_tangent_space(c, vec4f(n_enc(nts, dxf), 1.0));
    store_uvs(c, vec4f(fu, cov, 1.0));
}
