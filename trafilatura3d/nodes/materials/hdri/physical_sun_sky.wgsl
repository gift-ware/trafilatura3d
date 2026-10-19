fn perez(th: f32, g: f32, a: f32, b: f32, cc: f32, dd: f32, e: f32) -> f32 {
    let ct = max(cos(th), 0.01);
    return (1.0 + a * exp(b / ct)) * (1.0 + cc * exp(dd * g) + e * cos(g) * cos(g));
}

// Preetham sky radiance (linear sRGB, ~kcd/m2 / 10) towards a direction above the horizon.
fn sky(dir: vec3f, sun: vec3f, t: f32) -> vec3f {
    let ths = acos(clamp(sun.y, -1.0, 1.0));
    let th = acos(clamp(dir.y, 0.0, 1.0));
    let g = acos(clamp(dot(dir, sun), -1.0, 1.0));
    let ts = min(ths, 1.55);
    let chi = (4.0 / 9.0 - t / 120.0) * (PI - 2.0 * ts);
    let yz = max((4.0453 * t - 4.9710) * tan(chi) - 0.2155 * t + 2.4192, 0.0);
    let t2 = t * t;
    let s3 = ts * ts * ts;
    let s2 = ts * ts;
    let xz = t2 * (0.00166 * s3 - 0.00375 * s2 + 0.00209 * ts) + t * (-0.02903 * s3 + 0.06377 * s2 - 0.03202 * ts + 0.00394)
        + (0.11693 * s3 - 0.21196 * s2 + 0.06052 * ts + 0.25886);
    let yzc = t2 * (0.00275 * s3 - 0.00610 * s2 + 0.00317 * ts) + t * (-0.04214 * s3 + 0.08970 * s2 - 0.04153 * ts + 0.00516)
        + (0.15346 * s3 - 0.26756 * s2 + 0.06670 * ts + 0.26688);
    let fy = perez(th, g, 0.1787 * t - 1.4630, -0.3554 * t + 0.4275, -0.0227 * t + 5.3251, 0.1206 * t - 2.5771, -0.0670 * t + 0.3703)
        / perez(0.0, ts, 0.1787 * t - 1.4630, -0.3554 * t + 0.4275, -0.0227 * t + 5.3251, 0.1206 * t - 2.5771, -0.0670 * t + 0.3703);
    let fx = perez(th, g, -0.0193 * t - 0.2592, -0.0665 * t + 0.0008, -0.0004 * t + 0.2125, -0.0641 * t - 0.8989, -0.0033 * t + 0.0452)
        / perez(0.0, ts, -0.0193 * t - 0.2592, -0.0665 * t + 0.0008, -0.0004 * t + 0.2125, -0.0641 * t - 0.8989, -0.0033 * t + 0.0452);
    let fyc = perez(th, g, -0.0167 * t - 0.2608, -0.0950 * t + 0.0092, -0.0079 * t + 0.2102, -0.0441 * t - 1.6537, -0.0109 * t + 0.0529)
        / perez(0.0, ts, -0.0167 * t - 0.2608, -0.0950 * t + 0.0092, -0.0079 * t + 0.2102, -0.0441 * t - 1.6537, -0.0109 * t + 0.0529);
    let lum = yz * fy * 0.1;
    let x = xz * fx;
    let y = max(yzc * fyc, 1e-4);
    let big_x = x / y * lum;
    let big_z = (1.0 - x - y) / y * lum;
    let rgb = mat3x3f(vec3f(3.2406, -0.9689, 0.0557), vec3f(-1.5372, 1.8758, -0.2040), vec3f(-0.4986, 0.0415, 1.0570))
        * vec3f(big_x, lum, big_z);
    // The sun under the horizon: the sky fades to night.
    return max(rgb, vec3f(0.0)) * smoothstep(-0.1, 0.05, sun.y);
}

// Transmittance of sunlight through the atmosphere along an elevation.
fn sun_tint(elev: f32, t: f32) -> vec3f {
    let air = 1.0 / max(sin(max(elev, 0.0)) + 0.15 * pow(max(elev * 57.3 + 3.885, 0.01), -1.253), 0.03);
    return exp(-air * (0.02 + 0.012 * t) * vec3f(0.6, 1.2, 2.6));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let dir = ll_dir(uv_of(c, d));
    let sun = ll_dir(p.sun_position);
    let t = clamp(p.turbidity, 1.0, 10.0);
    let elev = asin(clamp(sun.y, -1.0, 1.0));
    let zenith = sky(vec3f(0.0, 1.0, 0.0), sun, t);
    // Sky above the horizon, ground albedo lifting its lower part.
    let up = max(dir.y, 0.0);
    var col = sky(vec3f(dir.x, max(dir.y, 0.0), dir.z), sun, t) * (1.0 + p.albedo * 0.5 * (1.0 - up) * (1.0 - up));
    col *= p.sky_intensity;
    // Sun disc (0.27 degrees of radius at Sun Size 1) with limb darkening.
    let rad = 0.0047 * max(p.sun_size, 0.0);
    let ang = acos(clamp(dot(dir, sun), -1.0, 1.0));
    if (rad > 0.0 && ang < rad && dir.y > -0.02) {
        let mu = sqrt(max(1.0 - (ang / rad) * (ang / rad), 0.0));
        col += sun_tint(elev, t) * 20000.0 * p.sun_intensity * (0.4 + 0.6 * mu);
    }
    // Ground: diffuse ground color lit by the sun disc and the sky dome (irradiance / pi).
    let e_sun = sun_tint(elev, t) * 20000.0 * p.sun_intensity * PI * 0.0047 * 0.0047 * max(sun.y, 0.0);
    let e_sky = PI * zenith * 0.8 * p.sky_intensity;
    let ground = p.ground_color.rgb * (e_sun + e_sky) / PI;
    col = mix(col, ground, smoothstep(0.0, -0.02, dir.y));
    store(c, vec4f(col * exp2(p.exposure), 1.0));
}
