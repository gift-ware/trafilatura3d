// Where water accumulates: everywhere, or the occluded areas (baked ambient occlusion) with AO Based Intensity.
fn fx(c: vec2i) {
    let occ = mx_sat((1.0 - load_baked_ambient_occlusion(c).r) * 2.5);
    store_water(c, vec4f(vec3f(mix(1.0, occ, p.ao_based_intensity)), 1.0));
}
