// Whether Pattern k (0-7) of a Shape Splatter node is connected.
fn ss_connected(k: i32) -> bool {
    switch k {
        case 0: { return PATTERN_1_CONNECTED; }
        case 1: { return PATTERN_2_CONNECTED; }
        case 2: { return PATTERN_3_CONNECTED; }
        case 3: { return PATTERN_4_CONNECTED; }
        case 4: { return PATTERN_5_CONNECTED; }
        case 5: { return PATTERN_6_CONNECTED; }
        case 6: { return PATTERN_7_CONNECTED; }
        case 7: { return PATTERN_8_CONNECTED; }
        default: { return false; }
    }
}
