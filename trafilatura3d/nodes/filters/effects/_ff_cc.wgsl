// Flood fill shapes by union-find on buffers. `parent`, per pixel (row-major index): a pixel of its shape with a
// smaller index, after the passes the shape's first pixel (its root); FF_NONE on black. Once every shape is one tree,
// each root takes a slot (FF_ROOT | slot in its own entry, ff_shape) in `box`, FF_BOX values per shape: the sectors
// of columns and of rows the shape covers (bit masks), then its min x, max x, min y, max y in the frame starting at a
// sector it does not cover (no shape crosses that seam, so the extents are its own).
const FF_NONE = 0xffffffffu;
const FF_ROOT = 0x80000000u;
const FF_BOX = 6u;

fn ff_index(q: vec2i) -> u32 { return u32(q.y) * u32(size().x) + u32(q.x); }
fn ff_pixel(i: u32) -> vec2i { let w = u32(size().x); return vec2i(i32(i % w), i32(i / w)); }

// A white pixel's shape once the roots have their slots: its root's index (x) and the first of its values in `box`.
fn ff_shape(i: u32) -> vec2u {
    let up = atomicLoad(&parent[i]);
    if ((up & FF_ROOT) != 0u) { return vec2u(i, (up & ~FF_ROOT) * FF_BOX); }
    return vec2u(up, (atomicLoad(&parent[up]) & ~FF_ROOT) * FF_BOX);
}

// The pixel one step `o` from c, across the tiled edges; -1 outside the image.
fn ff_step(c: vec2i, o: vec2i) -> i32 {
    let d = size();
    var q = c + o;
    if (q.x < 0 || q.x >= d.x) { if (!tile_h()) { return -1; } q.x = (q.x + d.x) % d.x; }
    if (q.y < 0 || q.y >= d.y) { if (!tile_v()) { return -1; } q.y = (q.y + d.y) % d.y; }
    return i32(ff_index(q));
}
fn ff_white(i: i32) -> bool { return i >= 0 && atomicLoad(&parent[u32(i)]) != FF_NONE; }

fn ff_find(i: u32) -> u32 {
    var x = i;
    var up = atomicLoad(&parent[x]);
    while (up != x) {
        x = up;
        up = atomicLoad(&parent[x]);
    }
    return x;
}

// Joins the shapes of pixels i and j: the larger root points to the smaller (atomicMin: a concurrent join that got
// there first is followed and the loop goes on from its root).
fn ff_union(i: u32, j: u32) {
    var a = i;
    var b = j;
    loop {
        a = ff_find(a);
        b = ff_find(b);
        if (a == b) { return; }
        if (a < b) {
            let old = atomicMin(&parent[b], a);
            if (old == b) { return; }
            b = old;
        } else {
            let old = atomicMin(&parent[a], b);
            if (old == a) { return; }
            a = old;
        }
    }
}

// Sectors along an axis of n pixels: 32 (fewer on a smaller image: each holds a pixel at least).
fn ff_sectors(n: i32) -> i32 { return min(32, n); }
fn ff_sector(x: i32, n: i32) -> u32 { return u32(x * ff_sectors(n) / n); }
// The first pixel of the first sector `mask` leaves empty, where the shape's frame starts; 0 on an axis that does
// not tile (nothing crosses its edge) or when the shape covers every sector (it goes round the image).
fn ff_seam(mask: u32, n: i32, tiled: bool) -> i32 {
    let k = i32(countTrailingZeros(~mask));
    if (!tiled || k >= ff_sectors(n)) { return 0; }
    return (k * n + ff_sectors(n) - 1) / ff_sectors(n);
}
