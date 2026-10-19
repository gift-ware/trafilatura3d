/* A flat C API over ufbx (https://github.com/ufbx/ufbx, MIT) for trafilatura3d/fbx.py (ctypes): a scene loaded
 * in meters, right handed, Y up; every mesh instance (Catmull-Clark subdivided first when asked, on its own
 * polygons, with the file's smooth mesh boundary rules and new smooth normals) triangulated into per corner
 * arrays in world space; the scene's materials and perspective cameras. Built with ufbx.c by
 * tools/build_ufbx.py. */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ufbx.h"

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

typedef struct {
    const char *name;
    int visible;
    uint32_t corners;  /* 3 per triangle */
    int has_uv, has_color;
    float *pos, *nrm, *uv, *color;
    uint32_t *material;  /* per triangle, into the scene's materials */
    ufbx_mesh *subdivided;  /* owned: the subdivided copy the arrays come from, or NULL */
} tf_mesh;

typedef struct {
    ufbx_scene *scene;
    tf_mesh *meshes;
    size_t mesh_count;
} tf_scene;

static void store(float *dst, ufbx_vec3 v) {
    dst[0] = (float)v.x;
    dst[1] = (float)v.y;
    dst[2] = (float)v.z;
}

static int build_mesh(tf_mesh *out, ufbx_node *node, ufbx_mesh *mesh) {
    size_t tris = mesh->num_triangles, k = 0;
    uint32_t *indices = (uint32_t *)malloc(sizeof(uint32_t) * mesh->max_face_triangles * 3);
    ufbx_matrix normal_matrix = ufbx_get_compatible_matrix_for_normals(node);
    out->name = node->name.data;
    out->visible = node->visible;
    out->has_uv = mesh->vertex_uv.exists;
    out->has_color = mesh->vertex_color.exists;
    out->pos = (float *)malloc(sizeof(float) * tris * 9);
    out->nrm = (float *)malloc(sizeof(float) * tris * 9);
    out->uv = (float *)malloc(sizeof(float) * tris * 6);
    out->color = (float *)malloc(sizeof(float) * tris * 12);
    out->material = (uint32_t *)malloc(sizeof(uint32_t) * tris);
    if (!indices || !out->pos || !out->nrm || !out->uv || !out->color || !out->material) {
        free(indices);
        return 0;
    }
    for (size_t f = 0; f < mesh->num_faces; f++) {
        ufbx_face face = mesh->faces.data[f];
        uint32_t n = ufbx_triangulate_face(indices, mesh->max_face_triangles * 3, mesh, face);
        ufbx_material *mat = NULL;
        if (mesh->face_material.count > f) {
            uint32_t m = mesh->face_material.data[f];
            if (m < node->materials.count) mat = node->materials.data[m];
            else if (m < mesh->materials.count) mat = mesh->materials.data[m];
        }
        for (uint32_t t = 0; t < n; t++) {
            out->material[k / 3] = mat ? mat->typed_id : UINT32_MAX;
            for (uint32_t c = 0; c < 3; c++, k++) {
                uint32_t i = indices[t * 3 + c];
                store(out->pos + k * 3, ufbx_transform_position(&node->geometry_to_world,
                                                                 ufbx_get_vertex_vec3(&mesh->vertex_position, i)));
                ufbx_vec3 nv = mesh->vertex_normal.exists ? ufbx_get_vertex_vec3(&mesh->vertex_normal, i)
                                                          : (ufbx_vec3){0, 0, 0};
                store(out->nrm + k * 3, ufbx_transform_direction(&normal_matrix, nv));
                if (out->has_uv) {
                    ufbx_vec2 uv = ufbx_get_vertex_vec2(&mesh->vertex_uv, i);
                    out->uv[k * 2] = (float)uv.x;
                    out->uv[k * 2 + 1] = (float)uv.y;
                } else {
                    out->uv[k * 2] = out->uv[k * 2 + 1] = 0.0f;
                }
                ufbx_vec4 col = out->has_color ? ufbx_get_vertex_vec4(&mesh->vertex_color, i) : (ufbx_vec4){1, 1, 1, 1};
                out->color[k * 4] = (float)col.x;
                out->color[k * 4 + 1] = (float)col.y;
                out->color[k * 4 + 2] = (float)col.z;
                out->color[k * 4 + 3] = (float)col.w;
            }
        }
    }
    out->corners = (uint32_t)k;
    free(indices);
    return 1;
}

API void tf_free(tf_scene *s) {
    if (!s) return;
    for (size_t i = 0; i < s->mesh_count; i++) {
        free(s->meshes[i].pos);
        free(s->meshes[i].nrm);
        free(s->meshes[i].uv);
        free(s->meshes[i].color);
        free(s->meshes[i].material);
        if (s->meshes[i].subdivided) ufbx_free_mesh(s->meshes[i].subdivided);
    }
    free(s->meshes);
    ufbx_free_scene(s->scene);
    free(s);
}

/* `subdivide`: Catmull-Clark levels applied to every mesh before it is triangulated (0: none). */
API tf_scene *tf_load(const char *path, uint32_t subdivide, char *error, size_t error_size) {
    ufbx_load_opts opts = {0};
    ufbx_error err;
    opts.target_axes = ufbx_axes_right_handed_y_up;
    opts.target_camera_axes = ufbx_axes_right_handed_y_up;  /* cameras look down -Z */
    opts.target_unit_meters = 1.0f;
    opts.space_conversion = UFBX_SPACE_CONVERSION_MODIFY_GEOMETRY;
    opts.generate_missing_normals = true;
    opts.ignore_animation = true;
    ufbx_scene *scene = ufbx_load_file(path, &opts, &err);
    if (!scene) {
        ufbx_format_error(error, error_size, &err);
        return NULL;
    }
    tf_scene *s = (tf_scene *)calloc(1, sizeof(tf_scene));
    s->scene = scene;
    s->meshes = (tf_mesh *)calloc(scene->nodes.count ? scene->nodes.count : 1, sizeof(tf_mesh));
    for (size_t i = 0; i < scene->nodes.count; i++) {
        ufbx_node *node = scene->nodes.data[i];
        if (!node->mesh || node->mesh->num_triangles == 0 || !node->mesh->vertex_position.exists) continue;
        tf_mesh *out = &s->meshes[s->mesh_count++];
        ufbx_mesh *mesh = node->mesh;
        if (subdivide > 0) {
            ufbx_subdivide_opts sopts = {0};  /* the file's boundary rules, new smooth normals */
            mesh = out->subdivided = ufbx_subdivide_mesh(node->mesh, subdivide, &sopts, &err);
            if (!mesh) {
                ufbx_format_error(error, error_size, &err);
                tf_free(s);
                return NULL;
            }
        }
        if (!build_mesh(out, node, mesh)) {
            strncpy(error, "out of memory", error_size);
            tf_free(s);
            return NULL;
        }
    }
    return s;
}

API size_t tf_mesh_count(const tf_scene *s) { return s->mesh_count; }
API const char *tf_mesh_name(const tf_scene *s, size_t i) { return s->meshes[i].name; }
API int tf_mesh_visible(const tf_scene *s, size_t i) { return s->meshes[i].visible; }
API int tf_mesh_has_uv(const tf_scene *s, size_t i) { return s->meshes[i].has_uv; }
API int tf_mesh_has_color(const tf_scene *s, size_t i) { return s->meshes[i].has_color; }

/* Per corner RGBA vertex colors (white when the mesh has none). */
API void tf_mesh_colors(const tf_scene *s, size_t i, float *color) {
    memcpy(color, s->meshes[i].color, sizeof(float) * s->meshes[i].corners * 4);
}

/* A material's base color (PBR base color, else FBX diffuse color, else white). */
API void tf_material_color(const tf_scene *s, size_t i, float *out) {
    const ufbx_material *m = s->scene->materials.data[i];
    ufbx_vec4 c = {1, 1, 1, 1};
    if (m->pbr.base_color.has_value) c = m->pbr.base_color.value_vec4;
    else if (m->fbx.diffuse_color.has_value) c = (ufbx_vec4){m->fbx.diffuse_color.value_vec3.x,
                                                             m->fbx.diffuse_color.value_vec3.y,
                                                             m->fbx.diffuse_color.value_vec3.z, 1};
    out[0] = (float)c.x; out[1] = (float)c.y; out[2] = (float)c.z; out[3] = (float)c.w;
}
API uint32_t tf_mesh_corners(const tf_scene *s, size_t i) { return s->meshes[i].corners; }

API void tf_mesh_data(const tf_scene *s, size_t i, float *pos, float *nrm, float *uv, uint32_t *material) {
    const tf_mesh *m = &s->meshes[i];
    memcpy(pos, m->pos, sizeof(float) * m->corners * 3);
    memcpy(nrm, m->nrm, sizeof(float) * m->corners * 3);
    memcpy(uv, m->uv, sizeof(float) * m->corners * 2);
    memcpy(material, m->material, sizeof(uint32_t) * (m->corners / 3));
}

API size_t tf_material_count(const tf_scene *s) { return s->scene->materials.count; }
API const char *tf_material_name(const tf_scene *s, size_t i) { return s->scene->materials.data[i]->name.data; }

/* Perspective cameras: out = eye xyz, forward xyz, vertical field of view in degrees. */
API size_t tf_camera_count(const tf_scene *s) {
    size_t n = 0;
    for (size_t i = 0; i < s->scene->nodes.count; i++) {
        ufbx_node *node = s->scene->nodes.data[i];
        if (node->camera && node->camera->projection_mode == UFBX_PROJECTION_MODE_PERSPECTIVE) n++;
    }
    return n;
}

API const char *tf_camera(const tf_scene *s, size_t index, float *out) {
    size_t n = 0;
    for (size_t i = 0; i < s->scene->nodes.count; i++) {
        ufbx_node *node = s->scene->nodes.data[i];
        if (!node->camera || node->camera->projection_mode != UFBX_PROJECTION_MODE_PERSPECTIVE) continue;
        if (n++ != index) continue;
        store(out, ufbx_transform_position(&node->node_to_world, (ufbx_vec3){0, 0, 0}));
        store(out + 3, ufbx_transform_direction(&node->node_to_world, (ufbx_vec3){0, 0, -1}));
        out[6] = (float)node->camera->field_of_view_deg.y;
        return node->name.data;
    }
    return NULL;
}
