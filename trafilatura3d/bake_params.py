"""The bakers' parameters: names, options and defaults. Defaults of our own choice are marked "ours". `when`
lists the conditions for a parameter to apply ("if X is ..."):
(other parameter, allowed values) or (other parameter, "set") for a file path that is given."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

BAKERS = {
    "ambient_occlusion": "Ambient Occlusion", "ambient_occlusion_from_mesh": "Ambient Occlusion from Mesh",
    "bent_normals_from_mesh": "Bent Normals from Mesh", "color_map_from_mesh": "Color Map from Mesh",
    "convert_uv_to_svg": "Convert UV to SVG", "curvature": "Curvature", "curvature_from_mesh": "Curvature from Mesh",
    "height_map_from_mesh": "Height Map from Mesh", "normal_map_from_mesh": "Normal Map from Mesh",
    "opacity_mask_from_mesh": "Opacity Mask from Mesh", "position": "Position",
    "position_map_from_mesh": "Position Map from Mesh", "thickness_map_from_mesh": "Thickness Map from Mesh",
    "transferred_texture_from_mesh": "Transferred Texture from Mesh", "world_space_direction": "World Space Direction",
    "world_space_normals": "World Space Normals",
}
ORIENTATION = ("OpenGL", "DirectX")


@dataclass(frozen=True)
class P:
    label: str
    default: Any
    options: tuple | None = None
    when: tuple = ()  # ((parameter, values or "set"), ...): all must hold
    kind: str = ""  # "file", "vec3" or inferred from the default


# Ours: an edge-aware filter on the secondary rays' noise (bake_wgsl.DENOISE).
DENOISE = P("Denoise", False)


def _occlusion(rays_default=64, backface=("Never", "Always", "By Mesh Name")):
    """The occlusion rays' parameters shared by Ambient Occlusion, Bent Normals and Thickness from Mesh."""
    return {
        "secondary_rays": P("Secondary Rays", rays_default),
        "min_occluder_distance": P("Min Occluder Distance", 0.00001),
        "max_occluder_distance": P("Max Occluder Distance", 0.1),
        "relative_to_bounding_box": P("Relative to Bounding Box", True),  # ours
        "spread_angle": P("Spread Angle", 180.0),
        "distribution": P("Distribution", "Cosine", ("Cosine", "Uniform")),
        "ignore_backface": P("Ignore Backface", "Never", backface) if backface else P("Ignore Backface", False),
        "self_occlusion": P("Self Occlusion", "Always", ("Always", "By Mesh Name")),
        "denoise": DENOISE,
    }


BAKER_PARAMS: dict[str, dict[str, P]] = {
    "ambient_occlusion": {
        "normal_map": P("Normal Map", "", kind="file"),
        "world_space": P("World Space", False, when=(("normal_map", "set"),)),
        "invert_normal": P("Invert Normal", False),
        "quality": P("Quality", "Medium (5 pass)",
                     ("Low (3 pass)", "Medium (5 pass)", "High (10 pass)", "Very High (16 pass)")),
        "precision_bias": P("Precision Bias", 0.001),  # ours (relative to the bounding box)
        "distance_fade": P("Distance Fade", 0.25),  # ours (relative to the bounding box)
        "denoise": DENOISE,
    },
    "ambient_occlusion_from_mesh": {
        **_occlusion(),
        "normal_map": P("Normal Map", "", kind="file"),
        "world_space": P("World Space", False, when=(("normal_map", "set"),)),
        "normal_orientation": P("Normal Orientation", "DirectX", ORIENTATION,
                                when=(("normal_map", "set"), ("world_space", (False,)))),
        "attenuation": P("Attenuation", "Linear", ("None", "Linear", "Smooth")),
        "ground_plane": P("Ground Plane", False),
        "ground_plane_offset": P("Ground Plane Offset", 0.0, when=(("ground_plane", (True,)),)),
    },
    "bent_normals_from_mesh": {
        **_occlusion(backface=None),
        "map_type": P("Map Type", "Tangent space", ("World space", "Tangent space")),
        "normal_orientation": P("Normal Orientation", "DirectX", ORIENTATION, when=(("map_type", ("Tangent space",)),)),
    },
    "color_map_from_mesh": {
        "color_source": P("Color Source", "Mesh ID",  # ours
                          ("Vertex Color", "Material Color", "Mesh ID", "Polygroup/Submesh ID")),
        "color_generator": P("Color Generator", "Random", ("Random", "Hue Shift", "Grayscale"),  # ours
                             when=(("color_source", ("Mesh ID", "Polygroup/Submesh ID")),)),
    },
    "convert_uv_to_svg": {
        "padding": P("Padding", 0.0),  # ours (pixels)
        "color_mode": P("Color Mode", "Random",  # ours
                        ("Random", "HUE Shift", "Grayscale", "Uniform color", "Material ID Color")),
    },
    "curvature": {
        # "Algorithm" is on the page without its options: not offered.
        "details": P("Details", 1.0),  # ours
        "enable_seams": P("Enable Seams", True),  # ours
        "seams_intensity": P("Seams Intensity", 1.0, when=(("enable_seams", (True,)),)),  # ours
    },
    "curvature_from_mesh": {
        "secondary_rays": P("Secondary Rays", 32),
        "sampling_radius": P("Sampling Radius", 0.01),  # ours
        "relative_to_bounding_box": P("Relative To Bounding Box", True),  # ours
        "self_intersection": P("Self Intersection", "Always", ("Always", "By Mesh Name")),
        "denoise": DENOISE,
        "automatic_tonemapping": P("Automatic Tonemapping bounds", True),  # ours
        "tonemapping_min": P("Tonemapping Min", -1.0, when=(("automatic_tonemapping", (False,)),)),  # ours
        "tonemapping_max": P("Tonemapping Max", 1.0, when=(("automatic_tonemapping", (False,)),)),  # ours
        "normal_map": P("Normal Map", "", kind="file"),
        "world_space": P("World Space", False, when=(("normal_map", "set"),)),
        "normal_orientation": P("Normal Orientation", "DirectX", ORIENTATION,
                                when=(("normal_map", "set"), ("world_space", (False,)))),
    },
    "height_map_from_mesh": {
        "normalization": P("Normalization", "Relative to low poly mesh (per UV Tile)",
                           ("Relative to ray distance", "Relative to low poly mesh (per UV Tile)",
                            "Relative to Min/Max (per UV Tile)", "Manual")),
        "scaling_divisor": P("Scaling divisor", 1.0, when=(("normalization", ("Manual",)),)),  # ours
    },
    "normal_map_from_mesh": {
        "map_type": P("Map Type", "Tangent Space", ("World Space", "Tangent Space")),
        "normal_orientation": P("Normal Orientation", "DirectX", ORIENTATION, when=(("map_type", ("Tangent Space",)),)),
    },
    "opacity_mask_from_mesh": {},
    "position": {
        "mode": P("Mode", "All axes", ("All axes", "One axis")),  # ours
        "axis": P("Axis", "X", ("X", "Y", "Z"), when=(("mode", ("One axis",)),)),  # ours (labels not listed)
        "normalization_type": P("Normalization Type", "BBox", ("BBox", "BSphere")),  # ours
        "normalization_scale": P("Normalization Scale", "Full Scene", ("Per Material", "Full Scene")),
    },
    "thickness_map_from_mesh": {
        **_occlusion(backface=None),
        "automatic_normalization": P("Automatic Normalization", False),  # ours
    },
    "transferred_texture_from_mesh": {
        "texture_file": P("Texture File", "", kind="file"),
        "filtering_mode": P("Filtering Mode", "Bilinear", ("Nearest", "Bilinear")),
        "normal_map": P("Normal Map", False),
        "map_type": P("Map Type", "Tangent Space", ("World Space", "Tangent Space"), when=(("normal_map", (True,)),)),
        "normal_orientation": P("Normal Orientation", "DirectX", ORIENTATION,
                                when=(("normal_map", (True,)), ("map_type", ("Tangent Space",)))),
    },
    "world_space_direction": {
        "input_direction": P("Input Direction", "From Uniform Vector", ("From Texture", "From Uniform Vector")),
        "normal_orientation": P("Normal Orientation", "DirectX", ORIENTATION),
        "direction": P("X Y Z", [0.0, 1.0, 0.0], when=(("input_direction", ("From Uniform Vector",)),),  # ours
                       kind="vec3"),
        "direction_file": P("Direction File", "", when=(("input_direction", ("From Texture",)),), kind="file"),
    },
    "world_space_normals": {
        "baking_type": P("Baking Type", "Normal", ("Normal", "Tangent", "Binormal")),
        "normal_map": P("Normal Map", "", kind="file"),
        "normal_orientation": P("Normal Orientation", "DirectX", ORIENTATION,
                                when=(("baking_type", ("Normal",)), ("normal_map", "set"))),
    },
}
BAKER_PARAMS["position_map_from_mesh"] = dict(BAKER_PARAMS["position"])  # same parameters, from the high poly


def baker_defaults(baker: str) -> dict:
    return {k: (list(p.default) if isinstance(p.default, list) else p.default)
            for k, p in BAKER_PARAMS.get(baker, {}).items()}


def resolved(baker: str, params: dict | None) -> dict:
    """The baker's parameters: defaults, then the given values of known parameters."""
    spec = BAKER_PARAMS.get(baker, {})
    return {**baker_defaults(baker), **{k: v for k, v in (params or {}).items() if k in spec}}


def applies(baker: str, name: str, values: dict) -> bool:
    """Whether a parameter applies given the others (the pages' "if X is ...")."""
    for other, allowed in BAKER_PARAMS[baker][name].when:
        v = values.get(other)
        if allowed == "set":
            if not v:
                return False
        elif v not in allowed:
            return False
    return True


def names_match(low: str, high: str, low_suffix: str = "_low", high_suffix: str = "_high") -> bool:
    """Matching by Name (case sensitive): the low name with its suffix replaced by the high one must equal
    the high name, or one must be the other followed by "_..." (body_low ~ body_high_top, Leg_low_top ~
    Leg_high; not body_top_high, nor Head_high for head_low)."""
    key = low.replace(low_suffix, high_suffix, 1) if low_suffix and low_suffix in low else low + high_suffix
    return high == key or high.startswith(key + "_") or key.startswith(high + "_")


# Files named by a bake's configuration: absolute in memory, relative to the document in its file, so a document
# and its meshes move together (a Paint project, a package).
def relative_file(path: str, folder: Path) -> str:
    if not path or not Path(path).is_absolute():
        return path
    try:
        return os.path.relpath(path, folder).replace(os.sep, "/")
    except ValueError:  # another drive (Windows): kept absolute
        return path


def absolute_file(path: str, folder: Path) -> str:
    return path if not path or Path(path).is_absolute() else str((folder / path).resolve())


def map_setting_files(settings: dict, change: Callable[[str], str]) -> None:
    """The mesh files of common settings (Settings as a dict: high definition meshes, cage) through `change`."""
    if "high_meshes" in settings:
        settings["high_meshes"] = [change(p) for p in settings["high_meshes"]]
    if "cage_file" in settings:
        settings["cage_file"] = change(settings["cage_file"])


def map_param_files(baker: str, params: dict, change: Callable[[str], str]) -> None:
    """A baker's file parameters (an input normal map, a texture...) through `change`."""
    for name, p in BAKER_PARAMS.get(baker, {}).items():
        if p.kind == "file" and params.get(name):
            params[name] = change(params[name])
