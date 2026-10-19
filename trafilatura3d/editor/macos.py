"""macOS: the application's name in the menu bar and the Dock. Started from Python, the process's main
bundle is Python's, so macOS shows "Python"; renaming it in the bundle's info dictionary before Qt starts
Cocoa shows "Trafilatura 3D" instead. Through the Objective-C runtime with ctypes: no extra dependency.

And the system allocator's cache of freed large blocks, turned off (`without_large_cache`)."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys

LARGE_CACHE = "MallocLargeCache"  # libmalloc's setting, read when a process starts


def without_large_cache() -> None:
    """The process started again with the allocator's cache of freed large blocks off. The cache keeps every
    large temporary freed (images loaded and converted, textures read back) for reuse, and it counts in the
    process's memory: a project of 4096 tiles held 2 GB more. Read only at a process's start: the process is
    replaced by a new one with the setting (once: the new one has it)."""
    if sys.platform != "darwin" or os.environ.get(LARGE_CACHE) == "0":
        return
    env = {**os.environ, LARGE_CACHE: "0"}
    args = list(sys.argv) if getattr(sys, "frozen", False) else [sys.executable, *sys.argv]
    os.execve(sys.executable, args, env)


def set_app_name(name: str) -> bool:
    """CFBundleName of the running process's main bundle; call before QApplication. True when set."""
    if sys.platform != "darwin":
        return False
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
    except OSError:
        return False
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    def send(receiver, selector, *args, restype=ctypes.c_void_p, argtypes=()):
        # objc_msgSend needs the exact prototype of each call (arm64 doesn't pass variadics in registers).
        fn = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(("objc_msgSend", objc))
        return fn(receiver, objc.sel_registerName(selector), *args)

    def nsstring(text):
        return send(objc.objc_getClass(b"NSString"), b"stringWithUTF8String:", text.encode(),
                    argtypes=(ctypes.c_char_p,))

    bundle = send(objc.objc_getClass(b"NSBundle"), b"mainBundle")
    info = send(bundle, b"infoDictionary") if bundle else None
    mutable = objc.objc_getClass(b"NSMutableDictionary")
    if not info or not send(info, b"isKindOfClass:", mutable, restype=ctypes.c_bool, argtypes=(ctypes.c_void_p,)):
        return False
    send(info, b"setObject:forKey:", nsstring(name), nsstring("CFBundleName"), restype=None,
         argtypes=(ctypes.c_void_p, ctypes.c_void_p))
    return True
