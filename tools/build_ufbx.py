"""Build ufbx (https://github.com/ufbx/ufbx, MIT) with trafilatura3d/native/ufbx_shim.c into a shared library
for this platform, from a pinned official release, into trafilatura3d/native/<platform>/ with its license:

    python3 tools/build_ufbx.py

macOS: clang (Xcode command line tools). Windows: run it from a "x64 Native Tools Command Prompt"
(MSVC's cl.exe). trafilatura3d/fbx.py loads the library through ctypes. Binaries are versioned in the depot,
so only whoever updates VERSION builds."""

from __future__ import annotations

import io
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

VERSION = "v0.23.1"  # official release tag
SOURCE = f"https://github.com/ufbx/ufbx/archive/refs/tags/{VERSION}.tar.gz"
NATIVE = Path(__file__).resolve().parent.parent / "trafilatura3d" / "native"


def platform_dir() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        return "macos-" + ("arm64" if machine == "arm64" else "x86_64")
    if sys.platform == "win32":
        return "windows-x64"
    return "linux-" + machine


def library_name() -> str:
    return {"darwin": "libtrafilatura3d_ufbx.dylib", "win32": "trafilatura3d_ufbx.dll"}.get(sys.platform, "libtrafilatura3d_ufbx.so")


def main() -> int:
    out = NATIVE / platform_dir()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        print(f"downloading ufbx {VERSION}")
        data = urllib.request.urlopen(SOURCE, timeout=60).read()
        tarfile.open(fileobj=io.BytesIO(data), mode="r:gz").extractall(tmp, filter="data")
        root = next(Path(tmp).iterdir())
        sources = [str(root / "ufbx.c"), str(NATIVE / "ufbx_shim.c")]
        target = out / library_name()
        if sys.platform == "win32":
            cmd = ["cl", "/nologo", "/O2", "/LD", f"/I{root}", *sources, f"/Fe{target}"]
            subprocess.run(cmd, check=True, cwd=tmp)
        else:
            cmd = ["clang" if sys.platform == "darwin" else "cc", "-O2", "-shared", "-fPIC", "-std=c99",
                   "-fvisibility=hidden", f"-I{root}", *sources, "-o", str(target), "-lm"]
            if sys.platform == "darwin":
                cmd += ["-install_name", "@rpath/" + library_name()]
            subprocess.run(cmd, check=True)
        shutil.copyfile(root / "LICENSE", out / "LICENSE-ufbx.txt")
        (out / "VERSION-ufbx.txt").write_text(f"{VERSION}\n{SOURCE}\n", encoding="utf-8")
    print(f"built {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
