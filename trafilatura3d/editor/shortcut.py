"""`trafilatura3d shortcut`: a Desktop shortcut that starts the installed editor, with the application's icon, and
the editor made the application of its documents (.trafilatura3d packages, .trafilatura3dpaint projects: a double
click opens them). Windows: a .lnk (through PowerShell's WScript.Shell), the extensions in the user's registry
classes; macOS: a small "Trafilatura 3D.app" whose executable starts the editor (its icon an .icns made with
iconutil), declaring the document types, registered with Launch Services; Linux: a .desktop launcher, the MIME types
and the launcher in the user's data folder. The icon files are written in the user folder (project.user_dir), drawn
by icons.app_icon."""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

NAME = "Trafilatura 3D"
BUNDLE_ID = "io.github.gift-ware.trafilatura3d"


def documents() -> list[tuple[str, str, str]]:
    """(suffix, description, id) of the editor's documents."""
    from trafilatura3d.package import SUFFIX
    from trafilatura3d.paint.model import PROJECT_SUFFIX
    return [(SUFFIX, f"{NAME} package", "package"), (PROJECT_SUFFIX, f"{NAME} Paint project", "paint")]


def editor_command() -> Path:
    """The installed `trafilatura3d-editor` of this environment."""
    exe = Path(sysconfig.get_path("scripts")) / ("trafilatura3d-editor.exe" if os.name == "nt" else "trafilatura3d-editor")
    if not exe.exists():
        raise FileNotFoundError(f"{exe} not found: install the editor (pip install \"trafilatura3d[editor]\")")
    return exe


def _icon_image(size: int):
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window: only the drawing
        _icon_image.app = QGuiApplication([sys.argv[0]])  # kept alive with the function
    from trafilatura3d.editor.icons import app_icon
    return app_icon(size).pixmap(size, size).toImage()


def _save(image, path: Path) -> None:
    if not image.save(str(path)):
        raise OSError(f"could not write {path}")


def desktop() -> Path:
    """The user's Desktop folder (Windows: the shell's, which may be redirected; Linux: xdg-user-dir's)."""
    if os.name == "nt":
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return Path(out)
    if sys.platform != "darwin" and shutil.which("xdg-user-dir"):
        out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True).stdout.strip()
        if out:
            return Path(out)
    return Path.home() / "Desktop"


def create(where: Path | None = None) -> Path:
    """The shortcut in `where` (default the Desktop), replacing one there; returns its path."""
    from trafilatura3d.project import user_dir
    where = Path(where) if where else desktop()
    exe = editor_command()
    icons = user_dir() / "icons"
    icons.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        return _windows(where, exe, icons)
    if sys.platform == "darwin":
        return _macos(where, exe)
    return _linux(where, exe, icons)


def _windows(where: Path, exe: Path, icons: Path) -> Path:
    ico = icons / "trafilatura3d.ico"
    _save(_icon_image(256), ico)
    link = where / f"{NAME}.lnk"
    script = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:T3D_LINK); "
              "$s.TargetPath = $env:T3D_EXE; $s.IconLocation = $env:T3D_ICON; "
              "$s.WorkingDirectory = $env:USERPROFILE; $s.Description = $env:T3D_NAME; $s.Save()")
    env = {**os.environ, "T3D_LINK": str(link), "T3D_EXE": str(exe), "T3D_ICON": str(ico), "T3D_NAME": NAME}
    subprocess.run(["powershell", "-NoProfile", "-Command", script], env=env, check=True)  # paths through the
    import winreg                                                                          # environment: no quoting
    for suffix, description, kind in documents():  # HKEY_CURRENT_USER: no administrator rights
        prog_id = f"Trafilatura3D.{kind}"
        winreg.SetValue(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{suffix}", winreg.REG_SZ, prog_id)
        winreg.SetValue(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{prog_id}", winreg.REG_SZ, description)
        winreg.SetValue(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{prog_id}\DefaultIcon", winreg.REG_SZ, str(ico))
        winreg.SetValue(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{prog_id}\shell\open\command",
                        winreg.REG_SZ, f'"{exe}" "%1"')
    import ctypes
    ctypes.windll.shell32.SHChangeNotify(0x08000000, 0, None, None)  # SHCNE_ASSOCCHANGED: Explorer picks them up
    return link


def _macos(where: Path, exe: Path) -> Path:
    app = where / f"{NAME}.app"
    if app.exists():
        shutil.rmtree(app)
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Resources").mkdir()
    with tempfile.TemporaryDirectory() as tmp:  # an .iconset of the sizes iconutil expects, then the .icns
        iconset = Path(tmp) / "trafilatura3d.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            _save(_icon_image(size), iconset / f"icon_{size}x{size}.png")
            _save(_icon_image(size * 2), iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(contents / "Resources" / "trafilatura3d.icns")],
                       check=True)
    launcher = contents / "MacOS" / "trafilatura3d"
    launcher.write_text(f'#!/bin/sh\nexec "{exe}" "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    types = [{"UTTypeIdentifier": f"{BUNDLE_ID}.{kind}", "UTTypeDescription": description,
              "UTTypeConformsTo": ["public.data", "public.content"],
              "UTTypeTagSpecification": {"public.filename-extension": [suffix.lstrip(".")]}}
             for suffix, description, kind in documents()]
    roles = [{"CFBundleTypeName": t["UTTypeDescription"], "CFBundleTypeRole": "Editor", "LSHandlerRank": "Owner",
              "LSItemContentTypes": [t["UTTypeIdentifier"]]} for t in types]
    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump({"CFBundleName": NAME, "CFBundleDisplayName": NAME, "CFBundleIdentifier": BUNDLE_ID,
                       "CFBundleExecutable": "trafilatura3d", "CFBundleIconFile": "trafilatura3d",
                       "CFBundlePackageType": "APPL", "CFBundleInfoDictionaryVersion": "6.0",
                       "NSHighResolutionCapable": True, "UTExportedTypeDeclarations": types,
                       "CFBundleDocumentTypes": roles,
                       # a script executable is otherwise started under Rosetta on Apple silicon: `uname -p` then says
                       # i386 and rubicon-objc (wgpu's Metal surface) looks for Intel-only symbols at start
                       "LSRequiresNativeExecution": True, "LSArchitecturePriority": ["arm64", "x86_64"]}, f)
    lsregister = Path("/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework"
                      "/Support/lsregister")
    subprocess.run([str(lsregister), "-f", str(app)], check=True)  # Finder knows the documents now, not at a reboot
    return app


def _linux(where: Path, exe: Path, icons: Path) -> Path:
    png = icons / "trafilatura3d.png"
    _save(_icon_image(256), png)
    mimes = [(f"application/x-trafilatura3d-{kind}", suffix, description) for suffix, description, kind in documents()]
    text = (f"[Desktop Entry]\nType=Application\nName={NAME}\nExec=\"{exe}\" %F\nIcon={png}\n"
            f"Terminal=false\nCategories=Graphics;3DGraphics;\nMimeType={';'.join(m for m, _, _ in mimes)};\n")
    entry = where / "trafilatura3d.desktop"
    entry.write_text(text, encoding="utf-8")
    entry.chmod(0o755)
    data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    (data / "mime" / "packages").mkdir(parents=True, exist_ok=True)
    (data / "applications").mkdir(parents=True, exist_ok=True)
    types = "".join(f'  <mime-type type="{m}"><comment>{d}</comment><glob pattern="*{s}"/></mime-type>\n'
                    for m, s, d in mimes)
    (data / "mime" / "packages" / "trafilatura3d.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">\n{types}</mime-info>\n',
        encoding="utf-8")
    (data / "applications" / "trafilatura3d.desktop").write_text(text, encoding="utf-8")
    for command in (["update-mime-database", str(data / "mime")],
                    ["update-desktop-database", str(data / "applications")],
                    *(["xdg-mime", "default", "trafilatura3d.desktop", m] for m, _, _ in mimes)):
        if shutil.which(command[0]):  # each one optional: the files alone work after the next login
            subprocess.run(command, capture_output=True)
    return entry
