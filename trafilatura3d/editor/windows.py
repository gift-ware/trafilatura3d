"""Windows: the application's own taskbar button. Started from Python, the process belongs to python.exe and the
taskbar groups it there, with Python's icon; an explicit AppUserModelID set before any window exists makes it an
application of its own, shown with the windows' icon (learn.microsoft.com/windows/win32/shell/appids). Through
ctypes: no extra dependency."""

from __future__ import annotations

import sys

APP_ID = "Trafilatura3D.Editor"  # CompanyName.ProductName, no spaces (the documented form)


def set_app_id(app_id: str = APP_ID) -> bool:
    """SetCurrentProcessExplicitAppUserModelID; call before QApplication. True when set."""
    if sys.platform != "win32":
        return False
    import ctypes
    try:
        return ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(app_id)) == 0
    except (AttributeError, OSError):
        return False
