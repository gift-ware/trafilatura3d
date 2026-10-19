@echo off
rem Install or update Trafilatura 3D from this folder (Windows): a virtual environment in .venv, the editor installed
rem in it, a Desktop shortcut, and .trafilatura3d / .trafilatura3dpaint files opening with a double click.
rem Run it again after a `git pull` to update. Another Python: set PYTHON=C:\path\to\python.exe before running it.
setlocal
cd /d "%~dp0"
if defined PYTHON (set "PY=%PYTHON%") else (where py >nul 2>nul && (set "PY=py -3") || (set "PY=python"))

%PY% -c "import sys; sys.exit(sys.version_info < (3, 10))" 2>nul
if errorlevel 1 (
    echo Trafilatura 3D needs Python 3.10 or newer: %PY% is missing or older ^(set PYTHON to another one^).
    exit /b 1
)
if not exist .venv (%PY% -m venv .venv || exit /b 1)
rem a venv made without pip gets it
.venv\Scripts\python -m pip --version >nul 2>nul || (.venv\Scripts\python -m ensurepip --upgrade || exit /b 1)
.venv\Scripts\python -m pip install -e ".[editor]" resvg-py || exit /b 1
.venv\Scripts\trafilatura3d shortcut || exit /b 1
echo Trafilatura 3D is installed: start it from the Desktop shortcut, or with .venv\Scripts\trafilatura3d-editor.
