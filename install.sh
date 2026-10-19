#!/bin/sh
# Install or update Trafilatura 3D from this folder (macOS, Linux): a virtual environment in .venv, the editor
# installed in it, a Desktop shortcut, and .trafilatura3d / .trafilatura3dpaint files opening with a double click.
# Run it again after a `git pull` to update. Another Python: PYTHON=/path/to/python3 ./install.sh
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    echo "Trafilatura 3D needs Python 3.10 or newer: $PY is missing or older (set PYTHON to another one)." >&2
    exit 1
fi
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip --version >/dev/null 2>&1 || .venv/bin/python -m ensurepip --upgrade  # a venv made without pip
.venv/bin/python -m pip install -e ".[editor]" resvg-py
.venv/bin/trafilatura3d shortcut
echo "Trafilatura 3D is installed: start it from the Desktop shortcut, or with .venv/bin/trafilatura3d-editor."
