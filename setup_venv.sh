#!/bin/bash
# Creates .venv/ with the minimal packages needed for visualization (host-side, no Docker).
set -e

python3 -m venv cgn_visualize_venv
cgn_visualize_venv/bin/pip install --upgrade pip
cgn_visualize_venv/bin/pip install open3d numpy matplotlib pillow

echo ""
echo "Venv ready. Run visualization with:"
echo "  make visualize"
