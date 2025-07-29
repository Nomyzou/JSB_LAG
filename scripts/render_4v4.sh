#!/bin/sh
echo "Running 4v4 Fixed Pairing Rendering..."
# We need to be in the root directory of the project for the imports to work correctly.
# This script assumes it's being run from the 'scripts' directory.
cd ..
# Execute the render script
python renders/render_4v4_fixed_pairing.py
echo "Rendering complete." 