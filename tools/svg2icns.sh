# Script to convert SVG files to macOS .icns format.
# Adapted from https://github.com/magnusviri/svg2icns.

#!/usr/bin/env bash

# Exit immediately if a command fails.
set -e

# Check if at least one argument is provided.
if [ $# -lt 1 ]; then
    echo "Usage: $0 <svg-file1> [<svg-file2> ...]"
    echo "Converts SVG files to macOS .icns format."
    exit 1
fi

# Check if the required command-line tools are available.
if [ ! -e "/usr/bin/sips" ]; then
    echo "sips is required (are you on macOS?)."
    exit 1
fi

if [ ! -e "/usr/bin/iconutil" ]; then
    echo "iconutil is required (are you on macOS?)."
    exit 1
fi

# Define required icon sizes for macOS iconset.
SIZES="
16,16x16
32,16x16@2x
64,32x32@2x
128,128x128
256,128x128@2x
512,256x256@2x
1024,512x512@2x
"

# Define maximum resolution for initial conversion.
MAX_SIZE=1024

# Process each SVG file passed as an argument.
for SVG in "$@"; do
    # Check if the SVG file exists.
    if [ ! -f "$SVG" ]; then
        echo "Error: File '$SVG' not found."
        continue
    fi

    # Extract the base filename without extension.
    BASE=$(basename "$SVG" | sed 's/\.[^\.]*$//')
    ICONSET="$BASE.iconset"
    
    echo "Processing: $SVG"
    
    # Create the temporary iconset directory.
    mkdir -p "$ICONSET"
    
    # Use trap to ensure cleanup on script exit or interruption.
    trap 'rm -rf "$ICONSET"; echo "Cleaned up temporary directory: $ICONSET"' EXIT
    
    # Convert the SVG file to PNG format using sips.
    sips -z $MAX_SIZE $MAX_SIZE -s format png "$SVG" --out "$ICONSET/$BASE.png"

    # Generate each icon size using sips (Scriptable Image Processing System).
    for PARAMS in $SIZES; do
        SIZE=$(echo $PARAMS | cut -d, -f1)  # Extract pixel size.
        LABEL=$(echo $PARAMS | cut -d, -f2) # Extract size label for filename.
        
        # Resize the image to the required dimensions.
        sips -z $SIZE $SIZE "$ICONSET/$BASE.png" --out "$ICONSET/icon_$LABEL.png"
    done

    # Create duplicate icons for standard sizes (required by macOS).
    cp "$ICONSET/icon_16x16@2x.png" "$ICONSET/icon_32x32.png"
    cp "$ICONSET/icon_128x128@2x.png" "$ICONSET/icon_256x256.png"
    cp "$ICONSET/icon_256x256@2x.png" "$ICONSET/icon_512x512.png"
    
    # Convert the iconset directory to a macOS .icns file.
    iconutil -c icns "$ICONSET"

    # Clean up the temporary iconset directory and reset trap.
    rm -rf "$ICONSET"
    trap - EXIT
    
    # Move the .icns to the directory where the SVG file is located.
    mv "$BASE.icns" "$(dirname "$SVG")/$BASE.icns"

    # Print success message.
    echo "Created $BASE.icns"
done
