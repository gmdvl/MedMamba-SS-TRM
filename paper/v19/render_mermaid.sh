#!/bin/sh
# Render paper/v19/build/*.mmd (written by build_v19.py) to paper/v19/figures/*.pdf with
# mermaid-cli on the HOST (the VS Code Flatpak sandbox has no browser). Run from anywhere:
#   flatpak-spawn --host sh <abs>/paper/v19/render_mermaid.sh
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
printf '{"executablePath":"/usr/bin/brave","args":["--no-sandbox","--headless=new"]}' > "$DIR/build/pup.json"
printf '{"theme":"neutral","flowchart":{"htmlLabels":true,"curve":"basis","wrappingWidth":320},"themeVariables":{"fontFamily":"Times New Roman, serif","fontSize":"15px"}}' > "$DIR/build/config.json"
cd "${TMPDIR:-/tmp}"
for f in "$DIR"/build/*.mmd; do
  n=$(basename "$f" .mmd)
  PUPPETEER_SKIP_DOWNLOAD=1 npx -y -p @mermaid-js/mermaid-cli mmdc -q -p "$DIR/build/pup.json" \
    -c "$DIR/build/config.json" -b white -i "$f" -o "$DIR/figures/$n.pdf"
  PUPPETEER_SKIP_DOWNLOAD=1 npx -y -p @mermaid-js/mermaid-cli mmdc -q -p "$DIR/build/pup.json" \
    -c "$DIR/build/config.json" -b white -s 3 -i "$f" -o "$DIR/figures/$n.png"
  echo "[mermaid] $n"
done
