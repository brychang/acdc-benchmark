#!/usr/bin/env bash
# Download the FlyWire VNC matching challenge inputs the benchmark needs.
#
# The challenge serves each file gzipped from a query parameter on the challenge
# page; this writes the .csv.gz and then decompresses it to the plain .csv name
# the harness expects. The data is NOT redistributed in this repository: much of
# it is unpublished, and the challenge page asks that it not be published or
# redistributed without first contacting flywire@princeton.edu.
#
# Usage: scripts/fetch_data.sh [dest_dir]        (default: bench/data)

set -euo pipefail

BASE="https://codex.flywire.ai/app/vnc_matching_challenge"
DEST="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/bench/data}"

# download key -> local base name
FILES=(
  "male_connectome_graph:male_connectome_graph"
  "female_connectome_graph:female_connectome_graph"
  "vnc_matching_submission_benchmark_5154247:vnc_matching_submission_benchmark_5154247"
)

mkdir -p "$DEST"

for entry in "${FILES[@]}"; do
  key="${entry%%:*}"
  name="${entry##*:}"
  gz="$DEST/$name.csv.gz"
  csv="$DEST/$name.csv"

  if [ -s "$csv" ]; then
    echo "have    $csv"
    continue
  fi

  if [ ! -s "$gz" ]; then
    echo "fetch   $key"
    curl -fL --retry 3 --retry-delay 2 -o "$gz.part" "$BASE?download=$key"
    mv "$gz.part" "$gz"
  fi

  echo "gunzip  $name.csv"
  gunzip -c "$gz" > "$csv.part"
  mv "$csv.part" "$csv"
done

echo
echo "Data in $DEST:"
ls -l "$DEST"/*.csv
echo
echo "Reminder: do not publish or redistribute these files."
