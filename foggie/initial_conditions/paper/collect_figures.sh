#!/bin/bash
# Copy the current analysis figures into paper/figures/.
#
# Figures are NOT edited here and NOT committed from here by hand: they are
# products of scripts in halocat/scripts, and this script records exactly which
# file each manuscript figure comes from. Re-run it whenever the analysis is
# re-run, so the manuscript never drifts from the data.
#
# PNG is fine for a draft and for Overleaf. Convert to vector before
# submission -- the plotting scripts can emit PDF by changing the --out
# extension, since matplotlib picks the backend from it.

set -u
SRC=/nobackupnfs1/jtumlins/25Mpc_new_cosmology/figures_z0
DST="$(cd "$(dirname "$0")" && pwd)/figures"
mkdir -p "$DST"

# manuscript name <- source file  (manuscript name matches \plotone in ms.tex)
FIGS="
sixpack_audit.png|sixpack_audit.png
sixpack_projections.png|sixpack_projections.png
nref7_vs_nref9.png|nref7_vs_nref9.png
sfh_fleet_z0.png|sfh_fleet_z0.png
quenched_fraction_z0.png|quenched_fraction_z0.png
environment_quenching_z0.png|environment_quenching_z0.png
scaling_relations_pooled-targets_z0.png|scaling_relations_pooled-targets_z0.png
"

n=0; miss=0
for row in $FIGS; do
  dst="${row%%|*}"; src="${row##*|}"
  if [ -f "$SRC/$src" ]; then
    cp -p "$SRC/$src" "$DST/$dst"
    printf "  %-46s <- %s\n" "$dst" "$src"
    n=$((n+1))
  else
    printf "  %-46s MISSING (%s)\n" "$dst" "$src"
    miss=$((miss+1))
  fi
done
echo "  $n copied, $miss missing"

cat <<'EOF'

  Figures referenced by ms.tex that do NOT yet exist:
    PLACEHOLDER_rank_occupancy   -- grids per rank and idle fraction vs AMR
                                    level. Must be generated from
                                    performance.out; it is the figure that
                                    motivates the whole method and is
                                    currently the largest gap in the draft.
EOF
