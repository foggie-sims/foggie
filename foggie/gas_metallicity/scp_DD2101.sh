#!/usr/bin/env bash

for TRACK in 10 7 6 5e-6 5e-7
do
    mkdir -p "/Users/acharyya/Work/astro/foggie_outputs/plots_halo_008508/feedback-track/feedback-$TRACK-track/figs/"
    scp "pfe:/nobackupp19/aachary2/foggie_outputs/plots_halo_008508/feedback-track/feedback-$TRACK-track/figs/DD2101*.png" "/Users/acharyya/Work/astro/foggie_outputs/plots_halo_008508/feedback-track/feedback-$TRACK-track/figs/."
done