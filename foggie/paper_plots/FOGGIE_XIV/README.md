# README
*Last Update: May 22, 2026*

This document provides a guide for reproducing plots in "Figuring Out Gas & Galaxies In Enzo (FOGGIE). XIV. The Observability of Emission Accretion and Feedback in the Circumgalactic Medium with Current and Future Instruments."

Notebooks for reproducing each figure can be found in the FOGGIE GitHub repository at: `paper_plots/FOGGIE_XIV/`

Notebooks are organized by paper section and figure number.

---

**`FOGGIE_XIV_sec234.ipynb`** — Sections 2, 3, and 4

Reproduces Figures 1–6 and Table 1. Also produces Appendix Figures 15 & 16 (full 8-ion version of Figure 5).
Requires running `foggie/emission/emission_mass_maps_dynamic.py` first for all halos at resolutions 0.183, 1, 3, and 6 kpc, with and without `--filter_type disk_cgm`. See the prerequisite instructions at the top of the notebook.

---

**`FOGGIE_XIV_sec5.ipynb`** — Section 5

Reproduces Figures 7–13. Requires the emission maps from `emission_mass_maps_dynamic.py` (see above) plus inflow/outflow maps. See prerequisite notes at the top of the notebook.
Figure 8 (bottom panel) is made using `plot_3x3_velocity_comparison_grid` in `foggie/emission/kinematics.py`.
Figure 10 (bottom left panel) is made using `velocityprofile.py` in `paper_plots/FOGGIE_XIV/`.

---

**`FOGGIE_XIV_sec6.ipynb`** — Section 6

Reproduces Figure 14 (instrument comparison plot). Instrument properties are hardcoded from Table 3 of the paper. No simulation data required.

---

**`FOGGIE_XIV_appendix.ipynb`** — Appendix

Reproduces Appendix Figure 17 (2D emissivity histograms).
Appendix Figures 15 & 16 are in `FOGGIE_XIV_sec234.ipynb`.
Appendix Figure 18 was generated using `emission_maps_dynamic.py` with `--plot emission_map_vbins` and assembled in Keynote — see the top of the appendix notebook for the exact commands.
