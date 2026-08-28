# Manuscript: multizoom dwarf simulations

Draft ApJ paper covering the 25 Mpc dwarf campaign — the multizoom method, its
validation, and first results.

    ms.tex              the manuscript (AASTeX 6.3.1, twocolumn, linenumbers)
    refs.bib            bibliography; several entries are placeholders
    figures/            figures, copied from the analysis by collect_figures.sh
    collect_figures.sh  refresh figures/ from figures_z0

## Getting it into Overleaf

Overleaf imports from GitHub, so once this branch is pushed:
**New Project → Import from GitHub → pick the repo → set the root document to
`foggie/initial_conditions/paper/ms.tex`.**

Overleaf's git bridge syncs the whole repository, which is large. If the import
is unwieldy, the alternative is to copy this directory into a standalone
Overleaf project and keep it in sync by hand — at the cost of losing the link
back to the analysis that produced the numbers.

`linenumbers` is on for co-author review; remove it before submission.

## Status markers

The draft is a scaffold with real numbers, not prose. Every gap is marked so
none of it can be mistaken for finished text:

| marker | meaning |
|---|---|
| `\todo{...}` | writing not yet done |
| `\pending{...}` | waiting on a run or measurement still in flight |
| `\check{...}` | number is in hand but must be re-derived when the sample freezes |

Compile and read the red/blue/orange to see what is left.

## Where the numbers come from

Every measured value in the draft traces to analysis already run:

| claim in ms.tex | source |
|---|---|
| 36–47% rank-time idle; L7 = 43% of wallclock | `perf_bench/AUDIT_SUMMARY.md` |
| 0.40× cost, cost decomposition, timestep tax | multizoom cost analysis |
| Median M200 ratio 0.997, N=77, ±4% | `figures_z0/SIXPACK_AUDIT.md` |
| Merger-timing excursions (24122, 48014) | `figures_z0/SIXPACK_AUDIT.md` |
| nref7 vs nref9: 10–200× stellar mass | `plot_nref_resolution.py` |
| Contamination arc, rvir_min 0→200→400 | `figures_z0/CONTAMINATION_L3.md`, `DECONTAM_STATUS.md` |
| N=1 regression, roundoff-level agreement | `n1_regression/RESULT.md` |
| Subgrid floor sweep, reported as unresolved | `perf_bench/BENCH_PHASE3.md` |

If an analysis is re-run, re-run `collect_figures.sh` and re-check the
`\check{}` values. The manuscript should never carry a number the repo cannot
reproduce.

## The largest gap

`PLACEHOLDER_rank_occupancy` does not exist. It is the figure that motivates
the entire method — grids per rank and idle fraction against AMR level — and it
needs to be generated from `performance.out` of a production single-zoom gas
run. Everything else in the methods section rests on it.

## Known open items in the science

- Gas-arm validation (sixpack L2 nref7 vs six isolated nref7) is running.
- The run-to-run noise floor (same halo, 32 vs 64 ranks) is running; until it
  lands, the quoted multizoom scatter is an **upper bound** on the method's
  effect, not a measurement of it. The draft says so.
- Tenpack N=10 scaling and its L3 DM comparison are in progress.
- The quenched-fraction sample is still growing; 6/13 is not the final number.
