DIRECTORY: `initial_conditions`
AUTHOR: JT and Claude, [Please add your names when you update the list below]
LAST UPDATED: 07/31/2026

This directory generates the initial conditions for cosmological zoom-in
simulations and runs them through their refinement levels. It contains the
automated pipeline, the templates it renders, the halo registry that drives it,
and vendored copies of MUSIC and enzo-mrp-music.

**Full documentation:** `doc/source/user_guide/generating_new_ICs.rst`, which is
the place to start. What follows is a map of the directory.

A zoom is built as a ladder of stages, one per refinement level. Each is two
jobs: a short build job that traces the halo's Lagrangian region back to z = 99
and generates the ICs, and a long Enzo run that evolves them to z = 0. Level N
cannot be built until level N-1 finishes, because enzo-mrp-music reads the
previous level's outputs. The pipeline chains those steps so no one has to sit
and watch for each level to complete.

To add a halo to the fleet, add a row to `halo_registry.ecsv` by hand. That is
the only file you normally edit.

| Folder/Module | Description |
|---------------|-------------|
| `halo_registry.ecsv` | **The one file you edit.** Hand-curated list of halos to run, one row each: refinement depth, whether to run gas, zoom radius floor, PBS hints. Nothing writes to it. |
| `pipeline/` | The pipeline itself, see below. |
| `templates_512/` | Templates rendered per stage for the 512 parent box: the Enzo parameter files, the PBS run script, the enzo-mrp-music config, `simrun.pl`, and the poller scripts. |
| `templates_512/baseline/` | Approved renderings of the templates. `validate-templates` diffs against these so an intentional edit is approved once while an accidental one still fails. |
| `enzo-mrp-music/` | Britton Smith and John Wise's tool for finding a halo's Lagrangian region and driving MUSIC. Locally modified: see below. |
| `music/` | MUSIC source. The compiled `MUSIC` binary is gitignored; build it in place on the machine you run on. |
| `planck18_cosmology/` | CAMB transfer functions for the adopted cosmology, fed to MUSIC. |
| `halo_catalogs_512/` | Rockstar z=0 catalog for the 512 parent box. Registry `halo_id` values must resolve here. |
| `halo_template_512/`, `halo_template_256/` | The previous hand-driven workflow, kept until the pipeline has been exercised through L3. Superseded by `pipeline/` and `templates_512/`. |
| `REFACTOR_PLAN.md`, `refactor_roadmap.html` | Design notes and diagrams for the pipeline. |

### `pipeline/`

| Module | Description |
|--------|-------------|
| `ic_pipeline.py` | The command line entry point. Run it **by path**, not with `python -m` -- importing the `foggie` package pulls in yt, and nothing here needs it. Subcommands: `status`, `advance`, `build`, `poll`, `validate-registry`, `validate-templates`. |
| `config.py` | Box definitions and the registry reader. Everything that used to be a literal in `script512.py` lives here: parent grid size, halo catalog, MUSIC and Enzo binaries, baryon fraction, PBS resources, Rvir floor. |
| `state.py` | Works out what state each stage is in, entirely from files Enzo and `simrun.pl` already write. Opens no HDF5 and imports no yt, so `status` stays fast. |
| `build.py` | Renders the configs, parameter files and run scripts, runs enzo-mrp-music or MUSIC, and submits jobs. |
| `ledger.py` | Per-halo record of what was submitted, the lock that makes concurrent triggers safe, and the guard that refuses to write into a hand-built directory. |
| `report.py` | Renders the progress table as text, ECSV and HTML. |
| `notify.py` | Emails stage state changes. Reports transitions, not status. |

### Local modifications to `enzo-mrp-music`

`enzo-mrp-music.py` is not upstream-clean. It gained a `new_ics_directory`
parameter, separating where the previous level's Enzo outputs are *read* from
where the new level's ICs are *written*. Previously one option did both jobs,
which put L1's ICs in the wrong directory. It also now takes the MUSIC binary
path from the config it already parses rather than a second hardcoded copy, and
raises on a MUSIC failure instead of continuing silently.

### Notes for anyone picking this up

* Status is never written into the registry. It goes to `status.ecsv`,
  `status_by_halo.ecsv` and `status.html` in `$FOGGIE_ICS_DIR`, regenerated
  every sweep and deliberately outside version control.
* `DONE` requires both the final redshift dump *and* `RunFinished`. Either
  alone is a false positive in the existing runs.
* A `.message` file saying "finished!" is not evidence of anything. `simrun.pl`
  leaves them behind across attempts.
* `cron` is accepted but never executed on the NAS front ends. The poller uses
  a self-rescheduling `at` chain instead.
* IC generation needs roughly 10 GB and must run on a compute node, which is
  what `build --as-job` is for.
