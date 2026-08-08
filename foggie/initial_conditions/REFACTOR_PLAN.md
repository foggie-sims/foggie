# Automating the 25 Mpc zoom IC → Enzo pipeline

## Context

Producing a dwarf-galaxy zoom in `/nobackupnfs1/jtumlins/25Mpc_new_cosmology` currently takes a
human in the loop at every refinement level. For each halo you must: submit `L1.sh`, wait for
`script512.py` to build the L1 ICs with enzo-mrp-music + MUSIC, wait hours-to-days for the L1 Enzo
run to reach `RD0265` (z = 0), notice that it finished, then hand-submit `L2.sh`, and repeat
through L3 and the gas run. Nothing tells you the state of the fleet except `tail`-ing
`OutputLog` by hand (`/nobackupnfs1/jtumlins/status.sh`, `isitrunning`).

The consequence is idle machine time between levels, silent stalls (`halo15097/25Mpc_DM_512-L1`
died at `RD0259` on a walltime kill and simply sat there), and a set of driver scripts that have
drifted into an unmaintainable state: `script512.py` and `script256.py` are ~90 % copy-paste of
each other, build config files by shelling out to `awk`/`sed`, dispatch on `'2' in args.level`, and carry dead
functions plus hardcoded absolute paths. The committed repo copy has also fallen behind what
production actually needs, to the point where it would fail at L1 (see the finding below).

The goal: keep the tools that work (enzo-mrp-music, MUSIC, Enzo, `simrun.pl`), replace the two
driver scripts with one clean parameterized module, make level L+1 fire automatically when level
L reaches its final redshift dump, add a progress table over all halos and levels, and document
the whole thing in the FOGGIE repo.

**Scope decisions taken with the user:** automate the DM ladder L1 → L3 now, with the gas run
designed in as a registry-driven extension; drive everything from a single hand-curated halo
registry; trigger via both a job-chained hook and a periodic poller; consolidate 512/256 into one
module with the box as configuration.

**The existing manually-run halos are frozen.** Every `halo?????` directory under
`$FOGGIE_ICS_DIR` — `halo15097`, `halo15134`, `halo39642`, `halo46205`, `halo52166`, and the rest —
is left untouched until the automation is proven. `halo11177` and `halo42189` have been relabelled
to `halo11177-manual` / `halo42189-manual` to preserve them and free their names. The first real
exercise of the pipeline is to **recreate those two from scratch** through the registry and the new
scripts, into fresh `halo11177/` and `halo42189/` directories, and check the result against the
preserved manual versions.

---

## Finding: what the repo tree is missing

**`/nobackupnfs1/jtumlins/initial_conditions` (a.k.a. `/home1/jtumlins/nobackup/initial_conditions`)
is deprecated and is not a source of truth.** The canonical tree is the repo copy,
`/nobackupnfs1/jtumlins/foggie/foggie/initial_conditions`. It was still worth diffing the two,
because the deprecated tree documents fixes the repo tree needs — the repo files get those fixes
applied directly, as new work, rather than being "merged" from anywhere.

`diff -rq` shows **only 7 differing files**; `script512.py` and the three DM `.enzo` templates are
byte-identical. Three of the seven are real gaps in the repo tree, three are hardware/toolchain
staleness that the rewrite parameterizes away anyway, and one is a case where the repo is already
ahead:

**Real gaps — fix the repo files.**

1. `enzo-mrp-music/enzo-mrp-music.py` has no `new_ics_directory` parameter. It uses
   `simulation_run_directory` for two different jobs at once: where previous-level Enzo outputs are
   *read* and where new ICs are *written*. With `halo_DM_NtoN.conf` setting
   `simulation_run_directory = FOGGIE_ICS_DIR`, the L1 IC directory lands in `$FOGGIE_ICS_DIR`
   rather than the halo directory, so `script512.py:206`'s `os.chdir('25Mpc_DM_512-L1')` fails.
   L2/L3 happen to work only because `set_1to2_conf`/`set_2to3_conf` rewrite
   `simulation_run_directory` to `os.getcwd()` (`script512.py:128,155`) — which then breaks the
   *read* side's ability to find the previous level. Fix: add a separate `new_ics_directory`
   (default `"."`), used at `sim_dir`, `output/filename`, and `new_config_file`.
2. `halo_template_512/halo_DM_NtoN.conf` needs the corresponding
   `new_ics_directory = FOGGIE_ICS_DIR/HALO_DIR` line.
3. Neither repo template resolves `HALO_DIR`. The per-halo production copies do
   (`/nobackupnfs1/jtumlins/25Mpc_new_cosmology/halo15097/script512.py:30-34`, a
   `sed s/HALO_DIR/<cwd basename>/g` step) — this must exist in the canonical code, not only in
   hand-edited copies.

That these three are genuinely required is confirmed by the generated production configs:
`halo15097/halo15097_DM_2to3.conf:8` carries
`new_ics_directory = /u/jtumlins/nobackup/25Mpc_new_cosmology/halo15097`. The running system
already depends on this behaviour; the repo simply never gained it.

**Staleness the rewrite removes anyway.** `RunScript.sh` in the repo requests Haswell
(`select=2:ncpus=16:mpiprocs=16:model=has`, `-np 32`, `module load mpi-hpe/mpt.2.23`) while
current production runs on Aitken/Milan (`select=1:ncpus=64:mpiprocs=64:model=mil_ait`, `-np 64`,
no mpt module); `simrun.pl:20` points at `enzo-foggie-opthigh` while production uses
`enzo-foggie-aitken-mpich`. Both become template keywords and box-config values
(`__SELECT__`/`__NRANKS__`/`__MODEL__`, `enzo_exe`), so hardware and binary choice stop being
baked into committed files. Defaults follow current production (Aitken, `aitken-mpich`).

**Where the repo is already ahead.** The `-gas.enzo` templates carry
`StarMakerMinimumMass = 10000.` plus a new `H2StarMakerMinimumMass = 10`; the deprecated tree had
`StarMakerMinimumMass = 10` and no `H2StarMakerMinimumMass`. Keep the repo values.

> **Flag (physics, not infrastructure):** those repo gas values differ from what `halo15097` and
> `halo42189` `-L3-gas-therm` actually ran (`StarMakerMinimumMass = 10`). Worth a conscious
> confirmation before the first new gas run — it does not block any DM work.

The deprecated tree is not copied, referenced, or vendored anywhere by this work.

---

## Design

### State is derived from disk, not from a database

Every decision the pipeline makes is a pure function of files already on disk. This is the single
most important property: it makes `advance` idempotent, makes the job-chained trigger and the
poller safe to run concurrently, and means **the frozen manual runs need no migration to be
readable** — `status` reports `halo42189-manual`, `halo11177-manual`, and `halo15097` correctly on
first run without touching them. That is what makes them usable as free ground truth for
validating the state detector in Phase 1.

A **stage** is the tuple `(halo_id, box, level, phase)` with `phase ∈ {DM, gas}`, living in
`$FOGGIE_ICS_DIR/halo<ID>/<sim_name>-L<N>[-gas]/`. Its state:

| State | Detected by |
|---|---|
| `DONE` | `OutputLog` last line names the final redshift dump **and** `RunFinished` exists |
| `RUNNING` / `QUEUED` | ledger holds a job id whose `qstat -x` state is `R` / `Q`,`H` |
| `BUILDING` | ledger holds a live job id for the IC-generation step |
| `STALLED` | IC dir exists, no live job, not `DONE`. Sub-reason from `pbs_output_*.txt` (walltime kill), `.message` ("in trouble!"), or `estd.out` tail |
| `READY` | prerequisite stage is `DONE`, no IC dir yet |
| `BLOCKED` | prerequisite stage is not `DONE` |

The final dump is **not** hardcoded to `RD0265`. It is `RD%04d` of the highest
`CosmologyOutputRedshift[i]` index parsed out of that stage's own `.enzo` file (266 entries today,
index 265 → z = 0). This survives any future change to the output list.

### The halo registry

One hand-curated file, `foggie/initial_conditions/halo_registry.ecsv`, is the sole source of
truth for what should run. ECSV (astropy) rather than YAML — the repo already depends on astropy
and reads catalogs this way (`script512.py:182`), and there is no `pyyaml` dependency.

Initial contents — only the two halos being recreated from scratch. Everything else stays out of
the registry until the automation is proven:

```
halo_id | box           | enabled | final_level | gas   | queue  | nodes | model   | notes
42189   | 25Mpc_DM_512  | True    | 3           | False | normal | 1     | mil_ait | regression vs halo42189-manual
11177   | 25Mpc_DM_512  | True    | 3           | False | normal | 1     | mil_ait | manual L1 never completed
```

`gas` is written and validated from day one but only acted on in the gas phase (below). The
per-halo stage plan is `[L1-DM … L<final_level>-DM]`, plus `[L<final_level>-gas]` when
`gas is True` and the gas phase is enabled.

**Guard against the frozen halos.** `build` refuses to write into a halo directory that already
contains stage directories it did not create — i.e. `<sim_name>-L*/` exists but `.pipeline/` does
not — unless `--adopt` is passed explicitly. This is what stops someone adding `15097` to the
registry and having the pipeline start writing into a hand-built run. The `-manual` directories
are invisible to the pipeline anyway, since their names do not match `halo<id>` for any registry
`halo_id`.

### Advance algorithm

```python
for halo in registry.enabled():
    with lock(halo):                      # flock on halo<ID>/.pipeline/lock
        for stage in stage_plan(halo):    # strictly ordered
            s = state(stage)
            if s is DONE:      continue
            if s is READY:     submit_build_job(stage)
            break                         # BLOCKED/QUEUED/RUNNING/STALLED → nothing to do
```

The `break` after the first non-`DONE` stage enforces the hard sequential dependency: level N's ICs
require level N−1's Enzo outputs, because enzo-mrp-music `yt.load_simulation`s them to trace the
Lagrangian region. `STALLED` deliberately does **not** auto-retry — a stall means something needs
a human, and silently resubmitting into the same wall would burn allocation.

### Two triggers, one engine

**Job-chained (primary, seconds of latency).** The generated `RunScript.sh` gains one line after
`simrun.pl` returns:

```bash
python -m foggie.initial_conditions.pipeline.ic_pipeline advance \
       --halo __HALO_ID__ >> __HALO_DIR__/pipeline.log 2>&1
```

No "did it finish?" logic in bash — `advance` re-derives state and is a no-op unless the stage is
`DONE`. That matters because `simrun.pl` exits in three different ways (finished, self-resubmitted
for walltime, died), and only one of them should advance. Submitting from a compute node is
already proven: `simrun.pl` itself calls `qsub $job_file` from inside the job.

**Poller (safety net, ~30 min latency).** `ic_pipeline poll` runs `advance --all` then
`status --write` and re-`qsub`s itself, reusing the self-resubmission pattern from `simrun.pl`. It
catches chains broken by hard node failure. A `crontab` alternative on `pfe` is documented as
equally valid. Concurrency between the two is safe via the per-halo `flock` plus the
`qstat`-checked ledger.

### Template consolidation

The three DM templates `25Mpc_DM_512-L{1,2,3}.enzo` were diffed and differ **only** in three
values (plus comments and whitespace), so they collapse to one parameterized `DM-LX.enzo`:

| Keyword | Value at level L |
|---|---|
| `__NUM_INITIAL_GRIDS__` | `L + 1` |
| `__MRP_REFINE_TO_LEVEL__` | `L` |
| `__MIN_OVERDENSITY__` | `8^-(L-1)` in the first two slots, `1.` thereafter (matches `bds_notes`) |
| `__GRID_PARAMETERS__` | the `CosmologySimulationGrid*` block read from MUSIC's `parameter_file.txt` |

`__GRID_PARAMETERS__` replaces the `grep … > grid_parameters.txt; cat template grid_parameters.txt
> pars.temp; mv` shell dance at `script512.py:79-103`, using the substitution approach already
proven in `foggie/utils/run_foggie_sim.py:184` (`COPYHERE`).

The gas templates genuinely diverge (L3 adds H2 star formation, tabular SNe yields, `nref9`, and
drops `ComputePotential`/`CosmologySimulationUseMetallicityField`), so they are **not** collapsed.
`gas-LX.enzo` is derived from the L3 gas file with the same level keywords plus
`__MAX_REFINE_LEVEL__`; the L1/L2 gas templates are retired rather than mechanically merged.

`RunScript.sh` becomes one template with `__JOBNAME__ __SELECT__ __NRANKS__ __QUEUE__
__WALLTIME__ __GROUP__ __PARAM_FILE__ __ENZO_EXE__ __PIPELINE_HOOK__`, replacing the
`RunScript.sh`/`RunScriptGas.sh` pair and the hand-maintained `L{1,2,3,4}[-gas].sh` files.

**Safety net for the collapse:** `ic_pipeline validate-templates` renders DM L1/L2/L3 and gas L3
and diffs them against the existing committed `.enzo` files, ignoring whitespace and comments.
This must come out clean before anything is switched over.

---

## Files

New package `foggie/initial_conditions/pipeline/`:

| File | Contents |
|---|---|
| `ic_pipeline.py` | CLI: `status`, `advance`, `build`, `poll`, `validate-templates`, `validate-registry`. Global `--dry-run`, `--registry`, `--ics-dir`. |
| `config.py` | `BOXES` dict + registry reader. Per box: `sim_name`, `parent_ngrid` (shift divisor is `ngrid - 1`: 511 for 512, 255 for 256), `catalog` path, `template_config`, `template_dir`, `boxsize_mpc`, `rvir_floor_kpc`, `omega_b`, `max_level`, `enzo_exe`. |
| `state.py` | `stage_state()`, `final_dump()`, `last_output()`, `redshift_of()`, `qstat_states()`. Read-only, no yt. |
| `build.py` | `write_mrp_config(level)` — the single function replacing `set_0to1_conf`/`set_1to2_conf`/`set_2to3_conf`/`set_3to4_conf`; `run_mrp_music()`, `run_music_gas()`, `render_enzo_param()`, `render_runscript()`, `submit()`, ledger append. |
| `report.py` | Table rendering: `text`, `ecsv`, `html`. |

New `foggie/initial_conditions/halo_registry.ecsv` — the curated registry.

New `foggie/initial_conditions/templates_512/` — rebuilt from the repo's `halo_template_512/`:
`halo_DM_NtoN.conf` (plus the `new_ics_directory` line), `DM-LX.enzo`, `gas-LX.enzo` (repo star
formation values), `RunScript.sh` (parameterized), `simrun.pl`. `templates_256/` gets the same
treatment; `halo_template_{512,256}/` are removed in the same commit so there is one tree, not two.

Modified `foggie/initial_conditions/enzo-mrp-music/enzo-mrp-music.py` — add the
`new_ics_directory` parameter per gap #1 above, and lift the hardcoded MUSIC binary path
(currently `/nobackupnfs1/jtumlins/foggie/foggie/initial_conditions/music/MUSIC` at ~line 228,
which ignores the config's own `music_exe_dir`) into the config it already parses.

Deleted: `halo_template_512/script512.py`, `halo_template_256/script256.py`,
`L{1,2,3,4}[-gas].sh` in both template dirs, `plots.py`.

### Reused, not rewritten

- `simrun.pl` — the Enzo restart/resubmit loop, `OutputLog` parsing, and `RunFinished` semantics
  are unchanged. The pipeline sits above it. Only the hardcoded `$enzo_executable` at line 20
  becomes a config value.
- `enzo-mrp-music.py` + `get_halo_initial_extent.py` + MUSIC — unchanged except `new_ics_directory`
  and the binary path.
- `replace_keywords_in_file` / `modify_lines_in_file` (`foggie/utils/run_foggie_sim.py:28,51`) and
  `execute_command` (`:369`) — lift these into `build.py` rather than reinventing. Do not import
  from `run_foggie_sim.py` itself: it does `from header import *` / `from util import *`, which
  only resolves when cwd is `foggie/utils/`.
- `foggie/fogghorn/run_halo_finding_incremental.py` — the existing `run_once` / `run_watch`
  incremental-scan idiom; `poll` should read like it.

### Bugs fixed by the rewrite

1. `script512.py:194-202` — level dispatch is `'2' in args.level` on a string; breaks for any level ≥ 10 and would match `'13'` twice. → integer levels.
2. `script512.py:52-59,209` — `get_0to1_shifts()` always greps `25Mpc_DM_512-L1.conf_log.txt` regardless of the level being built, and is called at every level.
3. `script512.py:31` — `os.getenv("FOGGIE_REPO")+"initial_conditions/…"`, missing `/`; works only because `.bash_profile` has a trailing slash and `.bashrc` does not. (Function `run_0to1_music` is dead anyway — delete.)
4. `script512.py:215` — hardcoded `/u/jtumlins/nobackup/foggie/…/music/MUSIC`. → box config.
5. `script512.py:166` — `convert_to_gas` hardcodes `Omega_b = 0.0461`, but `25Mpc_DM_512_planck18-gas.conf:22` and every `-gas.enzo` say `0.04576`. MUSIC and Enzo are being given different baryon fractions. → single `omega_b` in the box config, asserted against the template config at build time.
6. `script512.py:205,210,219` — unguarded `rm *temp` in the cwd.
7. Level scripts: `L2-gas.sh` is a copy of `L2.sh` still passing `--gas='no'`; `L3.sh`/`L4.sh` omit the `required=True` `--halo_id`; `halo11177/L1.sh:24` passes `--rvir_min=80`, an argument no version of `script512.py` defines. → all generated, none hand-maintained.
8. Rvir floor is silently inconsistent — `script256.py:16` floors at 200 kpc, `script512.py:188` does not. → explicit `rvir_floor_kpc` per box.
9. `--run` is parsed but unused (`script512.py:175`); the real switch is the module global `run = True` at line 7. → one `--dry-run`.
10. Build failures are not checked — `run_music()` return code is ignored and the script chdirs and `qsub`s regardless. → check returns, record failure in the ledger, leave the stage `STALLED`.

---

## Monitoring output

`ic_pipeline status` produces a per-stage table and a per-halo summary:

```
HALO            BOX            STAGE   STATE     LAST    z      CYCLE  JOBID     Q  NOTE
42189           25Mpc_DM_512   L1-DM   DONE      RD0265  0.000   1208  -         -
42189           25Mpc_DM_512   L2-DM   RUNNING   RD0131  1.104    602  9182734.  R
42189           25Mpc_DM_512   L3-DM   BLOCKED   -       -          -  -         -  waiting on L2-DM
11177           25Mpc_DM_512   L1-DM   RUNNING   RD0088  2.512    341  9182801.  R
42189-manual    25Mpc_DM_512   L1-DM   DONE      RD0265  0.000   1208  -         -  frozen
15097           25Mpc_DM_512   L1-DM   STALLED   RD0259  0.011   1180  -         -  frozen, walltime kill
```

Registry halos are listed first; frozen directories appear only under `--include-manual` and are
marked, so there is never any doubt about which rows the pipeline will act on.

Redshift is looked up from the `CosmologyOutputRedshift[i]` list in that stage's `.enzo` file —
no HDF5, no yt, so `status` stays instant on a login node. Written to
`$FOGGIE_ICS_DIR/pipeline_status.ecsv` and `pipeline_status.html` by the poller each sweep, so
there is always a current table without anyone running anything.

---

## Documentation

- **`doc/source/user_guide/ic_pipeline.rst`** — the main deliverable. Register it in the
  "User Guide - New Sims" toctree in `doc/source/user_guide/index.rst`, next to `enzo-foggie`,
  `clean_from_ICs`, `restart`. Follow the structure of `quick_halo_finding.rst` (Key features /
  Command-line usage / Required + Optional arguments / Primary functions / Outputs / Dependencies),
  with an author byline as in `analysis_scripts.rst:3`. Cover: the stage graph and state machine,
  adding a halo to the registry, the two triggers, reading the status table, recovering a stalled
  run, and what every generated file is.
- **`foggie/initial_conditions/README.md`** — does not exist today. Module table in the style of
  `foggie/utils/README.md` (`DIRECTORY:` / `AUTHOR:` / `LAST UPDATED:` header then a table).
- **Top-level `README.md`** — the `initial_conditions` row currently reads "Old initial conditions
  for a 25 Mpc simulation box"; update it.
- **Retire** `enzo-mrp-music/bds_notes` into the new doc — its per-level parameter edits are now
  encoded in the templates and should be documented as such rather than left as loose notes.

`doc/Makefile` sets `SPHINXOPTS = -W`, so any ReST warning fails the build.

---

## Phasing

Ordered so that nothing risky lands before it can be verified against known-good output.

0. **Fix the repo tree** per the finding above (the three `new_ics_directory`/`HALO_DIR` gaps);
   land `templates_512/` and `validate-templates`. No behaviour change yet.
1. **`status` only** — read-only, zero risk. Point it at the frozen manual directories and confirm
   it reproduces their known state: `halo42189-manual` L1/L2/L3 all `DONE` at RD0265 (cycles 1208,
   1208, 1209); `halo11177-manual` L1 `STALLED` at `DD0041` with no `RunFinished`; `halo15097` L1
   `STALLED` at RD0259 on a walltime kill with L2/L3 `DONE`. Between them these cover every state
   the detector must distinguish, on real data, before anything is able to submit a job.
2. **`build` + `submit`** for one stage, `--dry-run` first, then live: `halo11177` L1 into a fresh
   directory.
3. **`advance` + job-chain hook + poller.** Recreate `halo42189` end-to-end and unattended — L1 →
   L2 → L3 with no human in the loop. This is the headline test of the whole design, and it has a
   golden reference to check against (below).
4. **Regression check vs `halo42189-manual`** — see the three tiers in Verification.
5. **Gas extension** — honour the registry `gas` column, add the L3 gas stage to the plan.
6. **Docs.** Branch is then handed off — see below; the PR is not mine to open.

Only after all of that does anything get said about the other frozen halos; migrating them into
the registry is a separate decision for JT.

## Verification

```bash
# 0 — template collapse is exactly equivalent to what is committed today
python -m foggie.initial_conditions.pipeline.ic_pipeline validate-templates --box 25Mpc_DM_512
#   must report no differences vs 25Mpc_DM_512-L{1,2,3}.enzo and -L3-gas.enzo

# registry parses and every halo_id resolves in the catalog
python -m ...ic_pipeline validate-registry

# 1 — state detection against the frozen manual runs (read-only, no jobs submitted)
python -m ...ic_pipeline status --scan-dir $FOGGIE_ICS_DIR --include-manual
#   expect: halo42189-manual  L1/L2/L3-DM  DONE    @ RD0265  (cycles 1208, 1208, 1209)
#           halo11177-manual  L1-DM        STALLED @ DD0041  (no RunFinished)
#           halo15097         L1-DM        STALLED @ RD0259  (walltime kill in pbs_output_*.txt)
#           halo15097         L2/L3-DM     DONE    @ RD0265
#   cross-check: tail -1 <dir>/OutputLog ; cat <dir>/RunFinished ; /nobackupnfs1/jtumlins/status.sh

# 1 — the guard actually guards: a frozen halo must be refused, not written to
python -m ...ic_pipeline build --halo 15097 --level 4 --dry-run   # must refuse: no .pipeline/, stage dirs present

# 2 — build one stage without touching the queue, then for real
python -m ...ic_pipeline build --halo 11177 --level 1 --dry-run
python -m ...ic_pipeline build --halo 11177 --level 1
qstat -u $USER    # confirm one job; confirm halo11177/.pipeline/ledger.json recorded its id

# 3 — advance is a genuine no-op unless a stage is DONE (the critical safety property)
python -m ...ic_pipeline advance --halo 11177 --dry-run   # L1 mid-run -> must submit nothing

# 3 — full unattended chain: watch L1 finish and L2 appear with no human action
tail -f $FOGGIE_ICS_DIR/halo42189/pipeline.log
python -m ...ic_pipeline status          # L1 DONE -> L2 BUILDING -> L2 RUNNING -> ...

# 6 — docs build (warnings are fatal)
cd /nobackupnfs1/jtumlins/foggie/doc && make html
```

### Phase 4 — regression check against `halo42189-manual`

The recreated `halo42189` is compared with the preserved manual run in three tiers, because not
all of it can or should match bitwise:

**Tier 1 — must match exactly (deterministic text).** The generated MUSIC configs
(`25Mpc_DM_512-L{1,2,3}.conf`), the `l0_to_l1_shifts` / `l1_to_l2_shifts` values, the
`CosmologySimulationGrid*` geometry, and the final `.enzo` parameter files. Any difference here is
a bug in the rewrite, not physics — modulo the known-intended fixes, which must be enumerated and
justified rather than waved through (`Omega_b`, and any level-dependent value the template
collapse touches).

```bash
for L in 1 2 3; do
  diff halo42189/25Mpc_DM_512-L$L.conf halo42189-manual/25Mpc_DM_512-L$L.conf
  diff <(grep CosmologySimulationGrid halo42189/25Mpc_DM_512-L$L/parameter_file.txt) \
       <(grep CosmologySimulationGrid halo42189-manual/25Mpc_DM_512-L$L/parameter_file.txt)
done
```

**Tier 2 — expected to match bitwise (deterministic ICs).** The MUSIC output files
(`GridDensity.*`, `ParticleDisplacements_*`, `ParticleVelocities_*`, `RefinementMask.*`). MUSIC is
seeded from fixed `seed[5..13]` values in `25Mpc_DM_512_planck18.conf` and the shared
`wnoise_*.bin` fields, so identical configs should give identical ICs. Compare with `cmp`. A
mismatch here means an input drifted (seed, transfer function, or the Lagrangian region), and is
worth understanding before proceeding.

**Tier 3 — physical agreement only, never bitwise.** The Enzo outputs. These will *not* match
bit-for-bit: the manual L1 ran 64 MPI ranks on `mil_ait`, and floating-point reduction order
differs with decomposition, so trajectories diverge at round-off and amplify. Compare the science
instead — halo center, `Rvir`, and `Mvir` at RD0265, plus the cycle count as a sanity check
(manual: 1208 / 1208 / 1209). Agreement to well within a cell width is the pass criterion.

`halo11177` gets no such comparison — its manual L1 never finished (`DD0041`, no `RunFinished`), so
recreating it is a test that the pipeline completes a ladder the manual process did not, not a
reproduction.

## Step 0 — land this plan on the branch

Before any code, the plan itself goes onto `ics_refactor` so it is reviewable in place:

- `foggie/initial_conditions/REFACTOR_PLAN.md` — this document, verbatim.
- `foggie/initial_conditions/refactor_roadmap.html` — a graphical roadmap of the pathway: the
  stage dependency graph (L1-DM → L2-DM → L3-DM → L3-gas), the per-stage state machine, the two
  trigger paths, and the phase timeline.

The HTML must be **fully self-contained** — inline CSS and hand-written SVG, no CDN, no external
fonts, no `<script>` fetching anything. It will be opened over `file://` on a laptop with no
network, so anything remote silently renders as a blank box. Then push the branch.

## Branch and isolation

All repo changes live on a **new branch `ics_refactor`**, cut from a now-clean `master`
(`e20f8bf8`, in sync with `origin/master`). Nothing lands on `master`, and the existing
`ICs-generation-tweaks` branch is left untouched.

```bash
cd /nobackupnfs1/jtumlins/foggie
git checkout master && git checkout -b ics_refactor
```

Two consequences of the just-landed `e20f8bf8` ("added music source to repo, gitignore MUSIC
binary") that this work should respect:

- `foggie/initial_conditions/music/` is now **tracked** (78 source files) and sits inside the
  directory being restructured. The refactor touches `halo_template_{512,256}/`,
  `enzo-mrp-music/`, and adds `pipeline/` + `templates_{512,256}/` — `music/` is left entirely
  alone.
- `.gitignore` now carries `foggie/initial_conditions/music/MUSIC`, fixing the compiled binary at
  a known in-repo path. That makes the natural default for the box config's `music_exe`
  `$FOGGIE_REPO/initial_conditions/music/MUSIC` — built in place from tracked source, ignored by
  git. This replaces both hardcoded paths (`script512.py:215` and the one inside
  `enzo-mrp-music.py`) with a single defensible default rather than an arbitrary one.

The working tree is clean, so ordinary staging is safe; still prefer staging by path so the
gitignored 18 MB binary and any locally built MUSIC artifacts stay out.

**The work stops at the `ics_refactor` branch.** Do not open a pull request and do not merge into
`master` — the final PR into `foggie-sims/foggie:master` is JT's to make, at the end. Pushing the
branch to `origin` is fine when asked for; the review and merge are not part of this work.

---

## Appendix — Zooming a coarse parent box on a halo it does not resolve

Moved to its own file: **`REGION_TRANSPLANT.md`**, which records the technique,
the five tests run on halo80181, the measurements, and the risks. Demonstrated,
not adopted.
