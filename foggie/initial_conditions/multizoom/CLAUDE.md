# multizoom package — working context for Claude Code

Read `HANDOFF.md` (session state, decisions, next steps) and
`AUDIT_AND_PLAN.md` (full audit + staged plan) before making changes.

## Hard rules

- **Never modify anything outside this directory** in
  `foggie/initial_conditions/` — the legacy IC framework
  (`enzo-mrp-music/`, `halo_template_256/`, `halo_template_512/`,
  `music/`) must stay byte-for-byte untouched.  This package is a
  deliberate, fully isolated fork; changes to Enzo or MUSIC are carried
  only as patch files in `enzo_patches/` and `music_patches/`.
- All work happens on branch `claude/multi-zoom-single-domain-goqnjp`.
- Physics invariants the code enforces (do not weaken them): every MUSIC
  run in a merge set shares identical `[random]` seeds and one common
  domain frame (`no_shift = yes` or one shared `region_shift_override`);
  the merge tool hard-fails on any seed/shift/cosmology mismatch and on
  overlapping patches.  Multi-halo configs require `shape_type = exact`
  so the mask never refines the volume between halos.

## Quick orientation

- `mrp_music.py` orchestrates one level: union mode = one MUSIC run over
  the union point file; merge mode = one run per halo + `merge_music_ics`.
- `multizoom512.py` is the user-facing driver (rockstar catalog →
  configs → MUSIC → Enzo run dir).  Merge mode needs an Enzo built with
  the five patches in `enzo_patches/` (`git am` onto foggie-sims/enzo-foggie).
- Tests: `cd multizoom && python -m pytest` (numpy/h5py/pytest; no yt).
  The `pytest.ini` here is load-bearing — running pytest from the repo
  root resolves tests through the top-level `foggie` package, which
  imports matplotlib/yt.
