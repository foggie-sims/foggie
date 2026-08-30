"""
Diagnostic plots for a zoom, one row per refinement level.

Answers the two questions you actually need answered before trusting a zoom:

  Is the target a single object?  A halo that looks like one entry in the
  catalog can turn out to be two clumps mid-merger, or a chance superposition
  along the line of sight.  The wide panels show its surroundings.

  Is the high-resolution region clean?  A zoom is only usable if the coarse,
  heavy particles from the parent box have stayed out of it.  Contamination
  creeps in when the Lagrangian region was cut too small, and it is invisible
  in a density projection -- so the species panel colors particles by mass and
  the printed metric gives the closest approach of each coarse species in units
  of Rvir.

This is the one part of the pipeline that needs yt, and it reads particle data
rather than making AMR projections: for these questions a mass-weighted 2D
histogram of the particles is both faster and clearer than a ProjectionPlot.

The per-level re-centering is the subtle part.  Each MUSIC level shifts the
domain, so the halo sits at a different code position in each run.  That is
computed from the same cross-checked shift parser the builder uses, not
duplicated here.
"""

import os

import numpy as np

try:
    from . import build as _build
    from . import config as _config
    from . import state as _state
except ImportError:
    import build as _build
    import config as _config
    import state as _state


# Panel widths, in units of Rvir, from "where does it sit" to "the halo itself".
DEFAULT_WIDTHS_RVIR = (40.0, 8.0, 2.0)

# A particle is "coarse" if it is heavier than the finest species by more than
# this factor.  Refinement is by factors of 8 in mass, so 1.5 separates species
# cleanly without being fooled by round-off.
COARSE_FACTOR = 1.5


def center_in_run(box, halo_id, level, halo_dir, rvir_min=None, phase="DM"):
    """Halo center in the coordinate frame of the level-N *run*, and Rvir.

    Note this is not the same as the center written into the level-N build
    config.  A level-N run lives in the frame its own ICs were made in, so it
    carries level N's shift; the config that *builds* level N carries level
    N-1's.  Off by one, and easy to get wrong -- see build.center_for_level.
    """
    center, zoom_radius = _build.halo_center_and_radius(box, halo_id, rvir_min)
    # Diagnostics are quoted against the true virial radius, not the zoom
    # radius, which may have been floored up (80 kpc for this box) to give the
    # Lagrangian region a workable size.  Using the inflated one would make
    # contamination look further out than it is.
    rvir = _build.catalog_rvir(box, halo_id)
    if level == 0:
        return np.array(center), rvir, zoom_radius
    # THE GAS STAGE HAS ITS OWN SHIFT. Its ICs are a separate MUSIC run, and
    # MUSIC picks the domain shift per run: halo80181's L2 DM is
    # (-212,-69,-250) and its L2 gas is (-201,-96,+250) -- the z component
    # flips sign, half a box apart. Reading the DM conf log for a gas stage
    # therefore centres the panel on empty sky. It went unnoticed because L3
    # and L3-gas happen to share a shift, so only the L2-gas panel was blank.
    suffix = "-gas" if str(phase) == "gas" else ""
    conf_log = os.path.join(halo_dir, "%s-L%d%s.conf_log.txt"
                            % (box.sim_name, level, suffix))
    if not os.path.exists(conf_log):
        raise RuntimeError("No %s -- cannot place the halo in the L%d frame"
                           % (conf_log, level))
    shifts = _build.read_shifts(conf_log)
    # Wrap into [0,1) -- the box is periodic and the shift routinely carries a
    # center past an edge.  See build.center_for_level.
    return (np.array([(c + s / box.shift_divisor) % 1.0
                      for c, s in zip(center, shifts)]),
            rvir, zoom_radius)


def locate_halo(rel_kpc, mass, guess_radius_kpc, verbose=True):
    """Find the halo by shrinking spheres on the finest particles.

    The analytic center -- catalog position plus the MUSIC domain shift -- is
    only a starting guess.  The halo's z=0 position is not identical between the
    parent box and the zoom: refining changes the merger history slightly, and
    the object drifts.  Measured on halo79628 L1 the drift is ~260 kpc, which is
    ten times its virial radius, so quoting anything against the analytic center
    would be meaningless -- it lands in empty space with one particle inside
    Rvir instead of five thousand.

    Returns the offset from the guess, in kpc, as a vector.  That offset is
    itself worth looking at: a small one means the zoom reproduced the parent
    box, and a large one means the halo moved or a different object was
    targeted.
    """
    fine = mass < COARSE_FACTOR * mass.min()
    if fine.sum() < 50:
        return None

    pts = rel_kpc[fine]

    # Shrink from the analytic center rather than from the densest cell in the
    # whole region.  Seeding on the global peak finds whichever halo is biggest
    # nearby, which is often not the target: doing that moved halo79628 L0 by
    # 1.6 Mpc onto a neighbor, when its catalog center was already exact
    # (191 particles inside Rvir, enclosed mass 2.6e9 against a catalog Mvir of
    # 2.0e9).  Starting at the guess and contracting converges on the object
    # the guess refers to.
    center = np.zeros(3)
    radius = guess_radius_kpc
    converged = False
    while radius > 5.0:
        d = np.sqrt(((pts - center) ** 2).sum(axis=1))
        inside = d < radius
        if inside.sum() < 50:
            radius *= 0.8
            continue
        center = pts[inside].mean(axis=0)
        radius *= 0.8
        converged = True
    return center if converged else None


def last_output(stage_dir):
    """(path, name, redshift, is_final) for the best dump available.

    Prefers the final redshift dump.  Falls back to the newest dump of any kind
    so a run still in progress can be inspected -- but the caller is told it is
    not the final one, because the catalog position and Rvir are z = 0
    quantities and mean nothing at z = 4.  A half-built halo compared against
    its eventual virial radius produces a confident, meaningless answer.
    """
    log = _state.read_output_log(stage_dir)
    param = _state.find_param_file(stage_dir)
    final = _state.final_dump(param)
    redshifts = _state.output_redshifts(param)

    for name, is_final in ([(final, True)] if final else []) + \
                          ([(log.last_name, False)] if log.last_name else []):
        path = os.path.join(stage_dir, name, name)
        if name and os.path.exists(path):
            z = _state.dump_redshift(stage_dir, name, redshifts)
            return path, name, z, is_final
    return None, None, None, False


def load_particles(dataset_path, center, half_width_kpc):
    """Particle positions (kpc, relative to center) and masses (Msun) in a box.

    Reads a region rather than the whole domain: the parent box has 134 million
    particles and only the neighborhood matters here.
    """
    import yt

    ds = yt.load(dataset_path)

    # A sphere, not a box, and the separation computed periodically.  Halos sit
    # wherever they sit: halo79628 is at x = 0.998, hard against the domain
    # edge, and clamping the region at the boundary cut half its neighborhood
    # away -- which looks like a void in the plot rather than like a clipped
    # region.
    region = ds.sphere(center, (half_width_kpc, "kpc"))

    pos = np.array([region["all", "particle_position_%s" % ax].in_units("code_length").d
                    for ax in "xyz"])
    mass = region["all", "particle_mass"].in_units("Msun").d
    if mass.size == 0:
        return None, None, ds

    kpc_per_code = float(ds.quan(1.0, "code_length").in_units("kpc").d)
    delta = pos.T - np.array(center)
    delta -= np.round(delta)          # wrap to [-0.5, 0.5] in code units
    return delta * kpc_per_code, mass, ds


# Coarse mass fraction inside Rvir above which the zoom is called contaminated.
# Judging on a single particle's closest approach is too noisy -- one stray
# reads the same as a genuine incursion -- so the verdict uses the fraction of
# the mass inside Rvir that is in coarse species, with the closest approach
# reported alongside as context.
CONTAMINATION_TOLERANCE = 0.01


def contamination(rel_kpc, mass, rvir_kpc):
    """Coarse-particle intrusion, per species.

    Returns (rows, clean, coarse_fraction).  Each row is
    (mass, total count, closest approach in Rvir, count inside Rvir).
    """
    radius = np.sqrt((rel_kpc ** 2).sum(axis=1))
    inside = radius < rvir_kpc
    finest = mass.min()
    species = np.unique(np.round(mass / finest, 3))

    rows = []
    for ratio in species:
        sel = np.isclose(mass / finest, ratio, rtol=1e-3)
        closest = radius[sel].min() / rvir_kpc
        rows.append((finest * ratio, int(sel.sum()), closest, int((sel & inside).sum())))

    mass_inside = mass[inside]
    if mass_inside.size:
        coarse_inside = mass_inside[mass_inside > COARSE_FACTOR * finest]
        fraction = float(coarse_inside.sum() / mass_inside.sum())
    else:
        fraction = 0.0
    return rows, fraction <= CONTAMINATION_TOLERANCE, fraction


def make_qc_figure(box, halo_id, levels=None, out_path=None, rvir_min=None,
                   widths_rvir=DEFAULT_WIDTHS_RVIR, verbose=True):
    """Build the diagnostic figure for one halo across its levels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm

    halo_dir = box.halo_dir(halo_id)

    # Which levels actually have data.  L0 is the shared parent box.
    if levels is None:
        levels = []
        l0 = os.path.join(_config.foggie_ics_dir(), "%s-L0" % box.sim_name)
        if os.path.isdir(l0):
            levels.append(0)
        for lev in range(1, box.max_level + 1):
            if os.path.isdir(box.stage_dir(halo_id, lev, "DM")):
                levels.append(lev)
    if not levels:
        raise RuntimeError("No level directories found for halo %s" % halo_id)

    ncol = len(widths_rvir) + 1
    fig, axes = plt.subplots(len(levels), ncol,
                             figsize=(4.0 * ncol, 3.9 * len(levels)),
                             squeeze=False)

    report = []
    for irow, level in enumerate(levels):
        stage_dir = (os.path.join(_config.foggie_ics_dir(), "%s-L0" % box.sim_name)
                     if level == 0 else box.stage_dir(halo_id, level, "DM"))
        dataset, dump, zdump, is_final = last_output(stage_dir)
        center, rvir, zoom_radius = center_in_run(box, halo_id, level, halo_dir, rvir_min)

        if verbose:
            zlabel = "z=%.2f" % zdump if zdump is not None else "z=?"
            print("  L%d  %s  %s  center %s%s"
                  % (level, dump or "NO OUTPUT", zlabel, np.round(center, 6),
                     "" if is_final else "   [not the final dump]"))
        if dataset is None:
            for ax in axes[irow]:
                ax.text(0.5, 0.5, "L%d: no output yet" % level, ha="center",
                        va="center", transform=ax.transAxes, color="0.4")
                ax.set_xticks([]); ax.set_yticks([])
            report.append((level, dump, None, None, None, None))
            continue

        widest = max(widths_rvir) * rvir
        # Read a margin beyond the widest panel.  A box exactly as wide as the
        # panel leaves its corners empty, which reads as a hard edge in the
        # image and looks like a feature of the simulation rather than of the
        # region cut.  sqrt(2) covers the corners; 1.6 also absorbs the
        # re-centering shift.
        rel, mass, ds = load_particles(dataset, center, 1.6 * widest)
        if rel is None:
            report.append((level, dump, None, None, None, None))
            continue

        # Re-center on the halo itself; the analytic guess can be far off.
        # At L0 the catalog position is definitional -- Rockstar was run on this
        # very output -- so it is used as-is.  Verified on halo79628: 191
        # particles inside Rvir and an enclosed mass of 2.6e9 against a catalog
        # Mvir of 2.0e9.  Re-centering there can only walk onto a neighbor.
        #
        # Above L0 the halo has to be found: it does not sit where the parent
        # box put it.  halo79628 lands ~270 kpc away at both L1 and L2, and the
        # two agree, so the displacement is the object having moved rather than
        # the shift arithmetic being wrong.  Generous but bounded, so a nearby
        # cluster cannot capture the search.
        if level == 0:
            offset = np.zeros(3)
        else:
            offset = locate_halo(rel, mass, guess_radius_kpc=max(25.0 * rvir, 500.0))
        if offset is None:
            if verbose:
                print("       could not locate the halo -- too few fine particles")
            report.append((level, dump, None, None, None, None))
            for ax in axes[irow]:
                ax.text(0.5, 0.5, "L%d: halo not located" % level, ha="center",
                        va="center", transform=ax.transAxes, color="#c62828")
                ax.set_xticks([]); ax.set_yticks([])
            continue
        rel = rel - offset
        drift = float(np.sqrt((offset ** 2).sum()))
        if verbose:
            note = "" if drift < 5 * rvir else "   <-- large; check it is the same object"
            print("       halo found %.1f kpc (%.1f Rvir) from the analytic center%s"
                  % (drift, drift / rvir, note))

        rows, clean, frac = contamination(rel, mass, rvir)
        report.append((level, dump, rows, clean if is_final else None, frac, drift))

        finest = mass.min()
        coarse = mass > COARSE_FACTOR * finest

        for icol, wr in enumerate(widths_rvir):
            ax = axes[irow][icol]
            half = wr * rvir
            # Cut the depth as well as the width.  Projecting the full loaded
            # sphere puts megaparsecs of foreground and background into every
            # panel, which at L3 buried the halo under line-of-sight coarse
            # particles and made a clean zoom look contaminated.
            sel = ((np.abs(rel[:, 0]) < half) & (np.abs(rel[:, 1]) < half)
                   & (np.abs(rel[:, 2]) < half))
            nbin = 320
            H, xe, ye = np.histogram2d(rel[sel, 0], rel[sel, 1], bins=nbin,
                                       range=[[-half, half], [-half, half]],
                                       weights=mass[sel])
            H[H <= 0] = np.nan
            ax.imshow(H.T, origin="lower", extent=[-half, half, -half, half],
                      norm=LogNorm(), cmap="viridis", interpolation="nearest")
            ax.add_patch(plt.Circle((0, 0), rvir, fill=False, color="white",
                                    lw=1.0, ls="--", alpha=0.8))
            if zoom_radius > rvir * 1.01 and half > zoom_radius:
                ax.add_patch(plt.Circle((0, 0), zoom_radius, fill=False,
                                        color="#ffd54f", lw=0.9, ls=":", alpha=0.9))
            ax.set_xticks([]); ax.set_yticks([])
            if icol == 0:
                zlabel = "z=%.2f" % zdump if zdump is not None else ""
                ax.set_ylabel("L%d\n%s\n%s" % (level, dump, zlabel), fontsize=10,
                              color=("black" if is_final else "#a8571b"))
            ax.set_title("%g x Rvir  (%.0f kpc)" % (wr, 2 * half), fontsize=9)

        # Species panel: fine vs coarse particles, at the middle width.
        ax = axes[irow][-1]
        half = widths_rvir[1] * rvir if len(widths_rvir) > 1 else widest
        sel = ((np.abs(rel[:, 0]) < half) & (np.abs(rel[:, 1]) < half)
               & (np.abs(rel[:, 2]) < half))
        ax.scatter(rel[sel & ~coarse, 0], rel[sel & ~coarse, 1], s=0.05,
                   c="#2f6fb5", alpha=0.35, linewidths=0, label="fine")
        nc = int((sel & coarse).sum())
        if nc:
            ax.scatter(rel[sel & coarse, 0], rel[sel & coarse, 1], s=2.5,
                       c="#c62828", alpha=0.9, linewidths=0,
                       label="coarse (%d)" % nc)
        ax.add_patch(plt.Circle((0, 0), rvir, fill=False, color="0.2", lw=1.0, ls="--"))
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("particle species", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85, markerscale=6)
        verdict = ("CLEAN" if clean else "CONTAMINATED") if is_final else "IN PROGRESS"
        if not is_final:
            ax.text(0.5, 0.5,
                    "run has not reached z = 0\n(%s, z = %.2f)\nz = 0 Rvir and center\ndo not apply yet"
                    % (dump, zdump if zdump is not None else float("nan")),
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#a8571b",
                    bbox=dict(boxstyle="round", fc="white", ec="#a8571b", alpha=0.9))
        ax.text(0.03, 0.04,
                ("%s   coarse mass in Rvir: %.2f%%" % (verdict, 100 * frac)
                 if is_final else verdict),
                transform=ax.transAxes, fontsize=9, weight="bold",
                color=("#a8571b" if not is_final else ("#1f7a52" if clean else "#c62828")))
        ax.text(0.03, 0.93, "recentered %.0f kpc" % drift, transform=ax.transAxes,
                fontsize=8, color="0.35")

    subtitle = "dashed white = Rvir = %.1f kpc" % rvir
    if zoom_radius > rvir * 1.01:
        subtitle += ";  dotted yellow = zoom radius = %.1f kpc" % zoom_radius
    fig.suptitle("halo %s   %s   zoom diagnostics   (%s)"
                 % (halo_id, box.sim_name, subtitle), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = out_path or os.path.join(halo_dir, "qc_halo%s.png" % halo_id)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path, report


def format_report(halo_id, report):
    """Text summary of the contamination metrics."""
    lines = ["", "halo %s contamination by level" % halo_id,
             "  %-5s %-8s %-12s %-10s %-9s %-9s %s"
             % ("level", "dump", "mass/Msun", "particles", "in Rvir", "closest", "")]
    for level, dump, rows, clean, frac, drift in report:
        if rows is None:
            lines.append("  L%-4d %-8s  (no data)" % (level, dump or "-"))
            continue
        for i, (m, n, closest, n_in) in enumerate(rows):
            tag = "finest" if i == 0 else "coarse"
            flag = "  <-- inside Rvir" if (tag == "coarse" and n_in) else ""
            lines.append("  %-5s %-8s %-12.3e %-10d %-9d %6.2f Rvir  %s%s"
                         % ("L%d" % level if i == 0 else "", dump if i == 0 else "",
                            m, n, n_in, closest, tag, flag))
        verdict = ("CLEAN" if clean else "CONTAMINATED") if clean is not None else \
                  "IN PROGRESS -- not yet at z=0, so z=0 Rvir and center do not apply"
        lines.append("  %-5s %-8s coarse mass inside Rvir: %.3f%%   => %s"
                     % ("", "", 100 * frac, verdict))
        lines.append("  %-5s %-8s recentered %.1f kpc from the analytic center"
                     % ("", "", drift))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Projected density across the refinement ladder
# ---------------------------------------------------------------------------

# One width, not the three the contamination figure uses.  The question here is
# whether refinement resolves the same object or takes it apart, and 5 Rvir is
# wide enough to show the halo's immediate surroundings -- the filament it sits
# in, the subhalos falling down it -- while leaving the halo itself more than a
# third of the frame.
DENSITY_WIDTH_RVIR = 5.0

# CIC rather than ('deposit', 'all_density'): the cloud-in-cell kernel spreads
# each particle over the cells it straddles, which is what makes a coarse level
# legible at all.  Nearest-grid-point at L0 is 9 cells of shot noise.
DENSITY_FIELD = ("deposit", "all_cic")

DENSITY_NPIX = 800

# The context row.  Wide enough to show the filament and the neighbors a halo
# sits among, which is what lets you confirm by eye that every level captured
# the *same* object: the halo itself changes appearance a lot between L0 and
# L2, while the pattern of structure around it does not.  It is also the check
# that catches a zoom that centered on the wrong halo, which a 5 Rvir panel
# cannot -- at that width one dwarf looks much like another.
DENSITY_CONTEXT_MPC = 3.0

# Centring the panels on the mass-weighted centroid of the finest species was
# tried and rejected.  The finest species is the whole Lagrangian region, which
# by z = 0 is a lopsided Mpc-scale sprawl -- rms 0.6 to 1.8 Mpc for the halos
# measured -- so its centroid lands 320 to 480 kpc from the halo, further off
# than the catalog position is.  It is also awkward to compute honestly: over a
# bounded search box the answer depends on the box (85 kpc at +-400 kpc, 281 kpc
# at +-1500 kpc for halo42189 L1), and over the full domain it needs periodic
# wrapping or a region straddling the boundary reports a centroid a full box
# away.  It measures where the zoom region went, which is a real question, but
# not this figure's.


def density_stages(box, halo_id, include_gas=False):
    """[(label, level, phase, stage_dir), ...] for the ladder, coarsest first.

    L0 is the shared parent box and is the natural first panel: whatever the
    halo looks like there is what the zoom exists to improve on.
    """
    stages = []
    l0 = os.path.join(_config.foggie_ics_dir(), "%s-L0" % box.sim_name)
    if os.path.isdir(l0):
        stages.append(("L0", 0, "DM", l0))
    for lev in range(1, box.max_level + 1):
        d = box.stage_dir(halo_id, lev, "DM")
        if os.path.isdir(d):
            stages.append(("L%d" % lev, lev, "DM", d))
        if include_gas:
            g = box.stage_dir(halo_id, lev, "gas")
            if os.path.isdir(g):
                stages.append(("L%d-gas" % lev, lev, "gas", g))
    return stages


def _density_panel(box, halo_id, halo_dir, level, phase, stage_dir, width_rvir,
                   context_mpc=DENSITY_CONTEXT_MPC, recenter=False):
    """Project one stage.  Returns a dict of everything the figure needs, or None.

    By default every panel is on the *catalog* center carried into that stage's
    own frame -- no shrinking-sphere re-centering -- so the panels compare the
    same coordinate and the drift between levels stays visible.  The halo does
    not land in exactly the same place at every level: refining the Lagrangian
    region perturbs the merger history slightly.

    The drift is measured either way, because it decides whether the panel means
    anything.  It runs one to two hundred kpc, and that is roughly a *fixed
    physical scale* rather than a fixed fraction of Rvir -- so for a 127 kpc
    halo it is a fraction of the virial radius and the halo stays comfortably in
    frame, while for a 29 kpc dwarf it is five to seven Rvir and the halo leaves
    the frame entirely.  An uncentered panel of a dwarf therefore looks exactly
    like a halo that dissolved under refinement, which is the one conclusion
    this figure exists to test.  Hence recenter, and hence the drift printed on
    every panel whether or not it is used.
    """
    import yt

    snap, name, zdump, is_final = last_output(stage_dir)
    if snap is None:
        return None
    if not is_final:
        # The catalog center and Rvir are z = 0 quantities.  Framing a halo at
        # z = 5 with its eventual virial radius produces a confident, wrong
        # picture -- panels of mostly-empty space that read as a halo that
        # failed to form.  Skip the stage and say so rather than draw it.
        return dict(label=None, name=name, z=zdump, skipped="not at z = 0")

    center, rvir_kpc, _ = center_in_run(box, halo_id, level, halo_dir,
                                        phase=phase)
    center = np.array(center)
    # Rvir and the catalog positions carry the same h and the same comoving
    # convention, so dividing both by the box size in the same units cancels
    # both.  h never enters the geometry; it enters only the axis labels, which
    # yt writes in physical kpc.
    rvir_code = rvir_kpc / (box.boxsize_mpc * 1000.0)
    width_code = width_rvir * rvir_code

    ds = yt.load(snap)
    rvir_phys = float(ds.quan(rvir_code, "code_length").to("kpc").d)

    # Where the halo actually is, by shrinking spheres on the finest particles.
    # Searched out to well beyond the frame, since the whole point is that it
    # can be several frames away.
    drift, offset_code = None, np.zeros(3)
    search_kpc = max(6.0 * rvir_phys, 400.0)
    rel, mass, _ = load_particles(snap, center, search_kpc)
    if rel is not None:
        offset = locate_halo(rel, mass, guess_radius_kpc=search_kpc, verbose=False)
        if offset is not None:
            drift = float(np.sqrt((offset ** 2).sum()))
            kpc_per_code = float(ds.quan(1.0, "code_length").to("kpc").d)
            offset_code = np.asarray(offset) / kpc_per_code

    if recenter and drift is not None:
        center = center + offset_code

    field = DENSITY_FIELD
    if field not in ds.derived_field_list:
        field = ("deposit", "all_density")

    # Both rows come off one dataset load, which is the expensive part.
    widths = [(width_code, "%.0f Rvir" % width_rvir),
              (float(ds.quan(context_mpc, "Mpc").to("code_length").d),
               "%.1f Mpc" % context_mpc)]

    images = []
    for wcode, wlabel in widths:
        half = wcode / 2.0
        # A cube, not the full depth.  Projecting the whole 25 Mpc along the
        # line of sight buries a 0.4 Mpc halo in foreground and background, and
        # takes about twenty times as long.
        region = ds.region(center, center - half, center + half)
        frb = ds.proj(field, 2, data_source=region).to_frb(
            (wcode, "code_length"), DENSITY_NPIX, center=center)
        images.append(dict(img=np.array(frb[field]), width_label=wlabel,
                           half_kpc=float(ds.quan(half, "code_length").to("kpc").d)))

    return dict(name=name, z=abs(float(ds.current_redshift)), skipped=None,
                images=images, field=field,
                half_kpc=images[0]["half_kpc"],
                rvir_phys=rvir_phys, rvir_kpc=rvir_kpc,
                hubble=float(ds.hubble_constant),
                drift=drift, recentered=bool(recenter and drift is not None))


def make_density_figure(box, halo_id, out_path=None, width_rvir=DENSITY_WIDTH_RVIR,
                        context_mpc=DENSITY_CONTEXT_MPC, include_gas=False,
                        recenter=False, verbose=True):
    """Projected density for one halo at every level, coarsest to finest.

    Columns are refinement levels; rows are two widths on the same center.

    The top row, a few Rvir across, answers: does the same object appear, in the
    same place, at the same size, as resolution increases?  A halo that is a
    clean concentration at L0 and L1 but fragments or fades by L3 has not been
    resolved better, it has been resolved into something else -- the failure
    mode that matters most for the smallest halos, whose Lagrangian regions hold
    few enough particles that refinement can take the object apart.

    The bottom row, a few Mpc across, answers a question the top row cannot: is
    this the same halo at all?  At 5 Rvir one dwarf looks much like another, and
    a zoom that centered on a neighbor would still produce a plausible top row.
    The surrounding filaments and neighbors are the fingerprint -- they barely
    change between levels, so a level whose context does not match the others
    is not the same piece of the universe.

    Returns (out_path, rows) where rows is one record per stage, including the
    stages that were skipped and why.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    halo_dir = box.halo_dir(halo_id)
    stages = density_stages(box, halo_id, include_gas)
    if not stages:
        raise RuntimeError("No stage directories with data for halo %s" % halo_id)

    panels, rows = [], []
    for label, level, phase, stage_dir in stages:
        panel = _density_panel(box, halo_id, halo_dir, level, phase, stage_dir,
                               width_rvir, context_mpc=context_mpc,
                               recenter=recenter)
        if panel is None:
            rows.append(dict(label=label, name=None, z=None, drift=None,
                             note="no dump on disk"))
            if verbose:
                print("  %-8s skipped: no dump on disk" % label)
            continue
        if panel["skipped"]:
            rows.append(dict(label=label, name=panel["name"], z=panel["z"],
                             drift=None, note=panel["skipped"]))
            if verbose:
                print("  %-8s skipped: %s (%s, z = %s)"
                      % (label, panel["skipped"], panel["name"], panel["z"]))
            continue
        panel["label"] = label
        panels.append(panel)

        # A halo further from the frame center than the frame's half-width is
        # not in the picture at all.  Said plainly here because the panel itself
        # looks like a halo that dissolved.
        drift, half_kpc = panel["drift"], panel["half_kpc"]
        note = "plotted"
        if drift is None:
            note = "plotted; halo not located"
        elif panel["recentered"]:
            note = "plotted, re-centered %.0f kpc" % drift
        elif drift > half_kpc:
            note = ("OUT OF FRAME: halo is %.0f kpc away, frame half-width %.0f kpc"
                    % (drift, half_kpc))
        elif drift > 0.5 * half_kpc:
            note = "plotted; halo %.0f kpc off center" % drift
        rows.append(dict(label=label, name=panel["name"], z=panel["z"],
                         drift=drift, note=note))
        if verbose:
            print("  %-8s %s  z = %.2f   %s" % (label, panel["name"], panel["z"], note))

    if not panels:
        raise RuntimeError(
            "No stage of halo %s has reached z = 0, so there is nothing this "
            "figure can honestly compare." % halo_id)

    # Empty cells are genuinely zero, not missing data.  Left to the default
    # they take the figure's white background and read as holes in the map.
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    nrow = len(panels[0]["images"])
    fig, axes = plt.subplots(nrow, len(panels),
                             figsize=(5.0 * len(panels), 5.4 * nrow),
                             squeeze=False)

    for irow in range(nrow):
        # One color scale per row, shared across levels.  Across levels is the
        # comparison that matters: refinement redistributes mass into
        # substructure rather than creating it, so panels normalized
        # individually would look much the same however much the halo changed.
        # Across *rows* it would be wrong -- the two rows integrate through
        # different depths, so a shared scale would say more about the depth
        # than about the structure.  The top percentile rather than the maximum,
        # so one dense cell in the finest panel does not wash out the coarse
        # ones.
        row_imgs = [p["images"][irow]["img"] for p in panels]
        allpos = np.concatenate([im_[im_ > 0].ravel() for im_ in row_imgs])
        vmax = float(np.percentile(allpos, 99.99))
        vmin = vmax * 1e-4

        for ax, p in zip(axes[irow], panels):
            entry = p["images"][irow]
            half = entry["half_kpc"]
            extent = [-half, half, -half, half]
            # No transpose.  yt's FRB is already indexed [vertical, horizontal]
            # for imshow; .T silently swaps x and y, which looks entirely
            # plausible and puts every substructure on the wrong side.
            im = ax.imshow(entry["img"], origin="lower", extent=extent,
                           norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)
            ax.add_artist(plt.Circle((0, 0), p["rvir_phys"], fill=False,
                                     color="w", lw=1.2, ls="--", alpha=0.9))
            ax.set_title("%s   %s   z = %.2f   (%s)"
                         % (p["label"], p["name"], p["z"], entry["width_label"]),
                         fontsize=11)
            ax.set_xlabel("kpc")

            # The drift, on the panel itself.  An empty panel and an empty panel
            # whose halo is 210 kpc away look identical, and only one of them
            # means the halo dissolved.  Judged against this row's half-width,
            # since a halo out of frame in the 5 Rvir row is usually well inside
            # the 3 Mpc one -- which is exactly the point of the second row.
            drift = p["drift"]
            if drift is None:
                ax.text(0.03, 0.955, "halo not located", transform=ax.transAxes,
                        fontsize=9, color="#ffb74d", weight="bold")
            elif p["recentered"]:
                ax.text(0.03, 0.955, "re-centered %.0f kpc" % drift,
                        transform=ax.transAxes, fontsize=9, color="0.85")
            elif drift > half:
                ax.text(0.03, 0.955, "HALO OUT OF FRAME\n%.0f kpc = %.1f Rvir away"
                        % (drift, drift / p["rvir_phys"]), transform=ax.transAxes,
                        fontsize=9, color="#ff5252", weight="bold", va="top")
            else:
                ax.text(0.03, 0.955, "%.0f kpc = %.2f Rvir off center"
                        % (drift, drift / p["rvir_phys"]), transform=ax.transAxes,
                        fontsize=9, color="0.85")
        axes[irow][0].set_ylabel("kpc")
        fig.colorbar(im, ax=axes[irow].tolist(), fraction=0.023, pad=0.01,
                     label=r"projected DM density  (g cm$^{-2}$)")
    # Mvir alongside Rvir, both from the catalog and both quoted the same way:
    # the physical value first, the raw h-carrying catalog number after it.  The
    # registry notes quote the catalog value, so showing only the converted one
    # would read as a different halo.
    mvir_h = _build.catalog_mvir(box, halo_id)
    hubble = panels[0]["hubble"]
    fig.suptitle("halo%s   %s   Mvir = %.2e Msun (%.2e Msun/h)   "
                 "Rvir = %.0f kpc (%.1f kpc/h)\n"
                 "top row %.0f Rvir across, bottom row %.1f Mpc for context; "
                 "each projected through a cube of its own width   %s"
                 % (halo_id, box.sim_name, mvir_h / hubble, mvir_h,
                    panels[0]["rvir_phys"], panels[0]["rvir_kpc"],
                    width_rvir, context_mpc,
                    "re-centered on the halo" if recenter
                    else "catalog center, no re-centering"), fontsize=12)

    out_path = out_path or os.path.join(halo_dir, "qc_density_halo%s.png" % halo_id)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path, rows


def format_density_report(halo_id, rows):
    """Text summary of which stages went into the density figure, and the drift."""
    lines = ["", "halo %s density panels" % halo_id,
             "  %-9s %-8s %-7s %-9s %s"
             % ("stage", "dump", "z", "drift", "note")]
    for r in rows:
        lines.append("  %-9s %-8s %-7s %-9s %s"
                     % (r["label"], r["name"] or "-",
                        ("%.2f" % r["z"]) if r["z"] is not None else "-",
                        ("%.0f kpc" % r["drift"]) if r["drift"] is not None else "-",
                        r["note"]))
    if any(r["note"].startswith("OUT OF FRAME") for r in rows):
        lines += [
            "",
            "  Drift is the distance from the catalog center to where the halo",
            "  actually is at this level, and it is a roughly fixed physical",
            "  scale -- one to two hundred kpc -- rather than a fixed fraction",
            "  of Rvir.  A panel marked OUT OF FRAME is not showing a halo that",
            "  dissolved under refinement; it is showing the wrong piece of sky.",
            "  Re-run with --recenter to put the panels on the halo itself.",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Is the halo we asked for actually inside the region we built?
# ---------------------------------------------------------------------------

# How far the halo may sit from its analytic center before the region is judged
# to be built around something else.  Not a tuned number: measured across the
# twelve zooms that have gas runs, the ten sound ones land at 27-251 kpc/h and
# the two broken ones at 1315 and 1524, so anything from ~400 to ~1200 splits
# them identically.  600 sits in the middle of that gap.
TARGET_OFFSET_TOLERANCE_KPC = 600.0

# Below this the shrink has not found a halo at all, only noise.
MIN_FINE_INSIDE_RVIR = 50


def check_target_in_region(box, halo_id, level, halo_dir=None, rvir_min=None,
                           tolerance_kpc=TARGET_OFFSET_TOLERANCE_KPC):
    """Verify the level-N run refined the halo it was built for.

    Nothing else in the pipeline asks this.  MUSIC prints the convex-hull
    centroid it chose and moves on; check_region_points counts the particles
    that were traced but never asks where they ended up; the density figure
    measures the drift and only draws it.  So a zoom that refines the wrong
    object passes every gate and is caught, if at all, when someone fails to
    find the target in its catalogs -- which for halo79628 was after 3.6 days
    on 128 ranks, refining an 8.4e9 neighbour 1.9 Mpc/h from the 2.7e9 halo
    named in the registry.

    The test is the one that separates the fleet cleanly: start at the analytic
    center (catalog position plus this level's MUSIC shift) and shrink onto the
    finest particles.  If the region really was built around the target, the
    shrink stays put -- ten of the twelve zooms with gas runs converge within
    251 kpc/h.  If it was built around something else there is nothing to
    converge onto and the shrink either wanders off or finds too little mass.

    Deliberately does NOT seed on the densest clump in the region: that finds
    whichever halo is biggest nearby, which for a mis-built zoom is exactly the
    interloper, and it would report a confident success.  See locate_halo.

    Returns a dict; ``ok`` is False when the region cannot be trusted.
    """
    halo_dir = halo_dir or box.halo_dir(halo_id)
    center, rvir, zoom_radius = center_in_run(box, halo_id, level, halo_dir,
                                              rvir_min)
    stage_dir = box.stage_dir(halo_id, level, "DM")
    path, name, z, is_final = last_output(stage_dir)
    out = dict(halo_id=halo_id, level=level, dump=name, z=z, final=is_final,
               rvir_kpc=rvir, offset_kpc=None, n_fine_rvir=None,
               m_fine_rvir=None, m_catalog=None, coarse_fraction=None,
               ok=False, note="")
    if path is None:
        out["note"] = "no dump on disk"
        return out
    if not is_final:
        # Rvir and the catalog position are z = 0 quantities; measuring against
        # them at z = 4 gives a confident, meaningless answer.
        out["note"] = "run has not reached z = 0 (%s) -- not checkable yet" % name
        return out

    rel, mass, _ = load_particles(path, center, max(25.0 * rvir, 500.0))
    if mass is None or mass.size == 0:
        out["note"] = "no particles near the analytic center"
        return out

    offset = locate_halo(rel, mass, guess_radius_kpc=max(25.0 * rvir, 500.0),
                         verbose=False)
    if offset is None:
        out["note"] = "too few fine particles to locate anything"
        return out
    out["offset_kpc"] = float(np.sqrt((np.asarray(offset) ** 2).sum()))

    rel = rel - offset
    radius = np.sqrt((rel ** 2).sum(axis=1))
    fine = mass < COARSE_FACTOR * mass.min()
    inside = radius < rvir
    out["n_fine_rvir"] = int((fine & inside).sum())
    _, _, out["coarse_fraction"] = contamination(rel, mass, rvir)

    try:
        out["m_catalog"] = float(_build.catalog_mvir(box, halo_id))
    except Exception:
        out["m_catalog"] = None

    if out["n_fine_rvir"] < MIN_FINE_INSIDE_RVIR:
        out["note"] = ("only %d fine particles inside Rvir -- no halo here"
                       % out["n_fine_rvir"])
    elif out["offset_kpc"] > tolerance_kpc:
        out["note"] = ("halo is %.0f kpc from its analytic center (tolerance "
                       "%.0f) -- the region was built around something else"
                       % (out["offset_kpc"], tolerance_kpc))
    else:
        out["ok"] = True
        out["note"] = "target found %.0f kpc from its analytic center" % out["offset_kpc"]
    return out


def format_target_check(rows):
    lines = ["", "  %-8s %-6s %-8s %-9s %-9s %-8s %s"
             % ("halo", "level", "dump", "offset", "n_fine", "coarse", "verdict")]
    for r in rows:
        lines.append("  %-8s L%-5d %-8s %-9s %-9s %-8s %s"
                     % (r["halo_id"], r["level"], r["dump"] or "-",
                        ("%.0f kpc" % r["offset_kpc"]) if r["offset_kpc"] is not None else "-",
                        r["n_fine_rvir"] if r["n_fine_rvir"] is not None else "-",
                        ("%.3f" % r["coarse_fraction"]) if r["coarse_fraction"] is not None else "-",
                        ("OK   " if r["ok"] else "FAIL ") + r["note"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Projection with the Rockstar catalog overlaid
# ---------------------------------------------------------------------------

# Panel half-width, in Mpc/h comoving.  Wide enough to show the neighbourhood
# the zoom is embedded in, which is the point of the figure.
NEIGHBOR_PANEL_MPCH = 1.0

# Only circle halos big enough to matter; below this the panel is a thicket of
# labels and the IDs are unreadable.
NEIGHBOR_MIN_MVIR = 1.0e8


def catalog_neighbors(box, center_code, half_code, min_mvir=NEIGHBOR_MIN_MVIR,
                      level=0, halo_dir=None):
    """Rockstar halos inside the panel, in the frame of the level-N run.

    The IDs are the ORIGINAL parent-box Rockstar IDs, deliberately: the whole
    use of this figure is correlating what the zoom refined against the catalog
    the halo was selected from, and a zoom-local AHF numbering cannot do that.

    The catalog is in the unshifted parent frame and the run is not, so every
    position gets the same shift center_in_run applies to the target -- catalog
    position plus this level's MUSIC shift, wrapped.  Getting this wrong puts
    the circles in plausible but wrong places, which is worse than no circles.
    """
    halos = _build._read_catalog(box.catalog_path())
    pos = np.column_stack([np.asarray(halos[c], dtype=float) / box.boxsize_mpc
                           for c in ("X", "Y", "Z")])
    if level > 0:
        conf_log = os.path.join(halo_dir or box.halo_dir(0),
                                "%s-L%d.conf_log.txt" % (box.sim_name, level))
        shifts = _build.read_shifts(conf_log)
        pos = (pos + np.array(shifts) / box.shift_divisor) % 1.0

    d = pos - np.asarray(center_code)
    d -= np.round(d)                      # periodic
    mvir = np.asarray(halos["Mvir"], dtype=float)
    sel = (np.abs(d) <= half_code).all(axis=1) & (mvir >= min_mvir)
    return dict(ids=np.asarray(halos["ID"], dtype=np.int64)[sel],
                mvir=mvir[sel],
                rvir_code=np.asarray(halos["Rvir"], dtype=float)[sel]
                / (box.boxsize_mpc * 1000.0),
                rel=d[sel])


def make_neighbor_projection(box, halo_id, level=None, phase="DM", out_path=None,
                             half_mpch=NEIGHBOR_PANEL_MPCH,
                             min_mvir=NEIGHBOR_MIN_MVIR, rvir_min=None):
    """DM projection of one IC set with the Rockstar catalog circled on it.

    One panel per IC set, showing what the zoom actually refined and what sits
    around it, with every catalog halo drawn at its own Rvir and labelled by its
    parent-box Rockstar ID.  The target is labelled larger and its mass printed,
    so the figure answers "did we refine the halo we asked for, and what are its
    neighbours" without cross-referencing anything by hand.
    """
    import yt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import matplotlib.patheffects as _pe

    halo_dir = box.halo_dir(halo_id)
    level = box.final_level(halo_id) if level is None and hasattr(box, "final_level") else level
    stage_dir = box.stage_dir(halo_id, level, phase)
    snap, name, zdump, is_final = last_output(stage_dir)
    if snap is None:
        return None, "no dump on disk"

    # phase matters: the gas stage carries its own MUSIC domain shift.
    center, rvir_kpc, _ = center_in_run(box, halo_id, level, halo_dir, rvir_min,
                                        phase=phase)
    center = np.array(center)
    ds = yt.load(snap)
    half_code = float(half_mpch / box.boxsize_mpc)

    field = DENSITY_FIELD
    if field not in ds.derived_field_list:
        field = ("deposit", "all_density")
    region = ds.region(center, center - half_code, center + half_code)
    frb = ds.proj(field, 2, data_source=region).to_frb(
        (2 * half_code, "code_length"), DENSITY_NPIX, center=center)
    img = np.array(frb[field])

    nb = catalog_neighbors(box, center, half_code, min_mvir, level, halo_dir)
    # Axes in Mpc/h comoving, measured from the target rather than from the box
    # corner: the panel is about this halo, and absolute box coordinates change
    # under the MUSIC shift while offsets do not.
    ext = half_mpch
    fig, ax = plt.subplots(figsize=(9.0, 7.6))
    im = ax.imshow(img, origin="lower", extent=[-ext, ext, -ext, ext],
                   norm=matplotlib.colors.LogNorm(), cmap="magma")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"Projected DM Density  $\left(\mathrm{g\,cm^{-2}}\right)$")

    mpch = box.boxsize_mpc
    for hid, mv, rv, rel in zip(nb["ids"], nb["mvir"], nb["rvir_code"], nb["rel"]):
        x, y = rel[0] * mpch, rel[1] * mpch
        ax.add_patch(Circle((x, y), rv * mpch, fill=False, color="white",
                            lw=1.0, alpha=0.85))
        is_target = int(hid) == int(halo_id)
        # Offset in POINTS, not data units: a dwarf's Rvir is a few kpc and an
        # offset of that size in Mpc puts the label on top of the circle.
        ax.annotate(str(int(hid)), (x, y), color="white",
                    fontsize=11 if is_target else 6.5,
                    ha="left", va="bottom", xytext=(4, 4),
                    textcoords="offset points",
                    path_effects=[_pe.withStroke(linewidth=1.6, foreground="black")])
    tm = nb["mvir"][nb["ids"] == int(halo_id)]
    ax.text(0.03, 0.06, "%s\nM = %.2e" % (halo_id, tm[0] if len(tm) else float("nan")),
            transform=ax.transAxes, color="white", fontsize=13, va="bottom")
    ax.text(0.03, 0.955, "L%d %s   %s   z = %.2f   %d catalog halos > %.0e"
            % (level, phase, name, abs(zdump if zdump is not None else 0.0),
               len(nb["ids"]), min_mvir),
            transform=ax.transAxes, color="white", fontsize=8.5, va="top")
    ax.set_xlabel(r"$x$  (Mpc/h, comoving, from halo%s)" % halo_id)
    ax.set_ylabel(r"$y$  (Mpc/h, comoving, from halo%s)" % halo_id)
    fig.tight_layout()

    out_path = out_path or os.path.join(halo_dir, "qc_neighbors_halo%s_L%d%s.png"
                                        % (halo_id, level, phase))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path, "%d halos circled" % len(nb["ids"])
