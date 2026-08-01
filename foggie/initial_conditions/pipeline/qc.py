"""
Diagnostic plots for a zoom, one row per refinement level.

Answers the two questions you actually need answered before trusting a zoom:

  Is the target a single object?  A halo that looks like one entry in the
  catalog can turn out to be two clumps mid-merger, or a chance superposition
  along the line of sight.  The wide panels show its surroundings.

  Is the high-resolution region clean?  A zoom is only usable if the coarse,
  heavy particles from the parent box have stayed out of it.  Contamination
  creeps in when the Lagrangian region was cut too small, and it is invisible
  in a density projection -- so the species panel colours particles by mass and
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


def center_in_run(box, halo_id, level, halo_dir, rvir_min=None):
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
    conf_log = os.path.join(halo_dir, "%s-L%d.conf_log.txt" % (box.sim_name, level))
    if not os.path.exists(conf_log):
        raise RuntimeError("No %s -- cannot place the halo in the L%d frame"
                           % (conf_log, level))
    shifts = _build.read_shifts(conf_log)
    return (np.array([c + s / box.shift_divisor for c, s in zip(center, shifts)]),
            rvir, zoom_radius)


def locate_halo(rel_kpc, mass, guess_radius_kpc, verbose=True):
    """Find the halo by shrinking spheres on the finest particles.

    The analytic centre -- catalog position plus the MUSIC domain shift -- is
    only a starting guess.  The halo's z=0 position is not identical between the
    parent box and the zoom: refining changes the merger history slightly, and
    the object drifts.  Measured on halo79628 L1 the drift is ~260 kpc, which is
    ten times its virial radius, so quoting anything against the analytic centre
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

    # Shrink from the analytic centre rather than from the densest cell in the
    # whole region.  Seeding on the global peak finds whichever halo is biggest
    # nearby, which is often not the target: doing that moved halo79628 L0 by
    # 1.6 Mpc onto a neighbour, when its catalog centre was already exact
    # (191 particles inside Rvir, enclosed mass 2.6e9 against a catalog Mvir of
    # 2.0e9).  Starting at the guess and contracting converges on the object
    # the guess refers to.
    centre = np.zeros(3)
    radius = guess_radius_kpc
    converged = False
    while radius > 5.0:
        d = np.sqrt(((pts - centre) ** 2).sum(axis=1))
        inside = d < radius
        if inside.sum() < 50:
            radius *= 0.8
            continue
        centre = pts[inside].mean(axis=0)
        radius *= 0.8
        converged = True
    return centre if converged else None


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
    particles and only the neighbourhood matters here.
    """
    import yt

    ds = yt.load(dataset_path)

    # A sphere, not a box, and the separation computed periodically.  Halos sit
    # wherever they sit: halo79628 is at x = 0.998, hard against the domain
    # edge, and clamping the region at the boundary cut half its neighbourhood
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

        # Re-centre on the halo itself; the analytic guess can be far off.
        # At L0 the catalog position is definitional -- Rockstar was run on this
        # very output -- so it is used as-is.  Verified on halo79628: 191
        # particles inside Rvir and an enclosed mass of 2.6e9 against a catalog
        # Mvir of 2.0e9.  Re-centering there can only walk onto a neighbour.
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
            print("       halo found %.1f kpc (%.1f Rvir) from the analytic centre%s"
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
                    "run has not reached z = 0\n(%s, z = %.2f)\nz = 0 Rvir and centre\ndo not apply yet"
                    % (dump, zdump if zdump is not None else float("nan")),
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#a8571b",
                    bbox=dict(boxstyle="round", fc="white", ec="#a8571b", alpha=0.9))
        ax.text(0.03, 0.04,
                ("%s   coarse mass in Rvir: %.2f%%" % (verdict, 100 * frac)
                 if is_final else verdict),
                transform=ax.transAxes, fontsize=9, weight="bold",
                color=("#a8571b" if not is_final else ("#1f7a52" if clean else "#c62828")))
        ax.text(0.03, 0.93, "recentred %.0f kpc" % drift, transform=ax.transAxes,
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
                  "IN PROGRESS -- not yet at z=0, so z=0 Rvir and centre do not apply"
        lines.append("  %-5s %-8s coarse mass inside Rvir: %.3f%%   => %s"
                     % ("", "", 100 * frac, verdict))
        lines.append("  %-5s %-8s recentred %.1f kpc from the analytic centre"
                     % ("", "", drift))
    return "\n".join(lines)
