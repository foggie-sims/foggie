"""Driver for multi-halo zoom IC generation on the 512^3 25 Mpc/h box.

Replaces the awk/sed pipeline of halo_template_512/script512.py with a
self-contained Python workflow (nothing in the legacy template tree is
modified; the templates used here are the parameterized copies in
multizoom/templates/).

Per level::

    python multizoom512.py --halo_ids 5016,5033,2392 --level 1 --mode union \
        --music-exe-dir /path/to/music-build \
        --music-template /path/to/25Mpc_DM_512_planck18.conf \
        --workdir /path/to/rundir [--no-submit]

which (1) reads the rockstar catalog rows for every halo id, (2) writes the
multi-halo multizoom config, (3) traces every halo's Lagrangian region from
the previous level's run and generates the level's ICs via MUSIC (one union
run, or one run per halo merged into a multi-patch IC set), and (4)
assembles the Enzo run directory (parameter file, RunScript.sh, simrun.pl)
and submits it.

Level bookkeeping: --level n expects the level n-1 run to exist in the
workdir as <simulation_name>-L<n-1> with its <simulation_name>-L<n-1>.enzo
parameter file (as this driver lays it out).  In union mode the domain is
recentred by MUSIC at every level, so catalog-frame halo centers are
corrected by the accumulated shifts; in merge mode nothing ever shifts.
"""

import argparse
import configparser
import os
import shutil
import stat
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from multizoom import config as mzconfig  # noqa: E402
    from multizoom import mrp_music  # noqa: E402
else:
    from . import config as mzconfig
    from . import mrp_music

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(PACKAGE_DIR, "templates")
DEFAULT_CATALOG = os.path.normpath(os.path.join(
    PACKAGE_DIR, "..", "halo_catalogs_512", "512", "z0", "out_0.list"))


def accumulated_shift(workdir, simulation_name, level, levelmin):
    """Sum of domain shifts (code units) applied by levels 1..level-1."""
    total = [0.0, 0.0, 0.0]
    for prev in range(1, level):
        log_fn = os.path.join(workdir, "%s-L%d.conf_log.txt"
                              % (simulation_name, prev))
        if not os.path.exists(log_fn):
            raise RuntimeError(
                "Missing %s — the L%d run's MUSIC log is needed to correct "
                "halo centers for the domain shift." % (log_fn, prev))
        shift = [0, 0, 0]
        with open(log_fn) as fp:
            for line in fp:
                for i, ax in enumerate("xyz"):
                    if line.find("setup/shift_%s" % ax) >= 0:
                        shift[i] = int(line.split("=")[1])
        for i in range(3):
            total[i] += shift[i] / 2.0**levelmin
    return total


def write_multizoom_conf(args, centers, radii):
    """Write the multi-halo config consumed by mrp_music."""
    cf = configparser.ConfigParser()
    cf["setup"] = dict(
        music_exe_dir=args.music_exe_dir,
        simulation_name=args.simulation_name,
        template_config=args.music_template,
        original_config=args.music_template,
        simulation_run_directory=args.workdir,
        num_cores=str(args.num_cores) if args.num_cores else "None",
        mode=args.mode,
    )
    cf["region"] = dict(
        final_type="halo",
        final_redshift="%g" % args.final_redshift,
        radius_factor="%g" % args.radius_factor,
        shape_type="exact",
        halo_center_units="code_length",
        halo_radius_units="kpc",
    )
    for halo_id, center, radius in zip(args.halo_ids, centers, radii):
        cf["halo:%s" % halo_id] = dict(
            halo_center="%.8f, %.8f, %.8f" % tuple(center),
            halo_radius="%g" % radius,
        )
    tag = "+".join(args.halo_ids)
    conf_fn = os.path.join(args.workdir, "halos_%s_DM_%dto%d.conf"
                           % (tag, args.level - 1, args.level))
    with open(conf_fn, "w") as fp:
        cf.write(fp)
    return conf_fn


def read_grid_parameters(sim_dir):
    """Grid count and CosmologySimulationGrid* lines from parameter_file.txt."""
    n_grids = None
    grid_lines = []
    with open(os.path.join(sim_dir, "parameter_file.txt")) as fp:
        for line in fp:
            stripped = line.rstrip("\n")
            if stripped.startswith("CosmologySimulationNumberOfInitialGrids"):
                n_grids = int(stripped.split("=")[1])
            elif stripped.startswith("CosmologySimulationGrid"):
                grid_lines.append(stripped)
    if n_grids is None:
        raise RuntimeError("CosmologySimulationNumberOfInitialGrids not "
                           "found in %s/parameter_file.txt" % sim_dir)
    return n_grids, grid_lines


def assemble_run_directory(args, sim_dir):
    """Fill the Enzo parameter file and job script into the IC directory."""
    n_grids, grid_lines = read_grid_parameters(sim_dir)

    overdensity = ["%g" % (1.0 / 8.0**(args.level - 1))] * 2 + ["1."] * 8
    with open(os.path.join(TEMPLATE_DIR, "25Mpc_DM_512-LX.enzo.in")) as fp:
        enzo_text = fp.read()
    enzo_text = enzo_text \
        .replace("@NUM_INITIAL_GRIDS@", str(n_grids)) \
        .replace("@MRP_LEVEL@", str(args.level)) \
        .replace("@MIN_OVERDENSITY@", " ".join(overdensity))
    enzo_text += "\n# nested-grid geometry from MUSIC parameter_file.txt\n" \
        + "\n".join(grid_lines) + "\n"
    enzo_fn = os.path.join(sim_dir, "%s-L%d.enzo"
                           % (args.simulation_name, args.level))
    with open(enzo_fn, "w") as fp:
        fp.write(enzo_text)

    with open(os.path.join(TEMPLATE_DIR, "RunScript.sh.in")) as fp:
        job_text = fp.read()
    job_text = job_text \
        .replace("@JOB_NAME@", "halos%s-L%d"
                 % ("+".join(args.halo_ids), args.level)) \
        .replace("@ENZO_PARAM_FILE@", os.path.basename(enzo_fn)) \
        .replace("@NPROCS@", str(args.nprocs)) \
        .replace("@EMAIL@", args.email)
    job_fn = os.path.join(sim_dir, "RunScript.sh")
    with open(job_fn, "w") as fp:
        fp.write(job_text)
    os.chmod(job_fn, os.stat(job_fn).st_mode | stat.S_IXUSR)

    shutil.copyfile(os.path.join(TEMPLATE_DIR, "simrun.pl"),
                    os.path.join(sim_dir, "simrun.pl"))
    os.chmod(os.path.join(sim_dir, "simrun.pl"),
             os.stat(os.path.join(sim_dir, "simrun.pl")).st_mode |
             stat.S_IXUSR)
    return enzo_fn, job_fn


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate multi-halo zoom ICs for the 512^3 25 Mpc box.")
    parser.add_argument("--halo_ids", required=True,
                        help="comma-separated rockstar halo IDs")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--mode", choices=("union", "merge"), default="union")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--music-exe-dir", required=True,
                        help="directory holding the MUSIC executable")
    parser.add_argument("--music-template", required=True,
                        help="MUSIC template conf with the run's seeds "
                             "(e.g. your 25Mpc_DM_512_planck18.conf)")
    parser.add_argument("--workdir", default=os.getcwd(),
                        help="simulation run directory")
    parser.add_argument("--simulation-name", default="25Mpc_DM_512")
    parser.add_argument("--boxsize", type=float, default=25.0,
                        help="box size in Mpc/h (catalog X/Y/Z units)")
    parser.add_argument("--radius-factor", type=float, default=1.0)
    parser.add_argument("--min-rvir-kpc", type=float, default=200.0,
                        help="floor applied to catalog Rvir")
    parser.add_argument("--final-redshift", type=float, default=0.0)
    parser.add_argument("--num-cores", type=int, default=None)
    parser.add_argument("--nprocs", type=int, default=32)
    parser.add_argument("--email", default="")
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args(argv)
    args.halo_ids = [h.strip() for h in args.halo_ids.split(",") if h.strip()]
    args.workdir = os.path.abspath(args.workdir)

    catalog = mzconfig.read_rockstar_catalog(args.catalog)
    centers, radii = [], []
    for halo_id in args.halo_ids:
        center, rvir = mzconfig.halo_from_catalog(catalog, halo_id,
                                                  args.boxsize)
        radii.append(max(rvir, args.min_rvir_kpc))
        centers.append(center)
        print("Halo %s: center %s, Rvir %.1f kpc/h" % (halo_id, center,
                                                       radii[-1]))

    music_cf = configparser.ConfigParser()
    music_cf.read(args.music_template)
    levelmin = music_cf.getint("setup", "levelmin")
    if args.mode == "union" and args.level > 1:
        shift = accumulated_shift(args.workdir, args.simulation_name,
                                  args.level, levelmin)
        print("Applying accumulated domain shift (code units):", shift)
        centers = [[c[i] + shift[i] for i in range(3)] for c in centers]

    conf_fn = write_multizoom_conf(args, centers, radii)
    print("Wrote multizoom config:", conf_fn)

    params = mrp_music.startup(conf_fn, args.level)
    params = mrp_music.get_previous_run_params(params)
    params = mrp_music.find_lagrangian_regions(params)
    sim_dir = mrp_music.run_level(params)

    enzo_fn, job_fn = assemble_run_directory(args, sim_dir)
    print("Assembled run directory:", sim_dir)
    if args.no_submit:
        print("Skipping submission (--no-submit). Submit with: "
              "cd %s && qsub -koed RunScript.sh" % sim_dir)
    else:
        subprocess.run(["qsub", "-koed", "RunScript.sh"], cwd=sim_dir,
                       check=True)


if __name__ == "__main__":
    main()
