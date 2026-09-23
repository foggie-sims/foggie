import sys
import glob
import h5py as h5
import yt
import configparser as cp
import multiprocessing as mp
from get_halo_initial_extent import *
from particle_only_mask import *
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.rank
    my_size = comm.size
    parallel = True
    yt.enable_parallelism()
except:
    my_rank = 0
    my_size = 1
    parallel = False

def parse_config(config_fn):
    # Defaults
    params = dict(
        music_exe_dir = ".",
        simulation_name = "auto-wrapper",
        template_config = "template.conf",
        original_config = None,
        # Where the *previous* level's Enzo outputs are read from.
        simulation_run_directory = ".",
        # Where the *new* ICs for this level are written.  Kept separate from
        # simulation_run_directory so a zoom can read its parent level from a
        # shared directory while depositing its own ICs in the halo directory.
        # Defaults to "." so existing configs behave as before.
        new_ics_directory = ".",
        # Runtime environment for the MUSIC subprocess.
        music_ld_library_path = "/nasa/hdf5/1.8.18_serial/lib:/u/jtumlins/installs/gsl-2.4/lib",
        num_cores = None,
        final_type = "halo",
        final_redshift = 0.0,
        halo_center = None,
        halo_center_units = "code_length",
        halo_mass = None,
        halo_mass_units = "Msun/h",
        halo_radius = None,
        halo_radius_units = "kpc",
        radius_factor = 3.0,
        shape_type = "box")

    # Read config file
    mrp_cf = cp.ConfigParser()
    mrp_cf.read(config_fn)
    for section in mrp_cf.sections():
        for k,v in mrp_cf.items(section):
            if v != "None":
                params[k] = v
            else:
                params[k] = None

    params["radius_factor"] = float(params["radius_factor"])
    # Set the number of OpenMP threads
    if params["num_cores"] != None:
        params["num_cores"] = int(params["num_cores"])
    else:
        params["num_cores"] = mp.cpu_count()


    # Check whether halo properties have been set
    if (params["halo_center"] == None) or \
       (params["halo_mass"] == None and params["halo_radius"] == None):
        raise RuntimeError("Halo properties not set (either radius or mass must be set).\n"
                           "\t Center: %s\n"
                           "\t Mass:   %s\n"
                           "\t Radius: %s\n" % \
                           (params["halo_center"], params["halo_mass"], params["halo_radius"]))

    # Consolidate halo properties into a dict
    params["halo_center"] = np.array([float(p) for p in params["halo_center"].split(",")])
    if params["halo_mass"] != None:
        params["halo_info"] = dict(center = (params["halo_center"], params["halo_center_units"]),
                                   mass = (float(params["halo_mass"]), params["halo_mass_units"]),
                                   redshift = float(params["final_redshift"]))
    if params["halo_radius"] != None:
        if params["halo_mass"] == None:
            params["halo_info"] = dict(center = (params["halo_center"],
                                                 params["halo_center_units"]),
                                       radius = (float(params["halo_radius"]),
                                               params["halo_radius_units"]),
                                       redshift = float(params["final_redshift"]))
        else:
            raise RuntimeWarning("Halo mass and radius both set.  Defaulting to mass.")
    return params

def startup():
    # Obtain the next level from the command line
    #
    if len(sys.argv) != 3:
        raise RuntimeError("usage: %s config_file level\n"
                           "\t level: 0-based level of the next set of ICs" % \
                           (sys.argv[0]))
    config_fn = sys.argv[-2]
    level = int(sys.argv[-1])
    if not os.path.exists(config_fn):
        raise RuntimeError("Config file not found: %s" % (config_fn))

    params = parse_config(config_fn)
    params["level"] = level

    # Error check
    if params["level"] == 0:
        raise RuntimeError("level must be >0. "
                           "Please run the unigrid simulation first.")
    files_to_check = ["%s/MUSIC" % (params["music_exe_dir"]),
                      params["template_config"],
                      params["simulation_run_directory"]]
    if params["original_config"] != None: files_to_check += [params["original_config"]]
    for f in files_to_check:
        if not os.path.exists(f):
            raise RuntimeError("File/directory not found: %s" % (f))

    return params

def get_previous_run_params(params):
    # Set simulation directories.
    #
    # prev_sim_dir is where we look for the level N-1 outputs (e.g. level 0
    # when we are making level 1); sim_dir is where we deposit the new level N
    # ICs.  These are deliberately rooted in different config options.
    params["prev_sim_dir"] = os.path.join(params["simulation_run_directory"], "%s-L%d" %
                                          (params["simulation_name"], params["level"]-1))
    params["sim_dir"] = os.path.join(params["new_ics_directory"],
                                     "%s-L%d" % (params["simulation_name"], params["level"]))
    #
    # Obtain the maxlevel of the original run
    if params["original_config"] == None:
        original_config_file = "%s-L0.conf" % (params["simulation_name"])
    else:
        original_config_file = params["original_config"]
    music_cf0 = cp.ConfigParser()
    music_cf0.read(original_config_file)
    params["initial_min_level"] = music_cf0.getint("setup", "levelmin")
    params["initial_max_level"] = music_cf0.getint("setup", "levelmax")

    # Obtain the shift of the Lagrangian region from the previous zoom-in
    # (or unigrid) simulation
    params["region_shift"] = [0, 0, 0]
    if params["original_config"] != None and params["level"] == 1:
        prev_config_logfile = "%s_log.txt" % (params["original_config"])
    else:
        prev_config_logfile = "%s-L%d.conf_log.txt" % \
                                              (params["simulation_name"], params["level"]-1)
    with open(prev_config_logfile) as fp:
        for l in fp.readlines():
            if l.find("Domain") >= 0:
                params["region_shift"][0] = int(l.split('(')[1].split(',')[0])
                params["region_shift"][1] = int(l.split('(')[1].split(',')[1])
                params["region_shift"][2] = int(l.split('(')[1].split(',')[2].replace(')',''))
            if l.find("setup/levelmin") >= 0:
                params["region_point_levelmin"] = int(l.split("=")[1])

    # Rounding factor for the Lagrangian region if using a rectangular
    # prism.
    params["round_factor"] = 2**params["initial_max_level"]

    #
    # Get the inital dataset of the simulation and either
    # the final dataset or the dataset at the specified redshift.
    #
    sim_par_file = os.path.join(params["prev_sim_dir"], "%s-L%d.enzo" %
                                (params["simulation_name"], params["level"]-1))
    print("Opening Enzp param file: ", sim_par_file)
    es = yt.load_simulation(sim_par_file, "Enzo", find_outputs=True)

    params["enzo_initial_fn"] = es.all_outputs[0]["filename"]
    if "redshift" in params["halo_info"]:
        es.get_time_series(redshifts=[params["halo_info"]["redshift"]])
        ds = es[0]
        params["enzo_final_fn"] = os.path.join(ds.directory, ds.basename)
    else:
        params["enzo_final_fn"] = es.all_outputs[-1]["filename"]

    #<--- this is where the initial and final outputs are derived . . .



    return params

def find_lagrangian_region(params):
    particle_output_format = None if params["shape_type"] == "box" else "txt"
    params["region_center"], params["region_size"], params["lagr_particle_file"] = \
               get_center_and_extent(params["halo_info"],
                                     params["enzo_initial_fn"],
                                     params["enzo_final_fn"],
                                     round_size = params["round_factor"],
                                     radius_factor = params["radius_factor"],
                                     output_format = particle_output_format)
    return params


def trim_lagrangian_outliers(params, link_factor=6.0, max_fraction=0.10):
    """Drop Lagrangian material that is not connected to the halo's own region.

    The zoom region is the convex hull of the traced particles, so anything far
    from the rest drags the whole region out to enclose it.  Some traced
    particles are unbound or fast-moving: they sit inside the z = 0 selection
    sphere but started across the box.

    CONNECTIVITY, not radius.  The original version of this cut points beyond
    5x the 99th-percentile radius and refused to drop more than 2% of the
    cloud.  That works only while the contamination is a handful of points, and
    it fails silently the moment it is a clump:

      halo80181 L2, 2026-09-01.  5057 traced points, of which 194 (3.84%) form
      a second clump 2.9 Mpc/h away.  With 3.84% of the cloud out there the
      99th-percentile radius is 3128 ckpc/h -- ALREADY INSIDE the far clump --
      so the cut lands at 15638 ckpc/h, past the furthest point, and nothing is
      trimmed.  The 2% ceiling would have refused as well.  The result was a
      hull 26.8x too big: a 2.29 h L2 run against 0.85-1.23 h for its peers,
      and 147 GB, with no complaint from anything.

    Friends-of-friends has no such blind spot, because the linking length is
    set by the cloud's own local density (6x the median nearest-neighbour
    separation) rather than by a percentile the outliers themselves shift.
    Measured on the clouds on disk:

      halo80181 L2   4 groups, largest 96.2%, drops 3.84%, hull shrinks 26.8x
      halo39829      2 groups, largest 99.99%, drops 1 point, shrinks 4.0x
      halo59186      1 group,  largest 100.0%, drops 3 points, shrinks 1.00x
      halo543386     1 group,  100%, unchanged
      halo47314      1 group,  100%, unchanged

    halo59186 is the case that matters for safety: its region is a genuine
    4.1 Mpc/h filament, and a radius cut would shred it.  A filament is
    CONNECTED, so friends-of-friends keeps it whole -- which is exactly the
    distinction a percentile cannot draw.

    Refuses to drop more than max_fraction of the cloud: if that much material
    is detached, the region is genuinely multi-component and quietly discarding
    it would be wrong.
    """
    import numpy as np

    path = params.get("lagr_particle_file")
    if not path or not os.path.exists(path):
        return params

    pts = np.loadtxt(path)
    if pts.ndim != 2 or len(pts) < 50:
        return params

    try:
        from scipy.spatial import cKDTree
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        print("  region: scipy unavailable, skipping the connectivity trim")
        return params

    med = np.median(pts, axis=0)
    off = pts - med
    off -= np.round(off)                      # periodic, box is [0,1)

    tree = cKDTree(off)
    nn = tree.query(off, k=2, workers=-1)[0][:, 1]
    link = link_factor * np.median(nn)
    pairs = tree.query_pairs(link, output_type="ndarray")
    if not len(pairs):
        return params

    n = len(off)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(n, n))
    ncomp, labels = connected_components(graph, directed=False)
    if ncomp == 1:
        return params

    keep = labels == np.argmax(np.bincount(labels))
    n_drop = int((~keep).sum())
    if n_drop == 0:
        return params

    ext_before = (pts.max(axis=0) - pts.min(axis=0))
    ext_after = (pts[keep].max(axis=0) - pts[keep].min(axis=0))
    frac = n_drop / float(n)
    if frac > max_fraction:
        print("  region: %d of %d points (%.1f%%) are detached from the main "
              "group; that is more than %.0f%% so the region is genuinely "
              "multi-component -- NOT trimming.  Check the halo's environment."
              % (n_drop, n, 100 * frac, 100 * max_fraction))
        return params

    print("  region: %d groups at linking length %.5f; keeping the largest "
          "(%d points), dropping %d (%.2f%%) detached.  hull %s -> %s, "
          "volume shrinks %.1fx"
          % (ncomp, link, int(keep.sum()), n_drop, 100 * frac,
             np.round(ext_before, 4), np.round(ext_after, 4),
             np.prod(ext_before) / max(np.prod(ext_after), 1e-30)))
    np.savetxt(path, pts[keep], fmt="%.18e")
    return params


def name_region_file_by_level(params):
    """Give this level's region point file a name no other level can reuse.

    get_halo_initial_extent writes initial_particle_positions-<halo>-<snap>.dat.
    That name carries the halo and the snapshot but NOT the level, so every
    level of a halo writes the same file and each trace silently overwrites the
    last.  The conf that names it also records region_point_shift -- the frame
    that file was in at the moment the conf was written -- and MUSIC unapplies
    that shift before tracing.  So the instant a deeper level re-traces, every
    shallower conf's shift stops describing the file it names, and MUSIC will
    unapply a shift that no longer applies.

    halo80181, 2026-08-29: the L4 trace rewrote the shared file, after which
    the L3 and L2 confs both named a file neither had been built against.  What
    that costs is REPRODUCIBILITY -- a conf that can no longer be rebuilt into
    the region it describes.  It does NOT distort the region: measured
    2026-09-02, a cloud's bounding box is identical under every candidate
    shift, because unapplying a shift is a pure translation.  (halo80181's
    hulls really were 25x too big, but that was a detached clump in the traced
    cloud, removed by trim_lagrangian_outliers; the shift was a red herring and
    is recorded here so the wrong diagnosis is not made twice.)

    Appending -L<level> means each conf names a file that only that conf's own
    trace ever writes, so a conf and its points stay a matched pair for the life
    of the halo directory -- however many deeper levels are built afterwards.

    COPY, not rename.  The shared file has to survive: the 45 halos already on
    disk have confs that name it, and a fresh trace at one level would
    otherwise delete the file every other level's conf points at.  Leaving it
    means those confs are exactly as (un)reliable as they were before -- the
    shared file still holds whichever trace ran last -- while every conf
    written from now on names a file nothing else touches.
    """
    import shutil
    path = params.get("lagr_particle_file")
    if not path or not os.path.exists(path):
        return params
    stem, ext = os.path.splitext(path)
    tag = "-L%d" % params["level"]
    if stem.endswith(tag):
        return params
    dest = "%s%s%s" % (stem, tag, ext)
    shutil.copy2(path, dest)
    params["lagr_particle_file"] = dest
    print("  region: this level's points pinned to %s (the shared %s is kept "
          "for confs written before per-level naming)"
          % (os.path.basename(dest), os.path.basename(path)))
    return params


def run_music(params):
    #
    # Read the zoom-in MUSIC file, modify/add zoom-in parameters, and write out.
    #
    music_cf1 = cp.ConfigParser()
    # Turn-on case-sensitive for config files
    music_cf1.optionxform = str

    music_cf1.read(params["template_config"])
    # Delete some options if they exist.  If we need them, we'll create them again.
    for option in ["ref_offset", "ref_center", "ref_extent"]:
        if music_cf1.has_option("setup", option):
            music_cf1.remove_option("setup", option)

    music_cf1.set("setup", "levelmax", "%d" % (params["initial_min_level"] + params["level"]))
    music_cf1.set("output", "filename", os.path.join(params["new_ics_directory"], "%s-L%d" % (params["simulation_name"], params["level"])))
    music_cf1.set("setup", "region",
                  "convex_hull" if params["shape_type"] == "exact" else params["shape_type"])
    if params["shape_type"] == "box":
        music_cf1.set("setup", "ref_center", "%f, %f, %f" % \
                      (params["region_center"][0], params["region_center"][1],
                              params["region_center"][2]))
        music_cf1.set("setup", "ref_extent", "%f, %f, %f" % \
                      (params["region_size"][0], params["region_size"][1],
                       params["region_size"][2]))
    else:
        music_cf1.set("setup", "region_point_file", params["lagr_particle_file"])
        music_cf1.set("setup", "region_point_shift",
                      "%d, %d, %d" % (params["region_shift"][0], params["region_shift"][1],
                                      params["region_shift"][2]))
        music_cf1.set("setup", "region_point_levelmin", "%d" % (params["initial_min_level"]))

    os.makedirs(params["new_ics_directory"], exist_ok=True)
    new_config_file = os.path.join(params["new_ics_directory"], "%s-L%d.conf" % (params["simulation_name"], params["level"]))
    print('new_config_file: ', new_config_file)
    with open(new_config_file, "w") as fp:
        music_cf1.write(fp)

    os.environ["OMP_NUM_THREADS"] = "%d" % (params["num_cores"])
    os.environ["LD_LIBRARY_PATH"] = params["music_ld_library_path"]
    os.environ["DYLD_LIBRARY_PATH"] = params["music_ld_library_path"]
    # Use the MUSIC binary that startup() already verified exists, rather than
    # a second hardcoded copy of the path.
    music_exe = os.path.join(params["music_exe_dir"], "MUSIC")
    command = "%s %s" % (music_exe, new_config_file)
    print('about to run ', command)
    status = os.system(command)
    if status != 0:
        raise RuntimeError("MUSIC failed (exit status %d): %s" % (status, command))
    print('control has returned from MUSIC to the enzo_mrp script')

    # If we require the exact Lagrangian region, then we directly modify
    # the RefinementMask file that's written by MUSIC.
    #
    # smooth_edges: further smooth the CIC interpolation of the particles
    # in the Lagrangian region with a Gaussian over a 3x3x3 cell volume.
    #
    # backup: Copy original file with the suffix .bak
    if params["shape_type"] == "exact":
        particle_only_mask(new_config_file, smooth_edges=True, backup=True)

    # Modify the skeleton Enzo parameter file created by MUSIC to include
    # the parameters for must-refine particles.
    ic_dir = music_cf1.get("output", "filename")
    fp = open("%s/parameter_file.txt" % (ic_dir), "a")
    fp.write("\n"
             "#\n"
             "# must-refine particle parameters\n"
             "# *** must also include method 8 in CellFlaggingMethod ***\n"
             "# *** do NOT include the RefineRegion parameters above ***\n"
             "#\n"
             "MustRefineParticlesCreateParticles = 3\n"
             "MustRefineParticlesRefineToLevel   = %d\n"
             "CosmologySimulationParticleTypeName          = RefinementMask\n" \
             % (params["level"]))
    fp.close()

    # Copy initial conditions directory to the simulation run directory
    print ("Moving initial conditions to %s" % (params["sim_dir"]))
    os.rename(ic_dir, params["sim_dir"])

    return

if __name__ == "__main__":
    params = {}
    if yt.is_root():
        params = startup()
    if parallel:
        params = comm.bcast(params)
    params = get_previous_run_params(params)
    params = find_lagrangian_region(params)
    params = trim_lagrangian_outliers(params)
    # Pin the name before run_music writes region_point_file into the conf.
    params = name_region_file_by_level(params)
    if yt.is_root():
        run_music(params)
