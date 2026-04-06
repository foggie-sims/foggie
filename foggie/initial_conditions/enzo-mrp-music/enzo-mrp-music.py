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
        simulation_run_directory = ".",
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
    # Set simulation directories
    params["prev_sim_dir"] = os.path.join(params["simulation_run_directory"], "%s-L%d" %
                                          (params["simulation_name"], params["level"]-1))
    params["sim_dir"] = os.path.join(params["simulation_run_directory"],
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
    music_cf1.set("output", "filename", os.path.join(params["simulation_run_directory"],"%s-L%d" % (params["simulation_name"], params["level"])))
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

    new_config_file = os.path.join(params["simulation_run_directory"],"%s-L%d.conf" % (params["simulation_name"], params["level"]))
    with open(new_config_file, "w") as fp:
        music_cf1.write(fp)

    os.environ["OMP_NUM_THREADS"] = "%d" % (params["num_cores"])
    os.environ["LD_LIBRARY_PATH"] = "/nasa/hdf5/1.8.18_serial/lib:/u/jtumlins/installs/gsl-2.4/lib"
    os.environ["DYLD_LIBRARY_PATH"] = "/nasa/hdf5/1.8.18_serial/lib:/u/jtumlins/installs/gsl-2.4/lib"
    command = """ echo $LD_LIBRARY_PATH ;
                  /nobackupnfs1/jtumlins/foggie/foggie/initial_conditions/music/MUSIC """ + new_config_file
    print('about to run ', command)
    os.system(command)
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
    if yt.is_root():
        run_music(params)
