from __future__ import print_function

def get_run_loc_etc(args):
    if args.system == "oak":
        foggie_dir = "/astro/simulations/FOGGIE/"
        output_path = "/Users/molly/Dropbox/foggie/collab/"
        code_path = '/Users/molly/Dropbox/foggie/foggie/foggie/'
    elif args.system == "iris" or args.system == "palmetto":
        foggie_dir = "/Users/molly/foggie/"
        output_path = "/Users/molly/Dropbox/foggie/collab/"
        code_path = '/Users/molly/Dropbox/foggie/foggie/foggie/'
    elif args.system == "harddrive":
        foggie_dir = "/Volumes/foggie/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "nmearl":
        foggie_dir = "/Users/nearl/data/"
        output_path = "/Users/nearl/Desktop/"
    elif args.system == "pleiadesmolly":
        foggie_dir = "/nobackup/mpeeples/"
        output_path = "/nobackup/mpeeples/"
        code_path = '/pleiades/u/mpeeples/foggie/'
    elif args.system == "guy":
        foggie_dir = "/Users/tumlinson/Dropbox/FOGGIE/snapshots/"
        output_path = "/Users/tumlinson/Dropbox/foggie/collab/"
        code_path = '/Users/tumlinson/Dropbox/FOGGIE/foggie/foggie/'
    elif args.system == "jase":
        foggie_dir = "/Users/rsimons/Desktop/foggie/sims/"
        output_path = "/Users/rsimons/Desktop/foggie/"
        code_path = '/Users/rsimons/Dropbox/git/foggie/foggie/'
    elif args.system == "saje":
        foggie_dir = "/Users/rsimons/Desktop/foggie/sims/"
        output_path = "/Users/rsimons/Desktop/foggie/"
        code_path = '/Users/rsimons/Dropbox/git/foggie/foggie/'
    elif args.system == "laptop_raymond":
        foggie_dir = "/Users/rsimons/Desktop/foggie/sims/"
        output_path = "/Users/rsimons/Dropbox/foggie/"
        code_path = '/Users/rsimons/Dropbox/git/foggie/foggie/'
    elif args.system == "pegasus":
        foggie_dir = "/Volumes/pegasus/foggie/"
        output_path = "/User/rsimons/foggie/outputs"
        code_path = '/User/rsimons/Desktop/git/foggie/'
    elif args.system == "pleiades_raymond":
        foggie_dir = "/nobackup/mpeeples/"
        output_path = "/nobackupp2/rcsimons/foggie/"
        code_path = '/nobackupp2/rcsimons/git/foggie/foggie/'
    elif args.system == "pleiades_jt":
        foggie_dir = "/nobackup/jtumlins/"
        output_path = "/nobackupp/jtumlins/"
        code_path = '/nobackup/jtumlins/foggie/foggie/'
    elif args.system == "cassiopeia":
        foggie_dir = "/Users/clochhaas/Documents/Research/FOGGIE/Simulation_Data/"
        output_path = "/Users/clochhaas/Documents/Research/FOGGIE/Outputs/"
        code_path = "/Users/clochhaas/Documents/Research/FOGGIE/Analysis_Code/foggie/foggie/"
    elif args.system == "pleiades_cassi":
        if ('feedback' in args.run):
            foggie_dir = '/nobackup/clochhaa/'
        else:
            foggie_dir = "/nobackup/mpeeples/"
        output_path = "/nobackup/clochhaa/Outputs/"
        code_path = "/nobackup/clochhaa/foggie/foggie/"
    elif args.system == 'nnishimura':
        foggie_dir = '/Volumes/Student Project Drive/'
        code_path = '/Users/nnishimura/Desktop/FOGGIE/foggie/foggie'
        output_path = '/Users/nnishimura/sasp2021practice'
    elif args.system == "ramona":
        foggie_dir = "/Users/ramonaaugustin/WORK/SIMULATIONS/"
        output_path = "/Users/ramonaaugustin/WORK/Outputs/"
        code_path = "/Users/ramonaaugustin/foggie/foggie/"
    elif args.system == "ramona_astro":
        foggie_dir = "/astro/simulations/FOGGIE/"
        output_path = "/Users/raugustin/WORK/Outputs/"
        code_path = "/Users/raugustin/foggie/foggie/"
    elif args.system == "ramona_pleiades":
        foggie_dir = "/nobackup/mpeeples/"
        output_path = "/nobackupp13/raugust4/WORK/Outputs/"
        code_path = "/nobackupp13/raugust4/foggie/foggie/"
    elif args.system == "anna_pleiades":
        foggie_dir = "/nobackup/mpeeples/"
        output_path = "/nobackupp2/awright5/JHU/"
        code_path = "/nobackupp2/awright5/foggie/foggie/"
    elif args.system == "mogget":
        foggie_dir = "/Users/anna/foggie/foggie/"
        output_path = "/Users/anna/Research/Outputs/"
        code_path = "/Users/anna/Research/Simulations/"
    elif args.system == "ayan_local":
        foggie_dir = "/Users/acharyya/models/simulation_output/foggie/"
        output_path = "/Users/acharyya/Work/astro/foggie_outputs/"
        code_path = "/Users/acharyya/Work/astro/ayan_codes/foggie/foggie/"
    elif args.system == "ayan_hd":
        #foggie_dir = "/Volumes/Elements/foggieforayan/"
        foggie_dir = "/Volumes/Elements/acharyya_backup/models/simulation_output/foggie/"
        output_path = "/Volumes/Elements/acharyya_backup/Work/astro/foggie_outputs/"
        code_path = "/Users/acharyya/Work/astro/ayan_codes/foggie/foggie/"
    elif args.system == "claire_hpcc":
        foggie_dir = "/mnt/research/galaxies-REU/sims/FOGGIE/"
        output_path = "/mnt/scratch/kopenhaf/foggie_calcs/"
        code_path = "/mnt/home/kopenhaf/foggie/foggie/"
    elif args.system == "ayan_pleiades":
        foggie_dir = "/nobackup/mpeeples/" if args.foggie_dir is None else args.foggie_dir
        output_path = "/nobackupp19/aachary2/foggie_outputs/"
        code_path = "/nobackupp19/aachary2/ayan_codes/foggie/foggie/"
    elif args.system == "vida_local":
        foggie_dir = "/Users/vidasaeedzadeh/Projects/foggie_data/" 
        output_path = "/Users/vidasaeedzadeh/Projects/foggie_outputs/"
        code_path = "/Users/vidasaeedzadeh/Projects/repositories/foggie/foggie/"
    elif args.system == "vida_expanse":
        foggie_dir = "/expanse/lustre/projects/lal106/saeedzadeh/foggie_data" 
        output_path = "/expanse/lustre/scratch/saeedzadeh/temp_project/foggie_output"
        code_path = "/home/saeedzadeh/foggie/foggie"
    elif args.system == "cameron_local":
        foggie_dir = "/Volumes/FoggieCam/foggie_halos/"
        output_path = "/Users/ctrapp/Documents/foggie_analysis/default_output/"
        code_path = "/Users/ctrapp/Documents/GitHub/foggie/foggie/"
    elif args.system == "cameron_pleiades":
        foggie_dir = "/nobackup/mpeeples/"
        output_path = "/nobackup/cwtrapp/foggie_outputs/"
        code_path =  "/nobackup/cwtrapp/foggie/foggie/"

    if not args.pwd:
        if args.run == "natural":
            runname = "nref11n"
        elif args.run == "nref10f" or args.run == "nref11n_nref10f":
            runname = "nref11n_nref10f"
        elif args.run == "nref11c_nref9f" or args.run == "nref11c":
            runname = "nref11c_nref9f"
        elif ('feedback' in args.run) and (not 'track' in args.run) and (not args.forcepath):
            runname = "nref11c_nref9f"
        else:
            runname = args.run

        trackname = code_path + "halo_tracks/00" + args.halo + "/nref11n_selfshield_15/halo_track_200kpc_nref9"
        infofile = code_path  + "halo_infos/00" + args.halo + "/" + runname + "/halo_info"
        haloname = "halo_00" + args.halo + "_" + runname

        if ('feedback' in args.run) and (not 'track' in args.run) and (not args.forcepath):
            run_loc = args.run + "/"
        else:
            run_loc = "halo_00" + args.halo + "/" + runname + "/"

        output_dir = output_path + "plots_" + run_loc
        spectra_dir = output_dir + "spectra/"

        if args.system=='cassiopeia' or args.system=='pleiades_cassi':
            output_dir = output_path
            runname = args.run
            run_loc = "halo_00" + args.halo + "/" + runname + "/"

    if args.pwd:
        print('using pwd args')
        foggie_dir = '.'
        output_path = '.'
        output_dir = './'
        run_loc = '.'
        code_path = '.'
        trackname = 'halo_track'
        haloname = 'halo'
        infofile = 'halo_info'
        spectra_dir = '.'

    return foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile
