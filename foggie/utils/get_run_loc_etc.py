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
    elif args.system == "lefty":
        foggie_dir = "/astro/simulations/FOGGIE/"
        output_path = "/Users/tumlinson/Dropbox/foggie/collab/"
        code_path = '/Users/tumlinson/Dropbox/FOGGIE/foggie/'
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
    elif args.system == "cassiopeia":
        foggie_dir = "/Users/clochhaas/Documents/Research/FOGGIE/Simulation_Data/"
        output_path = "/Users/clochhaas/Documents/Research/FOGGIE/Outputs/"
        code_path = "/Users/clochhaas/Documents/Research/FOGGIE/Analysis_Code/foggie/foggie/"
    elif args.system == "pleiades_cassi":
        foggie_dir = "/nobackup/mpeeples/"
        output_path = "/nobackup/clochhaa/Outputs/"
        code_path = "/nobackup/clochhaa/foggie/foggie/"
    elif args.system == 'nnishimura':
        foggie_dir = '/Volumes/Student Project Drive/'
        code_path = '/Users/nnishimura/Desktop/FOGGIE/foggie/foggie'
        output_path = '/Users/nnishimura/sasp2021practice'
    elif args.system == "ramona":
        foggie_dir = "/Users/raugustin/WORK/SIMULATIONS/"
        output_path = "/Users/raugustin/WORK/Outputs/"
        code_path = "/Users/raugustin/foggie/foggie/"
    elif args.system == "ramona_astro":
        foggie_dir = "/astro/simulations/FOGGIE/"
        output_path = "/Users/raugustin/WORK/Outputs/"
        code_path = "/Users/raugustin/foggie/foggie/"
    elif args.system == "ramona_pleiades":
        foggie_dir = "/nobackup/mpeeples/"
        output_path = "/nobackupp13/raugust4/WORK/Outputs/"
        code_path = "/home5/raugust4/foggie/foggie/"
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
        foggie_dir = "/Volumes/Elements/foggieforayan/"
        output_path = "/Users/acharyya/Work/astro/foggie_outputs/"
        code_path = "/Users/acharyya/Work/astro/ayan_codes/foggie/foggie/"
    elif args.system == "msu_hpcc":
        foggie_dir = "/mnt/research/galaxies-REU/sims/FOGGIE/"
        output_path = "/mnt/home/kopenhaf/foggie_calcs/"
        code_path = "/mnt/home/kopenhaf/foggie/foggie/"
    elif args.system == "ayan_pleiades":
        foggie_dir = "/nobackup/mpeeples/"
        output_path = "/nobackup/aachary2/foggie_outputs/"
        code_path = "/home5/aachary2/Work/astro/ayan_codes/foggie/foggie/"

    if not args.pwd:
        if args.run == "natural":
            run_loc = "halo_00"+ args.halo + "/nref11n/"
            trackname = code_path + "halo_tracks/00"+ args.halo +"/nref11n_selfshield_15/halo_track_200kpc_nref9"
            infofile = code_path  + "halo_infos/00" + args.halo +"/nref11n/halo_info"
            haloname = "halo_00"+ args.halo + "_nref11n"
            path_part = run_loc
            output_dir = output_path + "plots_"+path_part
            spectra_dir = output_dir+"spectra/"
        elif args.run == "nref10f" or args.run == "nref11n_nref10f":
            run_loc = "halo_00"+ args.halo + "/nref11n_nref10f/"
            trackname = code_path + "halo_tracks/00"+ args.halo +"/nref11n_selfshield_15/halo_track_200kpc_nref10"
            infofile = code_path  + "halo_infos/00" + args.halo +"/nref11n_nref10f/halo_info"
            haloname = "halo_00"+ args.halo + "_nref11n_nref10f"
            path_part = run_loc
            output_dir = output_path + "plots_"+path_part
            spectra_dir = output_dir+"spectra/"
        elif args.run == "nref11c_nref9f" or args.run == "nref11c":
            run_loc = "halo_00"+ args.halo + "/nref11c_nref9f/"
            trackname = code_path + "halo_tracks/00"+ args.halo +"/nref11n_selfshield_15/halo_track_200kpc_nref9"
            infofile = code_path  + "halo_infos/00" + args.halo +"/nref11c_nref9f/halo_info"
            haloname = "halo_00"+ args.halo + "_nref11c_nref9f"
            path_part = run_loc
            output_dir = output_path + "plots_"+path_part
            spectra_dir = output_dir+"spectra/"
        if args.system=='cassiopeia' or args.system=='pleiades_cassi':
            output_dir = output_path

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
