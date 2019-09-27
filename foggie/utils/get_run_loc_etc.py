from __future__ import print_function

def get_run_loc_etc(args):
    print("for now I am assuming you are using the Tempest halo even if you passed in something different")


    if args.system == "oak":
        foggie_dir = "/astro/simulations/FOGGIE/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
        code_path = '/Users/molly/Dropbox/foggie/foggie/'
    elif args.system == "dhumuha" or args.system == "palmetto":
        foggie_dir = "/Users/molly/foggie/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
        code_path = '/Users/molly/Dropbox/foggie/foggie/'
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
        foggie_dir = "/User/rsimons/foggie/"
        output_path = "/User/rsimons/foggie/outputs"
        code_path = '/User/rsimons/Desktop/git/foggie/'
    elif args.system == "cassiopeia":
        #foggie_dir = "/astro/simulations/FOGGIE/"
        foggie_dir = "/Users/clochhaas/Documents/Research/FOGGIE/Simulation_Data/"
        output_path = "/Users/clochhaas/Documents/Research/FOGGIE/Outputs/"
        code_path = "/Users/clochhaas/Documents/Research/FOGGIE/foggie/"
    elif args.system == "pleiades_cassi":
        foggie_dir = "/nobackup/mpeeples/"
        output_path = "/home5/clochhaa/FOGGIE/Outputs/"
        code_path = "/home5/clochhaa/FOGGIE/Code/"

    if args.run == "natural":
        run_loc = "halo_00"+ args.halo + "/nref11n/"
        trackname = code_path + "halo_tracks/00"+ args.halo +"/nref11n_selfshield_15/halo_track_200kpc_nref9"
        haloname = "halo_00"+ args.halo + "_nref11n"
        path_part = run_loc
        output_dir = output_path + "plots_"+path_part
        spectra_dir = output_dir+"spectra/"
    elif args.run == "nref10f":
        run_loc = "halo_00"+ args.halo + "/nref11n_nref10f/"
        trackname = code_path + "halo_tracks/00"+ args.halo +"/nref11n_selfshield_15/halo_track_200kpc_nref10"
        haloname = "halo_00"+ args.halo + "_nref11n_nref10f"
        path_part = run_loc
        output_dir = output_path + "plots_"+path_part
        spectra_dir = output_dir+"spectra/"
    elif args.run == "nref11c_nref9f" or args.run == "nref11c":
        run_loc = "halo_00"+ args.halo + "/nref11c_nref9f/"
        trackname = code_path + "halo_tracks/00"+ args.halo +"/nref11n_selfshield_15/halo_track_200kpc_nref9"
        haloname = "halo_00"+ args.halo + "_nref11c_nref9f"
        path_part = run_loc
        output_dir = output_path + "plots_"+path_part
        spectra_dir = output_dir+"spectra/"

    if args.pwd:
        print('using pwd args')
        foggie_dir = '.'
        output_path = '.'
        output_dir = './'
        run_loc = '.'
        trackname = 'halo_track'
        haloname = 'halo'
        spectra_dir = '.'

    return foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir
