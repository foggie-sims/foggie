from __future__ import print_function

def get_run_loc_etc(args):
    print("for now I am assuming you are using the Tempest halo even if you passed in something different")

    if args.system == "oak":
        foggie_dir = "/astro/simulations/FOGGIE/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "dhumuha" or args.system == "palmetto":
        foggie_dir = "/Users/molly/foggie/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "harddrive":
        foggie_dir = "/Volumes/foggie/"
        output_path = "/Users/molly/Dropbox/foggie-collab/"
    elif args.system == "nmearl":
        foggie_dir = "/Users/nearl/data/"
        output_path = "/Users/nearl/Desktop/"
    elif args.system == "pleiades":
        foggie_dir = "/nobackup/mpeeples/"
        output_path = "/nobackup/mpeeples/"
    elif args.system == "pancho":
        foggie_dir = "/Users/tumlinson/Dropbox/foggie-test/"
        output_path = "/Users/tumlinson/Dropbox/foggie-collab/"
    elif args.system == "lefty":
        foggie_dir = "/Users/tumlinson/Dropbox/foggie-test/"
        output_path = "/Users/tumlinson/Dropbox/foggie-collab/"


    if args.run == "natural":
        run_loc = "halo_00"+ args.halo + "/nref11n/natural/"
        trackname = foggie_dir + "halo_00"+ args.halo +"/nref11n/nref11n_nref10f_refine200kpc/halo_track"
        haloname = "halo008508_nref11n"
        path_part = run_loc
        output_dir = output_path + "plots_"+path_part
        spectra_dir = output_dir+"spectra/"
        if args.system == 'pleiades':
            run_loc = 'orig/nref11n_orig/'
    elif args.run == "nref10f":
        run_loc = "halo_00"+ args.halo + "/nref11n/nref11n_nref10f_refine200kpc/"
        trackname = foggie_dir + "halo_00"+ args.halo +"/nref11n/nref11n_nref10f_refine200kpc/halo_track"
        haloname = "halo008508_nref11n_nref10f"
        path_part = run_loc
        output_dir = output_path + "plots_"+path_part
        spectra_dir = output_dir+"spectra/"
        if args.system == 'pleiades':
            run_loc = 'orig/nref11n_nref10f_orig/'
    elif args.run == "nref9f":
        run_loc = "halo_00"+ args.halo + "/nref11n/nref11n_nref9f_refine200kpc/"
        trackname = foggie_dir + "halo_00"+ args.halo +"/nref11n/nref11n_nref9f_refine200kpc/halo_track"
        haloname = "halo008508_nref11n_nref9f"
        path_part = "halo_008508/nref11n/nref11n_"+args.run+"_refine200kpc/"
        output_dir = output_path + "plots_"+path_part
        spectra_dir = output_dir+"spectra/"
    elif args.run == "nref11f":
        run_loc = "halo_00"+ args.halo + "nref11n/nref11f_refine200kpc/"
        trackname =  foggie_dir + "halo_00"+ args.halo + "/nref11n/nref11f_refine200kpc/halo_track"
        haloname = "halo008508_nref11f"
        output_dir = output_path + "plots_"+path_part
        spectra_dir = output_dir+"spectra/"
        if args.system == "pleiades":
            trackname = "halo_008508/orig/nref11f_refine200kpc_z4to2/halo_track"
            run_loc = "orig/nref11f_refine200kpc_z4to2/"
    elif args.run == "nref11n_selfshield":
        print('~!!!!!!!~~~!!!@!#!#!@!!!!~!!!!!!!~~~!!!@!#!#!@!!!!~!!!!!!!~~~!!!@!#!#!@!!!!THIS IS PROBABLY FINDING THE WRONG ONE HALT HALT HALT HALT HALT ~!!!!!!!~~~!!!@!#!#!@!!!!')
        run_loc = "halo_00"+ args.halo + "nref11n/nref11n_selfshield/"
        trackname = "halo_008508/nref11n/nref11n_selfshield/halo_track"
        haloname = "halo008508_nref11n_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref11f_refine200kpc/halo_track"
            run_loc = "nref11n_selfshield/"
    elif args.run == "nref11n_startest_selfshield":
        run_loc = "nref11n/nref11n_startest_selfshield/"
        trackname = "halo_008508/nref11n/nref11n_selfshield/halo_track"
        haloname = "halo008508_nref11n_startest_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref11f_refine200kpc/halo_track"
            run_loc = "nref11n_selfshield/"
    elif args.run == "nref10n_nref8f_selfshield":
        run_loc = "nref10n/nref10n_nref8f_selfshield/"
        trackname = "halo_008508/nref10n/nref10n_nref8f_selfshield/halo_track"
        haloname = "halo008508_nref10n_nref8f_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref10n_nref8f_selfshield/halo_track"
            run_loc = "nref10n_nref8f_selfshield/"
    elif args.run == "nref11n_nref9f_startest":
        run_loc = "nref11n/nref11n_nref9f_startest/"
        trackname = "halo_008508/nref11n_nref9f_startest/halo_track"
        haloname = "halo008508_nref11n_nref9f_startest"
    elif args.run == "nref10n_nref8f_startest_selfshield":
        run_loc = "nref10n/nref10n_nref8f_startest_selfshield/"
        trackname = "halo_008508/nref10n/nref10n_nref8f_startest_selfshield/halo_track"
        haloname = "halo008508_nref10n_nref8f_startest_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref10n_nref8f_startest_selfshield/halo_track"
            run_loc = "nref10n_nref8f_startest_selfshield/"
    elif args.run == "nref10n_nref8f_startest10000_selfshield":
        run_loc = "nref10n/nref10n_nref8f_startest10000_selfshield/"
        trackname = "halo_008508/nref10n/nref10n_nref8f_startest10000_selfshield/halo_track"
        haloname = "halo008508_nref10n_nref8f_startest10000_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref10n_nref8f_startest10000_selfshield/halo_track"
            run_loc = "nref10n_nref8f_startest10000_selfshield/"
    elif args.run == "nref10n_nref8f_startest5000_selfshield":
        run_loc = "nref10n/nref10n_nref8f_startest5000_selfshield/"
        trackname = "halo_008508/nref10n/nref10n_nref8f_startest5000_selfshield/halo_track"
        haloname = "halo008508_nref10n_nref8f_startest5000_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref10n_nref8f_startest5000_selfshield/halo_track"
            run_loc = "nref10n_nref8f_startest5000_selfshield/"
    elif args.run == "nref10n_nref8f_startest_selfshield":
        run_loc = "nref10n/nref10n_nref8f_startest_selfshield/"
        trackname = "halo_008508/nref10n/nref10n_nref8f_startest_selfshield/halo_track"
        haloname = "halo008508_nref10n_nref8f_startest_selfshield"
        if args.system == "pleiades":
            trackname = "halo_008508/nref10n_nref8f_startest_selfshield/halo_track"
            run_loc = "nref10n_nref8f_startest_selfshield/"
    elif args.run == "nref11n_selfshield_z15":
        run_loc = "nref11n_selfshield_z15/natural/"
        # trackname = "halo_008508/nref11n_selfshield_z15/nref11n_nref10f_selfshield_z6/halo_track"
        trackname = "halo_008508/nref11n/nref11n_nref10f_refine200kpc/halo_track"
        haloname = "halo008508_nref11n_selfshield_z15"
        path_part = "halo_008508/nref11n_selfshield_z15/nref11n_selfshield_z15/"
        output_dir = output_path + "plots_"+path_part
        spectra_dir = output_dir+"spectra/"
        if args.system == "pleiades":
            trackname = "halo_008508/nref11n_nref10f_selfshield_z6/halo_track"
            run_loc = "nref11n_selfshield_z15/"
    elif args.run == "nref10f_selfshield":
        run_loc = "halo_00"+ args.halo + "/nref11n_selfshield_z15/nref11n_nref10f_selfshield_z6/"
        trackname = "halo_008508/nref11n_selfshield_z15/nref11n_nref10f_selfshield_z6/halo_track"
        haloname = "halo008508_nref11n_nref10f_selfshield_z6"
        path_part = "halo_008508/nref11n_selfshield_z15/nref11n_"+args.run+"_z6/"
        output_dir = output_path + "plots_"+path_part
        spectra_dir = output_dir+"spectra/"
        if args.system == "pleiades":
            trackname = "halo_008508/nref11n_nref10f_selfshield_z6/halo_track"
            run_loc = "halo_008508/nref11n_nref10f_selfshield_z6/"
    elif args.run == "nref11c_nref9f":
        run_loc = "halo_00"+ args.halo + "/nref11n_selfshield_z15/nref11c_nref9f_selfshield_z6/"
        trackname = "halo_008508/nref11n_selfshield_z15/nref11c_nref9f_selfshield_z6/halo_track"
        haloname = "halo008508_nref11c_nref9f_selfshield_z6"
        path_part = "halo_008508/nref11n_selfshield_z15/"+args.run+"_selfshield_z6/"
        output_dir = output_path + "plots_"+path_part
        spectra_dir = output_dir+"spectra/"
        if args.system == "pleiades":
            trackname = "halo_008508/nref11c_nref9f_selfshield_z6/halo_track"
            run_loc = "halo_008508/nref11c_nref9f_selfshield_z6/"
    elif args.run == "nref11c_400kpc":
        run_loc = "nref11n_selfshield_z15/nref11c_nref5f_400kpc/"
        trackname = "halo_008508/nref11n_selfshield_z15/nref11c_nref5f_400kpc/halo_track"
        haloname = "halo008508_nref11c_nref5f_400kpc"
    elif args.run == "nref11c_600kpc":
        run_loc = "nref11n_selfshield_z15/nref11c_nref8f_600kpc/"
        trackname = "halo_008508/nref11n_selfshield_z15/nref11c_nref8f_600kpc/halo_track"
        haloname = "halo008508_nref11c_nref8f_600kpc"
        if args.system == "pleiades":
            trackname = "halo_008508/nref11c_nref8f_600kpc/halo_track"
            run_loc = "nref11c_nref8f_600kpc/"
    elif args.run == "nref11c_400kpc":
        run_loc = "nref11n_selfshield_z15/nref11c_nref8f_400kpc/"
        trackname = "halo_008508/nref11n_selfshield_z15/nref11c_nref8f_400kpc/halo_track"
        haloname = "halo008508_nref11c_nref8f_400kpc"
        if args.system == "pleiades":
            trackname = "halo_008508/nref11c_nref8f_600kpc/halo_track"
            run_loc = "nref11c_nref8f_400kpc/"

    return foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir
