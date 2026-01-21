"""

    Filename: fogghorn_analysis.py
    Authors: Cassi, Ayan,
    Created: 06-12-24
    Last modified: 04-28-25 by Cassi

    This "master" script calls the relevant functions to produce a set of basic analysis plots.
    There are two options for types of plots to make:
    1) Plots of every single snapshot -- useful for visualizations of analysis of single outputs
    2) Plots where multiple snapshots are collected on the same plot -- useful for population or time evolution plots

    For the first type of plot, this code will search through the specified save directory to determine
    if the plot requested for the snapshot requested already exists, and if not, it will make it.

    For the second type of plot, this code will create a data table called central_galaxy_info.txt, and add
    information for every snapshot to the data table before it creates the plot containing all that
    data at the end. It will check the table to see if the information needed already exists in the
    table so it doesn't re-calculate, but it will ALWAYS re-make the plot containing all snapshots in the file.

    The user can choose which plots or groups of plots to make. This script does the book-keeping for existing
    plots and multiprocessing, and is the script that should be called by the user.
    The actual plotting routines are in XXXX_plots.py.

    There is more detailed documentation in foggie/doc/source/user_guide/analysis_scripts.rst. That file 
    also describes how to add a plot to these scripts.

"""

from html import parser
from foggie.fogghorn.header import *
from foggie.fogghorn.util import *

start_time = datetime.now()

# --------------------------------------------------------------------------------------------------------------------
def parse_args():
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Produces analysis plots for FOGGHORN runs.')

    # Optional arguments:

    # These arguments are for file organization
    parser.add_argument('--directory', metavar='directory', type=str, action='store', default='./', help='What is the directory of the enzo outputs you want to make plots of?')
    parser.add_argument('--save_directory', metavar='save_directory', type=str, action='store', default=None, help='Where do you want to store the plots? Default is to put them in a plots/ directory inside the outputs directory.')
    parser.add_argument('--output', metavar='output', type=str, action='store', default=None, help='If you want to make the plots for specific output/s then specify those here separated by comma (e.g., DD0030,DD0040). Otherwise (default) it will make plots for ALL outputs in that directory')
    parser.add_argument('--output_step', metavar='output_step', type=int, action='store', default=1, help='If you are making plots for specific outputs, use this to specify every Nth output in the range given by --output.')
    parser.add_argument('--pwd', dest='pwd', action='store_true', default=False, help='Just use the working directory?, Default is no')
    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', default=1, help='How many processes do you want? Default is 1 (no parallelization), if multiple processors are specified, code will run one output per processor')
    parser.add_argument('--rockstar_directory', metavar='rockstar_directory', type=str, action='store', default=None, help='What is the directory where your rockstar outputs are located?')

    # These arguments are defined as in foggie_load for consistency - these will all be flowed down the the foggie_load call make in fogghorn_analysis
    parser.add_argument('--central_halo', metavar='central_halo', type=bool, action='store', default=True, help='Are you analyzing the central halo of the simulation? Default is True. Goes to foggie_load')
    parser.add_argument('--trackfile_name', metavar='trackfile_name ', type=str, action='store', default=None, help='What is the directory of the track file for this halo?\n' + 'This is needed to find the center of the galaxy of interest.')
    parser.add_argument('--halo_c_v_name', metavar='halo_c_v_name', type=str, action='store', default=None, help='What is the name of the halo catalog file to use for finding halo centers? Default is None')
    parser.add_argument('--root_catalog_name', metavar='root_catalog_name', type=str, action='store', default=None, help='What is the root name of the halo catalog files to use for finding halo centers? Default is None')
    parser.add_argument('--do_filter_particles', dest='do_filter_particles', action='store_true', default=True, help='Filter star particles to only those in high-res region? Default is yes. Goes to foggie_load')
    parser.add_argument('--disk_relative', dest='disk_relative', action='store_true', default=False, help='Load the dataset in a disk-relative frame? Default is no. Goes to foggie_load')
    parser.add_argument('--smooth_AM_name', metavar='smooth_AM_name', type=bool, action='store', default=False, help='If using a smoothed center file, what is the name of that file? Default is None.')
    parser.add_argument('--particle_type_for_angmom', metavar='particle_type_for_angmom', type=str, action='store', default='young_stars', help='Which particle type to use for calculating angular momentum for disk-relative loading? Default is stars. Options are stars or dark_matter.')
    parser.add_argument('--gravity', metavar='gravity', type=bool, action='store', default=False, help='Include gravity when loading the dataset? Default is True. Goes to foggie_load')
    parser.add_argument('--masses_dir', metavar='masses_dir', type=str, action='store', default='', help='Directory where particle masses files are located, if needed. Default is None. Goes to foggie_load')

    # These arguments are options for halo center finding
    parser.add_argument('--use_track_center', dest='use_track_center', action='store_true', default=False, help='Just use trackbox center instead of finding halo center? Default is no.')

    # These arguments are options for how this code is run
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Over-write existing plots? Default is no.')
    parser.add_argument('--silent', dest='silent', action='store_true', default=False, help='Suppress some generic pritn statements? Default is no.')
    
    # These arguments are plotting options
    ###### IF YOU ADD A PLOT STEP 6: if that plot has options, add an argument for it here ######
    parser.add_argument('--proj_width', metavar='proj_width', type=float, action='store', default=20., help='If making projection plots, use this to specify the width of the plots.')
    parser.add_argument('--upto_kpc', metavar='upto_kpc', type=float, action='store', default=None, help='Limit analysis out to a certain physical kpc. By default it does the entire refine box.')
    parser.add_argument('--projection', metavar='projection', type=str, action='store', default='x', help='Which projection do you want to plot, i.e., which axis is your line of sight? Default is x; but user can input multiple comma-separated values. Options are: x, y, z, x_disk, y_disk, z_disk')
    parser.add_argument('--use_cen_smoothed', dest='use_cen_smoothed', action='store_true', default=False, help='use Cassis new smoothed center file?, default is no')

    # These are for metallicity plots that are not ready yet
    parser.add_argument('--docomoving', dest='docomoving', action='store_true', default=False, help='Consider the input upto_kpc as a comoving quantity? Default is No.')
    parser.add_argument('--weight', metavar='weight', type=str, action='store', default=None, help='Name of quantity to weight the metallicity by. Default is None i.e., no weighting.')
    parser.add_argument('--use_density_cut', dest='use_density_cut', action='store_true', default=False, help='Impose a density cut to get just the disk? Default is no.')
    parser.add_argument('--nbins', metavar='nbins', type=int, action='store', default=100, help='Number of bins to use for the metallicity histogram plot. Default is 100')

    # These arguments are for choosing groups of plots to make
    ###### IF YOU ADD A PLOT STEP 3B: If your plot has a new category, add the argument for that category here. ######
    parser.add_argument('--all_plots', dest='all_plots', action='store_true', default=False, help='Make all the plots? Default is no.')
    parser.add_argument('--all_sf_plots', dest='all_sf_plots', action='store_true', default=False, help='Make all star formation plots? Default is no.')
    parser.add_argument('--all_fb_plots', dest='all_fb_plots', action='store_true', default=False, help='Make all feedback plots? Default is no.')
    parser.add_argument('--all_vis_plots', dest='all_vis_plots', action='store_true', default=False, help='Make all visualisation plots? Default is no.')
    parser.add_argument('--all_edge_plots', dest='all_edge_plots', action='store_true', default=False, help='Make all edge-on temperature plots? Default is no.')
    #parser.add_argument('--all_metal_plots', dest='all_metal_plots', action='store_true', default=False, help='Make all resolved metallicity plots? Default is no.') # Not ready yet
    parser.add_argument('--all_time_evol_plots', dest='all_time_evol_plots', action='store_true', default=False, help='Make all time-evolving central galaxy properties plots? Default is no.')
    parser.add_argument('--all_highz_halos_plots', dest='all_highz_halos_plots', action='store_true', default=False, help='Make all plots with all high-z halos on each plot (no central)? Default is no.')
    parser.add_argument('--all_shade_maps', dest='all_shade_maps', action='store_true', default=False, help='Make all shade maps plots? Default is no.')
    parser.add_argument('--all_diagnosis_plots', dest='all_diagnosis_plots', action='store_true', default=False, help='Make all diagnosis plots? Default is no.')
    # This argument is for specifying which individual plots you want to make
    parser.add_argument('--make_plots', metavar='make_plots', type=str, action='store', default='', help='Which plots to make? Comma-separated names of the plotting routines to call. Default is none.')

    # These arguments are used for backward compatibility, to find the trackfile for production runs, if a trackfile has not been explicitly specified
    parser.add_argument('--system', metavar='system', type=str, action='store', default='ayan_local', help='Which system are you on? This is used only when trackfile is not specified. Default is ayan_local')
    parser.add_argument('--halo', metavar='halo', type=str, action='store', default='8508', help='Which halo? Default is Tempest. This is used only when trackfile is not specified.')
    parser.add_argument('--run', metavar='run', type=str, action='store', default='nref11c_nref9f', help='Which run? Default is nref11c_nref9f. This is used only when trackfile is not specified.')

    # ------- wrap up and processing args ------------------------------
    ###### IF YOU ADD A PLOT STEP 5: Add the function name as a string to this plots_needing_projection list ######
    plots_needing_projection = ['gas_density_projection', 'gas_temperature_projection', 'gas_h1_projection', 'gas_h2_projection', 'gas_mg2_projection', 
                                'gas_o6_projection', 'young_stars_density_projection', 'KS_relation', 'gas_metallicity_projection', 
                                'edge_projection', 'edge_slice']
    args = parser.parse_args()
    args.projection_arr = [item for item in args.projection.split(',')]
    if (args.make_plots!=''):
        args.plots_asked_for = [item for item in args.make_plots.split(',')]
        plots_with_projections = []
        for i in range(len(args.plots_asked_for)):
            if (args.plots_asked_for[i] in plots_needing_projection):
                for p in args.projection_arr:
                    if ('edge' in args.plots_asked_for[i]) and ('z' in p): print('Not making edge plots for projection z')
                    else:
                        plots_with_projections.append(args.plots_asked_for[i] + '_' + p)
            else:
                plots_with_projections.append(args.plots_asked_for[i])
        args.plots_asked_for = plots_with_projections
    else: args.plots_asked_for = []

    return args

# --------------------------------------------------------------------------------------------------------------------
def update_table(ds, snap, args):
    '''
    Determines if the halo info table needs to be updated with information from this snapshot
    and either adds the information if needed or skips this snapshot if it already
    exists in the table.
    '''

    # Load the table
    data = Table.read(args.save_directory + '/central_galaxy_info.txt', format='ascii.ecsv')

    # Check if the snapshot is already in the table
    if snap in data['snapshot']:
        if not args.silent:
            print('Halo info for snapshot' + snap + 'already calculated.')
        if not args.clobber:
            if not args.silent:
                print('So we will skip it.')
            return
        else:
            if not args.silent:
                print('But we will re-calculate it...')

    row = get_halo_info(ds, snap, args)
    data = make_table()
    data.add_row(row)
    data.write(args.save_directory + '/' + snap + '_central_galaxy_info.txt', format='ascii.ecsv', overwrite=True)
    if not args.silent: print('Halo info for snapshot ' + snap + ' written to central_galaxy_info.txt')

# --------------------------------------------------------------------------------------------------------------------
def does_table_need_updating(outputs, args):
    '''This function returns True if the halo data table needs to be updated with info
    from any snapshots in 'outputs', and returns False otherwise.'''

    if (args.table_needed):
        data = Table.read(args.save_directory + '/central_galaxy_info.txt', format='ascii.ecsv')

        table_needed = []
        for snap in outputs:
            # Search for this snapshot in the table
            if (snap in data['snapshot']):
                if args.clobber:
                    table_needed.append(True)
                else:
                    table_needed.append(False)
            else:
                table_needed.append(True)
        if (True in table_needed):
            if not args.silent: print('central_galaxy_info.txt will be updated!')
            return True
        else:
            if not args.silent: print('central_galaxy_info.txt will not be updated!')
            return False

    else:
        return False

# --------------------------------------------------------------------------------------------------------------------
def which_plots_asked_for(args):
    '''
    Determines which plots have been asked for by the user, and then checks which of them already exists, and
    returns the list of plots that need to be still made.
    '''

    plots_asked_for = args.plots_asked_for

    ###### IF YOU ADD A PLOT STEP 3C: If your plot has a new category, add it to the below lists. ######
    if args.all_plots:
        plots_asked_for += np.hstack([args.sf_plots, args.fb_plots, args.vis_plots, args.edge_plots, args.highz_halos_plots, args.time_evol_plots]) # these *_plots variables are defined in main below
    else:
        if args.all_sf_plots: plots_asked_for += args.sf_plots
        if args.all_fb_plots: plots_asked_for += args.fb_plots
        if args.all_edge_plots: plots_asked_for += args.edge_plots
        if args.all_vis_plots: plots_asked_for += args.vis_plots
        #if args.all_metal_plots: plots_asked_for += args.metal_plots
        if args.all_highz_halos_plots: plots_asked_for += args.highz_halos_plots
        if args.all_time_evol_plots: plots_asked_for += args.time_evol_plots
        if args.all_shade_maps: plots_asked_for += args.shade_maps
        if args.all_diagnosis_plots: plots_asked_for += args.diagnosis_plots

    plots_asked_for = np.unique(plots_asked_for)
    print(plots_asked_for)

    return plots_asked_for

# ----------------------------------------------------------------------------
def generate_plot_filename(quantity, args, snap):
    '''
    Generates filename for a plot that is about to be made.
    This way the nomenclature is consistent.
    '''

    ###### IF YOU ADD A PLOT STEP 4: Add it to this dictionary as 'function_name':'output_filename.png' ######
    output_filename_dict = {'young_stars_density_projection_x':snap + '_Projection_young_stars3_cic_x.png', \
                            'young_stars_density_projection_y':snap + '_Projection_young_stars3_cic_y.png', \
                            'young_stars_density_projection_z':snap + '_Projection_young_stars3_cic_z.png', \
                            'young_stars_density_projection_x_disk':snap + '_Projection_young_stars3_cic_x-disk.png', \
                            'young_stars_density_projection_y_disk':snap + '_Projection_young_stars3_cic_y-disk.png', \
                            'young_stars_density_projection_z_disk':snap + '_Projection_young_stars3_cic_z-disk.png', \
                            'KS_relation_x': snap + '_KS-relation_x.png', \
                            'KS_relation_y': snap + '_KS-relation_y.png', \
                            'KS_relation_z': snap + '_KS-relation_z.png', \
                            'KS_relation_x_disk': snap + '_KS-relation_x-disk.png', \
                            'KS_relation_y_disk': snap + '_KS-relation_y-disk.png', \
                            'KS_relation_z_disk': snap + '_KS-relation_z-disk.png', \
                            'outflow_rates': snap + '_outflows.png', \
                            'gas_density_projection_x': snap + '_Projection_density_x.png', \
                            'gas_density_projection_y': snap + '_Projection_density_y.png', \
                            'gas_density_projection_z': snap + '_Projection_density_z.png', \
                            'gas_temperature_projection_x': snap + '_Projection_temperature_x.png', \
                            'gas_temperature_projection_y': snap + '_Projection_temperature_y.png', \
                            'gas_temperature_projection_z': snap + '_Projection_temperature_z.png', \
                            'gas_h1_projection_x': snap + '_Projection_HI_x.png', \
                            'gas_h1_projection_y': snap + '_Projection_HI_y.png', \
                            'gas_h1_projection_z': snap + '_Projection_HI_z.png', \
                            'gas_mg2_projection_x': snap + '_Projection_Mg2_x.png', \
                            'gas_mg2_projection_y': snap + '_Projection_Mg2_y.png', \
                            'gas_mg2_projection_z': snap + '_Projection_Mg2_z.png', \
                            'gas_o6_projection_x': snap + '_Projection_O6_x.png', \
                            'gas_o6_projection_y': snap + '_Projection_O6_y.png', \
                            'gas_o6_projection_z': snap + '_Projection_O6_z.png', \
                            'gas_H2_projection_x': snap + '_Projection_H2_x.png', \
                            'gas_H2_projection_y': snap + '_Projection_H2_y.png', \
                            'gas_H2_projection_z': snap + '_Projection_H2_z.png', \
                            'gas_density_projection_x_disk': snap + '_Projection_density_x-disk.png', \
                            'gas_density_projection_y_disk': snap + '_Projection_density_y-disk.png', \
                            'gas_density_projection_z_disk': snap + '_Projection_density_z-disk.png', \
                            'gas_metallicity_projection_x': snap + '_gas_metallicity_projection_x.png', \
                            'gas_metallicity_projection_y': snap + '_gas_metallicity_projection_y.png', \
                            'gas_metallicity_projection_z': snap + '_gas_metallicity_projection_z.png', \
                            'gas_metallicity_projection_x_disk': snap + '_gas_metallicity_projection_x-disk.png', \
                            'gas_metallicity_projection_y_disk': snap + '_gas_metallicity_projection_y-disk.png', \
                            'gas_metallicity_projection_z_disk': snap + '_gas_metallicity_projection_z-disk.png', \
                            'edge_projection_x_disk': snap + '_Projection_temperature_density-weighted_x-disk.png', \
                            'edge_projection_y_disk': snap + '_Projection_temperature_density-weighted_y-disk.png', \
                            'edge_slice_x_disk': snap + '_Slice_temperature_x-disk.png', \
                            'edge_slice_y_disk': snap + '_Slice_temperature_y-disk.png', \
                            'gas_metallicity_resolved_MZR': snap + '_resolved_gas_MZR' + args.upto_text + args.density_cut_text + '.png', \
                            'gas_metallicity_histogram': snap + '_gas_metallicity_histogram' + args.upto_text + args.density_cut_text + '.png', \
                            'gas_metallicity_radial_profile': snap + '_gas_metallicity_radial_profile' + args.upto_text + args.density_cut_text + '.png', \
                            'den_temp_phase': snap + '_density_temperature_phase_plot' + args.upto_text +'.png', \
                            'rad_vel_temp_colored': snap + '_radial-velocity_temperature.png', \
                            'halos_SFMS': snap + '_halos_SFMS.png', \
                            'halos_SMHM': snap + '_halos_SMHM.png', \
                            'halos_MZR': snap + '_halos_MZR.png', \
                            'halos_h2_frac': snap + '_halos_h2_fraction.png', \
                            'halos_gasMHM': snap + '_halos_gas-mass_halo-mass.png', \
                            'halos_ismMHM': snap + '_halos_ism-mass_halo-mass.png', \
                            'halos_cgmMHM': snap + '_halos_cgm-mass_halo-mass.png', \
                            'baryon_budget': snap + '_baryon_budget.png', \
                            'plot_SFMS': 'SFMS.png', \
                            'plot_SMHM': 'SMHM.png', \
                            'plot_MZR': 'MZR.png', \
                            'phase_shade': snap + '_shade', \
                            'diagnosis_plots': snap + '_diagnosis_plots.png'}

    output_filename = args.save_directory + '/' + output_filename_dict[quantity]
    return output_filename

# --------------------------------------------------------------------------------------------------------------------
def make_everysnap_plots(snap, args):
    '''
    Finds the halo center and other properties of the dataset and then calls the plotting scripts.
    Returns nothing. Saves outputs as multiple png files.
    '''

    # Figure out which plots are being asked for
    plots_asked_for = which_plots_asked_for(args)
    plots_to_make = []

    need_disk = args.disk_rel
    # Check if these file names already exist, and if not, add plots to the list of plots to make
    for thisplot in plots_asked_for:
        output_filename = generate_plot_filename(thisplot, args, snap)
        if (thisplot not in args.time_evol_plots) and (need_to_make_this_plot(output_filename, args)):
            plots_to_make += [thisplot]
            if ('disk' in thisplot): need_disk=True

    myprint('Total %d plots asked for, of which %d will be made, others already exist' %(len(plots_asked_for), len(plots_to_make)), args)

    if (len(plots_to_make) > 0) or (does_table_need_updating([snap], args)):
        # Read the snapshot
        filename = args.directory + '/' + snap + '/' + snap
        args.snap = snap

        #all of foggie_load's arguments are passed down from args so we include them all here
        ds, region = foggie_load(filename, central_halo = args.central_halo, trackfile_name=args.trackfile_name, halo_c_v_name=args.halo_c_v_name, 
                                 root_catalog_name=args.root_catalog_name, do_filter_particles=True, disk_relative=need_disk, 
                                 central_halo=args.central_halo, smooth_AM_name=args.smooth_AM_name, 
                                 particle_type_for_angmom=args.particle_type_for_angmom, gravity=args.gravity, 
                                 masses_dir=args.masses_dir) 

        #if args.trackfile_name == None:
        #    ds, region = foggie_load(filename, do_filter_particles=True, disk_relative=need_disk, central_halo=False) 
        #else:
        #    if (args.use_track_center):
        #        ds, region = foggie_load(filename, trackfile_name=args.trackfile_name, do_filter_particles=True, central_halo=False)
        #    else:
        #        ds, region = foggie_load(filename, trackfile_name=args.trackfile_name, do_filter_particles=True, disk_relative=need_disk)

        #  Make the plots
        for thisplot in plots_to_make:
            output_filename = generate_plot_filename(thisplot, args, snap)
            globals()[thisplot](ds, region, args, output_filename)

        # Update the halo info table
        if (does_table_need_updating([snap], args)):
            update_table(ds, snap, args)

    print('Yayyy you have completed making all plots for this snap ' + snap)

# --------------------------------------------------------------------------------------------------------------------
def make_manysnaps_plots(args):
    '''
    This function makes plots where information from multiple snapshots will be
    displayed on a single plot, using information from central_galaxy_info.txt.
    
   Unlike the plots that are made for every snapshot, it will remake the plots even if
   they already exist, since it is possible that central_galaxy_info.txt was updated since the plot was
   last made. It just reads from the data file to make the plots, so re-making is quick.'''

    for plot in args.time_evol_plots:
        if (plot in args.make_plots) or (args.all_time_evol_plots):
            output_filename = generate_plot_filename(plot, args, '')
            globals()[plot](args, output_filename)

# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    print('FOGGHORN_ANALYSIS: your trackfile is ', args.trackfile_name)
    ###### IF YOU ADD A PLOT STEP 3 ######
    # Add the function name to the appropriate grouping list, or make a new list
    # These plots make one plot per snapshot:
    args.sf_plots = []
    args.vis_plots = []
    args.edge_plots = []
    args.disk_rel = False
    for p in args.projection_arr:
        args.sf_plots.append('gas_density_projection_' + p)
        args.sf_plots.append('young_stars_density_projection_' + p)
        args.sf_plots.append('KS_relation_' + p)
        args.vis_plots.extend(['gas_density_projection_' + p, 'gas_temperature_projection_' + p, 'gas_h1_projection_' + p, 'gas_mg2_projection_' + p, 'gas_o6_projection_' + p, 'gas_H2_projection_' + p, 'gas_metallicity_projection_' + p])
        if ('disk' in p):
            args.disk_rel = True
    args.edge_plots = ['edge_projection_x_disk', 'edge_slice_x_disk', 'edge_projection_y_disk', 'edge_slice_y_disk']
    args.fb_plots = ['outflow_rates', 'rad_vel_temp_colored']
    #args.metal_plots = ['gas_metallicity_resolved_MZR', 'gas_metallicity_histogram', 'gas_metallicity_radial_profile'] # Not ready yet

    # These plots put all halos in the high-res area of the box on one plot and require
    # yt's HOP halo finder to be run (each will check if the halo catalog already exists).
    # It is not recommended to run these on snapshots lower than z = 2 because the halo finder
    # doesn't work well at low redshifts:
    args.highz_halos_plots = ['halos_SMHM','halos_SFMS','halos_MZR','halos_gasMHM', 'halos_ismMHM', 'halos_cgmMHM', 'baryon_budget', 'halos_h2_frac']

    # These plots add a line to the central_galaxy_info.txt table for each snapshot, then make
    # ONE plot at the end containing data from every snapshot:
    args.time_evol_plots = ['plot_SFMS', 'plot_SMHM'] #, 'plot_MZR'] plot_MZR isn't ready yet

    # These plots are datashader-style maps of physical quantities from earlier papers 
    args.shade_maps = ['phase_shade'] 

    # These plots show various quantities for a zoom/halo snapshot such as star particle mass vs. time, from Diagnosis scripts
    args.diagnosis_plots = ['diagnosis_plots'] 

    # ------------------ Figure out directory and outputs -------------------------------------
    if args.save_directory is None:
        args.save_directory = args.directory + '/plots'
        Path(args.save_directory).mkdir(parents=True, exist_ok=True)

    #This line is commented out so that the user has to explicitly provide a trackfile if needed, and we can work on runs without tracks 
    #if args.trackfile is None: _, _, _, args.code_path, args.trackfile, _, _, _ = get_run_loc_etc(args) # for FOGGIE production runs it knows which trackfile to grab
    #print('Using trackfile: ', args.trackfile, ' derived from get_run_loc_etc')

    if args.output is not None: # Running on specific output/s
        outputs = make_output_list(args.output, output_step=args.output_step)
    else: # Running on all snapshots in the directory
        outputs = []
        for fname in os.listdir(args.directory):
            folder_path = os.path.join(args.directory, fname)
            if os.path.isdir(folder_path) and ((fname[0:2]=='DD') or (fname[0:2]=='RD')):
                outputs.append(fname)
    print(outputs)

    # Determine if any of the asked-for plots require the halo info table
    args.table_needed = False
    for plot in args.time_evol_plots:
        if (plot in args.make_plots) or (args.all_time_evol_plots):
            args.table_needed = True
    # If table is needed but doesn't exist, initialize it
    if (args.table_needed) and (not os.path.exists(args.save_directory + '/central_galaxy_info.txt')):
        data = make_table()
        data.write(args.save_directory + '/central_galaxy_info.txt', format='ascii.ecsv', overwrite=True)

    # ----------------- Add some parameters to args that will be used throughout ----------------------------------
    args.density_cut_text = '_wdencut' if args.use_density_cut else ''
    args.upto_text = '' if args.upto_kpc is None else '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
 
    # --------- Loop over outputs, for either single-processor or parallel processor computing ---------------
    if (args.nproc == 1):
        for snap in outputs:
            make_everysnap_plots(snap, args)
        print('Serially: time taken for ' + str(len(outputs)) + ' snapshots with ' + str(args.nproc) + ' core was %s' % timedelta(seconds=(datetime.now() - start_time).seconds))
    else:
        # ------- Split into a number of groupings equal to the number of processors and run one process per processor ---------
        for i in range(len(outputs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outputs[args.nproc*i+j]
                print(snap)
                threads.append(multi.Process(target=make_everysnap_plots, args=[snap, args]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # Add each snapshot's halo data to master halo data file
            if (args.table_needed):
                data = Table.read(args.save_directory + '/central_galaxy_info.txt', format='ascii.ecsv')
                for j in range(args.nproc):
                    snap = outputs[args.nproc*i+j]
                    if (os.path.exists(args.save_directory + '/' + snap + '_central_galaxy_info.txt')):
                        snap_data = Table.read(args.save_directory + '/' + snap + '_central_galaxy_info.txt', format='ascii.ecsv')
                        data.add_row(snap_data[0])
                        os.remove(args.save_directory + '/' + snap + '_central_galaxy_info.txt')
                data.sort('time')
                data.write(args.save_directory + '/central_galaxy_info.txt', format='ascii.ecsv', overwrite=True)
        # ----- For any leftover snapshots, run one per processor ------------------
        threads = []
        for j in range(len(outputs) % args.nproc):
            snap = outputs[-(j+1)]
            threads.append(multi.Process(target=make_everysnap_plots, args=[snap, args]))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Add each snapshot's halo data to master halo data file
        if (args.table_needed):
            data = Table.read(args.save_directory + '/central_galaxy_info.txt', format='ascii.ecsv')
            for j in range(len(outputs) % args.nproc):
                snap = outputs[-(j+1)]
                if (os.path.exists(args.save_directory + '/' + snap + '_central_galaxy_info.txt')):
                    snap_data = Table.read(args.save_directory + '/' + snap + '_central_galaxy_info.txt', format='ascii.ecsv')
                    data.add_row(snap_data[0])
                    os.remove(args.save_directory + '/' + snap + '_central_galaxy_info.txt')
            data.sort('time')
            data.write(args.save_directory + '/central_galaxy_info.txt', format='ascii.ecsv', overwrite=True)
        print('Parallely: time taken for ' + str(len(outputs)) + ' snapshots with ' + str(args.nproc) + ' cores was %s' % timedelta(seconds=(datetime.now() - start_time).seconds))

    # If asked for, make time-evolution plots using data in central_galaxy_info.txt
    # If any of the time-evolution population plots were asked for, then central_galaxy_info.txt would
    # have already been created and/or updated in make_everysnap_plots
    if (args.table_needed):
        make_manysnaps_plots(args)
