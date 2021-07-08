##!/usr/bin/env python3

"""

    Title :      plot_mock_observables
    Notes :      Plots fitted (e.g. flux) and derived (e.g. metallicity) quantities
    Author :     Ayan Acharyya
    Started :    May 2021
    Examples :
run plot_mock_observables.py --system ayan_local --halo 5036 --output RD0020 --mergeHII 0.04 --galrad 6 --base_spatial_res 0.04 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 0.1 --obs_spec_res 60 --exptime 1200 --snr 5 --get_property flux --line H6562 --plot_property

run plot_mock_observables.py --system ayan_local --halo 4123 --output RD0031 --mergeHII 0.04 --galrad 6 --base_spatial_res 0.04 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 0.07 --obs_spec_res 60 --snr 1000 --iscolorlog --line H6562 --get_property metallicity --plot_property --cmax 1.2 --cmin -0.6 --keep --saveplot --doideal

run plot_mock_observables.py --system ayan_local --halo 4123 --output RD0020 --mergeHII 0.04 --galrad 20 --base_spatial_res 0.06 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 0.07 --obs_spec_res 60 --snr 0 --iscolorlog --line H6562 --get_property metallicity --plot_property --cmax 1.2 --cmin -0.6 --keep --saveplot --doideal
run plot_mock_observables.py --system ayan_local --halo 4123 --output RD0020 --mergeHII 0.04 --galrad 20 --base_spatial_res 0.06 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 0.07 --obs_spec_res 60 --snr 0 --iscolorlog --line H6562 --get_property metallicity --plot_property --cmax 1.2 --cmin -0.6 --keep --saveplot
run plot_mock_observables.py --system ayan_local --halo 4123 --output RD0020 --mergeHII 0.04 --galrad 20 --base_spatial_res 0.06 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 0.07 --obs_spec_res 60 --snr 1000 --iscolorlog --line H6562 --get_property metallicity --plot_property --cmax 1.2 --cmin -0.6 --keep --saveplot

run plot_mock_observables.py --system ayan_local --halo 4123 --output RD0020 --mergeHII 0.04 --galrad 2 --base_spatial_res 0.06 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 0.07 --obs_spec_res 60 --snr 0 --iscolorlog --line H6562 --get_property metallicity --plot_property --cmax 1.2 --cmin -0.6 --keep --saveplot --doideal
run plot_mock_observables.py --system ayan_local --halo 4123 --output RD0020 --mergeHII 0.04 --galrad 2 --base_spatial_res 0.06 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 0.07 --obs_spec_res 60 --snr 0 --iscolorlog --line H6562 --get_property metallicity --plot_property --cmax 1.2 --cmin -0.6 --keep --saveplot
run plot_mock_observables.py --system ayan_local --halo 4123 --output RD0020 --mergeHII 0.04 --galrad 2 --base_spatial_res 0.06 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 0.07 --obs_spec_res 60 --snr 1000 --iscolorlog --line H6562 --get_property metallicity --plot_property --cmax 1.2 --cmin -0.6 --keep --saveplot

run plot_mock_observables.py --system ayan_local --halo 4123 --output RD0020 --mergeHII 0.04 --galrad 2 --base_spatial_res 0.06 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 0.15 --obs_spec_res 60 --snr 0 --iscolorlog --line H6562 --get_property metallicity --plot_property --cmax 1.2 --cmin -0.6 --keep --saveplot --pix_per_beam 13

run plot_mock_observables.py --system ayan_local --halo 4123 --output RD0020 --mergeHII 0.04 --galrad 2 --base_spatial_res 0.06 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 0.07 --obs_spec_res 60 --snr 0 --iscolorlog --line H6562 --get_property metallicity --compare_property --cmax 1.5e1 --cmin 2.5e-1 --saveplot --doideal
"""
from header import *
from util import *
from fit_mock_spectra import fit_mock_spectra
from make_mock_measurements import compute_properties
plt.style.use('seaborn')

# -----------------------------------------------------------------------------
def get_property_name(args):
    '''
    Function to get the proper name (to display on plots) for a given property
    '''
    property_dict = {'flux': r'flux (ergs/s/cm$^2$)', \
                     'vel': r'velocity (km/s)', \
                     'vel_disp': r'velocity dispersion (km/s)', \
                     'metallicity': r'Z/Z$_\odot$'
                     }
    label = property_dict[args.get_property]
    if args.iscolorlog: label = r'$\log$' + label

    return label

# -----------------------------------------------------------------------------
def compare_property(property, args):
    '''
    Function to plot 2D maps of a derived property and the corresponding intrinsic property side by side using yt
    Based on: multi_plot_slice_and_proj.py in https://yt-project.org/doc/cookbook/complex_plots.html#advanced-multi-panel
    '''
    field_dict = {'metallicity': ('gas', 'metallicity')}
    unit_dict = {'metallicity': 'Zsun'}
    color_map = metal_color_map
    nticks = 5

    # load the simulation
    tmp = args.system
    args.system = 'ayan_hd' # to point to where the simulation data actually are
    ds, refine_box = load_sim(args, region='refine_box')
    args.system = tmp
    center = YTArray(args.halo_center.tolist(), kpc)
    x_width = 2 * args.galrad * kpc # ds.refine_width #
    small_box = ds.r[center[0] - x_width / 2.: center[0] + x_width / 2., center[1] - x_width / 2.: center[1] + x_width / 2., center[2] - x_width / 2.: center[2] + x_width / 2.]

    # set up the plot layout
    #fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12,6))
    #young_stars_ax, intrinsic_property_ax, derived_property_ax = axes
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,6))
    intrinsic_property_ax, derived_property_ax = axes
    fig.subplots_adjust(wspace=0.02, right=0.9, bottom=0.1, top=0.9, left=0.1)
    plt.locator_params(nbins=nticks - 1)
    fig.text(0.83, 0.22, halo_dict[args.halo] + '\n' + 'z = %.2F'% (pull_halo_redshift(args)), ha='right', va='bottom', size=args.fontsize, bbox=dict(boxstyle='round', ec='k', fc='salmon', alpha=0.7))
    '''
    # get the projected young stars distribution from the simulation
    proj = yt.ParticleProjectionPlot(ds, args.projection, ('young_stars', 'age'), center=center, data_source=small_box, width=x_width) #, weight_field=weight_field)
    proj.set_unit(('young_stars', 'age'), 'Myr')
    proj.set_buff_size(np.shape(property))

    # plotting the intrinsic young stars distribution via FRB
    proj_frb = proj.data_source.to_frb((x_width, 'kpc'), np.shape(property)[2])
    proj_frb_array = np.array(proj_frb[('young_stars', 'age')])
    plot_young_stars = young_stars_ax.imshow(proj_frb_array, origin='lower', extent=np.array([-1, 1, -1, 1]) * args.galrad)
    plot_young_stars.set_clim((0, 10))
    plot_young_stars.set_cmap(color_map)
    young_stars_ax.set_xlabel('y (kpc)', size=args.fontsize)
    young_stars_ax.set_ylabel('z (kpc)', size=args.fontsize)
    young_stars_ax.set_xticklabels(['%.1F'%item for item in young_stars_ax.get_xticks()[:-1]], size=args.fontsize)
    young_stars_ax.set_yticklabels(['%.1F'%item for item in young_stars_ax.get_yticks()], size=args.fontsize)
    fig.text(0.12, 0.78, 'Young stars', ha='left', va='top', size=args.fontsize, bbox=dict(boxstyle='round', ec='k', fc='salmon', alpha=0.7))

    cbar_ax = fig.add_axes([0.97, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(plot_young_stars, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=args.fontsize)
    cbar.set_label(get_property_name(args), size=args.fontsize)
    '''
    # get the intrinsic projected property (in this case metallicity) from the simulation
    proj = yt.ProjectionPlot(ds, args.projection, field_dict[args.get_property], center=center, data_source=small_box, width=x_width, weight_field=('gas', 'density')) #, weight_field=weight_field)
    proj.set_unit(field_dict[args.get_property], unit_dict[args.get_property])

    # plotting the intrinsic property via FRB
    proj_frb = proj.data_source.to_frb((x_width, 'kpc'), np.shape(property)[0])
    proj_frb_array = np.array(proj_frb[field_dict[args.get_property]])
    plot_intrinsinc = intrinsic_property_ax.imshow(proj_frb_array, origin='lower', norm=LogNorm() if args.iscolorlog else None, extent=np.array([-1, 1, -1, 1]) * args.galrad)
    plot_intrinsinc.set_clim((args.cmin, args.cmax))
    plot_intrinsinc.set_cmap(color_map)
    intrinsic_property_ax.set_xlabel('y (kpc)', size=args.fontsize)
    intrinsic_property_ax.set_ylabel('z (kpc)', size=args.fontsize)
    intrinsic_property_ax.set_xticklabels(['%.1F'%item for item in intrinsic_property_ax.get_xticks()[:-1]], size=args.fontsize)
    intrinsic_property_ax.set_yticklabels(['%.1F'%item for item in intrinsic_property_ax.get_yticks()], size=args.fontsize)
    fig.text(0.12, 0.78, 'Intrinsic', ha='left', va='top', size=args.fontsize, bbox=dict(boxstyle='round', ec='k', fc='salmon', alpha=0.7))

    # plotting the derived property
    derived_property_ax.set_facecolor('k')
    plot_derived =derived_property_ax.imshow(np.transpose(property), origin='lower', norm=LogNorm() if args.iscolorlog else None, extent=np.array([-1, 1, -1, 1]) * args.galrad)
    plot_derived.set_clim((args.cmin, args.cmax))
    plot_derived.set_cmap(color_map)
    derived_property_ax.set_xlabel('y (kpc)', size=args.fontsize)
    derived_property_ax.set_xticklabels(['%.1F'%item for item in derived_property_ax.get_xticks()], size=args.fontsize)
    fig.text(0.5, 0.78, 'Derived', ha='left', va='top', size=args.fontsize, bbox=dict(boxstyle='round', ec='k', fc='salmon', alpha=0.7))

    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(plot_intrinsinc, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=args.fontsize)
    cbar.set_label(get_property_name(args), size=args.fontsize)

    if args.saveplot:
        log_text = '_log' if args.iscolorlog else ''
        fig_output_dir = os.path.split(args.measured_cube_filename)[0].replace('/fits/', '/figs/') + '/'
        rootname = os.path.splitext(os.path.split(args.measured_cube_filename)[1])[0]
        saveplot(fig, args, 'comparison_' + rootname + log_text + '_' + str(args.get_property), outputdir=fig_output_dir)

    plt.show(block=False)
    return axes

# -----------------------------------------------------------------------------
def plot_property(property, args):
    '''
    Function to plot a given 2D map
    '''
    line_dict = {'H6562': r'H$\alpha$', 'NII6584':r'NII 6584', 'SII6717':r'SII 6717', 'SII6730':r'SII 6730'}
    isline = args.get_property in ['flux', 'flux_u', 'vel', 'vel_u', 'vel_disp', 'vel_disp_u']

    cmap_dict = defaultdict(lambda: density_color_map, vel= velocity_discrete_cmap, metallicity= metal_color_map)
    cmin_dict = defaultdict(lambda: None, metallicity= -6e-1, flux= -18, vel= -400)
    cmax_dict = defaultdict(lambda: None, metallicity= 1.2e0, flux= -15.5, vel= 400)

    cmap = cmap_dict[args.get_property]
    if args.cmin is None: args.cmin = cmin_dict[args.get_property]
    if args.cmax is None: args.cmax = cmax_dict[args.get_property]
    if 'vel' in args.get_property: args.iscolorlog = False

    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.95)

    if args.iscolorlog: property = np.log10(property)

    p = ax.imshow(np.transpose(property), cmap=cmap, origin='lower', extent=np.array([-1, 1, -1, 1]) * args.galrad, vmin=args.cmin, vmax=args.cmax)

    fig.text(0.2, 0.9, halo_dict[args.halo] + '\n' + 'z = %.2F'% (pull_halo_redshift(args)), ha='left', va='top', size=args.fontsize, bbox=dict(boxstyle='round', ec='k', fc='salmon', alpha=0.7))
    ax.set_xlabel(projection_dict[args.projection][0] + ' (kpc)', size=args.fontsize)
    ax.set_ylabel(projection_dict[args.projection][1] + ' (kpc)', size=args.fontsize)
    ax.tick_params(axis='both', labelsize=args.fontsize)
    if isline: fig.text(0.7, 0.9, line_dict[args.line], size=args.fontsize, va='top', ha='right', bbox=dict(boxstyle='round', ec='k', fc='salmon', alpha=0.7))

    cb = plt.colorbar(p)
    cb.set_label(get_property_name(args), size=args.fontsize)
    cb.ax.tick_params(labelsize=args.fontsize)

    if args.saveplot:
        log_text = '_log' if args.iscolorlog else ''
        line_text = '_' + args.line if isline else ''
        fig_output_dir = os.path.split(args.measured_cube_filename)[0].replace('/fits/', '/figs/') + '/'
        rootname = os.path.splitext(os.path.split(args.measured_cube_filename)[1])[0]
        saveplot(fig, args, rootname + log_text + line_text + '_' + str(args.get_property), outputdir=fig_output_dir)

    plt.show(block=False)
    return ax

# -----------------------------------------------------------------------------
def get_property(measured_cube, args):
    '''
    Function to read in given fitted property for a given emission line
    '''
    if args.get_property[-2:] == '_u': which_property = args.get_property[:-2]
    else: which_property = args.get_property

    if which_property in measured_cube.measured_quantities: # i.e. this is a fitted property, associated with a particular emission line
        property_index = np.where(measured_cube.measured_quantities == which_property)[0][0]
        line_prop, line_prop_u = measured_cube.get_line_prop(args.line)

        property = line_prop[:, :, property_index]
        property_u = line_prop_u[:, :, property_index]
    elif not hasattr(measured_cube, 'derived_quantities') or which_property not in measured_cube.derived_quantities: # i.e. this measured cube does not have any derived quantities (only has fitted quantities) yet
        myprint('Derived property ' + which_property + ' does not exist in measured_cube yet, so calling compute_property()..', args)
        tmp = args.plot_property
        args.write_property, args.plot_property, args.compute_property = True, False, which_property # to ensure this sub-call does not plot property (it will plot in the current routine anyway) and that the newly computed property is written to file
        property, property_u = compute_properties(measured_cube, args)
        args.plot_property = tmp
    else: # i.e. this is a derived property (not based on one particular emission line) AND it exists in the currently loaded cube's list of derived properties
        property, property_u = measured_cube.get_derived_prop(which_property)


    if 'flux' in args.get_property: property = np.ma.masked_less(property, 0) # masks negaitve flux values
    if 'metallicity' in args.get_property: property = np.ma.masked_outside(property, 1e-3, 1e3) # masks too unphysical metallicity values
    if 'vel' in args.get_property:
        flux, flux_u = line_prop[:, :, 0], line_prop_u[:, :, 0]
        property = np.ma.masked_where(flux == 0, property) # masks (otherwise = 0) velocity values where flux doesn't exist
        property_u = np.ma.masked_where(flux == 0, property_u) # masks (otherwise = 0) velocity values where flux doesn't exist

    if args.plot_property:
        if args.get_property[-2:] == '_u': ax = plot_property(property_u, args)
        else: ax = plot_property(property, args)
    if args.compare_property: ax = compare_property(property, args)

    return property, property_u

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    args = parse_args('8508', 'RD0042')
    if type(args) is tuple: args = args[0] # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    if not args.keep: plt.close('all')

    if args.doideal: fitted_file = args.idealcube_filename
    elif args.snr == 0: fitted_file = args.smoothed_cube_filename
    else: fitted_file = args.mockcube_filename
    args.measured_cube_filename = get_measured_cube_filename(fitted_file)
    if not os.path.exists(args.measured_cube_filename):
        myprint('measured_cube does not exist, so calling fit_mock_spectra.py to create one..', args)
        measured_cube_dummy = fit_mock_spectra(args)

    measured_cube = read_measured_cube(args.measured_cube_filename, args)
    property, property_u = get_property(measured_cube, args)

    print('Complete in %s minutes' % ((time.time() - start_time) / 60))
