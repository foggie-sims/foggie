##!/usr/bin/env python3

"""

    Title :      plot_mock_observables
    Notes :      Plots fitted (e.g. flux) and derived (e.g. metallicity) quantities
    Author :     Ayan Acharyya
    Started :    May 2021
    Example :    run plot_mock_observables.py --system ayan_local --halo 5036 --output RD0020 --mergeHII 0.04 --galrad 6 --base_spatial_res 0.04 --z 0.25 --obs_wave_range 0.8,0.85 --obs_spatial_res 0.1 --obs_spec_res 60 --exptime 1200 --snr 5 --get_property flux --line H6562 --plot_property

"""
from header import *
from util import *

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
    if args.islog: label = r'$\log$' + label

    return label

# -----------------------------------------------------------------------------
def plot_property(property, args):
    '''
    Function to plot a given 2D map
    '''
    line_dict = {'H6562': r'H$\alpha$', 'NII6584':r'NII 6584', 'SII6717':r'SII 6717', 'SII6730':r'SII 6730'}
    isline = args.get_property in ['flux', 'flux_u', 'vel', 'vel_u', 'vel_disp', 'vel_disp_u']

    cmap = 'BrBG' if 'vel' in args.get_property else 'viridis'
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.95)

    if args.islog: property = np.log10(property)

    p = ax.imshow(property, cmap=cmap, origin='lower', extent=np.array([-1, 1, -1, 1]) * args.galrad, vmin=args.cmin, vmax=args.cmax)

    ax.set_xlabel(projection_dict[args.projection][0] + ' (kpc)', size=args.fontsize)
    ax.set_ylabel(projection_dict[args.projection][1] + ' (kpc)', size=args.fontsize)
    ax.tick_params(axis='both', labelsize=args.fontsize)
    if isline: fig.text(0.7, 0.9, line_dict[args.line], size=args.fontsize, va='top', ha='right', bbox=dict(boxstyle='round', ec='k', fc='salmon', alpha=0.7))

    cb = plt.colorbar(p)
    cb.set_label(get_property_name(args), size=args.fontsize)
    cb.ax.tick_params(labelsize=args.fontsize)

    if args.saveplot:
        log_text = '_log' if args.islog else ''
        line_text = '_' + args.line if isline else ''
        figname = os.path.splitext(args.measured_cube_filename)[0] + log_text + line_text + '_' + str(args.get_property) + '.pdf'
        fig.savefig(figname)
        myprint('Saved plot as ' + figname, args)

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
    elif which_property in measured_cube.derived_quantities: # i.e. this is a derived property, not based on one particular emission line
        property, property_u = measured_cube.get_derived_prop(which_property)

    if 'flux' in args.get_property: property = np.ma.masked_less(property, 0) # masks negaitve flux values
    if 'metallicity' in args.get_property: property = np.ma.masked_outside(property, 1e-2, 1e2) # masks too unphysical metallicity values
    if 'vel' in args.get_property: property = np.ma.masked_outside(property, -1e4, 1e4) # masks too unphysical velocity values

    if args.plot_property:
        if args.get_property[-2:] == '_u': ax = plot_property(property_u, args)
        else: ax = plot_property(property, args)

    return property, property_u

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    args = parse_args('8508', 'RD0042')
    args.diag = args.diag_arr[0]
    args.Om = args.Om_arr[0]
    if not args.keep: plt.close('all')

    cube_output_path = get_cube_output_path(args)
    instrument = telescope(args)  # declare the instrument

    if args.snr == 0: args.measured_cube_filename = cube_output_path + 'measured_cube' + args.mergeHII_text + '.fits'
    else: args.measured_cube_filename = cube_output_path + instrument.path + 'measured_cube' + '_z' + str(args.z) + args.mergeHII_text + '_ppb' + str(args.pix_per_beam) + '_exp' + str(args.exptime) + 's_snr' + str(args.snr) + '.fits'

    if os.path.exists(args.measured_cube_filename): measured_cube = read_measured_cube(args.measured_cube_filename, args)
    else: raise ValueError(args.measured_cube_filename + ' does not exist, try creating the file first using fit_mock_spectra.py')

    property, property_u = get_property(measured_cube, args)

    print('Complete in %s minutes' % ((time.time() - start_time) / 60))
