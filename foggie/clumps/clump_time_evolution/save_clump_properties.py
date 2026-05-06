import yt
from yt import derived_field
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import numpy as np 

from astropy.table import Table
from scipy import stats


import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import cmasher as cmr
import os
import argparse

from scipy import stats

from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *

import h5py

from foggie.clumps.clump_finder.utils_clump_finder import halo_id_to_name
from foggie.clumps.clump_finder.utils_clump_finder import read_virial_mass_file
from foggie.clumps.clump_finder import *
from foggie.clumps.clump_finder.clump_finder import TqdmProgressBar


import time

def append_to_hierarchy_file(hf, key, value,delete_previous=True):
    if key in hf.keys():
        if delete_previous: del hf[key]
        else: return
    hf.create_dataset(key, data=value)


# --- START VIDA'S ADDITION : imports needed for emission field registration ---
import unyt                        
from scipy import interpolate      
# --- END VIDA'S ADDITION ---
start_time =time.time()

# --- START VIDA'S ADDITION : functions for emission field registration ---

def scale_by_metallicity(values, assumed_Z, wanted_Z):
    # Scales emission by the actual gas metallicity since CLOUDY assumes solar
    wanted_ratio = (10.**(wanted_Z)) / (10.**(assumed_Z))
    return values * wanted_ratio


def register_emission_fields(ds, cloudy_path, unit_system='photons'):
    # Registers all CLOUDY-based emission fields with yt

    emission_units     = 's**-1 * cm**-3 * steradian**-1'
    emission_units_ALT = 'erg * s**-1 * cm**-3 * arcsec**-2'
    ytEmU    = unyt.second**-1 * unyt.cm**-3 * unyt.steradian**-1
    ytEmUALT = unyt.erg * unyt.second**-1 * unyt.cm**-3 * unyt.arcsec**-2
    units    = emission_units if unit_system == 'photons' else emission_units_ALT

    # Build shared interpolation grid - must match the CLOUDY run settings
    hden_n_bins, hden_min, hden_max = 17, -6, 2
    T_n_bins,    T_min,    T_max    = 51,  3, 8
    hden = np.linspace(hden_min, hden_max, hden_n_bins)
    T    = np.linspace(T_min,    T_max,    T_n_bins)
    hden_pts, T_pts = np.meshgrid(hden, T)
    pts = np.array((hden_pts.ravel(), T_pts.ravel())).T

    def make_interp(table_index):
        table = np.zeros((hden_n_bins, T_n_bins))
        for i in range(hden_n_bins):
            table[i, :] = [float(l.split()[table_index])
                           for l in open(cloudy_path % (i+1)) if l[0] != '#']
        return interpolate.LinearNDInterpolator(pts, table.T.ravel())

    # Build one interpolator per ion (two for OVI since it has two lines)
    # The numbers are the column indices in the CLOUDY output files where the relevant line emissivity can be found
    # Make sure to double-check these if you change the CLOUDY output format or the lines you're interested in!
    bl_HA    = make_interp(2)    # H-Alpha 6563 
    bl_LA    = make_interp(1)    # Ly-Alpha 1216 
    bl_CII   = make_interp(10)   # CII 1335 
    bl_CIII  = make_interp(9)    # CIII 1910 
    bl_CIV   = make_interp(3)    # CIV 1548 
    bl_OVI_1 = make_interp(5)    # OVI 1032 
    bl_OVI_2 = make_interp(6)    # OVI 1038 
    bl_SiII  = make_interp(12)   # SiII 1260 
    bl_SiIII = make_interp(13)   # SiIII 1207 
    bl_SiIV  = make_interp(15)   # SiIV 1394 
    bl_MgII  = make_interp(17)   # MgII 2796 

    def apply_units(emission_line):
        if unit_system == 'photons':
            return emission_line * ytEmU
        else:
            return (emission_line / 4.25e10) * ytEmUALT

    def _Emission_HAlpha(field, data):
        H_N  = np.log10(np.array(data["H_nuclei_density"]))
        Temp = np.log10(np.array(data["Temperature"]))
        dia1 = bl_HA(H_N, Temp)
        dia1[np.isnan(dia1)] = -200.
        emission_line = (10.**dia1) * ((10.**H_N)**2.0)
        if unit_system == 'photons':
            emission_line = emission_line / (4. * np.pi * 3.03e-12)
        else:
            emission_line = emission_line / (4. * np.pi)
        return apply_units(emission_line)

    def _Emission_LyAlpha(field, data):
        H_N  = np.log10(np.array(data["H_nuclei_density"]))
        Temp = np.log10(np.array(data["Temperature"]))
        dia1 = bl_LA(H_N, Temp)
        dia1[np.isnan(dia1)] = -200.
        emission_line = (10.**dia1) * ((10.**H_N)**2.0)
        if unit_system == 'photons':
            emission_line = emission_line / (4. * np.pi * 1.63e-11)
        else:
            emission_line = emission_line / (4. * np.pi)
        return apply_units(emission_line)

    def _Emission_CII(field, data):
        H_N  = np.log10(np.array(data["H_nuclei_density"]))
        Temp = np.log10(np.array(data["Temperature"]))
        dia1 = bl_CII(H_N, Temp)
        dia1[np.isnan(dia1)] = -200.
        emission_line = (10.**dia1) * ((10.**H_N)**2.0)
        emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
        if unit_system == 'photons':
            emission_line = emission_line / (4. * np.pi * 2.03e-11)
        else:
            emission_line = emission_line / (4. * np.pi)
        return apply_units(emission_line)

    def _Emission_CIII(field, data):
        H_N  = np.log10(np.array(data["H_nuclei_density"]))
        Temp = np.log10(np.array(data["Temperature"]))
        dia1 = bl_CIII(H_N, Temp)
        dia1[np.isnan(dia1)] = -200.
        emission_line = (10.**dia1) * ((10.**H_N)**2.0)
        emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
        if unit_system == 'photons':
            emission_line = emission_line / (4. * np.pi * 2.03e-11)
        else:
            emission_line = emission_line / (4. * np.pi)
        return apply_units(emission_line)

    def _Emission_CIV(field, data):
        H_N  = np.log10(np.array(data["H_nuclei_density"]))
        Temp = np.log10(np.array(data["Temperature"]))
        dia1 = bl_CIV(H_N, Temp)
        dia1[np.isnan(dia1)] = -200.
        emission_line = (10.**dia1) * ((10.**H_N)**2.0)
        emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
        if unit_system == 'photons':
            emission_line = emission_line / (4. * np.pi * 1.28e-11)
        else:
            emission_line = emission_line / (4. * np.pi)
        return apply_units(emission_line)

    def _Emission_OVI(field, data):
        H_N  = np.log10(np.array(data["H_nuclei_density"]))
        Temp = np.log10(np.array(data["Temperature"]))
        dia1 = bl_OVI_1(H_N, Temp)
        dia2 = bl_OVI_2(H_N, Temp)
        dia1[np.isnan(dia1)] = -200.
        dia2[np.isnan(dia2)] = -200.
        emission_line = ((10.**dia1) + (10.**dia2)) * ((10.**H_N)**2.0)
        emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
        if unit_system == 'photons':
            emission_line = emission_line / (4. * np.pi * 1.92e-11)
        else:
            emission_line = emission_line / (4. * np.pi)
        return apply_units(emission_line)

    def _Emission_SiII(field, data):
        H_N  = np.log10(np.array(data["H_nuclei_density"]))
        Temp = np.log10(np.array(data["Temperature"]))
        dia1 = bl_SiII(H_N, Temp)
        dia1[np.isnan(dia1)] = -200.
        emission_line = (10.**dia1) * ((10.**H_N)**2.0)
        emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
        if unit_system == 'photons':
            emission_line = emission_line / (4. * np.pi * 2.03e-11)
        else:
            emission_line = emission_line / (4. * np.pi)
        return apply_units(emission_line)

    def _Emission_SiIII(field, data):
        H_N  = np.log10(np.array(data["H_nuclei_density"]))
        Temp = np.log10(np.array(data["Temperature"]))
        dia1 = bl_SiIII(H_N, Temp)
        dia1[np.isnan(dia1)] = -200.
        emission_line = (10.**dia1) * ((10.**H_N)**2.0)
        emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
        if unit_system == 'photons':
            emission_line = emission_line / (4. * np.pi * 2.03e-11)
        else:
            emission_line = emission_line / (4. * np.pi)
        return apply_units(emission_line)

    def _Emission_SiIV(field, data):
        H_N  = np.log10(np.array(data["H_nuclei_density"]))
        Temp = np.log10(np.array(data["Temperature"]))
        dia1 = bl_SiIV(H_N, Temp)
        dia1[np.isnan(dia1)] = -200.
        emission_line = (10.**dia1) * ((10.**H_N)**2.0)
        emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
        if unit_system == 'photons':
            emission_line = emission_line / (4. * np.pi * 2.03e-11)
        else:
            emission_line = emission_line / (4. * np.pi)
        return apply_units(emission_line)

    def _Emission_MgII(field, data):
        H_N  = np.log10(np.array(data["H_nuclei_density"]))
        Temp = np.log10(np.array(data["Temperature"]))
        dia1 = bl_MgII(H_N, Temp)
        dia1[np.isnan(dia1)] = -200.
        emission_line = (10.**dia1) * ((10.**H_N)**2.0)
        emission_line = scale_by_metallicity(emission_line, 0.0, np.log10(np.array(data['metallicity'])))
        if unit_system == 'photons':
            emission_line = emission_line / (4. * np.pi * 2.03e-11)
        else:
            emission_line = emission_line / (4. * np.pi)
        return apply_units(emission_line)

    # Register all fields with yt
    fields_to_register = [
        ('Emission_HAlpha',     _Emission_HAlpha),
        ('Emission_LyAlpha',    _Emission_LyAlpha),
        ('Emission_CII_1335',   _Emission_CII),
        ('Emission_CIII_1910',  _Emission_CIII),
        ('Emission_CIV_1548',   _Emission_CIV),
        ('Emission_OVI',        _Emission_OVI),
        ('Emission_SiII_1260',  _Emission_SiII),
        ('Emission_SiIII_1207', _Emission_SiIII),
        ('Emission_SiIV_1394',  _Emission_SiIV),
        ('Emission_MgII_2796',  _Emission_MgII),
    ]

    for field_name, func in fields_to_register:
        ds.add_field(
            ('gas', field_name),
            units=units,
            function=func,
            take_log=True,
            force_override=True,
            sampling_type='cell',
        )
    print(f"Registered {len(fields_to_register)} emission fields with unit system: {unit_system}")

def compute_clump_sb_per_los(mask, x, y, z, volumes, emission_data):
    """
    Computes total surface brightness of a clump along x, y, z sightlines.
    
    For each sightline, we project onto the transverse plane where each pixel
    corresponds to one cell size. For each pixel (unique transverse position):
        SB_pixel = (sum of emissivities of cells along LOS) * (sum of path lengths along LOS)
    Total SB = sum of SB_pixel over all pixels in the 2D map.
    
    Parameters
    ----------
    mask         : boolean array, True for cells belonging to this clump
    x, y, z      : yt arrays, cell positions in kpc (simulation axes)
    volumes      : yt array, cell volumes in kpc^3
    emission_data: dict, ion short name -> yt emissivity array [photons/s/cm^3/sr]
    
    Returns
    -------
    sb_results : dict
        keys are ion short names, values are dicts with keys 'xlos', 'ylos', 'zlos'
        each containing the total SB [photons/s/cm^2/sr]
    """

    kpc_to_cm = 3.0857e21  # cm per kpc

    # Get positions and path lengths for clump cells only
    x_clump = x[mask].in_units('kpc').v
    y_clump = y[mask].in_units('kpc').v
    z_clump = z[mask].in_units('kpc').v

    # Path length through each cell is V^(1/3) - same in all directions for cubic AMR cells
    # What changes between sightlines is how many cells are stacked in each transverse pixel
    cell_size_cm = volumes[mask].in_units('kpc**3').v**(1./3.) * kpc_to_cm

    # Define the three sightlines: (LOS axis, transverse coord 1, transverse coord 2)
    sightlines = {
        'xlos': (x_clump, y_clump, z_clump),  # looking along x, projecting onto y-z plane
        'ylos': (y_clump, x_clump, z_clump),  # looking along y, projecting onto x-z plane
        'zlos': (z_clump, x_clump, y_clump),  # looking along z, projecting onto x-y plane
    }

    sb_results = {}

    for ion_name, emissivity_array in emission_data.items():
        epsilon_clump = emissivity_array[mask].v  # emissivity of clump cells [photons/s/cm^3/sr]
        sb_results[ion_name] = {}

        for los_name, (los_coord, trans1, trans2) in sightlines.items():

            # Find unique transverse positions - each unique (trans1, trans2) pair is one pixel
            unique_pixels = np.unique(np.column_stack([trans1, trans2]), axis=0)

            total_sb = 0.0
            for t1_pix, t2_pix in unique_pixels:
                # Select all cells in this pixel (same transverse position, different LOS position)
                in_pixel = (trans1 == t1_pix) & (trans2 == t2_pix)

                # SB_pixel = (sum of emissivities along LOS) * (sum of path lengths along LOS)
                sum_epsilon = np.sum(epsilon_clump[in_pixel])   # [photons/s/cm^3/sr]
                sum_dl      = np.sum(cell_size_cm[in_pixel])    # [cm]
                total_sb   += sum_epsilon * sum_dl               # [photons/s/cm^2/sr]

            sb_results[ion_name][los_name] = total_sb  # total SB of clump from this sightline

    return sb_results

# --- END VIDA'S ADDITION ---

def parse_args():
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Makes basic kinematic plots for the disk (and CGM)')

    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='008508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--snapshot', metavar='snapshot', type=str, action='store', \
                        help='Which snapshot? Default is RD0042')
    parser.set_defaults(snapshot='RD0042')
    
    parser.add_argument('--snapshot_array_index', metavar='snapshot_array_index', type=str, action='store', \
                        help='Which snapshot number as fed in by the PBS Job array? Default is None, will override snapshot.')
    parser.set_defaults(snapshot_array_index=None)

    parser.add_argument('--clump_file', metavar='clump_file', type=str, action='store', \
                        help='Where is the clump hiearchy file')
    parser.set_defaults(clump_file=None)

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Where is the clump file to define the disk')
    parser.set_defaults(system='cameron_local')

    parser.add_argument('--pwd', metavar='pwd', type=bool, action='store', \
                        help='Use working directory in get_run_loc_etc. Default is False.')
    parser.set_defaults(pwd=False)

    parser.add_argument('--forcepath', metavar='forcepath', type=bool, action='store', \
                        help='Use forcepath in get_run_loc_etc. Default is False.')
    parser.set_defaults(forcepath=False)

    parser.add_argument('--is_rd', metavar='is_rd', type=bool, action='store', \
                        help='Are you analyzing RD snapshots instead of DD? Default is False.')
    parser.set_defaults(is_rd=False)

    parser.add_argument('--output_dir', metavar='output_dir', type=str, action='store', \
                        help='Where to save the histograms. Default is ./')
    parser.set_defaults(output_dir='./')

    parser.add_argument('--data_dir', metavar='data_dir', type=str, action='store', \
                        help='Override data directory location in get_run_loc_etc. Default is None.')
    parser.set_defaults(data_dir=None)

    parser.add_argument('--do_tracer_fluids', metavar='do_tracer_fluids', type=bool, action='store', \
                        help='Calculate tracer fluid stats? Default is False.')
    parser.set_defaults(do_tracer_fluids=False)

    parser.add_argument('--modify_existing_clump_hierarchy', metavar='modify_existing_clump_hierarchy', type=bool, action=argparse.BooleanOptionalAction, \
                        help='Add fields to the current clump tree. Will write separate clump stats file if False. Default is True.')
    parser.set_defaults(modify_existing_clump_hierarchy=False)

    parser.add_argument('--write_separate_stats_file', metavar='write_separate_stats_file', type=bool, action=argparse.BooleanOptionalAction, \
                        help='Write a separate stats file (outside of the clump hierarchy). Default is False. If modify_existing_clump_hierarchy is False, this will be set to True regardless.')
    parser.set_defaults(write_separate_stats_file=False)

    # --- VIDA'S ADDITION: flag to compute and save emission properties ---
    parser.add_argument('--add_emission', action='store_true',
                        help='Compute and save emissivity for each ion for each leaf clump. Default is False.')
    parser.add_argument('--unit_system', metavar='unit_system', type=str, action='store',
                        help='Which unit system for emission? Default is photons.Options are:\n' + \
                            'default - photons (photons * s**-1 * cm**-3 * sr**-1)\n' + \
                            'erg - erg (ergs * s**-1 * cm**-3 * arcsec**-2)')
    parser.set_defaults(unit_system='photons')
    # --- END VIDA'S ADDITION ---

    args = parser.parse_args()
    return args




args = parse_args()

if args.snapshot_array_index is not None:
    if args.is_rd:
        args.snapshot = "RD"+args.snapshot_array_index.zfill(4)
    else:
        args.snapshot = "DD"+args.snapshot_array_index.zfill(4)

data_dir, output_path, run_loc, code_dir,trackname,halo_name,spectra_dir,infofile = get_run_loc_etc(args)

if args.data_dir is not None:
    data_dir = args.data_dir

if args.clump_file is None:
    if args.system=='cameron_local':
        args.clump_file = '/Users/ctrapp/Documents/foggie_analysis/clump_project/clump_catalog/'
    elif args.system=='cameron_pleiades':
        args.clump_file = '/nobackup/cwtrapp/clump_catalogs/halo_'+args.halo+'/'

halo_id = args.halo #008508
snapshot = args.snapshot #RD0042
run = args.run #nref11c_nref9f

gal_name = halo_id_to_name(halo_id)
gal_name+="_"+snapshot+"_"+run

snap_name = data_dir + "halo_"+halo_id+"/"+run+"/"+snapshot+"/"+snapshot

trackname = code_dir+"/halo_tracks/"+halo_id+"/nref11n_selfshield_15/halo_track_200kpc_nref9"

halo_c_v_name = code_dir+"/halo_infos/"+halo_id+"/nref11c_nref9f/halo_c_v"

#particle_type_for_angmom = 'young_stars' ##Currently the default
particle_type_for_angmom = 'gas' #Should be defined by gas with Temps below 1e4 K

catalog_dir = code_dir + '/halo_infos/' + halo_id + '/'+run+'/'
#smooth_AM_name = catalog_dir + 'AM_direction_smoothed'
smooth_AM_name = None

ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, halo_c_v_name=halo_c_v_name, do_filter_particles=True,disk_relative=True,particle_type_for_angmom=particle_type_for_angmom,smooth_AM_name = smooth_AM_name)

GalName="UnknownHalo"
if halo_id == "008508":
    GalName="Tempest"
elif halo_id == "005036":
    GalName="Maelstrom"
elif halo_id == "005016":
    GalName="Squall"
elif halo_id == "004123":
    GalName="Blizzard"
elif halo_id == "002392":
    GalName="Hurricane"
elif halo_id == "002878":
    GalName="Cyclone"

# --- START VIDA'S ADDITION: register emission fields and load emission arrays ---
if args.add_emission:
    # Set the cloudy path
    cloudy_path = code_dir + "cgm_emission/cloudy_extended_z0_selfshield/TEST_z0_HM12_sh_run%i.dat"
    
    # Register the CLOUDY emission fields with yt
    register_emission_fields(ds, cloudy_path, unit_system=args.unit_system)

    # Define the unit string we will use when loading emission arrays
    emission_units = 's**-1 * cm**-3 * steradian**-1' if args.unit_system == 'photons' \
                     else 'erg * s**-1 * cm**-3 * arcsec**-2'

    # Load all emission fields from the refine box 'once' before the clump loop to avoid redundant calculations inside the loop. 
    # This will be a dictionary of yt arrays.
    print("Loading emission fields from refine box...")
    emission_data = {
        'halpha'  : refine_box['gas', 'Emission_HAlpha'].in_units(emission_units),     # H-Alpha 6563 
        'lyalpha' : refine_box['gas', 'Emission_LyAlpha'].in_units(emission_units),    # Ly-Alpha 1216 
        'cii'     : refine_box['gas', 'Emission_CII_1335'].in_units(emission_units),   # CII 1335 
        'ciii'    : refine_box['gas', 'Emission_CIII_1910'].in_units(emission_units),  # CIII 1910 
        'civ'     : refine_box['gas', 'Emission_CIV_1548'].in_units(emission_units),   # CIV 1548 
        'ovi'     : refine_box['gas', 'Emission_OVI'].in_units(emission_units),        # OVI 1032+1038 
        'siii'    : refine_box['gas', 'Emission_SiII_1260'].in_units(emission_units),  # SiII 1260 
        'siiii'   : refine_box['gas', 'Emission_SiIII_1207'].in_units(emission_units), # SiIII 1207 
        'siiv'    : refine_box['gas', 'Emission_SiIV_1394'].in_units(emission_units),  # SiIV 1394 
        'mgii'    : refine_box['gas', 'Emission_MgII_2796'].in_units(emission_units),  # MgII 2796 
    }
    print("Emission fields loaded successfully.")

    # Initialize empty lists to collect emissivity per clump
    leaf_halpha_emissivity  = []
    leaf_lyalpha_emissivity = []
    leaf_cii_emissivity     = []
    leaf_ciii_emissivity    = []
    leaf_civ_emissivity     = []
    leaf_ovi_emissivity     = []
    leaf_siii_emissivity    = []
    leaf_siiii_emissivity   = []
    leaf_siiv_emissivity    = []
    leaf_mgii_emissivity    = []
    # leaf SB empty lists, 3 sightlines per ion 
    # xlos = looking along x axis, projecting onto y-z plane
    # ylos = looking along y axis, projecting onto x-z plane
    # zlos = looking along z axis, projecting onto x-y plane
    leaf_halpha_sb_xlos  = [];  leaf_halpha_sb_ylos  = [];  leaf_halpha_sb_zlos  = []
    leaf_lyalpha_sb_xlos = [];  leaf_lyalpha_sb_ylos = [];  leaf_lyalpha_sb_zlos = []
    leaf_cii_sb_xlos     = [];  leaf_cii_sb_ylos     = [];  leaf_cii_sb_zlos     = []
    leaf_ciii_sb_xlos    = [];  leaf_ciii_sb_ylos    = [];  leaf_ciii_sb_zlos    = []
    leaf_civ_sb_xlos     = [];  leaf_civ_sb_ylos     = [];  leaf_civ_sb_zlos     = []
    leaf_ovi_sb_xlos     = [];  leaf_ovi_sb_ylos     = [];  leaf_ovi_sb_zlos     = []
    leaf_siii_sb_xlos    = [];  leaf_siii_sb_ylos    = [];  leaf_siii_sb_zlos    = []
    leaf_siiii_sb_xlos   = [];  leaf_siiii_sb_ylos   = [];  leaf_siiii_sb_zlos   = []
    leaf_siiv_sb_xlos    = [];  leaf_siiv_sb_ylos    = [];  leaf_siiv_sb_zlos    = []
    leaf_mgii_sb_xlos    = [];  leaf_mgii_sb_ylos    = [];  leaf_mgii_sb_zlos    = []
    # shell emissivity empty lists 
    shell_halpha_emissivity  = []
    shell_lyalpha_emissivity = []
    shell_cii_emissivity     = []
    shell_ciii_emissivity    = []
    shell_civ_emissivity     = []
    shell_ovi_emissivity     = []
    shell_siii_emissivity    = []
    shell_siiii_emissivity   = []
    shell_siiv_emissivity    = []
    shell_mgii_emissivity    = []
    
# --- END VIDA'S ADDITION ---


#load the leaf clumps
leaf_masses = []
leaf_vx = []
leaf_vy = []
leaf_vz = []
#leaf_vx_disk = []
#leaf_vy_disk = []
#leaf_vz_disk = []
leaf_hi_num_dense = []
leaf_mgii_num_dense = []
leaf_oi_num_dense = []
leaf_oii_num_dense = []
leaf_oiii_num_dense = []
leaf_oiv_num_dense = []
leaf_ov_num_dense = []
leaf_ovi_num_dense = []
leaf_volumes = []
leaf_x = []
leaf_y = []
leaf_z = []
#leaf_x_disk = []
#leaf_y_disk = []
#leaf_z_disk = []
leaf_metallicity = []
leaf_pressure = []
leaf_temperature = []

leaf_tf1_mass = []
leaf_tf2_mass = []
leaf_tf3_mass = []
leaf_tf4_mass = []
leaf_tf5_mass = []
leaf_tf6_mass = []
leaf_tf7_mass = []
leaf_tf8_mass = []

shell_masses = []
shell_volumes = []
shell_vx = []
shell_vy = []
shell_vz = []
shell_hi_num_dense = []
shell_mgii_num_dense = []
shell_oi_num_dense = []
shell_oii_num_dense = []
shell_oiii_num_dense = []
shell_oiv_num_dense = []
shell_ov_num_dense = []
shell_ovi_num_dense = []
shell_metallicity = []
shell_pressure = []
shell_temperature = []
shell_cooling_time = []
t_shear = []



hiearchy_file = args.clump_file#args.clump_dir + GalName+"_"+args.snapshot+"_"+args.run+"_ClumpTree.h5"
if args.modify_existing_clump_hierarchy:
    hf = h5py.File(hiearchy_file,'r+')
else:
    hf = h5py.File(hiearchy_file,'r')

leaf_clump_ids = hf['leaf_clump_ids'][...]

skip_adding_cell_ids = False
print("Adding cell ids...")
from foggie.clumps.clump_finder.utils_clump_finder import add_cell_id_field
add_cell_id_field(ds)

import trident
trident.add_ion_fields(ds, ions=['O II','O III','O IV','O V','O VI','Mg II'])



all_leaf_cell_ids = np.array([])
shell_ids_appended_to_all = True
for leaf_id in leaf_clump_ids:
    all_leaf_cell_ids = np.append(all_leaf_cell_ids,hf[str(leaf_id)]['cell_ids'][...])
    try:
        all_leaf_cell_ids = np.append(all_leaf_cell_ids,hf[str(leaf_id)]['shell_cell_ids'][...])
    except:
        shell_ids_appended_to_all=False

cell_ids = refine_box['index','cell_id_2']



initial_mask = np.isin(cell_ids,all_leaf_cell_ids)

#flatten a list of arrays
gas_masses = refine_box['gas','mass'].in_units('Msun')[initial_mask]
vx = refine_box['gas','velocity_x'].in_units('km/s')[initial_mask]
vy = refine_box['gas','velocity_y'].in_units('km/s')[initial_mask]
vz = refine_box['gas','velocity_z'].in_units('km/s')[initial_mask]
hi_num_dense = refine_box['gas','H_p0_number_density'].in_units('cm**-3')[initial_mask]
mgii_num_dense = refine_box['gas','Mg_p1_number_density'].in_units('cm**-3')[initial_mask]
oii_num_dense = refine_box['gas','O_p1_number_density'].in_units('cm**-3')[initial_mask]
oiii_num_dense = refine_box['gas','O_p2_number_density'].in_units('cm**-3')[initial_mask]
oiv_num_dense = refine_box['gas','O_p3_number_density'].in_units('cm**-3')[initial_mask]
ov_num_dense = refine_box['gas','O_p4_number_density'].in_units('cm**-3')[initial_mask]
ovi_num_dense = refine_box['gas','O_p5_number_density'].in_units('cm**-3')[initial_mask]
volumes = refine_box['gas','cell_volume'].in_units('kpc**3')[initial_mask]


x = refine_box['gas','x'].in_units('kpc')[initial_mask]
y = refine_box['gas','y'].in_units('kpc')[initial_mask]
z = refine_box['gas','z'].in_units('kpc')[initial_mask]

metallicity = refine_box['gas','metallicity'][initial_mask]
pressure = refine_box['gas','pressure'][initial_mask]
temperature = refine_box['gas','temperature'][initial_mask]

cooling_time = refine_box['gas','cooling_time'][initial_mask]

cell_ids = cell_ids[initial_mask]

code_density = ds.units.code_density
if args.do_tracer_fluids:
    #try:
        tf1 = refine_box['enzo','TracerFluid01'][initial_mask] * code_density
        tf2 = refine_box['enzo','TracerFluid02'][initial_mask] * code_density
        tf3 = refine_box['enzo','TracerFluid03'][initial_mask] * code_density
        tf4 = refine_box['enzo','TracerFluid04'][initial_mask] * code_density
        tf5 = refine_box['enzo','TracerFluid05'][initial_mask] * code_density
        tf6 = refine_box['enzo','TracerFluid06'][initial_mask] * code_density
        tf7 = refine_box['enzo','TracerFluid07'][initial_mask] * code_density
        tf8 = refine_box['enzo','TracerFluid08'][initial_mask] * code_density
    #except:
    #    args.do_tracer_fluids=False


if args.add_emission:
    emission_data['halpha']=emission_data['halpha'][initial_mask]
    emission_data['lyalpha']=emission_data['lyalpha'][initial_mask]
    emission_data['cii']=emission_data['cii'][initial_mask]
    emission_data['ciii']=emission_data['ciii'][initial_mask]
    emission_data['civ']=emission_data['civ'][initial_mask]
    emission_data['ovi']=emission_data['ovi'][initial_mask]
    emission_data['siii']=emission_data['siii'][initial_mask]
    emission_data['siiii']=emission_data['siiii'][initial_mask]
    emission_data['siiv']=emission_data['siiv'][initial_mask]
    emission_data['mgii']=emission_data['mgii'][initial_mask]

pbar = TqdmProgressBar("Calculating Leaf stats...",len(leaf_clump_ids),position=0)
itr=0



for leaf_id in leaf_clump_ids:
    leaf_cell_ids = hf[str(leaf_id)]['cell_ids'][...]
    if itr==0: print(hf[str(leaf_id)].keys())
    try:
        shell_cell_ids = hf[str(leaf_id)]['shell_cell_ids'][...]
    except:
        shell_cell_ids = None

    mask = np.isin(cell_ids, leaf_cell_ids)
    leaf_gas_mass = gas_masses[mask].in_units('Msun')
    norm = np.sum(leaf_gas_mass)
    leaf_masses.append(norm.in_units('Msun').v)
    
    leaf_gas_volume = volumes[mask].in_units('kpc**3')
    vol_norm = np.sum(leaf_gas_volume)
    leaf_volumes.append(vol_norm.in_units('kpc**3').v)

    leaf_vx.append( (np.sum( np.multiply(vx[mask],leaf_gas_mass)) / norm ).in_units('km/s').v)
    leaf_vy.append( (np.sum( np.multiply(vy[mask],leaf_gas_mass)) / norm ).in_units('km/s').v)
    leaf_vz.append( (np.sum( np.multiply(vz[mask],leaf_gas_mass)) / norm ).in_units('km/s').v)

    leaf_x.append(  (np.sum( np.multiply(x[mask], leaf_gas_mass)) / norm ).in_units('kpc').v)
    leaf_y.append(  (np.sum( np.multiply(y[mask], leaf_gas_mass)) / norm ).in_units('kpc').v)
    leaf_z.append(  (np.sum( np.multiply(z[mask], leaf_gas_mass)) / norm ).in_units('kpc').v)


    leaf_hi_num_dense.append(  (np.sum( np.multiply(hi_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)
    leaf_mgii_num_dense.append(  (np.sum( np.multiply(mgii_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)
    leaf_oii_num_dense.append(  (np.sum( np.multiply(oii_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)
    leaf_oiii_num_dense.append(  (np.sum( np.multiply(oiii_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)
    leaf_oiv_num_dense.append(  (np.sum( np.multiply(oiv_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)
    leaf_ov_num_dense.append(  (np.sum( np.multiply(ov_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)
    leaf_ovi_num_dense.append(  (np.sum( np.multiply(ovi_num_dense[mask], leaf_gas_volume)) / vol_norm ).in_units('cm**-3').v)

    leaf_metallicity.append(  (np.sum( np.multiply(metallicity[mask], leaf_gas_mass)) / norm ))
    leaf_pressure.append(  (np.sum( np.multiply(pressure[mask], leaf_gas_mass)) / norm ).in_units('Ba').v)
    leaf_temperature.append(  (np.sum( np.multiply(temperature[mask], leaf_gas_mass)) / norm ).in_units('K').v)


    # --- VIDA'S ADDITION: compute emissivity for each ion for this clump ---
    if args.add_emission:
        leaf_halpha_emissivity.append(  np.sum(emission_data['halpha'][mask].v)  )  # H-Alpha 6563
        leaf_lyalpha_emissivity.append( np.sum(emission_data['lyalpha'][mask].v) )  # Ly-Alpha 1216
        leaf_cii_emissivity.append(     np.sum(emission_data['cii'][mask].v)     )  # CII 1335 
        leaf_ciii_emissivity.append(    np.sum(emission_data['ciii'][mask].v)    )  # CIII 1910 
        leaf_civ_emissivity.append(     np.sum(emission_data['civ'][mask].v)     )  # CIV 1548 
        leaf_ovi_emissivity.append(     np.sum(emission_data['ovi'][mask].v)     )  # OVI 1032+1038 
        leaf_siii_emissivity.append(    np.sum(emission_data['siii'][mask].v)    )  # SiII 1260 
        leaf_siiii_emissivity.append(   np.sum(emission_data['siiii'][mask].v)   )  # SiIII 1207 
        leaf_siiv_emissivity.append(    np.sum(emission_data['siiv'][mask].v)    )  # SiIV 1394 
        leaf_mgii_emissivity.append(    np.sum(emission_data['mgii'][mask].v)    )  # MgII 2796 

        # Append SB results for each ion and each sightline
        sb_results = compute_clump_sb_per_los(mask, x, y, z, volumes, emission_data)
        # sb_results is a dict: ion_name then {'xlos': val, 'ylos': val, 'zlos': val}
        leaf_halpha_sb_xlos.append(  sb_results['halpha']['xlos']  )
        leaf_halpha_sb_ylos.append(  sb_results['halpha']['ylos']  )
        leaf_halpha_sb_zlos.append(  sb_results['halpha']['zlos']  )

        leaf_lyalpha_sb_xlos.append( sb_results['lyalpha']['xlos'] )
        leaf_lyalpha_sb_ylos.append( sb_results['lyalpha']['ylos'] )
        leaf_lyalpha_sb_zlos.append( sb_results['lyalpha']['zlos'] )

        leaf_cii_sb_xlos.append(     sb_results['cii']['xlos']     )
        leaf_cii_sb_ylos.append(     sb_results['cii']['ylos']     )
        leaf_cii_sb_zlos.append(     sb_results['cii']['zlos']     )

        leaf_ciii_sb_xlos.append(    sb_results['ciii']['xlos']    )
        leaf_ciii_sb_ylos.append(    sb_results['ciii']['ylos']    )
        leaf_ciii_sb_zlos.append(    sb_results['ciii']['zlos']    )

        leaf_civ_sb_xlos.append(     sb_results['civ']['xlos']     )
        leaf_civ_sb_ylos.append(     sb_results['civ']['ylos']     )
        leaf_civ_sb_zlos.append(     sb_results['civ']['zlos']     )

        leaf_ovi_sb_xlos.append(     sb_results['ovi']['xlos']     )
        leaf_ovi_sb_ylos.append(     sb_results['ovi']['ylos']     )
        leaf_ovi_sb_zlos.append(     sb_results['ovi']['zlos']     )

        leaf_siii_sb_xlos.append(    sb_results['siii']['xlos']    )
        leaf_siii_sb_ylos.append(    sb_results['siii']['ylos']    )
        leaf_siii_sb_zlos.append(    sb_results['siii']['zlos']    )

        leaf_siiii_sb_xlos.append(   sb_results['siiii']['xlos']   )
        leaf_siiii_sb_ylos.append(   sb_results['siiii']['ylos']   )
        leaf_siiii_sb_zlos.append(   sb_results['siiii']['zlos']   )

        leaf_siiv_sb_xlos.append(    sb_results['siiv']['xlos']    )
        leaf_siiv_sb_ylos.append(    sb_results['siiv']['ylos']    )
        leaf_siiv_sb_zlos.append(    sb_results['siiv']['zlos']    )

        leaf_mgii_sb_xlos.append(    sb_results['mgii']['xlos']    )
        leaf_mgii_sb_ylos.append(    sb_results['mgii']['ylos']    )
        leaf_mgii_sb_zlos.append(    sb_results['mgii']['zlos']    )
    # --- END VIDA'S ADDITION ---



    if args.do_tracer_fluids:
        leaf_tf1_mass.append( np.sum( np.multiply(tf1[mask] , volumes[mask] )) ) #Give tracer fluid mass in leaf clump
        leaf_tf2_mass.append( np.sum( np.multiply(tf2[mask] , volumes[mask] )) )
        leaf_tf3_mass.append( np.sum( np.multiply(tf3[mask] , volumes[mask] )) )
        leaf_tf4_mass.append( np.sum( np.multiply(tf4[mask] , volumes[mask] )) )
        leaf_tf5_mass.append( np.sum( np.multiply(tf5[mask] , volumes[mask] )) )
        leaf_tf6_mass.append( np.sum( np.multiply(tf6[mask] , volumes[mask] )) )
        leaf_tf7_mass.append( np.sum( np.multiply(tf7[mask] , volumes[mask] )) )
        leaf_tf8_mass.append( np.sum( np.multiply(tf8[mask] , volumes[mask] )) )

    if shell_cell_ids is not None:
        shell_mask = np.isin(cell_ids, shell_cell_ids)
        shell_gas_mass = gas_masses[shell_mask].in_units('Msun')
        shell_norm = np.sum(shell_gas_mass)

        shell_gas_volume = volumes[shell_mask].in_units('kpc**3')
        shell_vol_norm = np.sum(shell_gas_volume)

        shell_masses.append(shell_norm.in_units('Msun').v)
        shell_volumes.append(shell_vol_norm.in_units('kpc**3').v)

        shell_vx.append( (np.sum( np.multiply(vx[shell_mask],shell_gas_mass)) / shell_norm ).in_units('km/s').v)
        shell_vy.append( (np.sum( np.multiply(vy[shell_mask],shell_gas_mass)) / shell_norm ).in_units('km/s').v)
        shell_vz.append( (np.sum( np.multiply(vz[shell_mask],shell_gas_mass)) / shell_norm ).in_units('km/s').v)


        shell_hi_num_dense.append(  (np.sum( np.multiply(hi_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)
        shell_mgii_num_dense.append(  (np.sum( np.multiply(mgii_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)
        shell_oii_num_dense.append(  (np.sum( np.multiply(oii_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)
        shell_oiii_num_dense.append(  (np.sum( np.multiply(oiii_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)
        shell_oiv_num_dense.append(  (np.sum( np.multiply(oiv_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)
        shell_ov_num_dense.append(  (np.sum( np.multiply(ov_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)
        shell_ovi_num_dense.append(  (np.sum( np.multiply(ovi_num_dense[shell_mask], shell_gas_volume)) / shell_vol_norm ).in_units('cm**-3').v)

        shell_metallicity.append(  (np.sum( np.multiply(metallicity[shell_mask], shell_gas_mass)) / shell_norm ))
        shell_pressure.append(  (np.sum( np.multiply(pressure[shell_mask], shell_gas_mass)) / shell_norm ).in_units('Ba').v)
        shell_temperature.append(  (np.sum( np.multiply(temperature[shell_mask], shell_gas_mass)) / shell_norm ).in_units('K').v)

        # --- VIDA'S ADDITION: compute shell emissivity for each ion ---
        if args.add_emission:
            shell_halpha_emissivity.append(  np.sum(emission_data['halpha'][shell_mask].v)  )  # H-Alpha 6563 
            shell_lyalpha_emissivity.append( np.sum(emission_data['lyalpha'][shell_mask].v) )  # Ly-Alpha 1216 
            shell_cii_emissivity.append(     np.sum(emission_data['cii'][shell_mask].v)     )  # CII 1335 
            shell_ciii_emissivity.append(    np.sum(emission_data['ciii'][shell_mask].v)    )  # CIII 1910 
            shell_civ_emissivity.append(     np.sum(emission_data['civ'][shell_mask].v)     )  # CIV 1548 
            shell_ovi_emissivity.append(     np.sum(emission_data['ovi'][shell_mask].v)     )  # OVI 1032+1038 
            shell_siii_emissivity.append(    np.sum(emission_data['siii'][shell_mask].v)    )  # SiII 1260 
            shell_siiii_emissivity.append(   np.sum(emission_data['siiii'][shell_mask].v)   )  # SiIII 1207 
            shell_siiv_emissivity.append(    np.sum(emission_data['siiv'][shell_mask].v)    )  # SiIV 1394 
            shell_mgii_emissivity.append(    np.sum(emission_data['mgii'][shell_mask].v)    )  # MgII 2796 
        # --- END VIDA'S ADDITION ---

        shell_cooling_time.append( np.min(cooling_time[shell_mask]).in_units('s').v ) #units of seconds 

        dvx2 = np.power( (leaf_vx[-1] - shell_vx[-1]) , 2.) #Already in km/s
        dvy2 = np.power( (leaf_vy[-1] - shell_vy[-1]) , 2.)
        dvz2 = np.power( (leaf_vz[-1] - shell_vz[-1]) , 2.)

        t_shear.append( np.power(vol_norm.in_units('km**3').v, 1./3.) / np.sqrt(dvx2 + dvy2 + dvz2) ) #units of seconds

    pbar.update(itr)
    itr+=1
    skip_adding_cell_ids = True



    if args.modify_existing_clump_hierarchy:
        delete_previous = True
        append_to_hierarchy_file(hf[str(leaf_id)], 'leav_vx', np.array(leaf_vx[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_vy', np.array(leaf_vy[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_vz', np.array(leaf_vz[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_x', np.array(leaf_x[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_y', np.array(leaf_y[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_z', np.array(leaf_z[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_hi_num_dense', np.array(leaf_hi_num_dense[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_mgii_num_dense', np.array(leaf_mgii_num_dense[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_oii_num_dense', np.array(leaf_oii_num_dense[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_oiii_num_dense', np.array(leaf_oiii_num_dense[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_oiv_num_dense', np.array(leaf_oiv_num_dense[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_ov_num_dense', np.array(leaf_ov_num_dense[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_ovi_num_dense', np.array(leaf_ovi_num_dense[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_volume', np.array(leaf_volumes[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_metallicity', np.array(leaf_metallicity[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_pressure', np.array(leaf_pressure[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_temperature', np.array(leaf_temperature[-1]),delete_previous)
        append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_mass', np.array(leaf_masses[-1]),delete_previous)
        if args.do_tracer_fluids:
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_tf1_mass', np.array(leaf_tf1_mass[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_tf2_mass', np.array(leaf_tf2_mass[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_tf3_mass', np.array(leaf_tf3_mass[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_tf4_mass', np.array(leaf_tf4_mass[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_tf5_mass', np.array(leaf_tf5_mass[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_tf6_mass', np.array(leaf_tf6_mass[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_tf7_mass', np.array(leaf_tf7_mass[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_tf8_mass', np.array(leaf_tf8_mass[-1]),delete_previous)
        if shell_cell_ids is not None:
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_mass', np.array(shell_masses[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_volume', np.array(shell_volumes[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_vx', np.array(shell_vx[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_vy', np.array(shell_vy[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_vz', np.array(shell_vz[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_hi_num_dense', np.array(shell_hi_num_dense[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_mgii_num_dense', np.array(shell_mgii_num_dense[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_oii_num_dense', np.array(shell_oii_num_dense[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_oiii_num_dense', np.array(shell_oiii_num_dense[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_oiv_num_dense', np.array(shell_oiv_num_dense[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_ov_num_dense', np.array(shell_ov_num_dense[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_ovi_num_dense', np.array(shell_ovi_num_dense[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_metallicity', np.array(shell_metallicity[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_pressure', np.array(shell_pressure[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_temperature', np.array(shell_temperature[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'shell_cooling_time', np.array(shell_cooling_time[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 't_shear', np.array(t_shear[-1]),delete_previous)
            if args.add_emission:
                append_to_hierarchy_file(hf[str(leaf_id)], 'shell_halpha_emissivity', np.array(shell_halpha_emissivity[-1]),delete_previous)
                append_to_hierarchy_file(hf[str(leaf_id)], 'shell_lyalpha_emissivity', np.array(shell_lyalpha_emissivity[-1]),delete_previous)
                append_to_hierarchy_file(hf[str(leaf_id)], 'shell_cii_emissivity', np.array(shell_cii_emissivity[-1]),delete_previous)
                append_to_hierarchy_file(hf[str(leaf_id)], 'shell_ciii_emissivity', np.array(shell_ciii_emissivity[-1]),delete_previous)
                append_to_hierarchy_file(hf[str(leaf_id)], 'shell_civ_emissivity', np.array(shell_civ_emissivity[-1]),delete_previous)
                append_to_hierarchy_file(hf[str(leaf_id)], 'shell_ovi_emissivity', np.array(shell_ovi_emissivity[-1]),delete_previous)
                append_to_hierarchy_file(hf[str(leaf_id)], 'shell_siii_emissivity', np.array(shell_siii_emissivity[-1]),delete_previous)
                append_to_hierarchy_file(hf[str(leaf_id)], 'shell_siiii_emissivity', np.array(shell_siiii_emissivity[-1]),delete_previous)
                append_to_hierarchy_file(hf[str(leaf_id)], 'shell_siiv_emissivity', np.array(shell_siiv_emissivity[-1]),delete_previous)
                append_to_hierarchy_file(hf[str(leaf_id)], 'shell_mgii_emissivity', np.array(shell_mgii_emissivity[-1]),delete_previous)
        if args.add_emission:
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_halpha_emissivity', np.array(leaf_halpha_emissivity[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_lyalpha_emissivity', np.array(leaf_lyalpha_emissivity[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_cii_emissivity', np.array(leaf_cii_emissivity[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_ciii_emissivity', np.array(leaf_ciii_emissivity[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_civ_emissivity', np.array(leaf_civ_emissivity[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_ovi_emissivity', np.array(leaf_ovi_emissivity[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_siii_emissivity', np.array(leaf_siii_emissivity[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_siiii_emissivity', np.array(leaf_siiii_emissivity[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_siiv_emissivity', np.array(leaf_siiv_emissivity[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_mgii_emissivity', np.array(leaf_mgii_emissivity[-1]),delete_previous)

            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_halpha_sb_xlos', np.array(leaf_halpha_sb_xlos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_halpha_sb_ylos', np.array(leaf_halpha_sb_ylos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_halpha_sb_zlos', np.array(leaf_halpha_sb_zlos[-1]),delete_previous)

            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_lyalpha_sb_xlos', np.array(leaf_lyalpha_sb_xlos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_lyalpha_sb_ylos', np.array(leaf_lyalpha_sb_ylos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_lyalpha_sb_zlos', np.array(leaf_lyalpha_sb_zlos[-1]),delete_previous)

            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_cii_sb_xlos', np.array(leaf_cii_sb_xlos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_cii_sb_ylos', np.array(leaf_cii_sb_ylos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_cii_sb_zlos', np.array(leaf_cii_sb_zlos[-1]),delete_previous)

            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_ciii_sb_xlos', np.array(leaf_ciii_sb_xlos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_ciii_sb_ylos', np.array(leaf_ciii_sb_ylos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_ciii_sb_zlos', np.array(leaf_ciii_sb_zlos[-1]),delete_previous)

            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_civ_sb_xlos', np.array(leaf_civ_sb_xlos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_civ_sb_ylos', np.array(leaf_civ_sb_ylos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_civ_sb_zlos', np.array(leaf_civ_sb_zlos[-1]),delete_previous)

            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_ovi_sb_xlos', np.array(leaf_ovi_sb_xlos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_ovi_sb_ylos', np.array(leaf_ovi_sb_ylos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_ovi_sb_zlos', np.array(leaf_ovi_sb_zlos[-1]),delete_previous)

            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_siii_sb_xlos', np.array(leaf_siii_sb_xlos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_siii_sb_ylos', np.array(leaf_siii_sb_ylos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_siii_sb_zlos', np.array(leaf_siii_sb_zlos[-1]),delete_previous)

            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_siiii_sb_xlos', np.array(leaf_siiii_sb_xlos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_siiii_sb_ylos', np.array(leaf_siiii_sb_ylos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_siiii_sb_zlos', np.array(leaf_siiii_sb_zlos[-1]),delete_previous)

            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_siiv_sb_xlos', np.array(leaf_siiv_sb_xlos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_siiv_sb_ylos', np.array(leaf_siiv_sb_ylos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_siiv_sb_zlos', np.array(leaf_siiv_sb_zlos[-1]),delete_previous)

            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_mgii_sb_xlos', np.array(leaf_mgii_sb_xlos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_mgii_sb_ylos', np.array(leaf_mgii_sb_ylos[-1]),delete_previous)
            append_to_hierarchy_file(hf[str(leaf_id)], 'leaf_mgii_sb_zlos', np.array(leaf_mgii_sb_zlos[-1]),delete_previous)

 

hf.close()


if shell_cell_ids is not None:
    print("\nSuccessfully found clump and shell stats in clump hierarchy file:",hiearchy_file)
else:
    print("\nWarning: could not read shell data in clump hierarchy file. Only leaf clump stats found in:",hiearchy_file)

#Write a smaller, separate file for all clump stats
if args.write_separate_stats_file or not args.modify_existing_clump_hierarchy:  
    hf = h5py.File(args.output_dir + GalName+"_"+args.snapshot+"_"+args.run+"_clump_stats.h5",'w')
    hf.create_dataset('leaf_vx', data=np.array(leaf_vx))
    hf.create_dataset('leaf_vy', data=np.array(leaf_vy))
    hf.create_dataset('leaf_vz', data=np.array(leaf_vz))
    hf.create_dataset('leaf_x', data=np.array(leaf_x))
    hf.create_dataset('leaf_y', data=np.array(leaf_y))
    hf.create_dataset('leaf_z', data=np.array(leaf_z))
    hf.create_dataset('leaf_hi_num_dense', data=np.array(leaf_hi_num_dense))
    hf.create_dataset('leaf_mgii_num_dense', data=np.array(leaf_mgii_num_dense))
    hf.create_dataset('leaf_oii_num_dense', data=np.array(leaf_oii_num_dense))
    hf.create_dataset('leaf_oiii_num_dense', data=np.array(leaf_oiii_num_dense))
    hf.create_dataset('leaf_oiv_num_dense', data=np.array(leaf_oiv_num_dense))
    hf.create_dataset('leaf_ov_num_dense', data=np.array(leaf_ov_num_dense))
    hf.create_dataset('leaf_ovi_num_dense', data=np.array(leaf_ovi_num_dense))
    hf.create_dataset('leaf_volume', data=np.array(leaf_volumes))
    hf.create_dataset('leaf_metallicity', data=np.array(leaf_metallicity))
    hf.create_dataset('leaf_pressure', data=np.array(leaf_pressure))
    hf.create_dataset('leaf_temperature', data=np.array(leaf_temperature))
    hf.create_dataset('leaf_mass', data=np.array(leaf_masses))
    hf.create_dataset('leaf_clump_ids', data=np.array(leaf_clump_ids))
    if args.do_tracer_fluids:
        hf.create_dataset('leaf_tf1_mass', data=np.array(leaf_tf1_mass))
        hf.create_dataset('leaf_tf2_mass', data=np.array(leaf_tf2_mass))
        hf.create_dataset('leaf_tf3_mass', data=np.array(leaf_tf3_mass))
        hf.create_dataset('leaf_tf4_mass', data=np.array(leaf_tf4_mass))
        hf.create_dataset('leaf_tf5_mass', data=np.array(leaf_tf5_mass))
        hf.create_dataset('leaf_tf6_mass', data=np.array(leaf_tf6_mass))
        hf.create_dataset('leaf_tf7_mass', data=np.array(leaf_tf7_mass))
        hf.create_dataset('leaf_tf8_mass', data=np.array(leaf_tf8_mass))
    if shell_cell_ids is not None:
        hf.create_dataset('shell_mass', data=np.array(shell_masses))
        hf.create_dataset('shell_volume', data=np.array(shell_volumes))
        hf.create_dataset('shell_vx', data=np.array(shell_vx))
        hf.create_dataset('shell_vy', data=np.array(shell_vy))
        hf.create_dataset('shell_vz', data=np.array(shell_vz))
        hf.create_dataset('shell_hi_num_dense', data=np.array(shell_hi_num_dense))
        hf.create_dataset('shell_mgii_num_dense', data=np.array(shell_mgii_num_dense))
        hf.create_dataset('shell_oii_num_dense', data=np.array(shell_oii_num_dense))
        hf.create_dataset('shell_oiii_num_dense', data=np.array(shell_oiii_num_dense))
        hf.create_dataset('shell_oiv_num_dense', data=np.array(shell_oiv_num_dense))
        hf.create_dataset('shell_ov_num_dense', data=np.array(shell_ov_num_dense))
        hf.create_dataset('shell_ovi_num_dense', data=np.array(shell_ovi_num_dense))
        hf.create_dataset('shell_metallicity', data=np.array(shell_metallicity))
        hf.create_dataset('shell_pressure', data=np.array(shell_pressure))
        hf.create_dataset('shell_temperature', data=np.array(shell_temperature))
        hf.create_dataset('shell_cooling_time', data=np.array(shell_cooling_time[-1]))
        hf.create_dataset('t_shear', data=np.array(t_shear[-1]))
        # --- VIDA'S ADDITION: save shell emissivity into separate stats file ---
        if args.add_emission:
            hf.create_dataset('shell_halpha_emissivity',  data=np.array(shell_halpha_emissivity))   
            hf.create_dataset('shell_lyalpha_emissivity', data=np.array(shell_lyalpha_emissivity))  
            hf.create_dataset('shell_cii_emissivity',     data=np.array(shell_cii_emissivity))    
            hf.create_dataset('shell_ciii_emissivity',    data=np.array(shell_ciii_emissivity))    
            hf.create_dataset('shell_civ_emissivity',     data=np.array(shell_civ_emissivity))    
            hf.create_dataset('shell_ovi_emissivity',     data=np.array(shell_ovi_emissivity))     
            hf.create_dataset('shell_siii_emissivity',    data=np.array(shell_siii_emissivity))    
            hf.create_dataset('shell_siiii_emissivity',   data=np.array(shell_siiii_emissivity))   
            hf.create_dataset('shell_siiv_emissivity',    data=np.array(shell_siiv_emissivity))     
            hf.create_dataset('shell_mgii_emissivity',    data=np.array(shell_mgii_emissivity))    
        
    # save emissivity into separate stats file ---
    if args.add_emission:
        hf.create_dataset('leaf_halpha_emissivity',  data=np.array(leaf_halpha_emissivity))   
        hf.create_dataset('leaf_lyalpha_emissivity', data=np.array(leaf_lyalpha_emissivity))  
        hf.create_dataset('leaf_cii_emissivity',     data=np.array(leaf_cii_emissivity))      
        hf.create_dataset('leaf_ciii_emissivity',    data=np.array(leaf_ciii_emissivity))     
        hf.create_dataset('leaf_civ_emissivity',     data=np.array(leaf_civ_emissivity))      
        hf.create_dataset('leaf_ovi_emissivity',     data=np.array(leaf_ovi_emissivity))      
        hf.create_dataset('leaf_siii_emissivity',    data=np.array(leaf_siii_emissivity))     
        hf.create_dataset('leaf_siiii_emissivity',   data=np.array(leaf_siiii_emissivity))    
        hf.create_dataset('leaf_siiv_emissivity',    data=np.array(leaf_siiv_emissivity))     
        hf.create_dataset('leaf_mgii_emissivity',    data=np.array(leaf_mgii_emissivity))  


        hf.create_dataset('leaf_halpha_sb_xlos',  data=np.array(leaf_halpha_sb_xlos))    
        hf.create_dataset('leaf_halpha_sb_ylos',  data=np.array(leaf_halpha_sb_ylos))
        hf.create_dataset('leaf_halpha_sb_zlos',  data=np.array(leaf_halpha_sb_zlos))

        hf.create_dataset('leaf_lyalpha_sb_xlos', data=np.array(leaf_lyalpha_sb_xlos))   
        hf.create_dataset('leaf_lyalpha_sb_ylos', data=np.array(leaf_lyalpha_sb_ylos))
        hf.create_dataset('leaf_lyalpha_sb_zlos', data=np.array(leaf_lyalpha_sb_zlos))

        hf.create_dataset('leaf_cii_sb_xlos',     data=np.array(leaf_cii_sb_xlos))       
        hf.create_dataset('leaf_cii_sb_ylos',     data=np.array(leaf_cii_sb_ylos))
        hf.create_dataset('leaf_cii_sb_zlos',     data=np.array(leaf_cii_sb_zlos))

        hf.create_dataset('leaf_ciii_sb_xlos',    data=np.array(leaf_ciii_sb_xlos))      
        hf.create_dataset('leaf_ciii_sb_ylos',    data=np.array(leaf_ciii_sb_ylos))
        hf.create_dataset('leaf_ciii_sb_zlos',    data=np.array(leaf_ciii_sb_zlos))

        hf.create_dataset('leaf_civ_sb_xlos',     data=np.array(leaf_civ_sb_xlos))       
        hf.create_dataset('leaf_civ_sb_ylos',     data=np.array(leaf_civ_sb_ylos))
        hf.create_dataset('leaf_civ_sb_zlos',     data=np.array(leaf_civ_sb_zlos))

        hf.create_dataset('leaf_ovi_sb_xlos',     data=np.array(leaf_ovi_sb_xlos))       
        hf.create_dataset('leaf_ovi_sb_ylos',     data=np.array(leaf_ovi_sb_ylos))
        hf.create_dataset('leaf_ovi_sb_zlos',     data=np.array(leaf_ovi_sb_zlos))

        hf.create_dataset('leaf_siii_sb_xlos',    data=np.array(leaf_siii_sb_xlos))      
        hf.create_dataset('leaf_siii_sb_ylos',    data=np.array(leaf_siii_sb_ylos))
        hf.create_dataset('leaf_siii_sb_zlos',    data=np.array(leaf_siii_sb_zlos))

        hf.create_dataset('leaf_siiii_sb_xlos',   data=np.array(leaf_siiii_sb_xlos))     
        hf.create_dataset('leaf_siiii_sb_ylos',   data=np.array(leaf_siiii_sb_ylos))
        hf.create_dataset('leaf_siiii_sb_zlos',   data=np.array(leaf_siiii_sb_zlos))

        hf.create_dataset('leaf_siiv_sb_xlos',    data=np.array(leaf_siiv_sb_xlos))      
        hf.create_dataset('leaf_siiv_sb_ylos',    data=np.array(leaf_siiv_sb_ylos))
        hf.create_dataset('leaf_siiv_sb_zlos',    data=np.array(leaf_siiv_sb_zlos))

        hf.create_dataset('leaf_mgii_sb_xlos',    data=np.array(leaf_mgii_sb_xlos))      
        hf.create_dataset('leaf_mgii_sb_ylos',    data=np.array(leaf_mgii_sb_ylos))
        hf.create_dataset('leaf_mgii_sb_zlos',    data=np.array(leaf_mgii_sb_zlos))
    # --- END VIDA'S ADDITION ---

    hf.close()
