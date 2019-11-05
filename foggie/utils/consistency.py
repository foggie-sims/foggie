'''
a set of consistent colormaps, label names, etc.
'''

import matplotlib as mpl
import seaborn as sns
import collections
import astropy.units as u
from matplotlib.colors import to_hex
import numpy as np

c = 299792.458 * u.Unit('km/s')
c_kms = 299792.458

default_width = 85.  # kpc in projection

core_width = 20. # width of slice to use in show_velphase

axes_label_dict = {'density': 'log Density [g / cm$^3$]',
                    'Dark_Matter_Density': 'log DM Density [g / cm$^3$]',
                    'temperature': 'log Temperature [K]',
                    'cell_mass': r'log Cell Mass [M$_{\odot}$]',
                    'cell_size': 'Cell Size [physical pc]',
                    'x': '$x$ coordinate [physical kpc]',
                    'y': '$y$ coordinate [physical kpc]',
                    'z': '$z$ coordinate [physical kpc]',
                    'position_x': '$x$ coordinate [physical kpc]',
                    'position_y': '$y$ coordinate [physical kpc]',
                    'position_z': '$z$ coordinate [physical kpc]',
                    'radius': 'Radius [physical kpc]',
                    'mach_number': 'Mach Number',
                    'x_velocity': 'X velocity [km s$^{-1}$]',
                    'y_velocity': 'Y velocity [km s$^{-1}$]',
                    'z_velocity': 'Z velocity [km s$^{-1}$]',
                    'x-velocity': 'X velocity [km s$^{-1}$]',
                    'y-velocity': 'Y velocity [km s$^{-1}$]',
                    'z-velocity': 'Z velocity [km s$^{-1}$]',
                    'radial_velocity': 'Radial Velocity [km s$^{-1}$]',
                    'radial_velocity_corrected': 'Radial Velocity [km s$^{-1}$]', 
                    'tangential_velocity_corrected': 'Tangential Velocity [km s$^{-1}$]', 
                    'theta_velocity_corrected': 'Theta Velocity [km s$^{-1}$]', 
                    'phi_velocity_corrected': 'Phi Velocity [km s$^{-1}$]', 
                    'radius_corrected': 'Radius [physical kpc]', 
                    'relative_velocity': 'Relative Velocity [km s$^{-1}$]',
                    'velocity_spherical_radius': 'Radial Velocity [km s$^{-1}$]',
                    'metallicity': r'log Z/Z$_{\odot}$',
                    'pressure': 'log P [g cm$^{-1}$ s$^{-2}$ ]',
                    'entropy': 'log Entropy [cm$^2$ erg]',
                    'cooling_time': 'log Cooling Time [yr]',
                    'H_p0_ion_fraction': 'log [H I Ionization Fraction]',
                    'H_p0_number_density': 'log [H I Number Density]',
                    'H_p0_column_density': 'log [H I Cell Column Density]',
                    'O_p0_ion_fraction': 'O I Ionization Fraction',
                    'O_p0_number_density': 'log [O I Number Density]',
                    'O_p0_column_density': 'log [O I Cell Column Density]',
                    'O_p1_ion_fraction': 'O II Ionization Fraction',
                    'O_p1_number_density': 'log [O II Number Density]',
                    'O_p1_column_density': 'log [O II Cell Column Density]',
                    'O_p2_ion_fraction': 'O III Ionization Fraction',
                    'O_p2_number_density': 'log [O III Number Density]',
                    'O_p2_column_density': 'log [O III Cell Column Density]',
                    'O_p3_ion_fraction': 'O IV Ionization Fraction',
                    'O_p3_number_density': 'log [O IV Number Density]',
                    'O_p3_column_density': 'log [O IV Cell Column Density]',
                    'O_p4_ion_fraction': 'O V Ionization Fraction',
                    'O_p4_number_density': 'log [O V Number Density]',
                    'O_p4_column_density': 'log [O V Cell Column Density]',
                    'O_p5_ion_fraction': 'O VI Ionization Fraction',
                    'O_p5_number_density': 'log [O VI Number Density]',
                    'O_p5_column_density': 'log [O VI Cell Column Density]',
                    'O_p6_ion_fraction': 'O VII Ionization Fraction',
                    'O_p6_number_density': 'log [O VII Number Density]',
                    'O_p6_column_density': 'log [O VII Cell Column Density]',
                    'O_p7_ion_fraction': 'O VIII Ionization Fraction',
                    'O_p7_number_density': 'log [O VIII Number Density]',
                    'O_p7_column_density': 'log [O VIII Cell Column Density]',
                    'C_p3_ion_fraction': 'C IV Ionization Fraction',
                    'C_p3_number_density': 'log [C IV Number Density]',
                    'Si_p3_ion_fraction': 'Si IV Ionization Fraction',
                    'Si_p3_number_density': 'log [Si IV Number Density]',
                    'N_p4_number_denstiy': 'log [N V Number Density]'
                   }

# this is a dictionary of fields where we prefer to plot or
# visualize them in the log rather than the original yt / enzo field.
# Try "if field_name in logfields: field_name = log10(field_name)"
logfields = ('Dark_Matter_Density', 'density', 'temperature',
             'entropy', 'pressure', 'cooling_time',
             'H_p0_number_density', 'H_p0_column_density',
             'O_p0_number_density', 'O_p0_column_density',
             'O_p1_number_density', 'O_p1_column_density',
             'O_p2_number_density', 'O_p2_column_density',
             'O_p3_number_density', 'O_p3_column_density',
             'O_p4_number_density', 'O_p4_column_density',
             'O_p5_number_density', 'O_p5_column_density',
             'O_p6_number_density', 'O_p6_column_density',
             'O_p7_number_density', 'O_p7_column_density',
             'C_p3_number_density', 'Si_p3_number_density',
             'metallicity', 'cell_mass', 'cell_size')

species_dict = {'CIII': 'C_p2_number_density',
                'CIV': 'C_p3_number_density',
                'HI': 'H_p0_number_density',
                'MgII': 'Mg_p1_number_density',
                'OVI': 'O_p5_number_density',
                'SiII': "Si_p1_number_density",
                'SiIII': "Si_p2_number_density",
                'SiIV': "Si_p3_number_density",
                'NeVIII': 'Ne_p7_number_density',
                'FeXIV': 'Fe_p13_number_density',
                'NV': 'N_p4_number_density',
                'CII': 'C_p1_number_density', 
                'OVII': 'O_p6_number_density', 
                'OVIII': 'O_p7_number_density', 
                'NeVII': 'Ne_p6_number_density', 
                'NeVIII': 'Ne_p7_number_density'}

halo_dict = {   2392  :  'Hurricane' ,
                2878  :  'Cyclone' ,
                4123  :  'Wigshifter' ,
                5016  :  'Squall' ,
                5036  :  'Maelstrom' ,
                8508  :  'Tempest' }

linelist_jt = ['H I 1216', 'H I 919', \
                'Mg II 2796', 'Si II 1260', 'Si III 1206', 'Si IV 1394', \
                'C II 1335', 'C III 977', 'C IV 1548',\
                'O VI 1032', 'Ne VIII 770']
linelist_kodiaq  = ['H I 1216', 'H I 919', \
                'Si II 1260', 'Si III 1206', 'Si IV 1394',
                'C II 1335', 'C III 977', 'C IV 1548',
                 'O VI 1032']
linelist_long = ['H I 1216', 'H I 1026', 'H I 973',
               'H I 950', 'H I 919', 'Al II 1671', 'Al III 1855', \
               'Si II 1260', 'Si III 1206', 'Si IV 1394', \
               'C II 1335', 'C III 977', 'C IV 1548', \
               'O VI 1032', 'Ne VIII 770']

cgm_temperature_min = 1.5e4  #<---- in some FOGGIE codes this will be used to set a min
cgm_density_max = 2e-26
cgm_inner_radius = 10.
cgm_outer_radius = 200.

#These are strings that can be used to produce yt CutRegions with consistent cuts.
cgm_field_filter = ("(obj['temperature'] > {} ) | (obj['density'] < {})").format(cgm_temperature_min, cgm_density_max)
ism_field_filter = ("(obj['temperature'] < {} ) & (obj['density'] > {})").format(cgm_temperature_min, cgm_density_max)

# these are useful for the refinement levels maps
discrete_cmap = mpl.colors.ListedColormap(
    ['#565656', '#4daf4a', '#d73027', "#984ea3",
     '#ffe34d', '#4575b4', 'darkorange'])
discrete_cmap_rainbow = mpl.colors.ListedColormap(
    ['#4daf4a', "#ffe34d", 'darkorange', "#d73027",
     '#984ea3', '#4575b4', '#565656'])

density_color_map = sns.blend_palette(
    ("black", "#4575b4", "#4daf4a", "#ffe34d", "darkorange"), as_cmap=True)
density_proj_min = 5e-2  # msun / pc^2
density_proj_max = 1e4
density_slc_min = 5e-8  # msun / pc^3
density_slc_max = 5

dens_phase_min = 10.**-31
dens_phase_max = 10.**-21
metal_proj_min = 10.**54
metal_proj_max = 10.**61

metal_color_map = sns.blend_palette(
    ("black", "#4575b4", "#984ea3", "#984ea3", "#d73027",
     "darkorange", "#ffe34d"), as_cmap=True)
old_metal_color_map = sns.blend_palette(
    ("black", "#984ea3", "#4575b4", "#4daf4a",
     "#ffe34d", "darkorange"), as_cmap=True)
metal_min = 5.e-3
metal_max = 3.
metal_density_min = 1.e-5
metal_density_max = 250.

temperature_color_map = sns.blend_palette(
    ("black", "#d73027", "darkorange", "#ffe34d"), as_cmap=True)
temperature_max = 5e6
temperature_min = 1e4
temperature_max_datashader = 1.e8
temperature_min_datashader = 1.e2

entropy_color_map = "Spectral_r"
entropy_min = 1.e-4
entropy_max = 1.e3

pressure_color_map = "Spectral"
pressure_min = 1.e-16
pressure_max = 1.e-9

h1_color_map = sns.blend_palette(("white", "#ababab", "#565656", "black",
                                  "#4575b4", "#984ea3", "#d73027",
                                  "darkorange", "#ffe34d"), as_cmap=True)
h1_proj_min = 1.e12
h1_proj_max = 1.e24
h1_slc_min = 1.e-14
h1_slc_max = 1.e2

h1_color_map_mw = 'viridis' # same as figure 2 in HI4PI+2016 paper.
h1_proj_min_mw = 1e13 # for mocky way allsky map, YZ
h1_proj_max_mw = 1e23 # for mocky way allsky map, YZ, tuned for HI4PI

old_o6_color_map = sns.blend_palette(("white", "black", "#4daf4a",
                                      "#4575b4", "#984ea3", "#d73027",
                                      "darkorange"), as_cmap=True)
o6_color_map = "magma"
o6_min = 1.e11
o6_max = 1.e15

c4_color_map = "inferno"
c4_min = 1.e11
c4_max = 1.e16

mg2_color_map = "plasma"
mg2_min = 1.e10
mg2_max = 1.e17

c2_color_map = "plasma"
c2_min = 1.e10
c2_max = 1.e17

c3_color_map = "magma"
c3_min = 1.e11
c3_max = 1.e16

si2_color_map = "plasma"
si2_min = 1.e10
si2_max = 1.e17

si3_color_map = "magma"
si3_min = 1.e11
si3_max = 1.e16

si4_color_map = "inferno"
si4_min = 1.e11
si4_max = 1.e15

n5_color_map = "inferno"
n5_min = 1.e11
n5_max = 1.e15

o7_color_map = "magma"
o7_min = 1.e11
o7_max = 1.e15

o8_color_map = "magma"
o8_min = 1.e11
o8_max = 1.e15

ne7_color_map = "magma"
ne7_min = 1.e11
ne7_max = 1.e15

ne8_color_map = "magma"
ne8_min = 1.e11
ne8_max = 1.e15

fe14_color_map = "inferno"
fe14_min = 1.e10
fe14_max = 1.e15

#set up the ionization fraction colormap
def categorize_by_fraction(f_ion):
    """ define the ionization category strings"""
    frac = np.chararray(np.size(f_ion), 4)
    frac[f_ion > -10.] = b'all'
    frac[f_ion > 0.0001] = b'low'   # yellow
    frac[f_ion > 0.01]  = b'med'   # orange
    frac[f_ion > 0.1]  = b'high'  # red
    return frac

# I'm commenting this out because it produces a figure for no reason and doesn't appear to be
# used by any other files currently in the foggie repo. -Cassi
#ion_frac_color_key = sns.palplot(sns.blend_palette(("grey","#ff6600"), n_colors=10),size=1.5)
# Just in case this is needed, this might work instead without producing a figure:
ion_frac_color_key = sns.blend_palette(("grey","#ff6600"), n_colors=10)

############################# set up the temperature colormap
temp_colors = sns.blend_palette(
    ('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), n_colors=17)
temperature_discrete_cmap = mpl.colors.ListedColormap(temp_colors)
new_phase_color_key = collections.OrderedDict()

phase_color_labels = [b'cold1', b'cold2', b'cold3', b'cool', b'cool1', b'cool2',
                      b'cool3', b'warm', b'warm1', b'warm2', b'warm3', b'hot',
                      b'hot1', b'hot2', b'hot3']
for i in np.arange(np.size(phase_color_labels)):
    new_phase_color_key[phase_color_labels[i]] = to_hex(temp_colors[i])

def categorize_by_temp(temperature):
    """ define the temp category strings"""
    phase = np.chararray(np.size(temperature), 5)
    phase[temperature < 9.] = b'hot3'
    phase[temperature < 6.6] = b'hot2'
    phase[temperature < 6.4] = b'hot1'
    phase[temperature < 6.2] = b'hot'
    phase[temperature < 6.] = b'warm3'
    phase[temperature < 5.8] = b'warm2'
    phase[temperature < 5.6] = b'warm1'
    phase[temperature < 5.4] = b'warm'
    phase[temperature < 5.2] = b'cool3'
    phase[temperature < 5.] = b'cool2'
    phase[temperature < 4.8] = b'cool1'
    phase[temperature < 4.6] = b'cool'
    phase[temperature < 4.4] = b'cold3'
    phase[temperature < 4.2] = b'cold2'
    phase[temperature < 4.] = b'cold1'
    #print(phase)
    return phase

### I'm adding this logT color keys for mocky way. Yong Zheng, 10/10/2019. ##
### Still using the same temperature color plate ###
logT_color_labels_mw = [b'<4.0', b'4.0-4.5', b'4.5-5.0', b'5.0-5.5',
                        b'5.5-6.0', b'6.0-6.5', b'6.5-7.0', b'>7.0']
logT_colors_mw = sns.blend_palette(('salmon', "#984ea3", "#4daf4a",
                                    '#ffe34d', 'darkorange'),
                                    n_colors=len(logT_color_labels_mw))
logT_discrete_cmap_mw = mpl.colors.ListedColormap(logT_colors_mw)
logT_color_key_mw = collections.OrderedDict()
for i in np.arange(np.size(logT_color_labels_mw)):
    logT_color_key_mw[logT_color_labels_mw[i]] = to_hex(logT_colors_mw[i])

def categorize_by_logT_mw(logT):
    """ define the temp category strings"""
    phase = np.chararray(np.size(logT), 8)
    phase[logT>7] = b'>7.0'
    #phase[np.all([logT>=6.8, logT<7.0], axis=0)] = b'6.8-7.0'
    #phase[np.all([logT>=6.6, logT<6.8], axis=0)] = b'6.6-6.8'
    #phase[np.all([logT>=6.4, logT<6.6], axis=0)] = b'6.4-6.6'
    #phase[np.all([logT>=6.2, logT<6.4], axis=0)] = b'6.2-6.4'
    #phase[np.all([logT>=6.0, logT<6.2], axis=0)] = b'6.0-6.2'
    #phase[np.all([logT>=5.8, logT<6.0], axis=0)] = b'5.8-6.0'
    #phase[np.all([logT>=5.6, logT<5.8], axis=0)] = b'5.6-5.8'
    #phase[np.all([logT>=5.4, logT<5.6], axis=0)] = b'5.4-5.6'
    #phase[np.all([logT>=5.2, logT<5.4], axis=0)] = b'5.2-5.4'
    #phase[np.all([logT>=5.0, logT<5.2], axis=0)] = b'5.0-5.2'
    #phase[np.all([logT>=4.8, logT<5.0], axis=0)] = b'4.8-5.0'
    #phase[np.all([logT>=4.6, logT<4.8], axis=0)] = b'4.6-4.8'
    #phase[np.all([logT>=4.4, logT<4.6], axis=0)] = b'4.4-4.6'
    #phase[np.all([logT>=4.2, logT<4.4], axis=0)] = b'4.2-4.4'
    #phase[np.all([logT>=4.0, logT<4.2], axis=0)] = b'4.0-4.2'
    phase[np.all([logT>=6.5, logT<7.0], axis=0)] = b'6.5-7.0'
    phase[np.all([logT>=6.0, logT<6.5], axis=0)] = b'6.0-6.5'
    phase[np.all([logT>=5.5, logT<6.0], axis=0)] = b'5.5-6.0'
    phase[np.all([logT>=5.0, logT<5.5], axis=0)] = b'5.0-5.5'
    phase[np.all([logT>=4.5, logT<5.0], axis=0)] = b'4.5-5.0'
    phase[np.all([logT>=4.0, logT<4.5], axis=0)] = b'4.0-4.5'
    phase[logT<4] = b'<4.0'
    return phase

logT_color_labels_mw_fine = [b'<4.0', b'4.0-4.2', b'4.2-4.4', b'4.4-4.6',
                        b'4.6-4.8', b'4.8-5.0', b'5.0-5.2', b'5.2-5.4',
                        b'5.4-5.6', b'5.6-5.8', b'5.8-6.0', b'6.0-6.2',
                        b'6.2-6.4', b'6.4-6.6', b'6.6-6.8', b'6.8-7.0',
                        b'>7.0']
logT_colors_mw_fine = sns.blend_palette(('salmon', "#984ea3", "#4daf4a",
                                         '#ffe34d', 'darkorange'),
                                         n_colors=len(logT_color_labels_mw_fine))
logT_discrete_cmap_mw_fine = mpl.colors.ListedColormap(logT_colors_mw_fine)
###################################################################
metal_color_labels = [b'free', b'free1', b'free2', b'free3', b'poor',
                      b'poor1', b'poor2', b'poor3', b'low', b'low1',
                      b'low2', b'low3', b'solar', b'solar1', b'solar2',
                      b'solar3', b'high', b'high1', b'high2', b'high3', b'high4']
metallicity_colors = sns.blend_palette(("black", "#4575b4", "#984ea3", "#984ea3", "#d73027",
     "darkorange", "#ffe34d"), n_colors=21)
metal_discrete_cmap = mpl.colors.ListedColormap(metallicity_colors)
new_metals_color_key = collections.OrderedDict()
for i in np.arange(np.size(metal_color_labels)):
    new_metals_color_key[metal_color_labels[i]] = to_hex(metallicity_colors[i])

metal_labels = new_metals_color_key.keys()

def categorize_by_metals(metal):
    """ define the temp category strings"""
    metal_vals = np.power(10.0, np.linspace(start=np.log10(metal_min),
                                            stop=np.log10(metal_max), num=21))
    # make the highest value really high
    metal_vals[20] = 50. * metal_vals[20]
    phase = np.chararray(np.size(metal), 6)
    # need to do this by iterating over keys insteard of hard coding indices
    phase[metal < metal_vals[20]] = b'high4'
    phase[metal < metal_vals[19]] = b'high3'
    phase[metal < metal_vals[18]] = b'high2'
    phase[metal < metal_vals[17]] = b'high1'
    phase[metal < metal_vals[16]] = b'high'
    phase[metal < metal_vals[15]] = b'solar3'
    phase[metal < metal_vals[14]] = b'solar2'
    phase[metal < metal_vals[13]] = b'solar1'
    phase[metal < metal_vals[12]] = b'solar'
    phase[metal < metal_vals[11]] = b'low3'
    phase[metal < metal_vals[10]] = b'low2'
    phase[metal < metal_vals[9]] = b'low1'
    phase[metal < metal_vals[8]] = b'low'
    phase[metal < metal_vals[7]] = b'poor3'
    phase[metal < metal_vals[6]] = b'poor2'
    phase[metal < metal_vals[5]] = b'poor1'
    phase[metal < metal_vals[4]] = b'poor'
    phase[metal < metal_vals[3]] = b'free3'
    phase[metal < metal_vals[2]] = b'free2'
    phase[metal < metal_vals[1]] = b'free1'
    phase[metal < metal_vals[0]] = b'free'
    return phase

##############################################################
# I made a simplier categoy for mokcy way, 10/10/2019, Yong Zheng.
metal_color_labels_mw = [b'<0.01', b'[0.01, 0.1)',
                         b'[0.1, 0.5)', b'[0.5, 1.0)',
                         b'[1.0, 2.0)', b'>=2.0']
metal_colors_mw = sns.blend_palette(("black", "#4575b4", "#984ea3",
                                     "#984ea3", "#d73027", "darkorange",
                                     "#ffe34d"), n_colors=6)
metal_discrete_cmap_mw = mpl.colors.ListedColormap(metal_colors_mw)
metal_color_key_mw = collections.OrderedDict()
for i in np.arange(np.size(metal_color_labels_mw)):
    metal_color_key_mw[metal_color_labels_mw[i]] = to_hex(metal_colors_mw[i])

def categorize_by_metallicity_mw(metal):
    """
    define the metallicity category strings for mocky way. Yong Zheng. 10/10/2019.
    """
    phase = np.chararray(np.size(metal), 11)
    # need to do this by iterating over keys insteard of hard coding indices
    phase[metal<0.01] = b'<0.01'
    phase[np.all([metal>=0.01, metal<0.1], axis=0)] = b'[0.01, 0.1)'
    phase[np.all([metal>=0.1, metal<0.5], axis=0)] = b'[0.1, 0.5)'
    phase[np.all([metal>=0.5, metal<1.0], axis=0)] = b'[0.5, 1.0)'
    phase[np.all([metal>=1.0, metal<2.0], axis=0)] = b'[1.0, 2.0)'
    phase[metal>=2.0] = b'>=2.0'

    return phase

#######################################################################
hi_colors =  sns.blend_palette(("white", "#ababab", "#565656", "black",
                                  "#4575b4", "#984ea3", "#d73027",
                                  "darkorange", "#ffe34d"), n_colors=26)
hi_color_key = {b'free': to_hex(hi_colors[0]),
                        b'free1': to_hex(hi_colors[1]),
                        b'free2': to_hex(hi_colors[2]),
                        b'free3': to_hex(hi_colors[3]),
                        b'poor': to_hex(hi_colors[4]),
                        b'poor1': to_hex(hi_colors[5]),
                        b'poor2': to_hex(hi_colors[6]),
                        b'poor3': to_hex(hi_colors[7]),
                        b'low': to_hex(hi_colors[8]),  # blue
                        b'low1': to_hex(hi_colors[9]),
                        b'low2': to_hex(hi_colors[10]),
                        b'low3': to_hex(hi_colors[11]),
                        b'solar': to_hex(hi_colors[12]),
                        b'solar1': to_hex(hi_colors[13]),
                        b'solar2': to_hex(hi_colors[14]),
                        b'solar3': to_hex(hi_colors[15]),
                        b'high': to_hex(hi_colors[16]),
                        b'high1': to_hex(hi_colors[17]),
                        b'high2': to_hex(hi_colors[18]),
                        b'high3': to_hex(hi_colors[19]),
                        b'high4': to_hex(hi_colors[20]),
                        b'moar': to_hex(hi_colors[21]),
                        b'moar1': to_hex(hi_colors[22]),
                        b'moar2': to_hex(hi_colors[23]),
                        b'moar3': to_hex(hi_colors[24]),
                        b'moar4': to_hex(hi_colors[25])
                        }

hi_labels = hi_color_key.keys()
def categorize_by_hi(hi):
    """ define the temp category strings"""
    hi_vals = np.linspace(start=np.log10(h1_proj_min),stop=np.log10(h1_proj_max), num=26)
    # make the highest value really high
    hi_vals[25] = 50. * hi_vals[25]
    phase = np.chararray(np.size(hi), 6)
    # need to do this by iterating over keys insteard of hard coding indices
    phase[hi < hi_vals[25]] = b'moar4'
    phase[hi < hi_vals[24]] = b'moar3'
    phase[hi < hi_vals[23]] = b'moar2'
    phase[hi < hi_vals[22]] = b'moar1'
    phase[hi < hi_vals[21]] = b'moar'
    phase[hi < hi_vals[20]] = b'high4'
    phase[hi < hi_vals[19]] = b'high3'
    phase[hi < hi_vals[18]] = b'high2'
    phase[hi < hi_vals[17]] = b'high1'
    phase[hi < hi_vals[16]] = b'high'
    phase[hi < hi_vals[15]] = b'solar3'
    phase[hi < hi_vals[14]] = b'solar2'
    phase[hi < hi_vals[13]] = b'solar1'
    phase[hi < hi_vals[12]] = b'solar'
    phase[hi < hi_vals[11]] = b'low3'
    phase[hi < hi_vals[10]] = b'low2'
    phase[hi < hi_vals[9]] = b'low1'
    phase[hi < hi_vals[8]] = b'low'
    phase[hi < hi_vals[7]] = b'poor3'
    phase[hi < hi_vals[6]] = b'poor2'
    phase[hi < hi_vals[5]] = b'poor1'
    phase[hi < hi_vals[4]] = b'poor'
    phase[hi < hi_vals[3]] = b'free3'
    phase[hi < hi_vals[2]] = b'free2'
    phase[hi < hi_vals[1]] = b'free1'
    phase[hi < hi_vals[0]] = b'free'
    return phase

############### Yong Zheng add cat_radius for mocky way ########
### categorize halo gas by radius.
# radius_df_colname = 'cat_radius' # name of radius in dataframe
radius_color_labels = [b'0-10', b'10-20', b'20-30', b'30-40',
                       b'40-50', b'50-60', b'60-70', b'70-80',
                       b'80-90', b'90-100', b'100-110', b'110-120']
radius_colors = sns.blend_palette(('#691F5E', '#4FCEED', '#F76C1D', '#DAD10C'),
                                   n_colors=len(radius_color_labels))
radius_discrete_cmap = mpl.colors.ListedColormap(radius_colors)
radius_color_key = collections.OrderedDict()
for i, ilabel in enumerate(radius_color_labels):
    radius_color_key[ilabel] = to_hex(radius_colors[i])

def categorize_by_radius(radius):
    """ define the radius category strings"""
    cat_radius = np.chararray(np.size(radius), 8)
    cat_radius[np.all([radius>=0, radius<10], axis=0)] = b'0-10'
    cat_radius[np.all([radius>=10, radius<20], axis=0)] = b'10-20'
    cat_radius[np.all([radius>=20, radius<30], axis=0)] = b'20-30'
    cat_radius[np.all([radius>=30, radius<40], axis=0)] = b'30-40'
    cat_radius[np.all([radius>=40, radius<50], axis=0)] = b'40-50'
    cat_radius[np.all([radius>=50, radius<60], axis=0)] = b'50-60'
    cat_radius[np.all([radius>=60, radius<70], axis=0)] = b'60-70'
    cat_radius[np.all([radius>=70, radius<80], axis=0)] = b'70-80'
    cat_radius[np.all([radius>=80, radius<90], axis=0)] = b'80-90'
    cat_radius[np.all([radius>=90, radius<100], axis=0)] = b'90-100'
    cat_radius[np.all([radius>=100, radius<110], axis=0)] = b'100-110'
    cat_radius[np.all([radius>=110, radius<120], axis=0)] = b'110-120'
    return cat_radius

############### Yong Zheng add cat_velocity for mocky way ########
### categorize halo gas by velocity.
# velocity_df_colname = 'cat_velocity' # this is the name of velocity in dataframe
velocity_color_labels = [b'<-100', b'[-100, -50]', b'[-50, 0]',
                         b'[0, 50]', b'[50, 100]', b'>100']
velocity_colors=sns.blend_palette(('#C1BEB4', '#5FEAF0', '#3C92F9',
                                   '#F95B3C', '#FCA024', '#EFD96B'),
                                   n_colors=6)
velocity_discrete_cmap = mpl.colors.ListedColormap(velocity_colors)
velocity_color_key = collections.OrderedDict()
for i, ilabel in enumerate(velocity_color_labels):
    velocity_color_key[ilabel] = to_hex(velocity_colors[i])

def categorize_by_velocity(velocity):
    """ define the line of sight velocity category strings"""
    vv = velocity
    cat_vel = np.chararray(np.size(vv), 13)
    cat_vel[vv<-400] = b'<-400'
    cat_vel[np.all([vv>=-400, vv<-300], axis=0)] = b'[-400, -300)'
    cat_vel[np.all([vv>=-300, vv<-200], axis=0)] = b'[-300, -200)'
    cat_vel[np.all([vv>=-200, vv<-180], axis=0)] = b'[-200, -180)'
    cat_vel[np.all([vv>=-180, vv<-160], axis=0)] = b'[-180, -160)'
    cat_vel[np.all([vv>=-160, vv<-140], axis=0)] = b'[-160, -140)'
    cat_vel[np.all([vv>=-140, vv<-120], axis=0)] = b'[-140, -120)'
    cat_vel[np.all([vv>=-120, vv<-100], axis=0)] = b'[-120, -100)'
    cat_vel[np.all([vv>=-100, vv<-80], axis=0)] = b'[-100, -80)'
    cat_vel[np.all([vv>=-80, vv<-60], axis=0)] = b'[-80, -60)'
    cat_vel[np.all([vv>=-60, vv<-40], axis=0)] = b'[-60, -40)'
    cat_vel[np.all([vv>=-40, vv<-20], axis=0)] = b'[-40, -20)'
    cat_vel[np.all([vv>=-20, vv<0], axis=0)] = b'[-20, 0)'
    return cat_velocity

############### Yong Zheng add cat_outflow for mocky way ########
### categorize halo gas by velocity.
# outflow_df_colname = 'cat_vel' # this is the name of velocity in dataframe
outflow_color_labels = [b'[0, 20)', b'[20, 40)', b'[40, 60)', b'[60, 80)',
                        b'[80, 100)', b'[100, 120)', b'[120, 140)',
                        b'[140, 160)', b'[160, 180)', b'[180, 200)',
                        b'[200, 300)', b'[300, 400)', b'>400']
outflow_cmap = mpl.pyplot.cm.PuRd
outflow_colors = sns.color_palette("PuRd", len(outflow_color_labels))
#outflow_colors = sns.blend_palette((outflow_cmap(0.25),
#                                    outflow_cmap(0.4),
#                                    outflow_cmap(0.55),
#                                    outflow_cmap(0.7),
#                                    outflow_cmap(0.9)),
#                                    n_colors=len(outflow_color_labels))
outflow_discrete_cmap = mpl.colors.ListedColormap(outflow_colors)
outflow_color_key = collections.OrderedDict()
for i, ilabel in enumerate(outflow_color_labels):
    outflow_color_key[ilabel] = to_hex(outflow_colors[i])

def categorize_by_outflow(velocity):
    """ define the line of sight velocity category strings"""
    vv = velocity
    cat_vel = np.chararray(np.size(vv), 11)
    cat_vel[np.all([vv>=0, vv<20], axis=0)] = b'[0, 20)'
    cat_vel[np.all([vv>=20, vv<40], axis=0)] = b'[20, 40)'
    cat_vel[np.all([vv>=40, vv<60], axis=0)] = b'[40, 60)'
    cat_vel[np.all([vv>=60, vv<80], axis=0)] = b'[60, 80)'
    cat_vel[np.all([vv>=80, vv<100], axis=0)] = b'[80, 100)'
    cat_vel[np.all([vv>=100, vv<120], axis=0)] = b'[100, 120)'
    cat_vel[np.all([vv>=120, vv<140], axis=0)] = b'[120, 140)'
    cat_vel[np.all([vv>=140, vv<160], axis=0)] = b'[140, 160)'
    cat_vel[np.all([vv>=160, vv<180], axis=0)] = b'[160, 180)'
    cat_vel[np.all([vv>=180, vv<200], axis=0)] = b'[180, 200)'
    cat_vel[np.all([vv>=200, vv<300], axis=0)] = b'[200, 300)'
    cat_vel[np.all([vv>=300, vv<400], axis=0)] = b'[300, 400)'
    cat_vel[vv>400] = b'>400'
    return cat_vel

############### Yong Zheng add cat_inflow for mocky way ########
### categorize halo gas by velocity.
# inflow_df_colname = 'cat_inflow' # this is the name of velocity in dataframe
inflow_color_labels = [b'<-400',
                       b'[-400, -300)', b'[-300, -200)', b'[-200, -180)',
                       b'[-180, -160)', b'[-160, -140)', b'[-140, -120)',
                       b'[-120, -100)', b'[-100, -80)',  b'[-80, -60)',
                       b'[-60, -40)',   b'[-40, -20)',   b'[-20, 0)']
inflow_cmap = mpl.pyplot.cm.YlGnBu_r
inflow_colors = sns.color_palette("YlGnBu_r", len(inflow_color_labels))
#inflow_colors = sns.blend_palette((inflow_cmap(0.25),
#                                    inflow_cmap(0.4),
#                                    inflow_cmap(0.55),
#                                    inflow_cmap(0.7),
#                                    inflow_cmap(0.9)),
#                                    n_colors=len(inflow_color_labels))
inflow_discrete_cmap = mpl.colors.ListedColormap(inflow_colors)
inflow_color_key = collections.OrderedDict()
for i, ilabel in enumerate(inflow_color_labels):
    inflow_color_key[ilabel] = to_hex(inflow_colors[i])

def categorize_by_inflow(velocity):
    """ define the line of sight velocity category strings"""
    vv = velocity
    cat_vel = np.chararray(np.size(vv), 13)
    cat_vel[vv<-400] = b'<-400'
    cat_vel[np.all([vv>=-400, vv<-300], axis=0)] = b'[-400, -300)'
    cat_vel[np.all([vv>=-300, vv<-200], axis=0)] = b'[-300, -200)'
    cat_vel[np.all([vv>=-200, vv<-180], axis=0)] = b'[-200, -180)'
    cat_vel[np.all([vv>=-180, vv<-160], axis=0)] = b'[-180, -160)'
    cat_vel[np.all([vv>=-160, vv<-140], axis=0)] = b'[-160, -140)'
    cat_vel[np.all([vv>=-140, vv<-120], axis=0)] = b'[-140, -120)'
    cat_vel[np.all([vv>=-120, vv<-100], axis=0)] = b'[-120, -100)'
    cat_vel[np.all([vv>=-100, vv<-80], axis=0)] = b'[-100, -80)'
    cat_vel[np.all([vv>=-80, vv<-60], axis=0)] = b'[-80, -60)'
    cat_vel[np.all([vv>=-60, vv<-40], axis=0)] = b'[-60, -40)'
    cat_vel[np.all([vv>=-40, vv<-20], axis=0)] = b'[-40, -20)'
    cat_vel[np.all([vv>=-20, vv<=0], axis=0)] = b'[-20, 0)'
    return cat_vel

############################################################

colormap_dict = {'phase': new_phase_color_key,
                 'metal': new_metals_color_key,
                 'h1': hi_color_key,
                 'density': density_color_map,
                 'O_p5_number_density': o6_color_map,
                 'H_p0_number_density': h1_color_map,
                 'C_p1_number_density': c2_color_map,
                 'C_p3_number_density': c4_color_map,
                 'Mg_p1_number_density': mg2_color_map,
                 'Si_p1_number_density': si2_color_map,
                 'Si_p2_number_density': si3_color_map,
                 'Si_p3_number_density': si4_color_map,
                 'N_p4_number_density': n5_color_map, 
                 'O_p6_number_density': o7_color_map,
                 'O_p7_number_density': o8_color_map, 
                 'Ne_p6_number_density': ne7_color_map, 
                 'Ne_p7_number_density': ne8_color_map}


proj_max_dict = {'density': 1e-1, 
                 'H_p0_number_density': h1_proj_max, 
                 'C_p1_number_density': c2_max, 
                 'C_p2_number_density': c3_max,
                 'C_p3_number_density': c4_max,
                 'Si_p1_number_density': si2_max,
                 'Si_p2_number_density': si3_max,
                 'Si_p3_number_density': si4_max,
                 'Mg_p1_number_density': mg2_max,
                 'O_p5_number_density': o6_max,
                 'N_p4_number_density': n5_max,
                 'O_p6_number_density': o7_max, 
                 'O_p7_number_density': o8_max, 
                 'Ne_p6_number_density': ne7_max, 
                 'Ne_p7_number_density': ne8_max}


proj_min_dict = {'density':1e-6, 
                 'H_p0_number_density':h1_proj_min,
                 'C_p1_number_density':c2_min,
                 'C_p2_number_density':c3_min,
                 'C_p3_number_density':c4_min,
                 'Si_p1_number_density':si2_min,
                 'Si_p2_number_density':si3_min,
                 'Si_p3_number_density':si4_min,
                 'Mg_p1_number_density':mg2_min,
                 'O_p5_number_density':o6_min,
                 'N_p4_number_density': n5_min,
                 'O_p6_number_density': o7_min,
                 'O_p7_number_density': o8_min,  
                 'Ne_p6_number_density': ne7_min,
                 'Ne_p7_number_density':ne8_min}

background_color_dict = {'density':'black', \
                         'H_p0_number_density':'white', \
                         'C_p1_number_density':'black', \
                         'C_p2_number_density':'black', \
                         'C_p3_number_density':'black', \
                         'Si_p1_number_density':'black',\
                         'Si_p2_number_density':'black',\
                         'Si_p3_number_density':'black',\
                         'Mg_p1_number_density':'black',\
                         'O_p5_number_density':'black',\
                         'Ne_p7_number_density':'black'}
