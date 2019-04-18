import matplotlib as mpl
import seaborn as sns
import collections
import astropy.units as u
from matplotlib.colors import to_hex
import numpy as np
mpl.use('agg')

c = 299792.458 * u.Unit('km/s')
c_kms = 299792.458

default_width = 85.  # kpc in projection

axes_label_dict = {'density': 'log Density [g / cm$^3$]',
                   'Dark_Matter_Density': 'log DM Density [g / cm$^3$]',
                   'temperature': 'log Temperature [K]',
                   'cell_mass': 'log cell mass [M$_{\odot}$]',
                   'x': '$x$ coordinate [pkpc]',
                   'y': '$y$ coordinate [pkpc]',
                   'z': '$z$ coordinate [pkpc]',
                   'radius': 'Radius [pkpc]',
                   'x-velocity': 'X velocity [km s$^{-1}$]',
                   'y-velocity': 'Y velocity [km s$^{-1}$]',
                   'z-velocity': 'Z velocity [km s$^{-1}$]',
                   'relative_velocity': 'Relative Velocity [km s$^{-1}$]',
                   'metallicity': 'log Z/Z$_{\odot}$',
                   'pressure': 'log P [g cm$^{-1}$ s$^{-2}$ ]',
                   'O_p5_ion_fraction': 'log [O VI Ionization Fraction]',
                   'O_p5_number_density': 'log [O VI Number Density]',
                   'C_p3_ion_fraction': 'log [C IV Ionization Fraction]',
                   'C_p3_number_density': 'log [C IV Number Density]',
                   'Si_p3_ion_fraction': 'log [Si IV Ionization Fraction]',
                   'Si_p3_number_density': 'log [Si IV Number Density]',
                   }

# this is a dictionary of fields where we prefer to
# plot or visualize them in the log rather than the original yt / enzo
# field. Try "if field_name in logfields: field_name = log10(field_name)"
logfields = ('Dark_Matter_Density', 'density', 'temperature', 'entropy', 'pressure',
             'O_p5_ion_fraction', 'C_p3_ion_fraction', 'Si_p3_ion_fraction',
             'O_p5_number_density', 'C_p3_number_density',
             'Si_p3_number_density', 'metallicity')

phase_color_key = {b'cold': 'salmon',
                   b'hot': '#ffe34d',
                   b'warm': '#4daf4a',
                   b'cool': '#984ea3'}

metal_color_key = {b'high': 'darkorange',
                   b'solar': '#ffe34d',
                   b'low': '#4575b4',
                   b'poor': 'black'}

species_dict = {'CIII': 'C_p2_number_density',
                'CIV': 'C_p3_number_density',
                'HI': 'H_p0_number_density',
                'MgII': 'Mg_p1_number_density',
                'OVI': 'O_p5_number_density',
                'SiII': "Si_p1_number_density",
                'SiIII': "Si_p2_number_density",
                'SiIV': "Si_p3_number_density",
                'NeVIII': 'Ne_p7_number_density',
                'FeXIV': 'Fe_p13_number_density'}

ion_frac_color_key = {b'all': 'black',
                      b'low': 'yellow',
                      b'med': 'orange',
                      b'high': 'red',
                      b'phot': 'purple'}

discrete_cmap = mpl.colors.ListedColormap(
    ['#565656', '#4daf4a', '#d73027', "#984ea3",
     '#ffe34d', '#4575b4', 'darkorange'])
discrete_cmap_rainbow = mpl.colors.ListedColormap(
    ['#4daf4a', "#ffe34d", 'darkorange', "#d73027",
     '#984ea3', '#4575b4', '#565656'])

old_density_color_map = sns.blend_palette(
    ('black', '#984ea3', '#d73027', 'darkorange',
     '#ffe34d', '#4daf4a', 'white'), as_cmap=True)
density_color_map = sns.blend_palette(
    ("black", "#4575b4", "#4daf4a", "#ffe34d", "darkorange"), as_cmap=True)
density_proj_min = 5e-2  # msun / pc^2
density_proj_max = 1e4
density_slc_min = 5e-8  # msun / pc^3
density_slc_max = 5

metal_color_map = sns.blend_palette(
    ("black", "#4575b4", "#984ea3", "#984ea3", "#d73027",
     "darkorange", "#ffe34d"), as_cmap=True)
old_metal_color_map = sns.blend_palette(
    ("black", "#984ea3", "#4575b4", "#4daf4a",
     "#ffe34d", "darkorange"), as_cmap=True)
metal_min = 1.e-4
metal_max = 3.
metal_density_min = 1.e-5
metal_density_max = 250.

temperature_color_map = sns.blend_palette(
    ("black", "#d73027", "darkorange", "#ffe34d"), as_cmap=True)
temperature_min = 5.e6
temperature_max = 1.e4

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

old_o6_color_map = sns.blend_palette(
    ("white", "black", "#4daf4a", "#4575b4", "#984ea3", "#d73027",
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

si2_color_map = "plasma"
si2_min = 1.e10
si2_max = 1.e17

si3_color_map = "magma"
si3_min = 1.e11
si3_max = 1.e16

si4_color_map = "inferno"
si4_min = 1.e11
si4_max = 1.e15

ne8_color_map = "magma"
ne8_min = 1.e11
ne8_max = 1.e15

fe14_color_map = "inferno"
fe14_min = 1.e10
fe14_max = 1.e15

def categorize_by_fraction(f_ion, temperature):
    """ define the ionization category strings"""
    frac = np.chararray(np.size(f_ion), 4)
    frac[f_ion > -10.] = b'all'
    frac[f_ion > 0.01] = b'low'  # yellow
    frac[f_ion > 0.1] = b'med'  # orange
    frac[f_ion > 0.2] = b'high'  # red
    frac[(f_ion > 0.2) & (temperature < 1e5)] = b'phot'
    return frac


# set up the new temperature colormap
temp_colors = sns.blend_palette(
    ('salmon', "#984ea3", "#4daf4a", "#ffe34d", 'darkorange'), n_colors=17)
phase_color_labels = [b'cold1', b'cold2', b'cold3', b'cool', b'cool1', b'cool2',
                      b'cool3', b'warm', b'warm1', b'warm2', b'warm3', b'hot',
                      b'hot1', b'hot2', b'hot3']
temperature_discrete_cmap = mpl.colors.ListedColormap(temp_colors)
new_phase_color_key = collections.OrderedDict()
for i in np.arange(np.size(phase_color_labels)):
    new_phase_color_key[phase_color_labels[i]] = to_hex(temp_colors[i])

def new_categorize_by_temp(temp):
    """ define the temp category strings"""
    phase = np.chararray(np.size(temp), 5)
    phase[temp < 9.] = b'hot3'
    phase[temp < 6.6] = b'hot2'
    phase[temp < 6.4] = b'hot1'
    phase[temp < 6.2] = b'hot'
    phase[temp < 6.] = b'warm3'
    phase[temp < 5.8] = b'warm2'
    phase[temp < 5.6] = b'warm1'
    phase[temp < 5.4] = b'warm'
    phase[temp < 5.2] = b'cool3'
    phase[temp < 5.] = b'cool2'
    phase[temp < 4.8] = b'cool1'
    phase[temp < 4.6] = b'cool'
    phase[temp < 4.4] = b'cold3'
    phase[temp < 4.2] = b'cold2'
    phase[temp < 4.] = b'cold1'
    return phase

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

def new_categorize_by_metals(metal):
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
    print(phase)
    return phase




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
    #print(phase)
    return phase
