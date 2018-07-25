import matplotlib as mpl
from matplotlib.colors import to_hex
import numpy as np
mpl.use('agg')
import seaborn as sns

import astropy.units as u
c = 299792.458 * u.Unit('km/s')
c_kms = 299792.458

default_width = 85.  # kpc in projection

axes_label_dict = {'density':'log Density [g / cm$^3$]',
                   'Dark_Matter_Density':'log DM Density [g / cm$^3$]',
                   'temperature': 'log Temperature [K]',
                   'cell_mass': 'log Cell Mass [Msun]',
                   'x': 'X coordinate [comoving]',
                   'y': 'Y coordinate [comoving]',
                   'z': 'Z coordinate [comoving]',
                   'x-velocity': 'X velocity [units?]',
                   'y-velocity': 'Y velocity [units?]',
                   'z-velocity': 'Z velocity [units?]',
                   'O_p5_ion_fraction':'log [O VI Ionization Fraction]',
                   'O_p5_number_density':'log [O VI Number Density]',
                   'C_p3_ion_fraction':'log [C IV Ionization Fraction]',
                   'C_p3_number_density':'log [C IV Number Density]',
                   'Si_p3_ion_fraction':'log [Si IV Ionization Fraction]',
                   'Si_p3_number_density':'log [Si IV Number Density]',
                   }

#this is a dictionary of fields where we prefer to
# plot or visualize them in the log rather than the original yt / enzo
# field. Try "if field_name in logfields: field_name = log10(field_name)"
logfields = ('Dark_Matter_Density', 'density', 'temperature', 'entropy', 'O_p5_ion_fraction',
                'C_p3_ion_fraction', 'Si_p3_ion_fraction', 'O_p5_number_density',
                'C_p3_number_density', 'Si_p3_number_density')

phase_color_key = {b'cold':'salmon',
                   b'hot':'#ffe34d',
                   b'warm':'#4daf4a',
                   b'cool':'#984ea3'}

metal_color_key = {b'high':'darkorange',
                   b'solar':'#ffe34d',
                   b'low':'#4575b4',
                   b'poor':'black'}

species_dict =  {'CIII':'C_p2_number_density',
                 'CIV':'C_p3_number_density',
                 'HI':'H_p0_number_density',
                 'MgII':'Mg_p1_number_density',
                 'OVI':'O_p5_number_density',
                 'SiII':"Si_p1_number_density",
                 'SiIII':"Si_p2_number_density",
                 'SiIV':"Si_p3_number_density",
                 'NeVIII':'Ne_p7_number_density',
                 'FeXIV':'Fe_p13_number_density'}

ion_frac_color_key = {b'all':'black',
                      b'low':'yellow',
                      b'med':'orange',
                      b'high':'red'}

discrete_cmap = mpl.colors.ListedColormap(['#565656','#4daf4a',"#d73027","#984ea3","#ffe34d",'#4575b4','darkorange'])
discrete_cmap_rainbow = mpl.colors.ListedColormap(['#4daf4a',"#ffe34d",'darkorange',"#d73027","#984ea3",'#4575b4','#565656'])

density_color_map = sns.blend_palette(("black","#984ea3","#d73027","darkorange","#ffe34d","#4daf4a","white"), n_colors=60, as_cmap=True)
density_proj_min = 5e-2  ## msun / pc^2
density_proj_max = 1e4
density_slc_min = 5e-8  ## msun / pc^3
density_slc_max = 5

metal_color_map = sns.blend_palette(("black","#984ea3","#4575b4","#4daf4a",
                    "#ffe34d","darkorange"), as_cmap=True)
metal_min = 1.e-4
metal_max = 3.
metal_density_min = 1.e-5
metal_density_max = 250.

temperature_color_map = sns.blend_palette(("black","#d73027","darkorange","#ffe34d"), n_colors=50, as_cmap=True)
temperature_min = 5.e6
temperature_max = 1.e4

entropy_color_map = "Spectral"
entropy_min = 1.e-4
entropy_max = 1.e3

pressure_color_map = "Spectral"
pressure_min = 1.e-16
pressure_max = 1.e-9

h1_color_map = sns.blend_palette(("white","#ababab","#565656","black","#4575b4","#984ea3","#d73027","darkorange","#ffe34d"), as_cmap=True)
h1_proj_min = 1.e12
h1_proj_max = 1.e24
h1_slc_min = 1.e-14
h1_slc_max = 1.e2

o6_color_map = sns.blend_palette(("white","black","#4daf4a","#4575b4","#984ea3","#d73027","darkorange"), as_cmap=True)
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

def categorize_by_temp(temp):
    """ define the temp category strings"""
    phase = np.chararray(np.size(temp), 4)
    phase[temp < 9.] = b'hot'
    phase[temp < 6.] = b'warm'
    phase[temp < 5.] = b'cool'
    phase[temp < 4.] = b'cold'
    return phase

def categorize_by_fraction(f_ion):
    """ define the ionization category strings"""
    frac = np.chararray(np.size(f_ion), 4)
    frac[f_ion > -10.] = b'all'
    frac[f_ion > 0.01] = b'low' # yellow
    frac[f_ion > 0.1] = b'med'  # orange
    frac[f_ion > 0.2] = b'high' # red
    return frac

def categorize_by_metallicity(metallicity):
    """ define the metallicity category strings"""
    metal_label = np.chararray(np.size(metallicity), 5)
    metal_label[metallicity < 10.] = b'high'
    metal_label[metallicity < 0.05] = b'solar'
    metal_label[metallicity < 0.001] = b'low'
    metal_label[metallicity < 0.0001] = b'poor'
    return metal_label


temp_colors = sns.blend_palette(('salmon',"#984ea3","#4daf4a","#ffe34d",'darkorange'), n_colors=17)
new_phase_color_key = {b'cold':to_hex(temp_colors[0]),
                    b'cold1':to_hex(temp_colors[1]),
                    b'cold2':to_hex(temp_colors[2]),
                    b'cold3':to_hex(temp_colors[3]),
                    b'cool':to_hex(temp_colors[4]), ## purple
                    b'cool1':to_hex(temp_colors[5]),
                    b'cool2':to_hex(temp_colors[6]),
                    b'cool3':to_hex(temp_colors[7]),
                    b'warm':to_hex(temp_colors[8]), ## green
                    b'warm1':to_hex(temp_colors[9]),
                    b'warm2':to_hex(temp_colors[10]),
                    b'warm3':to_hex(temp_colors[11]),
                    b'hot':to_hex(temp_colors[12]), ## yellow
                    b'hot1':to_hex(temp_colors[13]),
                    b'hot2':to_hex(temp_colors[14]),
                    b'hot3':to_hex(temp_colors[15])
}

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
