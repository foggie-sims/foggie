import matplotlib as mpl
mpl.use('agg')
import seaborn as sns

default_width = 85.  # kpc in projection

axes_label_dict = {'density':'log Density [g / cm$^3$]',
                   'temperature': 'log Temperature [K]',
                   'O_p5_ion_fraction':'log [O VI Ionization Fraction]',
                   'O_p5_number_density':'log [O VI Number Density]',
                   'C_p3_ion_fraction':'log [C IV Ionization Fraction]',
                   'C_p3_number_density':'log [C IV Number Density]',
                   'Si_p3_ion_fraction':'log [Si IV Ionization Fraction]',
                   'Si_p3_number_density':'log [Si IV Number Density]',
                   }

phase_color_key = {b'cold':'salmon',
                   b'hot':'#ffe34d',
                   b'warm':'#4daf4a',
                   b'cool':'#984ea3'}

metal_color_key = {b'high':'yellow',
                   b'solar':'green',
                   b'low':'purple',
                   b'poor':'salmon'}

species_dict =  {'CIII':'C_p2_number_density',
                 'CIV':'C_p3_number_density',
                 'HI':'H_p0_number_density',
                 'MgII':'Mg_p1_number_density',
                 'OVI':'O_p5_number_density',
                 'SiII':"Si_p1_number_density",
                 'SiIII':"Si_p2_number_density",
                 'SiIV':"Si_p3_number_density",
                 'NeVIII':'Ne_p7_number_density'}

ion_frac_color_key = {b'all':'black',
                      b'low':'yellow',
                      b'med':'orange',
                      b'high':'red'}

discrete_cmap = mpl.colors.ListedColormap(['#565656','#4daf4a',"#d73027","#984ea3","#ffe34d",'#4575b4','darkorange'])

density_color_map = sns.blend_palette(("black","#984ea3","#d73027","darkorange","#ffe34d","#4daf4a","white"), n_colors=60, as_cmap=True)
density_proj_min = 5e-2  ## msun / pc^2
density_proj_max = 1e4
density_slc_min = 5e-7  ## msun / pc^3
density_slc_max = 5

metal_color_map = sns.blend_palette(("black","#984ea3","#4575b4","#4daf4a",
                    "#ffe34d","darkorange"), as_cmap=True)
metal_min = 1.e-4
metal_max = 2.
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
c4_max = 1.e15

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
