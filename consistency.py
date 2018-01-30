import matplotlib as mpl
mpl.use('agg')
import seaborn as sns

default_width = 50.  # kpc in projection

phase_color_key = {b'cold':'salmon',
                   b'hot':'#ffe34d',
                   b'warm':'#4daf4a',
                   b'cool':'#984ea3'}

density_color_map = sns.blend_palette(("black","#984ea3","#d73027","darkorange","#ffe34d","#4daf4a","white"), n_colors=60, as_cmap=True)
density_proj_min = 5e-2  ## msun / pc^2
density_proj_max = 1e4
density_slc_min = 5e-7  ## msun / pc^3
density_slc_max = 5

metal_color_map = sns.blend_palette(("black","#984ea3","#4575b4","#4daf4a","#ffe34d","darkorange"), as_cmap=True)
metal_min = 1.e-4
metal_max = 2.

temperature_color_map = sns.blend_palette(("black","#d73027","darkorange","#ffe34d"), n_colors=50, as_cmap=True)
temperature_min = 5.e6
temperature_max = 1.e4

entropy_color_map = "rainbow"
entropy_min = 1.e-4
entropy_max = 1.e3

h1_color_map = sns.blend_palette(("white","#ababab","#565656","black","#4575b4","#984ea3","#d73027","darkorange","#ffe34d"), as_cmap=True)
h1_proj_min = 1.e11
h1_proj_max = 1.e21
h1_slc_min = 1.e-14
h1_slc_max = 1.e2

o6_color_map = sns.blend_palette(("white","black","#4daf4a","#4575b4","#984ea3","#d73027","darkorange"), as_cmap=True)
o6_min = 1.e11
o6_max = 1.e15

c4_color_map = "algae"
c4_min = 1.e11
c4_max = 1.e15

si2_color_map = "algae"
si2_min = 1.e10
si2_max = 1.e16

si3_color_map = "algae"
si3_min = 1.e11
si3_max = 1.e16
