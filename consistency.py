import seaborn as sns

default_width = 250.  # kpc in projection 

density_color_map = "RdBu"
density_min = 1e-6
density_max = 0.1

metal_color_map = "rainbow"
metal_min = 1.e-4
metal_max = 2.

temperature_color_map = "rainbow"
temperature_min = 5.e6
temperature_max = 1.e4

entropy_color_map = "rainbow"
entropy_min = 1.e-4
entropy_max = 1.e3

h1_color_map = sns.blend_palette(("white","#ababab","#565656","black","#4575b4","#984ea3","#d73027","darkorange","#ffe34d"), as_cmap=True)
h1_min = 1.e11
h1_max = 1.e18

o6_color_map = "algae"
o6_min = 1.e11
o6_max = 1.e15

c4_color_map = "algae"
c4_min = 1.e11
c4_max = 1.e15

si3_color_map = "algae"
si3_min = 1.e11
si3_max = 1.e15
