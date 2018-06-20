import yt
yt.enable_parallelism()

ds = yt.load("DD0007/DD0007")
# v, c = ds.find_max("density")
# print(v, c)
s = yt.SlicePlot(ds, "x", "density", center=[0.5, 0.5, 0.5])
s.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
s.save()

p = yt.ProjectionPlot(ds, "x", "density", center=[0.5, 0.5, 0.5])
p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
p.save()
