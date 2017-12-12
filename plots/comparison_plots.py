import yt
from yt.units import kpc, Msun, km, s, cm, g, Gyr, yr
from yt.mods import *

import trident

import matplotlib as mpl

import os
import numpy as np

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'stixgeneral'

def make_plots(ds, c, boxside):
    box = ds.box(c-boxside,c+boxside)
    c = ds.arr(c,'code_length')
    basename = ds.basename + "_nref" + str(ds.get_parameter('MaximumRefinementLevel'))
    width = (197./ds.hubble_constant)/(1+ds.current_redshift)
    print "width = ", width, "kpc"

    # for axis in ("x", "y", "z"):
    if False:
        for axis in ("y"):
            dpy = yt.ProjectionPlot(ds,axis,('gas','density'),center=c,width=(width,"kpc"),data_source=box)
            cmap = sns.blend_palette(("black","#984ea3","#d73027","darkorange","#ffe34d","#4daf4a","white"), n_colors=60, as_cmap=True)
            dpy.set_zlim(("gas","density"),5e-2,1e4)
            dpy.set_cmap(("gas","density"), cmap)
            dpy.annotate_scale(size_bar_args={'color':'white'})
            dpy.hide_axes()
            dpy.set_unit(('gas','density'),'Msun/pc**2')
            plotname = dpy.save()
            newname = plotname[0].replace(ds.basename, basename)
            os.rename(plotname[0], newname)

            if True:
                dty = yt.ProjectionPlot(ds,axis,('gas','temperature'),weight_field=("gas","density"), center=c,width=(width,"kpc"),data_source=box)
                dty.set_zlim(("gas","temperature"),1e3, 6e5)
                cmap = sns.blend_palette(("black","#d73027","darkorange","#ffe34d"), n_colors=50, as_cmap=True)
                dty.set_cmap(("gas", "temperature"),cmap)
                plotname = dty.save()
                newname = plotname[0].replace(ds.basename, basename)
                os.rename(plotname[0], newname)

                dpz = yt.ProjectionPlot(ds,axis,('gas','metallicity'),weight_field=("gas","density"), center=c,width=(width,"kpc"),data_source=box)
                dpz.set_zlim(("gas","metallicity"),0.005,1.5)
                cmap = sns.blend_palette(("black","#984ea3","#4575b4","#4daf4a","#ffe34d","darkorange"), as_cmap=True)
                dpz.set_cmap(("gas", "metallicity"),cmap)
                plotname = dpz.save()
                newname = plotname[0].replace(ds.basename, basename)
                os.rename(plotname[0], newname)

    trident.add_ion_fields(ds, ions=['O VI'])
    trident.add_ion_fields(ds, ions=['C II'])
    for axis in "y":
        ### HI
        dph = yt.ProjectionPlot(ds,axis,('gas','H_p0_number_density'),center=c,width=(width,"kpc"),data_source=box)
        dph.set_zlim(("gas","H_p0_number_density"),1e13,1e23)
        cmap = sns.blend_palette(("white","#ababab","#565656","black","#4575b4","#984ea3","#d73027","darkorange","#ffe34d"), as_cmap=True)
        dph.set_cmap(("gas", "H_p0_number_density"),cmap)
        dph.annotate_scale(size_bar_args={'color':'black'})
        # dph.annotate_marker((25, 25), coord_system='plot')
        dph.hide_axes()
        # dph.hide_colorbar()
        p = dph.plots['H_p0_number_density']
        colorbar = p.cb(orientation='horizontal')
        dph._setup_plots()
        colorbar.set_ticks([1e13,1e15,1e17,1e19,1e21,1e23])
        colorbar.set_ticklabels(['13','15','17','19','21','23'])
        colorbar.ax.tick_params(labelsize=20)
        plotname = dph.save()
        newname = plotname[0].replace(ds.basename, basename)
        os.rename(plotname[0], newname)

        ### CII
        dpo = yt.ProjectionPlot(ds,axis,('gas','C_p1_number_density'),center=c,width=(width,"kpc"),data_source=box)
        dpo.set_zlim(("gas","C_p1_number_density"),1e5,1e20)
        cmap = sns.blend_palette(("white","#ababab","#565656","black","#4575b4","#984ea3","#d73027","darkorange","#ffe34d"), as_cmap=True)
        dpo.set_cmap(("gas", "C_p1_number_density"),cmap)
        dpo.hide_axes()
        plotname = dpo.save()
        newname = plotname[0].replace(ds.basename, basename)
        os.rename(plotname[0], newname)

        ### OVI
        dpo = yt.ProjectionPlot(ds,axis,('gas','O_p5_number_density'),center=c,width=(width,"kpc"),data_source=box)
        dpo.set_zlim(("gas","O_p5_number_density"),1e12,1e15)
        cmap = sns.blend_palette(("white","black","#4daf4a","#4575b4","#984ea3","#d73027","darkorange"), as_cmap=True)
        dpo.set_cmap(("gas", "O_p5_number_density"),cmap)
        dpo.hide_axes()
        plotname = dpo.save()
        newname = plotname[0].replace(ds.basename, basename)
        os.rename(plotname[0], newname)

    if False:
        dTm = yt.PhasePlot(box,('gas', 'density'),('gas','temperature'),['cell_mass'],weight_field=None, x_bins=512, y_bins=512)
        dTm.set_cmap("cell_mass","plasma")
        dTm.set_xlim(5e-31,1e-20)
        dTm.set_ylim(100,3e8)
        dTm.set_zlim("cell_mass",1e32,5e40)
        plotname = dTm.save()
        newname = plotname[0].replace(ds.basename, basename)
        os.rename(plotname[0], newname)


if __name__ == "__main__":

    # args = parse_args()
    boxside = np.array([0.001,0.001,0.001])

    c = np.array([0.493970873304, 0.488697442131, 0.502242657694]) # z=2.0
#    c = np.array([0.494760515267, 0.491315845011, 0.501333235879]) # z=2.75


    ds = yt.load("/astro/simulations/FOGGIE/halo_008508/nref11/RD0020/RD0020")
    make_plots(ds, c, boxside)

#    ds = yt.load("/astro/simulations/FOGGIE/halo_008508/nref10_refine_z4to2/RD0020/RD0020")
#    make_plots(ds, c, boxside)

#    ds = yt.load("/astro/simulations/FOGGIE/halo_008508/nref11_refine200kpc_z4to2/RD0020/RD0020")
#    make_plots(ds, c, boxside)

#    ds = yt.load("/astro/simulations/FOGGIE/halo_008508/nref11_refine200kpc_z4to2/RD0017/RD0017")
#    make_plots(ds, c, boxside)

#    ds = yt.load("/astro/simulations/FOGGIE/halo_008508/nref11/RD0017/RD0017")
#    make_plots(ds, c, boxside)
