from __future__ import print_function

import numpy as np

import yt
from yt.visualization.base_plot_types import get_multi_plot
import trident

import argparse
import os
import glob
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

from astropy.table import Table

from consistency import *
from get_refine_box import get_refine_box
from get_halo_center import get_halo_center
from get_proper_box_size import get_proper_box_size
from get_run_loc_etc import get_run_loc_etc

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams['font.size'] = 30.

from matplotlib.colors import LogNorm

#-----------------------------------------------------------------------------------------------------

old_density_color_map = sns.blend_palette(("black","#984ea3","#d73027","darkorange","#ffe34d","#4daf4a","white"), as_cmap=True)
density_color_map = sns.blend_palette(("black","#984ea3","#4575b4","#4daf4a","#ffe34d","darkorange"), as_cmap=True)
density_color_map = sns.blend_palette(("black","#4575b4","#4daf4a","#ffe34d","darkorange"), as_cmap=True)
density_proj_min = 5e-2  ## msun / pc^2
density_proj_max = 1e4
density_slc_min = 5e-8  ## msun / pc^3
density_slc_max = 0.1

metal_color_map = sns.blend_palette(("black","#4575b4","#984ea3","#984ea3","#d73027","darkorange","#ffe34d"), as_cmap=True)
old_metal_color_map = sns.blend_palette(("black","#984ea3","#4575b4","#4daf4a","#ffe34d","darkorange"), as_cmap=True)
metal_min = 1.e-4
metal_max = 3.
metal_density_min = 1.e-5
metal_density_max = 250.

temperature_color_map = sns.blend_palette(("black","#d73027","darkorange","#ffe34d"), as_cmap=True)
temperature_min = 5.e6
temperature_max = 1.e4


def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    parser = argparse.ArgumentParser(description="makes a bunch of plots")

    ## optional arguments
    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 ')
    parser.set_defaults(halo="8508")

    ## what are we plotting and where is it
    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is natural')
    parser.set_defaults(run="natural")

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")

    parser.add_argument('--system', metavar='system', type=str, action='store',
                        help='which system are you on? default is oak')
    parser.set_defaults(system="oak")

    args = parser.parse_args()
    return args

#-----------------------------------------------------------------------------------------------------

def make_slice_plot_no_colorbar(ds, output_dir, field, zmin, zmax, cmap, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    ision  = kwargs.get("ision", False)
    timestamp = kwargs.get('timestamp', False)
    basename = output_dir + ds.basename + appendix
    for ax in axis:
        if ision:
            print("field = ", species_dict[field])
            s = yt.SlicePlot(ds, ax, species_dict[field], center=center,  width=(1.5*width, 'kpc'))
            s.set_zlim(species_dict[field], zmin, zmax)
            s.set_cmap(field=species_dict[field], cmap=cmap)
        else:
            s = yt.SlicePlot(ds, ax, field, center=center,  width=(1.5*width, 'kpc'))
            s.set_zlim(field, zmin, zmax)
            s.set_cmap(field=field, cmap=cmap)
        if timestamp:
            s.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        if field == "density" or field == "metal_density":
            s.set_unit(('gas','density'),'Msun/pc**3')
        s.annotate_scale(size_bar_args={'color':'white'}, text_args={'size':28})
        s.hide_axes()
        s.hide_colorbar()
        s.save(basename + '_Slice_' + ax + '_' + field + '.png')
        s.save(basename + '_Slice_' + ax + '_' + field + '.pdf')


#-----------------------------------------------------------------------------------------------------

def make_slice_plot(ds, output_dir, field, zmin, zmax, cmap, **kwargs):
    axis = kwargs.get("axis", ['x','y','z']) # if axis not set, do all
    box = kwargs.get("box", "")
    center = kwargs.get("center", "")
    appendix = kwargs.get("appendix", "")
    width = kwargs.get("width", default_width)
    resolution = kwargs.get("resolution", (1048,1048)) # correct for the nref11f box
    ision  = kwargs.get("ision", False)
    timestamp = kwargs.get('timestamp', False)
    basename = output_dir + ds.basename + appendix
    for ax in axis:
        if ision:
            print("field = ", species_dict[field])
            s = yt.SlicePlot(ds, ax, species_dict[field], center=center,  width=(1.5*width, 'kpc'))
            s.set_zlim(species_dict[field], zmin, zmax)
            s.set_cmap(field=species_dict[field], cmap=cmap)
        else:
            s = yt.SlicePlot(ds, ax, field, center=center,  width=(1.5*width, 'kpc'))
            s.set_zlim(field, zmin, zmax)
            s.set_cmap(field=field, cmap=cmap)
        if timestamp:
            s.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        if field == "density" or field == "metal_density":
            s.set_unit(('gas','density'),'Msun/pc**3')
        s.annotate_scale(size_bar_args={'color':'white'})
        s.hide_axes()
        s.save(basename + '_Slice_' + ax + '_' + field + '.png')
        s.save(basename + '_Slice_' + ax + '_' + field + '.pdf')
        #frb = s.data_source.to_frb(width, resolution, center=center)
        #cPickle.dump(frb[field], open(basename + '_Slice_' + ax + '_' + field + '.cpkl','wb'), protocol=-1)

#-----------------------------------------------------------------------------------------------------

def ignore_this_it_is_crap():

    ## OK, set up the grid
    ## adopted from http://yt-project.org/doc/cookbook/complex_plots.html
    orient = 'vertical'
    fig, axes, colorbars = get_multi_plot(2, 3, colorbar=orient, bw = 5, dpi=300)

    slcn = yt.SlicePlot(dsn, 'x', fields=["density","temperature","metallicity"],
                        center=centern,  width=(1.5*width, 'kpc'))
    slcr = yt.SlicePlot(dsr, 'x', fields=["density","temperature","metallicity"],
                        center=centerr,  width=(1.5*width, 'kpc'))

    slcn.set_unit(('gas','density'),'Msun/pc**3')
    slcr.set_unit(('gas','density'),'Msun/pc**3')
    slcn_frb = slcn.data_source.to_frb((1.5*width, 'kpc'), 2048)
    slcr_frb = slcr.data_source.to_frb((1.5*width, 'kpc'), 2048)

    dens_axes = [axes[0][0], axes[0][1]]
    temp_axes = [axes[1][0], axes[1][1]]
    metal_axes = [axes[2][0], axes[2][1]]

    for dax, tax, zax in zip(dens_axes, temp_axes, metal_axes) :
        dax.xaxis.set_visible(False)
        dax.yaxis.set_visible(False)
        tax.xaxis.set_visible(False)
        tax.yaxis.set_visible(False)
        zax.xaxis.set_visible(False)
        zax.yaxis.set_visible(False)

    slc_densn = np.array(slcn_frb['density'])
    slc_tempn = np.array(slcn_frb['temperature'])
    slc_metaln = np.array(slcn_frb['metallicity'])
    slc_densr = np.array(slcr_frb['density'])
    slc_tempr = np.array(slcr_frb['temperature'])
    slc_metalr = np.array(slcr_frb['metallicity'])

    plots = [dens_axes[0].imshow(slc_densn, origin='lower', norm=LogNorm()),
             dens_axes[1].imshow(slc_densr, origin='lower', norm=LogNorm()),
             temp_axes[0].imshow(slc_tempn, origin='lower', norm=LogNorm()),
             temp_axes[1].imshow(slc_tempr, origin='lower', norm=LogNorm()),
             metal_axes[0].imshow(slc_metaln, origin='lower', norm=LogNorm()),
             metal_axes[1].imshow(slc_metalr, origin='lower', norm=LogNorm())]

    plots[0].set_clim((density_slc_min, density_slc_max))
    plots[0].set_cmap(density_color_map)
    plots[1].set_clim((density_slc_min, density_slc_max))
    plots[1].set_cmap(density_color_map)
    plots[2].set_clim((temperature_min, temperature_max))
    plots[2].set_cmap(temperature_color_map)
    plots[3].set_clim((temperature_min, temperature_max))
    plots[3].set_cmap(temperature_color_map)
    plots[4].set_clim((metal_min, metal_max))
    plots[4].set_cmap(metal_color_map)
    plots[5].set_clim((metal_min, metal_max))
    plots[5].set_cmap(metal_color_map)

    titles=[r'$\mathrm{Density}\ (\mathrm{g\ cm^{-3}})$',
            r'$\mathrm{Temperature}\ (\mathrm{K})$',
            r'$\mathrm{Metallicity}\ (\mathrm{Z_{\odot}})$']

    for p, cax, t in zip(plots[0:6:2], colorbars, titles):
        cbar = fig.colorbar(p, cax=cax, orientation=orient)
        cbar.set_label(t)

    outfilename = output_dir + 'six_slices.png'
    fig.savefig(outfilename)


#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    print(args.run, args.output)

    ## actually ...
    args.run = 'natural'
    args.output = 'RD0018'
    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir = get_run_loc_etc(args)
    run_dir = foggie_dir + run_loc
    print('run_dir = ', run_dir)
    outs = args.output + '/' + args.output
    print("outs = ", outs)
    snap = run_dir + outs
    dsn = yt.load(snap)

    args.run = 'nref10f'
    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir = get_run_loc_etc(args)
    run_dir = foggie_dir + run_loc
    print('run_dir = ', run_dir)
    outs = args.output + '/' + args.output
    print("outs = ", outs)
    snap = run_dir + outs
    dsr = yt.load(snap)

    args.run = 'nref11f'
    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir = get_run_loc_etc(args)
    run_dir = foggie_dir + run_loc
    print('run_dir = ', run_dir)
    outs = args.output + '/' + args.output
    print("outs = ", outs)
    snap = run_dir + outs
    dsh = yt.load(snap)

    print("opening track: " + trackname)
    track = Table.read(trackname, format='ascii')
    track.sort('col1')
    zsnap = dsn.get_parameter('CosmologyCurrentRedshift')
    proper_box_size = get_proper_box_size(dsn)

    refine_boxh, refine_box_centerh, refine_width = get_refine_box(dsh, zsnap, track)
    refine_boxr, refine_box_centerr, refine_width = get_refine_box(dsr, zsnap, track)
    refine_boxn, refine_box_centern, refine_width = get_refine_box(dsn, zsnap, track)
    refine_width = refine_width * proper_box_size

    # center is trying to be the center of the halo
    search_radius = 10.
    this_search_radius = search_radius / (1+dsr.get_parameter('CosmologyCurrentRedshift'))
    #centerh, velocity = get_halo_center(dsh, refine_box_centerh, radius=this_search_radius)
    centerh = [0.4946298599243164, 0.49077510833740234, 0.5014429092407227]  # RD0018 nref11f
    centerr, velocity = get_halo_center(dsr, refine_box_centerr, radius=this_search_radius)
    centern, velocity = get_halo_center(dsn, refine_box_centern, radius=this_search_radius)

    print('halo center = ', centerh, ' and refine_box_center = ', refine_box_centerh)
    print('halo center = ', centerr, ' and refine_box_center = ', refine_box_centerr)
    print('halo center = ', centern, ' and refine_box_center = ', refine_box_centern)

    width = default_width
    width_code = width / proper_box_size ## needs to be in code units
    boxh = dsh.r[centerh[0] - 0.5*width_code : centerh[0] + 0.5*width_code, \
              centerh[1] - 0.5*width_code : centerh[1] + 0.5*width_code, \
                  centerh[2] - 0.5*width_code : centerh[2] + 0.5*width_code]
    boxr = dsr.r[centerr[0] - 0.5*width_code : centerr[0] + 0.5*width_code, \
              centerr[1] - 0.5*width_code : centerr[1] + 0.5*width_code, \
                  centerr[2] - 0.5*width_code : centerr[2] + 0.5*width_code]
    boxn = dsn.r[centern[0] - 0.5*width_code : centern[0] + 0.5*width_code, \
              centern[1] - 0.5*width_code : centern[1] + 0.5*width_code, \
                  centern[2] - 0.5*width_code : centern[2] + 0.5*width_code]


    output_dir = '/Users/molly/Dropbox/foggie-collab/papers/absorption_peeples/Figures/'


    axis = 'z'
    make_slice_plot_no_colorbar(dsh, output_dir, "density", \
                    density_slc_min, density_slc_max, density_color_map, \
                    ision=False, axis=axis, center=centerh, box=refine_boxh, \
                    width=refine_width, appendix="_nref11f")

    make_slice_plot_no_colorbar(dsh, output_dir, "metallicity", \
                    metal_min, metal_max, metal_color_map, \
                    ision=False, axis=axis, center=centerh, box=refine_boxh, \
                    width=refine_width, appendix="_nref11f")

    make_slice_plot_no_colorbar(dsh, output_dir, "temperature", \
                    temperature_min, temperature_max, temperature_color_map, \
                    ision=False, axis=axis, center=centerh, box=refine_boxh, \
                    width=refine_width, appendix="_nref11f")

    make_slice_plot_no_colorbar(dsn, output_dir, "density", \
                    density_slc_min, density_slc_max, density_color_map, \
                    ision=False, axis=axis, center=centern, box=refine_boxn, \
                    width=refine_width, appendix="_natural", timestamp=False)

    make_slice_plot_no_colorbar(dsn, output_dir, "metallicity", \
                    metal_min, metal_max, metal_color_map, \
                    ision=False, axis=axis, center=centerr, box=refine_boxn, \
                    width=refine_width, appendix="_natural")

    make_slice_plot_no_colorbar(dsn, output_dir, "temperature", \
                    temperature_min, temperature_max, temperature_color_map, \
                    ision=False, axis=axis, center=centern, box=refine_boxn, \
                    width=refine_width, appendix="_natural")

    make_slice_plot_no_colorbar(dsr, output_dir, "density", \
                    density_slc_min, density_slc_max, density_color_map, \
                    ision=False, axis=axis, center=centerr, box=refine_boxr, \
                    width=refine_width, appendix="_nref10f")

    make_slice_plot_no_colorbar(dsr, output_dir, "metallicity", \
                    metal_min, metal_max, metal_color_map, \
                    ision=False, axis=axis, center=centerr, box=refine_boxr, \
                    width=refine_width, appendix="_nref10f")

    make_slice_plot_no_colorbar(dsr, output_dir, "temperature", \
                    temperature_min, temperature_max, temperature_color_map, \
                    ision=False, axis=axis, center=centerr, box=refine_boxr, \
                    width=refine_width, appendix="_nref10f")
