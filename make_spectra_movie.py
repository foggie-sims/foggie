import trident
import numpy as np
import yt
import MISTY
import sys
import os

import matplotlib as mpl
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
mpl.rcParams['font.family'] = 'stixgeneral'
mpl.rcParams['font.size'] = 6.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid

from astropy.table import Table
from astropy.io import fits

from modular_plots import get_refine_box
from consistency import *

def extract_spectra(ds, impact, **kwargs):
    read_fits_file = kwargs.get('read_fits_file', False)
    out_fits_name = kwargs.get('out_fits_name', "temp.fits")
    xmin = kwargs.get("xmin", 0)
    xmax = kwargs.get("xmax", 1)
    halo_center = kwargs.get("halo_center", [0.5, 0.5, 0.5])
    if read_fits_file:
        print "opening ", out_fits_name
        hdulist = fits.open(out_fits_name)
        ray_start_str, ray_end_str = hdulist[0].header['RAYSTART'], hdulist[0].header['RAYEND']
        ray_start = [float(ray_start_str.split(",")[0]), \
               float(ray_start_str.split(",")[1]), \
               float(ray_start_str.split(",")[2])]
        ray_end = [float(ray_end_str.split(",")[0]), \
               float(ray_end_str.split(",")[1]), \
               float(ray_end_str.split(",")[2])]
        return hdulist, ray_start, ray_end
    else:
        proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') / ds.get_parameter('CosmologyHubbleConstantNow') * 1000. # in kpc
        # line_list = ['H I 1216', 'Ly b', 'Ly c', 'Ly d', 'Ly 10', 'Si II 1260', 'C IV 1548', 'O VI 1032']
        line_list = ['H I 1216', 'H I 1026', 'H I 973', 'H I 950', 'H I 919', 'Si II 1260', 'Si III 1207', 'C II 1335', 'C IV 1548', 'O VI 1032']

        ray_start = np.zeros(3)
        ray_end = np.zeros(3)
        ray_start[0] = xmin
        ray_end[0] = xmax
        ray_start[1] = halo_center[1] - (impact/proper_box_size)
        ray_end[1] = ray_start[1]
        ray_start[2] = halo_center[2]
        ray_end[2] = ray_start[2]
        rs, re = np.array(ray_start), np.array(ray_end)
    #    out_fits_name = "hlsp_misty_foggie_"+haloname+"_"+ds.basename.lower()+"_i"+"{:05.1f}".format(impacts[i]) + \
    #                    "_dx"+"{:4.2f}".format(dx[i])+"_v2_los.fits"
        rs = ds.arr(rs, "code_length")
        re = ds.arr(re, "code_length")
        ray = ds.ray(rs, re)

        triray = trident.make_simple_ray(ds, start_position=rs.copy(),
                          end_position=re.copy(),
                          data_filename="test.h5",
                          lines=line_list,
                          ftype='gas')

        hdulist = MISTY.write_header(triray,start_pos=ray_start,end_pos=ray_end,
                        lines=line_list, impact=impact)
        tmp = MISTY.write_parameter_file(ds,hdulist=hdulist)
        for line in line_list:
            sg = MISTY.generate_line(triray,line,write=True,use_spectacle=False,hdulist=hdulist)
            MISTY.write_out(hdulist,filename=out_fits_name)

        return hdulist, ray_start, ray_end

def make_movie():
    # load the simulation
#     ds = yt.load("/Users/molly/foggie/halo_008508/nref11n_nref10f_refine200kpc_z4to2/RD0020/RD0020")
#     track_name = "/Users/molly/foggie/halo_008508/nref11n_nref10f_refine200kpc_z4to2/halo_track"
#     output_dir = "/Users/molly/Dropbox/foggie-collab/plots/halo_008508/nref11_refine200kpc_z4to2/spectra/"
    ds = yt.load("/Users/molly/foggie/halo_008508/natural/nref11/RD0020/RD0020")
    track_name = "/Users/molly/foggie/halo_008508/nref11n_nref10f_refine200kpc_z4to2/halo_track"
    output_dir = "/Users/molly/Dropbox/foggie-collab/plots/halo_008508/natural/nref11/spectra/"
    os.chdir(output_dir)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    # interpolate the center from the track
    centerx = 0.5 * ( np.interp(zsnap, track['col1'], track['col2']) + np.interp(zsnap, track['col1'], track['col5']))
    centery = 0.5 * ( np.interp(zsnap, track['col1'], track['col3']) + np.interp(zsnap, track['col1'], track['col6']))
    centerz = 0.5 * ( np.interp(zsnap, track['col1'], track['col4']) + np.interp(zsnap, track['col1'], track['col7']))
    halo_center = [centerx, centery, centerz]
    xmin = np.interp(zsnap, track['col1'], track['col2'])
    xmax = np.interp(zsnap, track['col1'], track['col5'])
    ymin = np.interp(zsnap, track['col1'], track['col3'])
    ymax = np.interp(zsnap, track['col1'], track['col6'])
    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') / ds.get_parameter('CosmologyHubbleConstantNow') * 1000. # in kpc
    x_width = np.abs((np.interp(zsnap, track['col1'], track['col2']) - np.interp(zsnap, track['col1'], track['col5'])))


    # slice will be repeated, so let's make it first
    slc = yt.SlicePlot(ds,'z',('gas','density'),center=halo_center,width=x_width)
    ## slc = ds.r[xmin:xmax, ymin:ymax, halo_center[2]]
    res = [1000,1000]
    # frb = slc.frb(x_width, res)
    frb = slc.frb['gas','density']
    # image = np.array(frb['gas', 'density'])
    image = np.array(frb)
    print "min, max density = ", np.min(np.log10(image)), np.max(np.log10(image))
    # extent = [float(x.in_units('code_length')) for x in (pro.xlim + pro.ylim)]
    extent = [xmin, xmax, ymin, ymax]


    # get a spectrum!
    # impact = 25.
    impacts = np.arange(25.,45.,0.5)
    for impact in impacts:
        out_fits_name = "hlsp_misty_foggie_halo008508_"+ds.basename.lower()+"_i"+"{:05.1f}".format(impact) + \
                    "_dx"+"{:4.2f}".format(0.)+"_v2_los.fits"
        out_plot_name = "slice_with_spectra_halo008508_"+ds.basename.lower()+"_i"+"{:05.1f}".format(impact) + \
                    "_dx"+"{:4.2f}".format(0.)+".png"
        hdulist, ray_start, ray_end = extract_spectra(ds, impact, read_fits_file=True,
                                out_fits_name=out_fits_name,
                                xmin=xmin, xmax=xmax, halo_center=halo_center)
        print "(impact/proper_box_size)/x_width = ", (impact/proper_box_size)/x_width

        # start by setting up plots
        fig = plt.figure(figsize=(12,6), dpi=100)

        # creates grid on which the figure will be plotted
        gs = gridspec.GridSpec(3, 2)

        ## this will be the slice
        ax_slice = fig.add_subplot(gs[:,0])
        ax_slice.imshow(np.log10(image), extent=extent, cmap=density_color_map, \
                                        vmin = -29.6, vmax=-23.5)
        ax_slice.plot([ray_start[0], ray_end[0]], [ray_start[1], ray_end[1]], color="white", lw=2.)

        ## print ray_start, ray_end, extent
        ax_slice.set_aspect('equal')

        zmin, zmax = 1.998, 2.004
        ## these will be the spectra
        ax_spec1 = fig.add_subplot(gs[0,1])
        ax_spec1.plot(hdulist["H I 1216"].data['redshift'], hdulist["H I 1216"].data['flux'], color="#4575b4",lw=2)
        ax_spec1.text(zmin+0.0003, 0, "H I 1216", fontsize=12.)
        plt.xlim(zmin, zmax)
        plt.ylim(-0.05, 1.05)

        ## these will be the spectra
        ax_spec2 = fig.add_subplot(gs[1,1])
        ax_spec2.plot(hdulist["C IV 1548"].data['redshift'], hdulist["C IV 1548"].data['flux'], color="#4575b4",lw=2)
        ax_spec2.text(zmin+0.0003, 0, "C IV 1548", fontsize=12.)
        plt.xlim(zmin, zmax)
        plt.ylim(-0.05, 1.05)

        ## these will be the spectra
        ax_spec3 = fig.add_subplot(gs[2,1])
        ax_spec3.plot(hdulist["O VI 1032"].data['redshift'], hdulist["O VI 1032"].data['flux'], color="#4575b4",lw=2)
        ax_spec3.text(zmin+0.0003, 0, "O VI 1032", fontsize=12.)
        plt.xlim(zmin, zmax)
        plt.ylim(-0.05, 1.05)

        fig.tight_layout()
        plt.savefig(out_plot_name)
        plt.close()


if __name__ == "__main__":
    make_movie()
    sys.exit("~~~*~*~*~*~*~all done!!!! spectra are fun!")
