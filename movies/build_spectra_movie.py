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
#mpl.rcParams['font.size'] = 6.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from astropy.table import Table
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve

from get_proper_box_size import get_proper_box_size
from get_refine_box import get_refine_box
from get_halo_center import get_halo_center
from consistency import *

def get_short_spectrum(ds, ray_start, ray_end, **kwargs):
    out_fits_name = kwargs.get('out_fits_name', "temp.fits")
    impact = kwargs.get('impact', -1.)
    rs, re = np.array(ray_start), np.array(ray_end)
    rs = ds.arr(rs, "code_length")
    re = ds.arr(re, "code_length")
    print "~~~OK FOR THE SPECTRUM", rs, re
    ray = ds.ray(rs, re)
    line_list = ['H I 1216', 'Si II 1260', 'Si III 1207', 'C II 1335', 'C IV 1548', 'O VI 1032']
    triray = trident.make_simple_ray(ds, start_position=rs.copy(),
                      end_position=re.copy(),
                      data_filename="temp.h5",
                      lines=line_list,
                      ftype='gas')

    hdulist = MISTY.write_header(triray, start_pos=ray_start, end_pos=ray_end,
                    lines=line_list, impact=impact)
    tmp = MISTY.write_parameter_file(ds,hdulist=hdulist)
    for line in line_list:
        sg = MISTY.generate_line(triray, line, write=True, use_spectacle=False, hdulist=hdulist)
        MISTY.write_out(hdulist,filename=out_fits_name)

    return hdulist, ray


def extract_spectra(ds, impact, **kwargs):
    read_fits_file = kwargs.get('read_fits_file', False)
    out_fits_name = kwargs.get('out_fits_name', "temp.fits")
    xmin = kwargs.get("xmin", 0)
    xmax = kwargs.get("xmax", 1)
    halo_center = kwargs.get("halo_center", [0.5, 0.5, 0.5])
    refine_box_center = kwargs.get("refine_box_center", halo_center)
    if read_fits_file:
        print "opening ", out_fits_name
        hdulist = fits.open(out_fits_name)
        ray_start_str, ray_end_str = hdulist[0].header['RAYSTART'], hdulist[0].header['RAYEND']
        ray_start = [float(ray_start_str.split(",")[0].strip('unitary')), \
               float(ray_start_str.split(",")[1].strip('unitary')), \
               float(ray_start_str.split(",")[2].strip('unitary'))]
        ray_end = [float(ray_end_str.split(",")[0].strip('unitary')), \
               float(ray_end_str.split(",")[1].strip('unitary')), \
               float(ray_end_str.split(",")[2].strip('unitary'))]
        rs, re = np.array(ray_start), np.array(ray_end)
        rs = ds.arr(rs, "code_length")
        re = ds.arr(re, "code_length")
        ray = ds.ray(rs, re)

        return hdulist, ray_start, ray_end, ray
    else:
        proper_box_size = get_proper_box_size(ds)
        # line_list = ['H I 1216', 'Ly b', 'Ly c', 'Ly d', 'Ly 10', 'Si II 1260', 'C IV 1548', 'O VI 1032']
        line_list = ['H I 1216', 'Si II 1260', 'Si III 1207', 'C II 1335', 'C IV 1548', 'O VI 1032']

        ray_start = np.zeros(3)
        ray_end = np.zeros(3)
        ray_start[0] = xmin
        ray_end[0] = xmax
        ray_start[1] = refine_box_center[1] + (impact/proper_box_size)
        ray_end[1] = refine_box_center[1] + (impact/proper_box_size)
        ray_start[2] = refine_box_center[2]
        ray_end[2] = refine_box_center[2]
        rs, re = np.array(ray_start), np.array(ray_end)
    #    out_fits_name = "hlsp_misty_foggie_"+haloname+"_"+ds.basename.lower()+"_i"+"{:05.1f}".format(impacts[i]) + \
    #                    "_dx"+"{:4.2f}".format(dx[i])+"_v2_los.fits"
        rs = ds.arr(rs, "code_length")
        re = ds.arr(re, "code_length")
        ray = ds.ray(rs, re)

        triray = trident.make_simple_ray(ds, start_position=rs.copy(),
                          end_position=re.copy(),
                          data_filename="temp.h5",
                          lines=line_list,
                          ftype='gas')

        hdulist = MISTY.write_header(triray, start_pos=ray_start, end_pos=ray_end,
                        lines=line_list, impact=impact)
        tmp = MISTY.write_parameter_file(ds,hdulist=hdulist)
        for line in line_list:
            sg = MISTY.generate_line(triray, line, write=True, use_spectacle=False, hdulist=hdulist)
            MISTY.write_out(hdulist,filename=out_fits_name)

        return hdulist, ray_start, ray_end, ray

def make_movie():
    # load the simulation
    ds = yt.load("/Users/molly/foggie/halo_008508/nref11n/nref11n_nref10f_refine200kpc/RD0020/RD0020")
    output_dir = "/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/"
    # ds = yt.load("/Users/molly/foggie/halo_008508/nref11n/natural/RD0020/RD0020")
    track_name = "/Users/molly/foggie/halo_008508/nref11n/nref11n_nref10f_refine200kpc/halo_track"
    # output_dir = "/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/natural/spectra/"
    os.chdir(output_dir)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    proper_box_size = get_proper_box_size(ds)
    refine_box, refine_box_center, x_width = get_refine_box(ds, zsnap, track)
    halo_center, v = get_halo_center(ds, refine_box_center)
    x_left = np.float(refine_box.left_edge[0].value)
    x_right = np.float(refine_box.right_edge[0].value)
    y_left = np.float(refine_box.left_edge[1].value)
    y_right = np.float(refine_box.right_edge[1].value)


    # slice will be repeated, so let's make it first
    print refine_box_center[0], refine_box_center[1], halo_center[2]
    #z_center = [refine_box_center[0], refine_box_center[1], halo_center[2]]
    z_center = refine_box_center
    slc = yt.SlicePlot(ds, 'z', ('gas','metallicity'), center=z_center, width=x_width)
    ## slc = ds.r[xmin:xmax, ymin:ymax, halo_center[2]]
    res = [1000,1000]
    # frb = slc.frb(x_width, res)
    frb = slc.frb['gas','metallicity']
    # image = np.array(frb['gas', 'density'])
    image = np.array(frb)
    print "min, max H I density = ", np.min(np.log10(image)), np.max(np.log10(image))
    # extent = [float(x.in_units('code_length')) for x in (pro.xlim + pro.ylim)]
    extent = [x_left, x_right, y_left, y_right]

    # spectral features
    #g = Gaussian1DKernel((7/0.0267)/2.355)  # HIRES v_fwhm = 7 km/s
    #snr = 30.


    ### start with the full ray
    impact = -20.
    out_fits_name = "hlsp_misty_foggie_halo008508_"+ds.basename.lower()+"_i"+"{:05.1f}".format(impact) + \
                "_dy"+"{:4.2f}".format(0.)+"_build_lots_los.fits"
    out_plot_name = "metallicity_slice_build_spectrum_halo008508_"+ds.basename.lower()+"_i"+"{:05.1f}".format(impact) + \
                "_dy"+"{:4.2f}".format(0.)+".png"
    hdulist, ray_start, ray_end, ray = extract_spectra(ds, impact, read_fits_file=True,
                            out_fits_name=out_fits_name,
                            xmin=x_left, xmax=x_right, halo_center=halo_center, refine_box_center=refine_box_center)
    ray_sort = np.argsort(ray['t'])
    ray_min = np.min(ray['x'][ray_sort])
    ray_max = np.max(ray['x'][ray_sort])
    print "~~~~~~~~~~~~~~THIS IS THE ORIGINAL RAY START AND END:", ray_start, ray_end
    full_ray_end = ray_end

    ### OK, now we are going to loop over a range of dx's
    fracs = np.arange(1.0,0.00,-0.01)
    print fracs
    for dx in fracs:
        this_ray_end = np.zeros(3)
        this_ray_end[0] = ray_start[0] + (full_ray_end[0] - ray_start[0])*dx
        this_ray_end[1] = full_ray_end[1]
        this_ray_end[2] = full_ray_end[2]
        print (full_ray_end[0] - ray_start[0])*dx
#        print "THIS IS THE RAY START AND END:", ray_start, this_ray_end
        print "dx:", dx
#        print "ray start:", ray_start
        print "ray end:", this_ray_end[0]

        out_fits_name = "hlsp_misty_foggie_halo008508_"+ds.basename.lower()+"_i"+"{:05.1f}".format(impact) + \
                    "_dy"+"{:4.2f}".format(0.)+"_build_lots_los_dx"+"{:0.3f}".format(dx)+".fits"
        out_plot_name = "metallicity_slice_build_spectrum_halo008508_"+ds.basename.lower()+"_i"+"{:05.1f}".format(impact) + \
                    "_dy"+"{:4.2f}".format(0.)+"_dx"+"{:0.3f}".format(dx)+".png"
        hdulist, ray = get_short_spectrum(ds, ray_start, this_ray_end, out_fits_name=out_fits_name, impact=impact)
        ray_sort = np.argsort(ray['t'])
        print "(impact/proper_box_size)/x_width = ", (impact/proper_box_size)/x_width

        # print "refine box center:", refine_box_center
        # print "halo center:", halo_center
        print "z_center:", z_center
        print "refine left:", refine_box.left_edge
        print "refine right:", refine_box.right_edge

        # start by setting up plots
        fig = plt.figure(figsize=(16,6), dpi=100)

        # creates grid on which the figure will be plotted
        gs = gridspec.GridSpec(6, 3,
                               width_ratios=[0.05, 1, 1.5])

        ## this will be the slice
        ax_slice = fig.add_subplot(gs[:,1])
        slc = ax_slice.imshow(np.log10(image), extent=extent, cmap=metal_color_map, \
                                        vmin=np.log10(metal_min), vmax=np.log10(metal_max))
        ax_slice.plot([ray_start[0], this_ray_end[0]], [ray_start[1], this_ray_end[1]], color="white", lw=2.)
        ax_slice.set_aspect('equal')
        ax_slice.xaxis.set_major_locator(ticker.NullLocator())
        ax_slice.yaxis.set_major_locator(ticker.NullLocator())
        # cbar = fig.colorbar(cax, ax=ax_cbar, orientation='vertical', pad=0.01, shrink=0.8)
        ax_cbar = fig.add_subplot(gs[:,0])
        cbar = fig.colorbar(slc, cax=ax_cbar, extend='both')
        ax_cbar.yaxis.set_ticks_position('left')
        ax_cbar.yaxis.set_label_position('left')
        cbar.set_label(r'log metallicity', fontsize=16.)

        ## these will be the spectra
        zmin, zmax = 1.998, 2.004
        vmin, vmax = -1000, 1000

        ax_spec1 = fig.add_subplot(gs[0,2])
        ax_spec1.plot(ray['x'][ray_sort], np.log10(ray['density'][ray_sort]), color='#984ea3',lw=1)
        plt.xlim(ray_min, ray_max)
        plt.ylim(-29, -25)
        ymin, ymax = ax_spec1.get_ylim()
        ax_spec1.text(ray_min + (ray_max - ray_min)*0.05, ymax - (ymax-ymin)*0.2, "density", fontsize=12.)
        ax_spec1.xaxis.set_major_locator(ticker.NullLocator())

        ax_spec2 = fig.add_subplot(gs[1,2])
        ax_spec2.plot(ray['x'][ray_sort], np.log10(ray['temperature'][ray_sort]), color='#984ea3',lw=1)
        plt.xlim(ray_min, ray_max)
        plt.ylim(4,6.5)
        ax_spec2.text(ray_min + (ray_max - ray_min)*0.05, 4.2, "temperature", fontsize=12.)
        ax_spec2.xaxis.set_major_locator(ticker.NullLocator())

        ax_spec3 = fig.add_subplot(gs[2,2])
        velocity = ((hdulist["H I 1216"].data['wavelength'] / hdulist['H I 1216'].header['restwave']) - 1 - ds.current_redshift) * 299792.458 - 250
        flux = hdulist["H I 1216"].data['flux']
        ax_spec3.plot(velocity, flux, color='darkorange',lw=1)
        ax_spec3.text(vmin + 100, 0, "H I 1216", fontsize=12.)
        ax_spec3.xaxis.set_major_locator(ticker.NullLocator())
        plt.xlim(vmin, vmax)
        plt.ylim(-0.05, 1.05)

        ax_spec4 = fig.add_subplot(gs[3,2])
        velocity = ((hdulist["Si III 1207"].data['wavelength'] / hdulist['Si III 1207'].header['restwave']) - 1 - ds.current_redshift) * 299792.458 - 250
        flux = hdulist["Si III 1207"].data['flux']
        ax_spec4.plot(velocity, flux, color='darkorange',lw=1)
        ax_spec4.text(vmin + 100, 0, "Si III 1207", fontsize=12.)
        ax_spec4.xaxis.set_major_locator(ticker.NullLocator())
        plt.xlim(vmin, vmax)
        plt.ylim(-0.05, 1.05)

        ax_spec5 = fig.add_subplot(gs[4,2])
        velocity = ((hdulist["C IV 1548"].data['wavelength'] / hdulist['C IV 1548'].header['restwave']) - 1 - ds.current_redshift) * 299792.458 - 250
        flux = hdulist["C IV 1548"].data['flux']
        ax_spec5.plot(velocity, flux, color='darkorange',lw=1)
        ax_spec5.text(vmin + 100, 0, "C IV 1548", fontsize=12.)
        ax_spec5.xaxis.set_major_locator(ticker.NullLocator())
        plt.xlim(vmin, vmax)
        plt.ylim(-0.05, 1.05)

        ax_spec6 = fig.add_subplot(gs[5,2])
        velocity = ((hdulist["O VI 1032"].data['wavelength'] / hdulist['O VI 1032'].header['restwave']) - 1 - ds.current_redshift) * 299792.458 - 250
        flux = hdulist["O VI 1032"].data['flux']
        ax_spec6.plot(velocity, flux, color='darkorange',lw=1)
        ax_spec6.text(vmin + 100, 0, "O VI 1032", fontsize=12.)
        plt.xlim(vmin, vmax)
        plt.ylim(-0.05, 1.05)

        fig.tight_layout()
        plt.savefig(out_plot_name)
        plt.close()


if __name__ == "__main__":
    make_movie()
    sys.exit("~~~*~*~*~*~*~all done!!!! spectra are fun!")
