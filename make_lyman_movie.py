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
import matplotlib.ticker as ticker

from astropy.table import Table
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve

from get_proper_box_size import get_proper_box_size
from modular_plots import get_refine_box
from get_halo_center import get_halo_center
from consistency import *

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
        ray_start = [float(ray_start_str.split(",")[0]), \
               float(ray_start_str.split(",")[1]), \
               float(ray_start_str.split(",")[2])]
        ray_end = [float(ray_end_str.split(",")[0]), \
               float(ray_end_str.split(",")[1]), \
               float(ray_end_str.split(",")[2])]
        return hdulist, ray_start, ray_end
    else:
        proper_box_size = get_proper_box_size(ds)
        # line_list = ['H I 1216', 'Ly b', 'Ly c', 'Ly d', 'Ly 10', 'Si II 1260', 'C IV 1548', 'O VI 1032']
        line_list = ['H I 1216', 'H I 1026', 'H I 973', 'H I 950', 'H I 919', 'Si II 1260', 'Si III 1207', 'C II 1335', 'C IV 1548', 'O VI 1032']

        ray_start = np.zeros(3)
        ray_end = np.zeros(3)
        ray_start[0] = xmin
        ray_end[0] = xmax
        ray_start[1] = halo_center[1] + (impact/proper_box_size)
        ray_end[1] = halo_center[1] + (impact/proper_box_size)
        ray_start[2] = halo_center[2]
        ray_end[2] = halo_center[2]
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
            sg = MISTY.generate_line(triray,line,write=True,use_spectacle=True,hdulist=hdulist)
            MISTY.write_out(hdulist,filename=out_fits_name)

        return hdulist, ray_start, ray_end

def make_movie():
    # load the simulation
    ds = yt.load("/Users/molly/foggie/halo_008508/nref11n/nref11n_nref10f_refine200kpc_z4to2/RD0020/RD0020")
    output_dir = "/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc_z4to2/spectra/"
    # ds = yt.load("/Users/molly/foggie/halo_008508/natural/nref11/RD0020/RD0020")
    track_name = "/Users/molly/foggie/halo_008508/nref11n/nref11n_nref10f_refine200kpc_z4to2/halo_track"
    # output_dir = "/Users/molly/Dropbox/foggie-collab/plots_halo_008508/nref11n/natural/spectra/"
    os.chdir(output_dir)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    proper_box_size = get_proper_box_size(ds)
    refine_box, refine_box_center, x_width = get_refine_box(ds, zsnap, track)
    halo_center = get_halo_center(ds, refine_box_center)
    x_left = np.float(refine_box.left_edge[0].value)
    x_right = np.float(refine_box.right_edge[0].value)
    y_left = np.float(refine_box.left_edge[1].value)
    y_right = np.float(refine_box.right_edge[1].value)


    # slice will be repeated, so let's make it first
    z_center = [refine_box_center[0], refine_box_center[1], halo_center[2]]
    slc = yt.SlicePlot(ds,'z',('gas','H_p0_number_density'),center=z_center,width=1.5*x_width)
    ## slc = ds.r[xmin:xmax, ymin:ymax, halo_center[2]]
    res = [1000,1000]
    # frb = slc.frb(x_width, res)
    frb = slc.frb['gas','H_p0_number_density']
    # image = np.array(frb['gas', 'density'])
    image = np.array(frb)
    print "min, max H I density = ", np.min(np.log10(image)), np.max(np.log10(image))
    # extent = [float(x.in_units('code_length')) for x in (pro.xlim + pro.ylim)]
    extent = [x_left, x_right, y_left, y_right]

    # spectral features
    #g = Gaussian1DKernel((7/0.0267)/2.355)  # HIRES v_fwhm = 7 km/s
    #snr = 30.


    # get a spectrum!
    # impact = 25.
    # impacts = np.arange(25.,50.5,0.5)
    #impacts = np.arange(-50., -24.5, 0.5)
    impacts = [-50.]
    for impact in impacts:
        out_fits_name = "hlsp_misty_foggie_halo008508_"+ds.basename.lower()+"_i"+"{:05.1f}".format(impact) + \
                    "_dx"+"{:4.2f}".format(0.)+"_v2_los.fits"
        out_plot_name = "HI_slice_with_lots_spectra_halo008508_"+ds.basename.lower()+"_i"+"{:05.1f}".format(impact) + \
                    "_dx"+"{:4.2f}".format(0.)+".png"
        hdulist, ray_start, ray_end = extract_spectra(ds, impact, read_fits_file=False,
                                out_fits_name=out_fits_name,
                                xmin=x_left, xmax=x_right, halo_center=halo_center, refine_box_center=refine_box_center)
        print "(impact/proper_box_size)/x_width = ", (impact/proper_box_size)/x_width

        print "ray start:", ray_start
        print "ray end:", ray_end
        print "refine box center:", refine_box_center
        print "halo center:", halo_center
        print "z_center:", z_center

        # start by setting up plots
        fig = plt.figure(figsize=(16,6), dpi=100)

        # creates grid on which the figure will be plotted
        gs = gridspec.GridSpec(6, 3,
                               width_ratios=[0.05, 1, 1])

        ## this will be the slice
        ax_slice = fig.add_subplot(gs[:,1])
        slc = ax_slice.imshow(np.log10(image), extent=extent, cmap=h1_color_map, \
                                        vmin=np.log10(h1_slc_min), vmax=np.log10(h1_slc_max))
        ax_slice.plot([ray_start[0], ray_end[0]], [ray_start[1], ray_end[1]], color="white", lw=2.)
        ax_slice.set_aspect('equal')
        ax_slice.xaxis.set_major_locator(ticker.NullLocator())
        ax_slice.yaxis.set_major_locator(ticker.NullLocator())
        # cbar = fig.colorbar(cax, ax=ax_cbar, orientation='vertical', pad=0.01, shrink=0.8)
        ax_cbar = fig.add_subplot(gs[:,0])
        cbar = fig.colorbar(slc, cax=ax_cbar, extend='both')
        ax_cbar.yaxis.set_ticks_position('left')
        ax_cbar.yaxis.set_label_position('left')
        cbar.set_label(r'log HI density', fontsize=16.)

        ## these will be the spectra
        zmin, zmax = 1.998, 2.004
        vmin, vmax = -1000, 1000

        ax_spec1 = fig.add_subplot(gs[0,2])
        velocity = ((hdulist["H I 1216"].data['wavelength'] / hdulist['H I 1216'].header['restwave']) - 1 - ds.current_redshift) * 299792.458 - 250
        flux = hdulist["H I 1216"].data['flux']
        # v_conv = convolve(velocity, g)
        # ax_spec1.step(v_conv, flux, color="#4575b4",lw=2)
        ax_spec1.plot(velocity, flux, color='darkorange',lw=1)
        ax_spec1.text(vmin + 100, 0, "H I 1216", fontsize=12.)
        ax_spec1.xaxis.set_major_locator(ticker.NullLocator())
        plt.xlim(vmin, vmax)
        plt.ylim(-0.05, 1.05)

        ax_spec2 = fig.add_subplot(gs[1,2])
        velocity = ((hdulist["H I 973"].data['wavelength'] / hdulist['H I 973'].header['restwave']) - 1 - ds.current_redshift) * 299792.458 - 250
        flux = hdulist["H I 973"].data['flux']
        # v_conv = convolve(velocity, g)
        # ax_spec2.step(v_conv, flux, color="#4575b4",lw=2)
        ax_spec2.plot(velocity, flux, color='darkorange',lw=1)
        ax_spec2.text(vmin + 100, 0, "Ly c", fontsize=12.)
        ax_spec2.xaxis.set_major_locator(ticker.NullLocator())
        plt.xlim(vmin, vmax)
        plt.ylim(-0.05, 1.05)

        ax_spec3 = fig.add_subplot(gs[2,2])
        velocity = ((hdulist["H I 919"].data['wavelength'] / hdulist['H I 919'].header['restwave']) - 1 - ds.current_redshift) * 299792.458 - 250
        flux = hdulist["H I 973"].data['flux']
        # v_conv = convolve(velocity, g)
        # ax_spec3.step(v_conv, flux, color="#4575b4",lw=2)
        ax_spec3.plot(velocity, hdulist["H I 919"].data['flux'], color='darkorange',lw=1)
        ax_spec3.text(vmin + 100, 0, "Ly 10", fontsize=12.)
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
