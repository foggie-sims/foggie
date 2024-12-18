'''
generates random spectra, through just the refine region, along just the cardinal axes
modified for generating asdi spectra ...

usage:
python random_misty_spectra.py --linelist linelist --run run --system system --axis axis --Nrays N --seed seed
so, e.g.:
python ~/Dropbox/foggie/foggie/random_misty_spectra.py --linelist jt --run nref11c --system palmetto --axis x --Nrays 10 --seed 101

probably last really updated for the FOGGIE I paper?

issues: requires MISTY import that depends on molly's path

'''

from __future__ import print_function
import trident
import numpy as np
import yt
import os

os.sys.path.insert(0, '/Users/molly/Dropbox/misty/MISTY-pipeline/MISTY')
import MISTY
import sys
import os
import argparse

from astropy.table import Table
from astropy.io import ascii
from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_run_loc_etc import get_run_loc_etc

# import show_velphase as sv

import getpass

from math import pi

def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    parser = argparse.ArgumentParser(description="extracts spectra from refined region")

    ## what are we plotting and where is it
    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--velocities', dest='velocities', action='store_true',
                            help='make the velocity plots?, default is no')
    parser.set_defaults(velocities=False)

    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo="8508")

    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is natural')
    parser.set_defaults(run="natural")

    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")

    parser.add_argument('--system', metavar='system', type=str, action='store',
                        help='which system are you on? default is palmetto')
    parser.set_defaults(system="palmetto")

    parser.add_argument('--Nrays', metavar='Nrays', type=int, action='store',
                        help='how many sightlines do you want? default is 1')
    parser.set_defaults(Nrays="1")

    parser.add_argument('--seed', metavar='seed', type=int, action='store',
                        help='random seed? default is 17')
    parser.set_defaults(seed=17)

    parser.add_argument('--axis', metavar='axis', type=str, action='store',
                        help='which axis? default is x')
    parser.set_defaults(axis="x")

    parser.add_argument('--linelist', metavar='linelist', type=str, action='store',
                        help='which linelist: long, kodiaq, jt, or short? default is short')
    parser.set_defaults(linelist="short")

    parser.add_argument('--pixdv', metavar='pixdv', type=float, action='store',
                        help='what km/s size of pixels? default is 0.1')
    parser.set_defaults(axis=0.1)


    args = parser.parse_args()
    return args


def get_random_ray_endpoints(ds, halo_center, track, axis, **kwargs):
    '''
    returns ray_start and ray_end for a ray along a given axis,
    within the refined region. returns the ray endpoints, the
    offsets, and the impact parameter to the center (that is passed in)
    '''
    refine_box, refine_box_center, x_width = get_refine_box(ds, ds.current_redshift, track)
    proper_box_size = get_proper_box_size(ds)
    dy = x_width * (0.05 + 0.9 * np.random.uniform())  ## don't want to be too close to box edges
    dz = x_width * (0.05 + 0.9 * np.random.uniform())
    dy_prop = proper_box_size * dy
    dz_prop = proper_box_size * dz

    ray_start = np.zeros(3)
    ray_end = np.zeros(3)
    if axis == 'x' or axis == 0:
        ray_ax = 0
        axy = 1
        axz = 2
        deltas = "_dy"+"{:05.1f}".format(dy_prop) + "_dz"+"{:05.1f}".format(dz_prop)
    elif axis == 'y' or axis == 1:
        ray_ax = 1
        axy = 0
        axz = 2
        deltas = "_dx"+"{:05.1f}".format(dy_prop) + "_dz"+"{:05.1f}".format(dz_prop)
    elif axis == 'z' or axis == 2:
        ray_ax = 2
        axy = 0
        axz = 1
        deltas = "_dx"+"{:05.1f}".format(dy_prop) + "_dy"+"{:05.1f}".format(dz_prop)

    ray_start[ray_ax] = np.float(refine_box.left_edge[ray_ax].value)
    ray_end[ray_ax] = np.float(refine_box.right_edge[ray_ax].value)
    ray_start[axy] = np.float(refine_box.left_edge[axy].value) + dy
    ray_end[axy] = np.float(refine_box.left_edge[axy].value) + dy
    ray_start[axz] = np.float(refine_box.left_edge[axz].value) + dz
    ray_end[axz] = np.float(refine_box.left_edge[axz].value) + dz

    impact = proper_box_size * np.sqrt((halo_center[axy] - ray_start[axy])**2 + (halo_center[axz] - ray_start[axz])**2)

    return np.array(ray_start), np.array(ray_end), deltas, impact

def quick_spectrum(ds, triray, filename, **kwargs):

    line_list = kwargs.get("line_list", ['H I 1216', 'Si II 1260', 'Mg II 2796', 'C III 977', 'C IV 1548', 'O VI 1032'])
    redshift = ds.get_parameter('CosmologyCurrentRedshift')

    ldb = trident.LineDatabase('data/atom_wave_gamma_f.dat')
    sg = trident.SpectrumGenerator(lambda_min=1000.,
                                       lambda_max=4000.,
                                       dlambda=0.01,
                                       line_database='data/atom_wave_gamma_f.dat')

    sg.make_spectrum(triray, line_list, min_tau=1.e-5,store_observables=True)

    restwave = sg.lambda_field / (1. + redshift)
    out_spectrum = Table([sg.lambda_field, restwave, sg.flux_field])
    out_spectrum.write(filename+'.fits')

def generate_random_rays(ds, halo_center, **kwargs):
    '''
    generate some random rays
    '''
    track = kwargs.get("track","halo_track")
    infoname = kwargs.get("infoname","halo_info")
    Nrays = kwargs.get("Nrays",2)
    seed = kwargs.get("seed",17)
    axis = kwargs.get("axis",'x')
    pixdv = kwargs.get("pixdv",0.1)
    output_dir = kwargs.get("output_dir", ".")
    haloname = kwargs.get("haloname","somehalo")
    line_list = kwargs.get("line_list", ['H I 1216', 'Si II 1260',  'O VI 1032'])

    proper_box_size = get_proper_box_size(ds)
    refine_box, refine_box_center, x_width = get_refine_box(ds, zsnap, track)
    proper_x_width = x_width*proper_box_size

    ## get halo info
    print("opening halo_info file", infoname)
    t = ascii.read(infoname, format='fixed_width')
    thisid = t['redshift'] ==  zsnap
    print('grabbing physical information from \n', t[thisid])
    assert len(t[thisid]) == 1
    Mvir = t['Mvir'][thisid][0] # 'Msun'
    Rvir = t['Rvir'][thisid][0] # 'kpc'
    Mstar = t['Mstar'][thisid][0] # 'Msun'
    Mism = t['Mism'][thisid][0] #  'Msun'
    SFR = t['SFR'][thisid][0] # 'Msun/yr'

    ## for now, assume all are z-axis
    np.random.seed(seed)
    out_ray_basename = haloname + "_" + ds.basename + "_ray_" + axis

    i = 0
    while i < Nrays:
        os.chdir(output_dir)
        rs, re, deltas, impact = get_random_ray_endpoints(ds, halo_center, track, axis)
        this_out_ray_basename = out_ray_basename + deltas
        out_ray_name =  this_out_ray_basename + ".h5"
        out_fits_name = "hlsp_misty_foggie_"+haloname.lower()+"_"+ds.basename.lower()+"_ax"+axis+deltas.replace('.','')+"_v1_los.fits.gz"
        out_plot_name = "hlsp_misty_foggie_"+haloname.lower()+"_"+ds.basename.lower()+"_ax"+axis+deltas.replace('.','')+"_v1_los.png"
        rs = ds.arr(rs, "code_length")
        re = ds.arr(re, "code_length")
        ray = ds.ray(rs, re)
        ray.save_as_dataset(out_ray_name, fields=["density","temperature", "metallicity"])

        out_tri_name = this_out_ray_basename + "_tri.h5"
        fields = []
        triray = trident.make_simple_ray(ds, start_position=rs.copy(),
                                  end_position=re.copy(),
                                  data_filename=out_tri_name,
                                  lines=line_list,
                                  ftype='gas',
                                  fields=fields)

        hi_col = np.log10((triray.r['H_p0_number_density']*triray.r['dl']).sum().d)
        print('log HI column = ', hi_col, '...')

        ray_start = triray.light_ray_solution[0]['start']
        ray_end = triray.light_ray_solution[0]['end']
        print("final start, end = ", ray_start, ray_end)
        filespecout_base = this_out_ray_basename + '_spec'
        print(ray_start, ray_end, filespecout_base)

        hdulist = MISTY.write_header(triray,start_pos=ray_start,end_pos=ray_end,
                      lines=line_list, impact=impact, redshift=ds.current_redshift,
                      haloname=halo_dict[args.halo], Mvir=Mvir, Rvir=Rvir, Mstar=Mstar, Mism=Mism, SFR=SFR)
        tmp = MISTY.write_parameter_file(ds,hdulist=hdulist)

        # quick_spectrum(ds, triray, filespecout_base)

        for line in line_list:
            sg = MISTY.generate_line(triray, line,
                                     zsnap=ds.current_redshift,
                                     write=True,
                                     hdulist=hdulist,
                                     pixdv=pixdv,
                                     use_spectacle=False)
            # the trident plots are not needed ; just take up lots of space
            ## filespecout = filespecout_base+'_'+line.replace(" ", "_")+'.png'
            ## sg.plot_spectrum(filespecout,flux_limits=(0.0,1.0))

        MISTY.write_out(hdulist,filename=out_fits_name)
        # plot_misty_spectra(hdulist, outname=out_plot_name)
        i = i+1
        print('''
                    ~~~~~~~~~~~~ i = ''',i,''' done  ~~~~~~~~~~~~~~~~~~~
              ''')

    print('Nrays = ',Nrays,' and i = ', i)



if __name__ == "__main__":

    args = parse_args()
    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir, infoname = get_run_loc_etc(args)
    run_dir = foggie_dir + run_loc

    if args.linelist == 'long':
        line_list = linelist_long
    elif args.linelist == 'kodiaq':
        line_list = linelist_kodiaq
    elif args.linelist == 'jt':
        line_list = linelist_jt
    elif args.linelist == 'short':
        line_list = linelist_short
    elif args.linelist == 'high':
        line_list = linelist_high
    else: ## short --- these are what show_velphase has
        line_list = ['H I 1216', 'Si II 1260', 'O VI 1032']

    ds_loc = run_dir + args.output + "/" + args.output
    print(ds_loc)
    ds = yt.load(ds_loc)
    trident.add_ion_fields(ds, line_list)

    print("opening track: " + trackname)
    track = Table.read(trackname, format='ascii')
    track.sort('col1')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    refine_box, refine_box_center, x_width = get_refine_box(ds, zsnap, track)
    halo_center, halo_velocity = get_halo_center(ds, refine_box_center)
    halo_center = get_halo_center(ds, refine_box_center)[0]
    asdi_dir = '/Users/molly/Dropbox/foggie/collab/spectra_for_asdi/'

    ## get the halo string
    halostring = "halo_" + halo_dict[args.halo] + "_"  + args.run
    print("halostring is ", halostring)

    generate_random_rays(ds, halo_center, haloname=halostring, track=track, infoname=infoname, \
                          axis=args.axis, line_list=line_list,\
                         output_dir=asdi_dir, seed=args.seed, Nrays=args.Nrays, pixdv=args.pixdv)

    # generate_random_rays(ds, halo_center, line_list=["H I 1216"], haloname="halo008508", Nrays=100)
    sys.exit("~~~*~*~*~*~*~all done!!!! spectra are fun!")
