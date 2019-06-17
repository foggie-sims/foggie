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

from utils.get_refine_box import get_refine_box
from utils.get_proper_box_size import get_proper_box_size
from get_halo_center import get_halo_center
#from plot_misty_spectra import plot_misty_spectra
from get_run_loc_etc import get_run_loc_etc

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
                        help='which system are you on? default is oak')
    parser.set_defaults(system="oak")

    parser.add_argument('--Nrays', metavar='Nrays', type=int, action='store',
                        help='how many sightlines do you want? default is 1')
    parser.set_defaults(Nrays="1")

    parser.add_argument('--seed', metavar='seed', type=int, action='store',
                        help='random seed? default is 17')
    parser.set_defaults(seed=17)

    parser.add_argument('--axis', metavar='axis', type=str, action='store',
                        help='which axis? default is x')
    parser.set_defaults(seed="x")

    parser.add_argument('--linelist', metavar='linelist', type=str, action='store',
                        help='which linelist: long, kodiaq, or short? default is short')
    parser.set_defaults(axis="short")

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
    Nrays = kwargs.get("Nrays",2)
    seed = kwargs.get("seed",17)
    axis = kwargs.get("axis",'x')
    output_dir = kwargs.get("output_dir", ".")
    haloname = kwargs.get("haloname","somehalo")
    line_list = kwargs.get("line_list", ['H I 1216', 'Si II 1260',  'O VI 1032'])

    proper_box_size = get_proper_box_size(ds)
    refine_box, refine_box_center, x_width = get_refine_box(ds, zsnap, track)
    proper_x_width = x_width*proper_box_size
    if args.velocities:
        trident.add_ion_fields(ds, ions=['Si II', 'Si III', 'Si IV', 'C II', 'C III', 'C IV', 'O VI', 'Mg II', 'Ne VIII'])
    if args.linelist == 'jt':
        trident.add_ion_fields(ds, ions=['Si II', 'Si III', 'Si IV', 'C II', 'C III', 'C IV', 'O VI', 'Mg II', 'Ne VIII'])

    ## for now, assume all are z-axis
    np.random.seed(seed)
    out_ray_basename = ds.basename + "_ray_" + axis

    i = 0
    while i < Nrays:
        os.chdir(spectra_dir + 'random/')
        rs, re, deltas, impact = get_random_ray_endpoints(ds, halo_center, track, axis)
        this_out_ray_basename = out_ray_basename + deltas
        out_ray_name =  this_out_ray_basename + ".h5"
        out_fits_name = "hlsp_misty_foggie_"+haloname+"_"+ds.basename.lower()+"_ax"+axis+deltas.replace('.','')+"_vjt_los.fits.gz"
        out_plot_name = "hlsp_misty_foggie_"+haloname+"_"+ds.basename.lower()+"_ax"+axis+deltas.replace('.','')+"_vjt_los.png"
        rs = ds.arr(rs, "code_length")
        re = ds.arr(re, "code_length")
        ray = ds.ray(rs, re)
        ray.save_as_dataset(out_ray_name, fields=["density","temperature", "metallicity"])

        if args.velocities:
            ray_df =  ray.to_dataframe(["x","y","z","density","temperature","metallicity","HI_Density",
                                    "x-velocity", "y-velocity", "z-velocity",
                                    "C_p2_number_density", "C_p3_number_density", "H_p0_number_density",
                                    "Mg_p1_number_density", "O_p5_number_density","Si_p2_number_density"])

        out_tri_name = this_out_ray_basename + "_tri.h5"
        if args.linelist == 'jt':
            fields = ['metallicity', 'H_p0_number_density',
                      'C_p1_number_density', 'C_p2_number_density', 'C_p3_number_density',
                      'Mg_p1_number_density', 'Ne_p7_number_density',
                      'O_p5_number_density', 'Si_p1_number_density', 'Si_p2_number_density',
                      'Si_p3_number_density', 'cell_mass']
        else:
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
                      lines=line_list, impact=impact, redshift=ds.current_redshift)
        tmp = MISTY.write_parameter_file(ds,hdulist=hdulist)

        # quick_spectrum(ds, triray, filespecout_base)

        for line in line_list:
            sg = MISTY.generate_line(triray, line,
                                     zsnap=ds.current_redshift,
                                     write=True,
                                     hdulist=hdulist,
                                     use_spectacle=False,
                                     resample=True)
            # the trident plots are not needed ; just take up lots of space
            ## filespecout = filespecout_base+'_'+line.replace(" ", "_")+'.png'
            ## sg.plot_spectrum(filespecout,flux_limits=(0.0,1.0))
            if args.velocities and ('H' in line):
                sv.show_velphase(ds, ray_df, rs, re, triray, filespecout_base)

        MISTY.write_out(hdulist,filename=out_fits_name)
        # plot_misty_spectra(hdulist, outname=out_plot_name)
        i = i+1
        print('''
                    ~~~~~~~~~~~~ i = ''',i,''' done  ~~~~~~~~~~~~~~~~~~~
              ''')

    print('Nrays = ',Nrays,' and i = ', i)



if __name__ == "__main__":

    args = parse_args()
    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir = get_run_loc_etc(args)
    run_dir = foggie_dir + run_loc

    if args.linelist == 'long':
        line_list = ['H I 1216', 'H I 1026', 'H I 973',
                       'H I 950', 'H I 919', 'Al II 1671', 'Al III 1855', \
                       'Si II 1260', 'Si III 1206', 'Si IV 1394', \
                       'C II 1335', 'C III 977', 'C IV 1548', \
                       'O VI 1032', 'Ne VIII 770']
    elif args.linelist == 'kodiaq':
        line_list = ['H I 1216', 'H I 919', \
                        'Si II 1260', 'Si III 1206', 'Si IV 1394',
                        'C II 1335', 'C III 977', 'C IV 1548',
                         'O VI 1032']
    elif args.linelist == 'jt':
        line_list = ['H I 1216', 'H I 919', \
                        'Mg II 2796', 'Si II 1260', 'Si III 1206', 'Si IV 1394', \
                        'C II 1335', 'C III 977', 'C IV 1548',\
                        'O VI 1032', 'Ne VIII 770']
    else: ## short --- these are what show_velphase has
        line_list = ['H I 1216', 'Si II 1260', 'O VI 1032']

    ds_loc = run_dir + args.output + "/" + args.output
    print(ds_loc)
    ds = yt.load(ds_loc)

    print("opening track: " + trackname)
    track = Table.read(trackname, format='ascii')
    track.sort('col1')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    refine_box, refine_box_center, x_width = get_refine_box(ds, zsnap, track)
    halo_center, halo_velocity = get_halo_center(ds, refine_box_center)
    halo_center = get_halo_center(ds, refine_box_center)[0]
    random_dir = output_dir + "random/"

    generate_random_rays(ds, halo_center, haloname=haloname, track=track, axis=args.axis, line_list=line_list,\
                         output_dir=random_dir, seed=args.seed, Nrays=args.Nrays)

    # generate_random_rays(ds, halo_center, line_list=["H I 1216"], haloname="halo008508", Nrays=100)
    sys.exit("~~~*~*~*~*~*~all done!!!! spectra are fun!")
