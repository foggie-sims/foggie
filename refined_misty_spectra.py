import trident
import numpy as np
import yt
import MISTY
import sys
import os

from astropy.table import Table

from modular_plots import get_refine_box

import getpass

from math import pi

def get_refined_ray_endpoints(ds, halo_center, track, **kwargs):
    '''
    returns ray_start and ray_end for a ray with a given
    impact parameter along a given axis, only within the refined box
    '''
    impact = kwargs.get("impact", 25.)
    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') / ds.get_parameter('CosmologyHubbleConstantNow') * 1000. # in kpc

    ray_start = np.zeros(3)
    ray_end = np.zeros(3)
    #### for now, ray has random y (sets impact), goes from -z to +z, in y direction, x is random
    x_width = np.abs((np.interp(zsnap, track['col1'], track['col2']) - np.interp(zsnap, track['col1'], track['col5'])))
    x_min = np.min((np.interp(zsnap, track['col1'], track['col2']), np.interp(zsnap, track['col1'], track['col5'])))
    ray_start[0] = x_width * np.random.uniform() + x_min
    ray_end[0] = ray_start[0]

    ray_start[1] = (impact/proper_box_size) + halo_center[1]
    ray_end[1] = ray_start[1]

    ray_start[2] = np.interp(zsnap, track['col1'], track['col4'])
    ray_end[2] = np.interp(zsnap, track['col1'], track['col7'])

    return np.array(ray_start), np.array(ray_end)

def generate_random_rays(ds, halo_center, **kwargs):
    '''
    generate some random rays
    '''
    low_impact = kwargs.get("low_impact", 10.)
    high_impact = kwargs.get("high_impact", 50.)
    track = kwargs.get("track","halo_track")
    Nrays = kwargs.get("Nrays",50)
    output_dir = kwargs.get("output_dir",".")
    haloname = kwargs.get("haloname","somehalo")
    # line_list = kwargs.get("line_list", ['H I 1216', 'Si II 1260', 'C II 1334', 'Mg II 2796', 'C III 977', 'Si III 1207','C IV 1548', 'O VI 1032'])
    line_list = kwargs.get("line_list", ['H I 1216', 'Si II 1260', 'C II 1335', 'C III 977', 'Si III 1207','C IV 1548', 'O VI 1032'])
    # line_list = kwargs.get("line_list", ['H I 1216', 'Si II 1260','O VI 1032'])

    ## for now, assume all are z-axis
    axis = "z"
    np.random.seed(17)
    impacts = np.random.uniform(low=low_impact, high=high_impact, size=Nrays)
    angles = np.random.uniform(low=0, high=2*pi, size=Nrays)
    out_ray_basename = ds.basename + "_ray_" + axis

    for i in range(Nrays):
        os.chdir(output_dir)
        this_out_ray_basename = out_ray_basename + "_imp"+"{:05.1f}".format(impacts[i]) + \
                        "_ang"+"{:4.2f}".format(angles[i])
        out_ray_name =  this_out_ray_basename + ".h5"
        rs, re = get_refined_ray_endpoints(ds, halo_center, track, impact=impacts[i])
        out_fits_name = "hlsp_misty_foggie_"+haloname+"_"+ds.basename.lower()+"_i"+"{:05.1f}".format(impacts[i]) + \
                        "_v2_los.fits"
        rs = ds.arr(rs, "code_length")
        re = ds.arr(re, "code_length")
        ray = ds.ray(rs, re)
        ray.save_as_dataset(out_ray_name, fields=["density","temperature", "metallicity"])
        out_tri_name = this_out_ray_basename + "_tri.h5"
        triray = trident.make_simple_ray(ds, start_position=rs.copy(),
                                  end_position=re.copy(),
                                  data_filename=out_tri_name,
                                  lines=line_list,
                                  ftype='gas')

        ray_start = triray.light_ray_solution[0]['start']
        ray_end = triray.light_ray_solution[0]['end']
        print "final start, end = ", ray_start, ray_end
        filespecout_base = this_out_ray_basename + '_spec'
        print ray_start, ray_end, filespecout_base

        hdulist = MISTY.write_header(triray,start_pos=ray_start,end_pos=ray_end,
                      lines=line_list, impact=impacts[i], angle=angles[i])
        tmp = MISTY.write_parameter_file(ds,hdulist=hdulist)

        for line in line_list:
            sg = MISTY.generate_line(triray,line,write=True,hdulist=hdulist)
            filespecout = filespecout_base+'_'+line.replace(" ", "_")+'.png'
            ## if we write our own plotting routine, we can overplot the spectacle fits
            sg.plot_spectrum(filespecout,flux_limits=(0.0,1.0))

        MISTY.write_out(hdulist,filename=out_fits_name)



if __name__ == "__main__":

    # args = parse_args()
    # ds = yt.load("/Users/molly/foggie/halo_008508/symmetric_box_tracking/nref10f_50kpc/RD0042/RD0042")
    # ds = yt.load("/Users/molly/foggie/halo_008508/symmetric_box_tracking/nref10f_50kpc/DD0165/DD0165")
    ds = yt.load("/astro/simulations/FOGGIE/halo_008508/symmetric_box_tracking/nref10f_50kpc/DD0165/DD0165")
    ### halo_center =  [0.4898, 0.4714, 0.5096]
    track_name = "/astro/simulations/FOGGIE/halo_008508/symmetric_box_tracking/nref10f_50kpc/halo_track"
    output_dir = "/Users/molly/Dropbox/foggie-collab/plots/halo_008508/symmetric_box_tracking/nref10f_50kpc/spectra"
    print("opening track: " + track_name)
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    # interpolate the center from the track
    centerx = 0.5 * ( np.interp(zsnap, track['col1'], track['col2']) + np.interp(zsnap, track['col1'], track['col5']))
    ### np.interp(zsnap, track['col1'], track['col2'])
    centery = 0.5 * ( np.interp(zsnap, track['col1'], track['col3']) + np.interp(zsnap, track['col1'], track['col6']))
    #### np.interp(zsnap, track['col1'], track['col3'])
    centerz = 0.5 * ( np.interp(zsnap, track['col1'], track['col4']) + np.interp(zsnap, track['col1'], track['col7']))

    halo_center = [centerx, centery, centerz]

    generate_random_rays(ds, halo_center, haloname="halo008508", track=track, output_dir=output_dir, Nrays=20)
    # generate_random_rays(ds, halo_center, line_list=["H I 1216"], haloname="halo008508", Nrays=100)
    sys.exit("~~~*~*~*~*~*~all done!!!! spectra are fun!")
