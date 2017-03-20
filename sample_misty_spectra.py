import trident
import numpy as np
import yt
import MISTY
import sys

import getpass

from math import pi

def get_ray_endpoints(ds, halo_center, **kwargs):
    '''
    returns ray_start and ray_end for a ray with a given
    impact parameter along a given axis, with either specified
    or random angle
    '''
    axis = kwargs.get("axis", 2)
    impact = kwargs.get("impact", 100.)
    angle = kwargs.get("angle", 2*pi*np.random.uniform())
    dl = kwargs.get("dl", 500.)  # pathlength of ray
    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') / ds.get_parameter('CosmologyHubbleConstantNow') * 1000. # in kpc

    ray_start = np.zeros(3)
    ray_end = np.zeros(3)
    if axis == "x" or axis == 0:
        axis = 0
        dx = 0
        dy = impact * np.cos(angle)
        dz = impact * np.sin(angle)
    elif axis == "y" or axis == 1:
        axis = 1
        dx = impact * np.cos(angle)
        dy = 0
        dz = impact * np.sin(angle)
    elif axis == "z" or axis == 2:
        axis = 2
        dx = impact * np.cos(angle)
        dy = impact * np.sin(angle)
        dz = 0
    else:
        print "---> your axis does not make any sense :-("

    delta = [dx,dy,dz]
    for ax in (0,1,2):
        ray_start[ax] = halo_center[ax] + delta[ax]/proper_box_size
        ray_end[ax] = halo_center[ax] + delta[ax]/proper_box_size
    ray_start[axis] -= 0.5*dl/proper_box_size
    ray_end[axis] += 0.5*dl/proper_box_size

    return np.array(ray_start), np.array(ray_end)

def generate_random_rays(ds, halo_center, **kwargs):
    '''
    generate some random rays
    '''
    low_impact = kwargs.get("low_impact", 10.)
    high_impact = kwargs.get("high_impact", 200.)
    Nrays = kwargs.get("Nrays",50)
    line_list = kwargs.get("line_list", ['H I','Si III','C IV', 'O VI'])

    ## for now, assume all are z-axis
    axis = "z"
    impacts = np.random.uniform(low=low_impact, high=high_impact, size=Nrays)
    out_ray_basename = ds.basename + "_ray_" + axis

    for i in range(Nrays):
        out_ray_name = out_ray_basename + "_imp"+"{:05.1f}".format(impacts[i]) + \
                        "_ang"+"{:4.2f}".format(angles[i])+".h5"
        out_fits_name = out_ray_basename + "_imp"+"{:05.1f}".format(impacts[i]) + \
                        "_ang"+"{:4.2f}".format(angles[i])+".fits"
        rs, re = get_ray_endpoints(ds, halo_center, impact=impacts[i], angle=angles[i], axis=axis)
        rs = ds.arr(rs, "code_length")
        re = ds.arr(re, "code_length")
        ray = ds.ray(rs, re)
        ray.save_as_dataset(out_ray_name)
        triray = trident.make_simple_ray(ds, start_position=rs.copy(),
                                  end_position=re.copy(),
                                  data_filename=out_ray_name,
                                  lines=line_list,
                                  ftype='gas')

        ray_start = triray.light_ray_solution[0]['start']
        ray_end = triray.light_ray_solution[0]['end']
        filespecout_base = 'spectrum_'+triray.light_ray_solution[0]['filename']
        print ray_start, ray_end, filespecout_base

        hdulist = MISTY.write_header(triray,start_pos=ray_start,end_pos=ray_end,
                      lines=line_list, author=getpass.getuser())
        tmp = MISTY.write_parameter_file(ds,hdulist=hdulist)

        for line in line_list:
            sg = MISTY.generate_line(triray,line,write=True,hdulist=hdulist)
            # filespecout = filespecout_base+'_'+line.replace(" ", "_")+'.png'
            # sg.plot_spectrum(filespecout,flux_limits=(0.0,1.0))

        MISTY.write_out(hdulist,filename=out_fits_name)



if __name__ == "__main__":

    # args = parse_args()
    ds = yt.load("/Users/molly/foggie/halo_008508/nref10/RD0042/RD0042")
    halo_center =  [0.4898, 0.4714, 0.5096]
    generate_random_rays(ds, halo_center, Nrays=5)
    sys.exit("~~~*~*~*~*~*~all done!!!! spectra are fun!")
