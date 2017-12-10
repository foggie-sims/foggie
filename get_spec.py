import trident
import numpy as np
import yt
import analysis

from get_proper_box_size import get_proper_box_size

def get_spec(ds, halo_center, rho_ray):

    proper_box_size = get_proper_box_size(ds)
    line_list = ['H', 'O', 'C', 'N', 'Si']

    # big rectangle box
    ray_start = [halo_center[0]-250./proper_box_size, halo_center[1]+71./proper_box_size, halo_center[2]-71./proper_box_size]
    ray_end =   [halo_center[0]+250./proper_box_size, halo_center[1]+71./proper_box_size, halo_center[2]-71./proper_box_size]

    ray = trident.make_simple_ray(ds, start_position=ray_start,
                                  end_position=ray_end,
                                  data_filename="ray.h5",
                                  lines=line_list,
                                  ftype='gas')

    sg = trident.SpectrumGenerator(lambda_min=1020, lambda_max=1045, dlambda=0.002)
    trident.add_ion_fields(ds, ions=['C IV', 'O VI','H I','Si III'])

    sg.make_spectrum(ray, lines=line_list)
    sg.plot_spectrum('spec_final.png')
    sg.save_spectrum('spec_final.txt')

    # this makes for convenient field extraction from the ray
    ad = ray.all_data()
    dl = ad["dl"] # line element length

    out_dict = {'nhi':np.sum(ad['H_number_density']*dl), 'no6':np.sum(ad['O_p5_number_density']*dl), 'nsi3':np.sum(ad['Si_p2_number_density']*dl)}
    print np.sum(ad['H_number_density']*dl)
    print np.sum(ad['O_p5_number_density']*dl)
    print np.sum(ad['Si_p2_number_density']*dl)

    return ray, out_dict
