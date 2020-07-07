#
# Written by Brendan Boyd: boydbre1@msu.edu
# https://github.com/biboyd/salsa
#
# Last Edits Made 07/07/2020
#
#
# File for generating an absorber catalog in the same vein as in
# extract_sample_absorber, but using salsa package
#
#
import numpy as np
import yt
from mpi4py import MPI
import argparse

from foggie.absorber_extraction import salsa
from foggie.absorber_extraction.salsa.utils.functions import parse_cut_filter

from foggie.utils.consistency import units_dict, default_spice_fields, min_absorber_dict
from foggie.utils.foggie_load import foggie_load

box_trackfile = '/mnt/home/boydbre1/data/track_files/halo_track_200kpc_nref10'
hcv_file='/mnt/home/boydbre1/Repo/foggie/foggie/halo_infos/008508/nref11c_nref9f/halo_c_v'

def main(args):

    ds_filename= args.ds
    n_rays = args.n_rays
    raydir = args.raydir
    ion = args.ion
    rlength = args.rlength
    max_imp = args.max_impact
    cuts = args.cut

    outfile = args.outfile


    # set rand seed
    np.random.seed(2020 + 16*int(ds_filename[-2:]))
    comm = MPI.COMM_WORLD

    #load dataset
    ds, reg_foggie = foggie_load(ds_filename, box_trackfile,
                                 halo_c_v_name=hcv_file, disk_relative=True)

    # broadcast dataset to each process
    comm.Barrier()

    cut_filters= parse_cut_filter(cuts)
    ext_kwargs = {'absorber_min':min_absorber_dict[ion]}

    df = salsa.generate_catalog(ds, n_rays, raydir, [ion],
                                center=ds.halo_center_code,
                                impact_param_lims=(0, max_imp),
                                cut_region_filters=cut_filters,
                                ray_length=rlength, fields=default_spice_fields,
                                extractor_kwargs=ext_kwargs,
                                units_dict=units_dict)

    #now save it
    if comm.rank == 0:
        df.to_csv(outfile)

if __name__ == '__main__':
    #create parser
    parser = argparse.ArgumentParser(description='Process cuts and stuff')
    parser.add_argument("--ds", type=str, help="The dataset path",
                        required=True)
    parser.add_argument("--raydir", type=str, help="path to dir where rays are/will be saved",
                        required=True)
    parser.add_argument("-o","--outfile", type=str, help="filename where catalog is saved",
                        required=True)
    parser.add_argument("-i", "--ion", type=str,
                        help='The ion to look at (ie "H I", "O VI")',
                        required=True)
    parser.add_argument("-m", "--max-impact", type=int,
                        help="Max impact parameter sampled", default=200)
    parser.add_argument("-c", "--cut", type=str, default='cgm')

    parser.add_argument("-n", "--n-rays", type=int, help="number of rays to create",
                        default=10)

    parser.add_argument("-l", "--rlength", type=int, help="length of lray in kpc",
                            default=200)



    args=parser.parse_args()

    main(args)
