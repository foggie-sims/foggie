import yt
from yt.data_objects.level_sets.api import *
from foggie.utils.foggie_load import foggie_load as fl
from foggie.utils.foggie_load import load_sim
import os
import matplotlib.pyplot as plt
import numpy as np
from foggie.utils.get_run_loc_etc import get_run_loc_etc
import argparse
import trident

def parse_args():
    '''Parse command line arguments. Returns args object.
    NOTE: Need to move command-line argument parsing to separate file.'''

    parser = argparse.ArgumentParser()
    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='8508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f. Alternative: nref11n_nref10f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='Which output? Default is RD0027 = redshift 2')
    parser.set_defaults(output='RD0020')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is ramona_pleiades')
    parser.set_defaults(system='ramona_pleiades')

    parser.add_argument('--pwd', dest='pwd', action='store_true', \
                        help='Just use the working directory? Default is no')
    parser.set_defaults(pwd=False)

    args = parser.parse_args()
    return args

args = parse_args()
foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
output_dir = output_dir+"plots_for_keerthi/"
if not (os.path.exists(output_dir)): os.system('mkdir -p ' + output_dir)
os.chdir(output_dir)
halo = args.halo
sim = args.run
snap = args.output

filename = foggie_dir+'halo_00'+halo+'/'+sim+'/'+snap+'/'+snap
track_name = trackname

#ds, region = fl(filename,trackname)
ds, region = fl(filename, trackname, \
                        particle_type_for_angmom=False, do_filter_particles=False, \
                        region='refine_box') # this *SHOULD* work better, I just hope I'm not losing anything important
# if halo_c_v file does not include the halo center, foggie_load will try to calculate it (which doesnt work without problems in yt4 so here is the workaround from Ayan using load_sim)


#ds, refine_box = load_sim(args, region='refine_box')
#args.halo_center = ds.halo_center_kpc
#args.halo_velocity = ds.halo_velocity_kms
#[centerx,centery,centerz] = ds.halo_center
#args.halo_velocity = ds.halo_velocity_kms



#for chosenion in ['O VI','C II','Ca II','C IV','Si II','Si III','Si IV', 'Mg I', 'Mg II', 'H I']:
for chosenion in ['C II','Ca II','Si II','Si III','Si IV', 'Mg I', 'Mg II', 'H I']:
    trident.add_ion_fields(ds, ions=[chosenion])

fields_of_interest = ["density","temperature", "metallicity", \
                      "radial_velocity_corrected", \
                      'Si_p1_number_density', 'Si_p2_number_density', 'Si_p3_number_density', 'C_p1_number_density', \
                      'Ca_p1_number_density', 'Mg_p0_number_density', 'Mg_p1_number_density', 'H_p0_number_density' \
                      ]

[centerx,centery,centerz]=region.center


leftedge = region.center - ds.quan(20/2., 'kpc').in_units('code_length')
rightedge = region.center + ds.quan(20/2., 'kpc').in_units('code_length')

data_source = ds.box(leftedge, rightedge)
#data_source = ds.sphere(region.center, (20, 'kpc'))

for field in fields_of_interest:
    for or in ['x','y','z']:


        prj = yt.ProjectionPlot(ds, or, ("gas", field),
                               # center=chosencenter, width=(chosenwidth,'kpc'),weight_field=("gas", "density"), data_source=data_source)
                                center=region.center, width=(20,'kpc'), data_source=data_source)

        prj.annotate_clumps(leaf_clumps)
        plotsdir = output_dir +'plots'
        if not (os.path.exists(plotsdir)): os.system('mkdir -p ' + plotsdir)
        prj.save(plotsdir+'/halo_00'+halo+'_'+sim+'_'+snap+'_'+or+'_'+field+'_proj_20kpc_box.png')




"""
#PBS -S /bin/sh
#PBS -N z2proj
#PBS -l select=1:ncpus=16:model=ldan:mem=750GB
#PBS -q ldan
#PBS -l walltime=5:00:00
#PBS -j oe
#PBS -o /home5/raugust4/WORK/Outputs/z2proj.out
#PBS -m abe
#PBS -V
#PBS -W group_list=s2358
#PBS -l site=needed=/home5+/nobackupp13
#PBS -M raugustin@stsci.edu
#PBS -e /home5/raugust4/WORK/Outputs/z2proj.err
#PBS -koed


source /home5/raugust4/.bashrc

/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/plotsforkeerthi.py --halo=2392 > /home5/raugust4/z2proj.log
/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/plotsforkeerthi.py --halo=2878 > /home5/raugust4/z2proj.log
/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/plotsforkeerthi.py --halo=4123 > /home5/raugust4/z2proj.log
/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/plotsforkeerthi.py --halo=5016 > /home5/raugust4/z2proj.log
/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/plotsforkeerthi.py --halo=5036 > /home5/raugust4/z2proj.log
/home5/raugust4/anaconda3/bin/python3 /home5/raugust4/offlinescripts/plotsforkeerthi.py --halo=8508 > /home5/raugust4/z2proj.log




"""
