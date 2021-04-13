import yt
from yt.data_objects.level_sets.api import *
from foggie.utils.foggie_load import foggie_load as fl
import os
import matplotlib.pyplot as plt
import numpy as np
from foggie.utils.get_run_loc_etc import get_run_loc_etc
import argparse

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
                        help='Which output? Default is RD0027 = redshift 1')
    parser.set_defaults(output='RD0027')

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
output_dir = output_dir+"clumps/"
if not (os.path.exists(output_dir)): os.system('mkdir -p ' + output_dir)
os.chdir(output_dir)
halo = args.halo
sim = args.run
snap = args.output

filename = foggie_dir+'halo_00'+halo+'/'+sim+'/'+snap+'/'+snap
track_name = trackname





#output_dir = "/Users/raugustin/"
#os.chdir(output_dir)

clumpmasses = []
clumpvolumes = []
elongations = []

for i in range(100):
    i=i+1
    if i != 17:
        print(i)
        clumpfile=str(i)+"_single_clump.h5"
        if (os.path.exists(clumpfile)):
            clump1 = yt.load(clumpfile)
            ad = clump1.all_data()
            clumpmass = ad["gas", "cell_mass"].sum().in_units("Msun")
            clumpvolume = ad["gas", "cell_volume"].sum().in_units("kpc**3")
            clumpmasses.append(clumpmass)
            clumpvolumes.append(clumpvolume)

            x_extend = (ad["grid", "x"].max().in_units("kpc") - ad["gas", "x"].min().in_units("kpc"))
            y_extend = (ad["grid", "y"].max().in_units("kpc") - ad["gas", "y"].min().in_units("kpc"))
            z_extend = (ad["grid", "z"].max().in_units("kpc") - ad["gas", "z"].min().in_units("kpc"))
            maxex=max([x_extend,y_extend,z_extend]).value + ad["grid", "dx"].mean().in_units("kpc").value
            minex=min([x_extend,y_extend,z_extend]).value + ad["grid", "dx"].mean().in_units("kpc").value
            elo=(np.max([x_extend.value,y_extend.value,z_extend.value])-np.min([x_extend.value,y_extend.value,z_extend.value]))/(np.max([x_extend.value,y_extend.value,z_extend.value])+np.min([x_extend.value,y_extend.value,z_extend.value]))
            elo = minex/maxex
            elongations.append(elo)

clumpmasses=np.array(clumpmasses)
clumpvolumes=np.array(clumpvolumes)
elongations=np.array(elongations)


plt.figure()
plt.hist(clumpvolumes)
plt.ylabel('counts')
plt.xlabel('clump volume [kpc^3]')
plt.savefig('clumpvolumes.png')

plt.figure()
plt.hist(clumpmasses)
plt.ylabel('counts')
plt.xlabel('clump mass [Msol]')
plt.savefig('clumpmasses.png')


clumpradii = (3/4/np.pi * clumpvolumes)**(1/3)

plt.figure()
plt.hist(clumpradii)
plt.ylabel('counts')
plt.xlabel('clump radius [kpc]')
plt.savefig('clumpradii.png')


plt.figure()
plt.hist(elongations)
plt.ylabel('counts')
plt.xlabel('spherical <- elongation -> filamentary')
plt.savefig('elongations.png')
