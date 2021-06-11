import yt
from yt.data_objects.level_sets.api import *
from foggie.utils.foggie_load import foggie_load as fl
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
                        help='Which output? Default is RD0027 = redshift 1')
    parser.set_defaults(output='RD0027')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is ramona_pleiades')
    parser.set_defaults(system='ramona_pleiades')

    parser.add_argument('--pwd', dest='pwd', action='store_true', \
                        help='Just use the working directory? Default is no')
    parser.set_defaults(pwd=False)

    parser.add_argument('--width', metavar='width', type=float, action='store', \
                        help='Width of the box around the halo center in kpc. default = 30')
    parser.set_defaults(width=30.)

    parser.add_argument('--step', metavar='step', type=float, action='store', \
                        help='clumpfinder step parameter. default = 2. ')
    parser.set_defaults(step=2.)

    parser.add_argument('--patchname', metavar='patchname', type=str, action='store', \
                        help='Name  for the patch to find clumps? Default is central_30kpc')
    parser.set_defaults(patchname='box1')

    parser.add_argument('--center', metavar='center', type=float, action='store', \
                        help='Center of the box in the halo center in code units. default = center1')
    parser.set_defaults(center=center1)

    args = parser.parse_args()
    return args

args = parse_args()
foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
patchname = args.patchname
output_dir = output_dir+"clumps/"+patchname+'/'
if not (os.path.exists(output_dir)): os.system('mkdir -p ' + output_dir)
os.chdir(output_dir)
halo = args.halo
sim = args.run
snap = args.output

filename = foggie_dir+'halo_00'+halo+'/'+sim+'/'+snap+'/'+snap
track_name = trackname
ds, region = fl(filename,trackname)

for chosenion in ['O VI','C II','C IV','Si II','Si III','Si IV', 'Mg I', 'Mg II', 'H I']:
    trident.add_ion_fields(ds, ions=[chosenion])



chosenwidth = args.width

dx= ds.quan(chosenwidth,'kpc').in_units('code_length')

dy= ds.quan(chosenwidth,'kpc').in_units('code_length')

dz= ds.quan(chosenwidth,'kpc').in_units('code_length')


[centerx,centery,centerz]=region.center


center1 = [centerx+dx,centery,centerz]
center2 = [centerx+dx,centery+dy,centerz]
center3 = [centerx+dx,centery+dy,centerz+dz]
center4 = [centerx+dx,centery+dy,centerz-dz]
center5 = [centerx+dx,centery,centerz+dz]
center6 = [centerx+dx,centery,centerz-dz]
center7 = [centerx+dx,centery-dy,centerz]
center8 = [centerx+dx,centery-dy,centerz+dz]
center9 = [centerx+dx,centery-dy,centerz-dz]
center10 = [centerx,centery,centerz]
center11 = [centerx,centery+dy,centerz]
center12 = [centerx,centery+dy,centerz+dz]
center13 = [centerx,centery+dy,centerz-dz]
center14 = [centerx,centery,centerz+dz]
center15 = [centerx,centery,centerz-dz]
center16 = [centerx,centery-dy,centerz]
center17 = [centerx,centery-dy,centerz+dz]
center18 = [centerx,centery-dy,centerz-dz]
center19 = [centerx-dx,centery,centerz]
center20 = [centerx-dx,centery+dy,centerz]
center21 = [centerx-dx,centery+dy,centerz+dz]
center22 = [centerx-dx,centery+dy,centerz-dz]
center23 = [centerx-dx,centery,centerz+dz]
center24 = [centerx-dx,centery,centerz-dz]
center25 = [centerx-dx,centery-dy,centerz]
center26 = [centerx-dx,centery-dy,centerz+dz]
center27 = [centerx-dx,centery-dy,centerz-dz]

print(center)
chosencenter =  args.center
data_source = ds.sphere(chosencenter, (chosenwidth, 'kpc'))

#yt.ProjectionPlot(ds, 2, ("gas", "density"), center=chosencenter, width=(chosenwidth,'kpc'),data_source=data_source, weight_field=("gas", "density")).show()

#yt.ProjectionPlot(ds, 2, ("gas", "temperature"), center=chosencenter, width=(chosenwidth,'kpc'),data_source=data_source, weight_field=("gas", "density")).show()

#yt.ProjectionPlot(ds, 2, ("gas", "metallicity"), center=chosencenter, width=(chosenwidth,'kpc'),data_source=data_source, weight_field=("gas", "density")).show()


master_clump1 = Clump(data_source, ("gas", "density"))
master_clump1.add_validator("min_cells", 20)
c_min = data_source["gas", "density"].min()
c_max = data_source["gas", "density"].max()
step = args.step #100. #2.0
find_clumps(master_clump1, c_min, c_max, step)

leaf_clumps = master_clump1.leaves
prj = yt.ProjectionPlot(ds, 0, ("gas", "density"),
                       # center=chosencenter, width=(chosenwidth,'kpc'),weight_field=("gas", "density"), data_source=data_source)
                        center=chosencenter, width=(chosenwidth,'kpc'), data_source=data_source)

prj.annotate_clumps(leaf_clumps)
plotsdir = output_dir +'plots'
if not (os.path.exists(plotsdir)): os.system('mkdir -p ' + plotsdir)
prj.save(plotsdir+'/halo_00'+halo+'_'+sim+'_'+snap+'_'+snap+'_clumps_density.png')
#prj.show()

master_clump=master_clump1
master_clump.add_info_item("total_cells")
master_clump.add_info_item("cell_mass")
master_clump.add_info_item("mass_weighted_jeans_mass")
master_clump.add_info_item("volume_weighted_jeans_mass")
master_clump.add_info_item("max_grid_level")
master_clump.add_info_item("min_number_density")
master_clump.add_info_item("max_number_density")
master_clump.add_info_item("center_of_mass")
master_clump.add_info_item("distance_to_main_clump")



fields_of_interest = [("gas", "density"),("gas", "temperature"), ("gas", "metallicity"),"particle_mass",'particle_position',("gas", 'cell_mass'),("gas", "cell_volume"), \
                      ("gas", 'radial_velocity_corrected'), \
                      ("gas", 'Si_p1_number_density'), ("gas", 'Si_p2_number_density'), ("gas", 'Si_p3_number_density'), ("gas", 'C_p1_number_density'), ("gas", 'C_p3_number_density'), ("gas", 'O_p5_number_density'), ("gas", 'Mg_p0_number_density'),("gas", 'Mg_p1_number_density'),("gas", 'H_p0_number_density'), \
                      ("gas", 'Si_p1_mass'), ("gas", 'Si_p2_mass'), ("gas", 'Si_p3_mass'), ("gas", 'C_p1_mass'), ("gas", 'C_p3_mass'), ("gas", 'O_p5_mass'), ("gas", 'Mg_p0_mass'),("gas", 'Mg_p1_mass'),("gas", 'H_p0_mass') \
                      ]


fn = master_clump.save_as_dataset(filename='halo_00'+halo+'_'+sim+'_'+snap+'_'+snap+'_clumps_tree',fields=fields_of_interest)
leaf_clumps = master_clump.leaves

indclumpdir = output_dir +'individual_clumps'
if not (os.path.exists(indclumpdir)): os.system('mkdir -p ' + indclumpdir)
for clump in leaf_clumps:
    clumpfn=str(clump.clump_id)+'_single_clump'
    #clump.save_as_dataset(filename=clumpfn,fields=["density", "particle_mass",'particle_position'])
    clump.data.save_as_dataset(filename=indclumpdir+'/'+clumpfn,fields=fields_of_interest)


filename = 'halo_00'+halo+'_'+sim+'_'+snap+'_'+snap+'_clumps_cut_region'
master_clump.data.save_as_dataset(filename=filename,fields=fields_of_interest)

"""
clumpmasses = []
clumpvolumes = []
failedclumps = []
for i in range(100):
    i=i+1
    clumpfile=str(i)+"_single_clump.h5"
    if (os.path.exists(clumpfile)):
        clump1 = yt.load(clumpfile)
        ad = clump1.all_data()
        try:
            clumpmass = ad["gas", "cell_mass"].sum().in_units("Msun")
            clumpvolume = ad["gas", "cell_volume"].sum().in_units("kpc**3")
            print(i)
            print(clumpmass)
            print(clumpvolume)
            clumpmasses.append(clumpmass)
            clumpvolumes.append(clumpvolume)

        except ValueError:
            failedclumps.append(i)
            pass

print('Failed clumps: ')
print(failedclumps)

clumpmasses=np.array(clumpmasses)
clumpvolumes=np.array(clumpvolumes)


plt.figure()
plt.hist(clumpvolumes)
plt.savefig('clumpvolumes.png')

plt.figure()
plt.hist(clumpmasses)
plt.savefig('clumpmasses.png')


clumpradii = (3/4/np.pi * clumpvolumes)**(1/3)

plt.figure()
plt.hist(clumpradii)
plt.savefig('clumpradii.png')
"""
