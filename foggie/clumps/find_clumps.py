import yt
from yt.data_objects.level_sets.api import *
from foggie.utils.foggie_load import foggie_load as fl
import trident

filename = '/Users/raugustin/WORK/SIMULATIONS/halo_008508/nref11n_nref10f/RD0025/RD0025'
trackname = "/Users/raugustin/foggie/foggie/halo_tracks/008508/nref11n_selfshield_15/halo_track_200kpc_nref10"
ds, region = fl(filename,trackname)

[centerx,centery,centerz]=region.center
dx= ds.quan(10.,'kpc').in_units('code_length')

dy= ds.quan(12.,'kpc').in_units('code_length')

dz= ds.quan(19.,'kpc').in_units('code_length')

print(dx)
chosencenter=[centerx+dx,centery+dy,centerz+dz]
chosencenter = region.center
chosenwidth = 5
data_source = ds.sphere(chosencenter, (chosenwidth, 'kpc'))

#yt.ProjectionPlot(ds, 2, ("gas", "density"), center=chosencenter, width=(chosenwidth,'kpc'),data_source=data_source, weight_field=("gas", "density")).show()

#yt.ProjectionPlot(ds, 2, ("gas", "temperature"), center=chosencenter, width=(chosenwidth,'kpc'),data_source=data_source, weight_field=("gas", "density")).show()

#yt.ProjectionPlot(ds, 2, ("gas", "metallicity"), center=chosencenter, width=(chosenwidth,'kpc'),data_source=data_source, weight_field=("gas", "density")).show()


master_clump1 = Clump(data_source, ("gas", "density"))
master_clump1.add_validator("min_cells", 7)
c_min = data_source["gas", "density"].min()
c_max = data_source["gas", "density"].max()
step = 100 #100. #2.0
find_clumps(master_clump1, c_min, c_max, step)

leaf_clumps = master_clump1.leaves
prj = yt.ProjectionPlot(ds, 0, ("gas", "density"),
                       # center=chosencenter, width=(chosenwidth,'kpc'),weight_field=("gas", "density"), data_source=data_source)
                        center=chosencenter, width=(chosenwidth,'kpc'), data_source=data_source)

prj.annotate_clumps(leaf_clumps)
prj.save('/Users/raugustin/Desktop/clumps_density.png')
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


for chosenion in ['O VI','C II','C IV','Si II','Si III','Si IV', 'Mg I', 'Mg II', 'H I']:
    trident.add_ion_fields(ds, ions=[chosenion])

fields_of_interest = ["density","temperature", "metallicity","particle_mass",'particle_position','cell_mass',"cell_volume", \
                      'radial_velocity_corrected', \
                      'Si_p1_number_density', 'Si_p2_number_density', 'Si_p3_number_density', 'C_p1_number_density', 'C_p3_number_density', 'O_p5_number_density', 'Mg_p0_number_density','Mg_p1_number_density','H_p0_number_density' \
                      'Si_p1_number_mass', 'Si_p2_number_mass', 'Si_p3_number_mass', 'C_p1_number_mass', 'C_p3_number_mass', 'O_p5_number_mass', 'Mg_p0_number_mass','Mg_p1_number_mass','H_p0_number_mass' \
                      ]
fn = master_clump.save_as_dataset(filename='/Users/raugustin/Desktop/clumpplots/RD0025_clumps_10kpc',fields=fields_of_interest)
leaf_clumps = master_clump.leaves
for clump in leaf_clumps:
    clumpfn=str(clump.clump_id)+'_single_clump'
    #clump.save_as_dataset(filename=clumpfn,fields=["density", "particle_mass",'particle_position'])
    clump.data.save_as_dataset(filename=clumpfn,fields=fields_of_interest)



"""

totmass=0.
totmass2=0.
for clump in master_clump:
    print(clump.clump_id)
    #print(clump.quantities.total_mass().in_units("Msun"))
    print(clump.quantities.total_quantity(["cell_mass"]).in_units("Msun"))
    print(clump.info["cell_mass"])
    totmass+=clump.info["cell_mass"][1].value
    totmass2+=clump.quantities.total_quantity(["cell_mass"]).in_units("Msun").value

print(totmass)
print(totmass2)


masses =[]
distances = []
for clump in master_clump:
    print(clump.clump_id)
    #print(clump["gas","density"])
    print(clump.quantities.total_quantity(["cell_mass"]).in_units("Msun").value)
    masses.append(clump.quantities.total_quantity(["cell_mass"]).in_units("Msun").value)
    print(clump.info["distance_to_main_clump"][1].value)
    distances.append(clump.info["distance_to_main_clump"][1].value)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.plot(distances, masses,'o')
plt.yscale('log')
plt.ylabel('clump mass [Msol]')
plt.xlabel('separation from main clump [pc]')
plt.savefig('/Users/raugustin/Desktop/distmass.png')
#plt.show()

plt.figure()
plt.hist(masses)
plt.savefig('/Users/raugustin/Desktop/masshist.png')

"""
filename = '/Users/raugustin/WORK/RD0025_clumps_test'
#master_clump.save_as_dataset(filename=filename,fields=[('gas','x'),('gas','y'),('gas','z')])
