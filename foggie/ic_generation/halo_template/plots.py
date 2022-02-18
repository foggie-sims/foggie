
import yt
from astropy.table import Table
import numpy as np
from foggie.utils.consistency import *

halos = Table.read('25Mpc_256_shielded-L0/BigBox_z2_rockstar/out_0.list', format='ascii', header_start=0)
halos.sort('Mvir')
halos.reverse()

id_halo = 4954
#l1 =  (  -41,  -12, -118)
#l2 =  (  -41,  -12, -118)
#l3 =  (  -41,  -12, -118)
a = halos[[halos['ID'] == id_halo]]
center0 = [ a['X'][0]/25., a['Y'][0]/25, a['Z'][0]/25.]

#obtain the l1 shifts
file = open('halo'+str(id_halo)+'/l0_to_l1_shifts')
line = file.read()
l1 = ( int(str.split(line)[0]), int(str.split(line)[1]), int(str.split(line)[2]))

#obtain the l2 shifts
file = open('halo'+str(id_halo)+'/l1_to_l2_shifts')
line = file.read()
l2 = ( int(str.split(line)[0]), int(str.split(line)[1]), int(str.split(line)[2]))

l3 = l2 #<---- THIS IS A KLUDGE BECAUSE THE SCRIPT DOES NOT GENERATE L3 SHIFTS

print("L0 to L1 shifts: ", l1)
print("L1 to L2 shifts: ", l2)

# L0 Gas
ds = yt.load('25Mpc_256_shielded-L0/RD0111/RD0111')
box = ds.r[(a['X']-1.)/25.:(a['X']+1.)/25., 0:1, 0:1] #<---- L0 gas uses the halo catalog center
p = yt.ProjectionPlot(ds, 'x', 'density', center=center0, data_source=box, width=(1, 'Mpc') )
p.annotate_sphere(center0, radius=(a['Rvir'], "kpc"), circle_args={"color": "white"})
p.set_cmap('density', cmap = density_color_map)
p.annotate_text((0.06, 0.08), "M = "+"{0:.2e}".format(a['Mvir'][0]), coord_system="axis")
p.annotate_text((0.06, 0.12), a['ID'][0], coord_system="axis")
p.annotate_text(center0, a['ID'][0], coord_system='data')
p.save('halo'+str(id_halo)+'/halo'+str(id_halo)+'_L0_gas')

# L0 DM from gas run
p = yt.ProjectionPlot(ds, 'x', 'all_density', center=center0, data_source=box, width=(1, 'Mpc') )
p.annotate_sphere(center0, radius=(a['Rvir'], "kpc"), circle_args={"color": "white"})
p.annotate_text((0.06, 0.08), "M = "+"{0:.2e}".format(a['Mvir'][0]), coord_system="axis")
p.annotate_text((0.06, 0.12), a['ID'][0], coord_system="axis")
p.annotate_text(center0, a['ID'][0], coord_system='data')
p.set_zlim('all_density', 1e-4, 0.1)
p.set_cmap('all_density', cmap = 'arbre')
p.save('halo'+str(id_halo)+'/halo'+str(id_halo)+'_L0_gas')


# ## L0 DM
dm0 = yt.load('25Mpc_DM_256-L0/RD0111/RD0111')
box0 = dm0.r[ (center0[0]-1./25.):(center0[0]+1./25.), 0:1, 0:1]    #<---- L0 DM uses the halo catalog center
p = yt.ProjectionPlot(dm0, 'x', 'all_density', data_source=box0, center=center0, width=(1, 'Mpc') )
p.annotate_sphere(center0, radius=(a['Rvir'], "kpc"), circle_args={"color": "white"})
p.annotate_text((0.06, 0.08), "M = "+"{0:.2e}".format(a['Mvir'][0]), coord_system="axis")
p.annotate_text((0.06, 0.12), a['ID'][0], coord_system="axis")
p.annotate_text(center0, a['ID'][0], coord_system='data')
p.set_zlim('all_density', 1e-4, 0.1)
p.set_cmap('all_density', cmap = 'arbre')
p.save('halo'+str(id_halo)+'/halo'+str(id_halo)+'_L0_dm')


# ## L1 DM
dm1 = yt.load('halo'+str(id_halo)+'/25Mpc_DM_256-L1/RD0111/RD0111')
center1 = [center0[0] + l1[0] / 255., center0[1] + l1[1]/255., center0[2] + l1[2]/255.] # <------ L1 coords
print("Center 1 = ", center1)
box1 = dm1.r[ center1[0]-1./25.:center1[0]+1./25., 0:1, 0:1]
p = yt.ProjectionPlot(dm1, 'x', 'all_density', data_source=box1, center=center1, width=(1, 'Mpc') )
p.annotate_sphere(center1, radius=(a['Rvir'], "kpc"), circle_args={"color": "white"})
p.annotate_text((0.06, 0.08), "M = "+"{0:.2e}".format(a['Mvir'][0]), coord_system="axis")
p.annotate_text((0.06, 0.12), a['ID'][0], coord_system="axis")
p.annotate_text(center1, a['ID'][0], coord_system='data')
p.set_zlim('all_density', 1e-4, 0.1)
p.set_cmap('all_density', cmap = 'arbre')
p.save('halo'+str(id_halo)+'/halo'+str(id_halo)+'_L1_dm')


# ## L2 DM
dm2 = yt.load('halo'+str(id_halo)+'/25Mpc_DM_256-L2/RD0111/RD0111')
center2 = [center0[0] + l2[0] / 255.,center0[1] + l2[1]/255., center0[2] + l2[2]/255.] # <------ L2 coords
print("Center 2 = ", center2)
box2 = dm2.r[ center2[0]-1./25.:center2[0]+1./25., 0:1, 0:1]
p = yt.ProjectionPlot(dm2, 'x', 'all_density', data_source=box2, center=center2, width=(1, 'Mpc') )
p.annotate_sphere(center2, radius=(a['Rvir'], "kpc"), circle_args={"color": "white"})
p.annotate_text((0.06, 0.08), "M = "+"{0:.2e}".format(a['Mvir'][0]), coord_system="axis")
p.annotate_text((0.06, 0.12), a['ID'][0], coord_system="axis")
p.annotate_text(center2, a['ID'][0], coord_system='data')
p.set_cmap('all_density', cmap = 'arbre')
p.set_zlim('all_density', 1e-4, 0.1)
p.save('halo'+str(id_halo)+'/halo'+str(id_halo)+'_L2_dm')


# ## L3 DM
dm3 = yt.load('halo'+str(id_halo)+'/25Mpc_DM_256-L3/RD0111/RD0111')
center3 = [center0[0] + l3[0] / 255.,center0[1] + l3[1]/255., center0[2] + l3[2]/255.] # <------ L2 coords
print("Center 3 = ", center3)
box3 = dm3.r[ center3[0]-1./25.:center3[0]+1./25., 0:1, 0:1]
p = yt.ProjectionPlot(dm3, 'x', 'all_density', data_source=box3, center=center2, width=(1, 'Mpc') )
p.annotate_sphere(center3, radius=(a['Rvir'], "kpc"), circle_args={"color": "white"})
p.annotate_text((0.06, 0.08), "M = "+"{0:.2e}".format(a['Mvir'][0]), coord_system="axis")
p.annotate_text((0.06, 0.12), a['ID'][0], coord_system="axis")
p.annotate_text(center3, a['ID'][0], coord_system='data')
p.set_zlim('all_density', 1e-4, 0.1)
p.set_cmap('all_density', cmap = 'arbre')
p.save('halo'+str(id_halo)+'/halo'+str(id_halo)+'_L3_dm')
