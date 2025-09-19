
import yt
from astropy.table import Table
import numpy as np
from foggie.utils.consistency import *

halos = Table.read('/nobackup/jtumlins/CGM_bigbox/25Mpc_512_shielded-L0/BigBox_z2_rockstar/out_0.list', format='ascii', header_start=0)
halos.sort('Mvir')
halos.reverse()

ds = yt.load('/nobackup/jtumlins/CGM_bigbox/25Mpc_512_shielded-L0/RD0111/RD0111')

id_halo = XXXX  
index = [halos['ID'] == id_halo]
a = halos[index]

#plot position of this halo in the 512^3 gas run
center0 = [ a['X'][0]/25., a['Y'][0]/25, a['Z'][0]/25.] #<-----L0 gas run, halo catalog
box = ds.r[(a['X']-0.5)/25.:(a['X']+0.5)/25., 0:1, 0:1]
p = yt.ProjectionPlot(ds, 'x', 'density', center=center0, data_source=box, width=(1, 'Mpc') )
p.annotate_sphere(center0, radius=(a['Rvir'], "kpc"), circle_args={"color": "white"})
p.set_cmap('density', cmap = density_color_map)
p.annotate_text((0.06, 0.08), "M = "+"{0:.2e}".format(a['Mvir'][0]), coord_system="axis")
p.annotate_text((0.06, 0.12), a['ID'][0], coord_system="axis")
p.annotate_text(center0, a['ID'][0], coord_system='data')
p.save('RD0111_L0_gas')


#now plot this in the DM only run
dm = yt.load('/nobackup/jtumlins/CGM_bigbox/25Mpc_DM_515120/RD0211/RD0111')
box = dm.r[ (center0[0]-0.5/25.):(center0[0]+0.5/25.), 0:1, 0:1]
p = yt.ProjectionPlot(dm, 'x', 'all_density', data_source=box, center=center0, width=(1, 'Mpc') )
p.annotate_sphere(center0, radius=(a['Rvir'], "kpc"), circle_args={"color": "white"})
p.annotate_text((0.06, 0.08), "M = "+"{0:.2e}".format(a['Mvir'][0]), coord_system="axis")
p.annotate_text((0.06, 0.12), a['ID'][0], coord_system="axis")
p.annotate_text(center0, a['ID'][0], coord_system='data')
p.set_zlim('all_density', 1e-4, 0.1)
p.save('RD0111_L0_dm')



center1 = [center0[0] - 118. / 511.,center0[1] - 77./511., center0[2] + 45./511.] # <------ L1 coords
