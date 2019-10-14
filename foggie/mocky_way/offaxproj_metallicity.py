import yt
from core_funcs import prepdata

dd_name = 'DD2175'
sim_name = 'nref11n_nref10f'

ds, ds_paras = prepdata(dd_name, sim_name=sim_name)

from core_funcs import obj_source_all_disk_cgm
obs_point = 'halo_center'
obj_tag = 'all'
observer_location = ds_paras[obs_point]
obj_source = obj_source_all_disk_cgm(ds, ds_paras, obj_tag)

from foggie.utils import consistency
cmap = consistency.metal_discrete_cmap
field = 'metallicity'
width = 2*ds.quan(120, 'kpc')
zmin = 0.01
zmax = 2.0
##################
pj = yt.OffAxisProjectionPlot(ds, ds_paras['L_vec'], field,
                              center=ds_paras['halo_center'],
                              width=width,
                              data_source=obj_source,
                              weight_field='cell_mass',
                              north_vector=ds_paras['phi_vec']
                              )
pj.set_cmap(field=field, cmap=cmap)
pj.set_zlim(field, zmin=zmin, zmax=zmax)
pj.save('figs/offaxproj_dshader/%s_%s_yt_faceon_Z.pdf'%(sim_name, dd_name))

################
pj = yt.OffAxisProjectionPlot(ds, ds_paras['phi_vec'], field,
                              center=ds_paras['halo_center'],
 			      width=width,
                              data_source=obj_source,
                              weight_field='cell_mass',
                              north_vector=ds_paras['L_vec']
                              )
pj.set_cmap(field=field, cmap=cmap)
pj.set_zlim(field, zmin=zmin, zmax=zmax)
pj.save('figs/offaxproj_dshader/%s_%s_yt_edgeon1_Z.pdf'%(sim_name, dd_name))

################
pj = yt.OffAxisProjectionPlot(ds, ds_paras['sun_vec'], field,
                              center=ds_paras['halo_center'],
                              width=width,
                              data_source=obj_source,
                              weight_field='cell_mass',
                              north_vector=ds_paras['L_vec']
                              )
pj.set_cmap(field=field, cmap=cmap)
pj.set_zlim(field, zmin=zmin, zmax=zmax)
pj.save('figs/offaxproj_dshader/%s_%s_yt_edgeon2_Z.pdf'%(sim_name, dd_name))
