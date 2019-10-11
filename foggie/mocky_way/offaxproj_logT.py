import yt
from core_funcs import prepdata

dd_name = 'DD2175'
sim_name = 'nref11n_nref10f'

ds, ds_paras = prepdata(dd_name)

from core_funcs import obj_source_all_disk_cgm
obs_point = 'halo_center'
obj_tag = 'all'
observer_location = ds_paras[obs_point]
obj_source = obj_source_all_disk_cgm(ds, ds_paras, obj_tag)

from foggie.utils import consistency
cmap = consistency.logT_discrete_cmap_mw
field = 'temperature'
width = 2*ds.quan(130, 'kpc')
zmin = 10**3.5
zmax = 10**6.5
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
pj.save('figs/offaxproj_dshader/%s_%s_yt_faceon_logT.pdf'%(sim_name, dd_name))

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
pj.save('figs/offaxproj_dshader/%s_%s_yt_edgeon1_logT.pdf'%(sim_name, dd_name))

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
pj.save('figs/offaxproj_dshader/%s_%s_yt_edgeon2_logT.pdf'%(sim_name, dd_name))
