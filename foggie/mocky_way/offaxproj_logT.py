import yt
from foggie.mocky_way.core_funcs import prepdata

dd_name = 'DD2175'
sim_name = 'nref11n_nref10f'
obs_point = 'halo_center'
vel_tag = 'vel_pos'

### get the data source, cut the region with velocity constrains
ds, ds_paras = prepdata(dd_name, sim_name=sim_name)
sp = ds.sphere(ds_paras['halo_center'], (120, 'kpc'))
width = 2*ds.quan(120, 'kpc')

#### set field parasmeters for the los_velocity_mw
observer_location = ds_paras[obs_point]
if obs_point == 'halo_center':
    observer_bulkvel = ds_paras['disk_bulkvel']
elif obs_point == 'offcenter_location':
    observer_bulkvel = ds_paras['offcenter_bulkvel']
else:
    print("Have not idea what obs_point is, please check")
    sys.exit()
sp.set_field_parameter('observer_location', observer_location)
sp.set_field_parameter('observer_bulkvel', observer_bulkvel)

if vel_tag == 'vel_pos':
    obj_source = sp.cut_region(["obj['los_velocity_mw'] >= 0"])
elif vel_tag == 'vel_neg':
    obj_source = sp.cut_region(["obj['los_velocity_mw'] < 0"])
else:
    print("I have no idea what you want to proj with vel_tag, please check.")
    sys.exit()

from foggie.utils import consistency
cmap = consistency.logT_discrete_cmap_mw
field = 'temperature'
zmin = 10**3.5
zmax = 10**6.5
figdir = 'figs/offaxproj'

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
figname = '%s_%s_%s_faceon_%s_logT.pdf'%(sim_name, dd_name, obs_point, vel_tag)
pj.save('%s/%s'%(figdir, figname))

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
figname = '%s_%s_%s_edgeon1_%s_logT.pdf'%(sim_name, dd_name, obs_point, vel_tag)
pj.save('%s/%s'%(figdir, figname))

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
figname = '%s_%s_%s_edgeon2_%s_logT.pdf'%(sim_name, dd_name, obs_point, vel_tag)
pj.save('%s/%s'%(figdir, figname))
