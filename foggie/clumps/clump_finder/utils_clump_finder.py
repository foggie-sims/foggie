import numpy as np
from foggie.utils.consistency import *
from yt.data_objects.level_sets.api import * #TODO - Not super slow here, but replace with custom clump finding to find disk?
import h5py
from functools import partial
global internal_clump_counter
internal_clump_counter = 0

halo_dict = {   '2392'  :  'Hurricane' ,
                '2878'  :  'Cyclone' ,
                '4123'  :  'Blizzard' ,
                '5016'  :  'Squall' ,
                '5036'  :  'Maelstrom' ,
                '8508'  :  'Tempest',
                '002392'  :  'Hurricane' ,
                '002878'  :  'Cyclone' ,
                '004123'  :  'Blizzard' ,
                '005016'  :  'Squall' ,
                '005036'  :  'Maelstrom' ,
                '008508'  :  'Tempest' }

def halo_id_to_name(halo_id):
    return halo_dict[str(halo_id)]
    
def read_virial_mass_file(halo_id,snapshot,refinement_scheme,codedir,key="radius"):
    '''
    Read in single entries from the virial mass file for a certain key for 1 snapshot
    Keys include: ['redshift', 'snapshot', 'radius', 'total_mass', 'dm_mass', 'stars_mass',
    'young_stars_mass', 'old_stars_mass', 'gas_mass', 'gas_metal_mass', 'gas_H_mass',
    'gas_HI_mass', 'gas_HII_mass', 'gas_CII_mass', 'gas_CIII_mass', 'gas_CIV_mass',
    'gas_OVI_mass', 'gas_OVII_mass', 'gas_MgII_mass', 'gas_SiII_mass', 'gas_SiIII_mass', 
    'gas_SiIV_mass', 'gas_NeVIII_mass']
    '''
    from astropy.table import Table
    masses_dir = codedir+"halo_infos/"+halo_id+"/"+refinement_scheme+"/rvir_masses.hdf5"
    rvir_masses = Table.read(masses_dir, path='all_data')
    
    return rvir_masses[key][rvir_masses['snapshot']==snapshot][-1]
    
    

def get_cgm_density_cut(ds,cut_type="comoving_density",additional_factor=2.,run="nref11c_nref9f",code_dir=None,halo=None,snapshot=None, cut_field=('gas','density')):
    '''
    Get a density cutoff to separate the galaxy from the CGM
    '''


    if cut_type=="comoving_density":
        z = ds.get_parameter('CosmologyCurrentRedshift')
        cgm_density_cut = 0.1 *additional_factor* cgm_density_max * (1+z)**3
        if cut_field==('gas','H_p0_number_density'):
            column_density_threshold = additional_factor * 1e17 / ds.units.cm / ds.units.cm
            cgm_density_cut = column_density_threshold / np.min(ds.all_data()['gas','dx'].in_units('cm')) * (1+z)**3 #in units cm^-3


    elif cut_type=="relative_density":
        try: Rvir = read_virial_mass_file(halo, "RD0042",run,code_dir)
        except:
            print("Warning: Could not read rvir file for this halo...")
            Rvir = 300.
        print("Rvir set to:",Rvir)
        #ds, refine_box = foggie_load(snap_name, trackname, halo_c_v_name=halo_c_v_name, do_filter_particles=True,disk_relative=True,particle_type_for_angmom=particle_type_for_angmom,smooth_AM_name = smooth_AM_name)
        sphere = ds.sphere(center=ds.halo_center_kpc, radius=(Rvir, 'kpc')) 
        z = ds.get_parameter('CosmologyCurrentRedshift')
        mask = ((sphere['gas','radius_corrected']>50./(1+z)) & (sphere['gas','density']<=cgm_density_max*(1+z)**3))
        
        mean_density = np.average( sphere['gas','density'][mask], weights=sphere['cell_volume'][mask])
        stdv_density = np.sqrt( np.average( np.power(sphere['gas','density'][mask] - mean_density , 2) , weights=sphere['cell_volume'][mask]))
        cgm_density_cut = mean_density + additional_factor * stdv_density 


        if cut_field==('gas','H_p0_number_density'):
            '''
            column_density_threshold = additional_factor * 1e17 / ds.units.cm / ds.units.cm
            mask = ((sphere['gas','radius_corrected']>50./(1+z)))# & (sphere['gas','H_p0_number_density']<=1e17*(1+z)**3))
            
        
            mean_density = np.average( sphere['gas','H_p0_number_density'][mask], weights=sphere['cell_volume'][mask])
            stdv_density = np.sqrt( np.average( np.power(sphere['gas','H_p0_number_density'][mask] - mean_density , 2) , weights=sphere['cell_volume'][mask]))
            print("Mean density is:",mean_density)
            print("stdv_density=",stdv_density)

            cgm_density_cut = mean_density + additional_factor * stdv_density
            '''

            min_dx = np.min(ds.all_data()['gas','dx'])
            disk_thickness_factor = min_dx.in_units('kpc') / ds.units.kpc

            column_density_threshold = additional_factor * 1e17 / ds.units.cm / ds.units.cm
            cgm_density_cut = column_density_threshold / min_dx.in_units('cm') * disk_thickness_factor * (1+z)**3#in units cm^-3

            mask = sphere['gas','H_p0_number_density'] <= cgm_density_cut * 1. # Less than 10**18 cm^2
            mean_density = np.average( sphere['gas','H_p0_number_density'][mask], weights=sphere['cell_volume'][mask])
            stdv_density = np.sqrt( np.average( np.power(sphere['gas','H_p0_number_density'][mask] - mean_density , 2) , weights=sphere['cell_volume'][mask]))
            
            #mean_density = np.average( sphere['gas','H_p0_number_density'], weights=sphere['cell_volume'])
            #stdv_density = np.sqrt( np.average( np.power(sphere['gas','H_p0_number_density'] - mean_density , 2) , weights=sphere['cell_volume']))
            

            print("cgm_density_cut was:",cgm_density_cut,"stdv=",stdv_density)
            cgm_density_cut += 100. * stdv_density * (1+z)**3



    else:
        # Define the density cut between disk and CGM to vary smoothly between 1 and 0.1 between z = 0.5 and z = 0.25,
        # with it being 1 at higher redshifts and 0.1 at lower redshifts
        # Cassi's original definition
        current_time = ds.current_time.in_units('Myr').v
        if (current_time<=7091.48):
            density_cut_factor = 20. - 19.*current_time/7091.48
        elif (current_time<=8656.88):
            density_cut_factor = 1.
        elif (current_time<=10787.12):
            density_cut_factor = 1. - 0.9*(current_time-8656.88)/2130.24
        else:
            density_cut_factor = 0.1
        
        cgm_density_cut = cgm_density_max * density_cut_factor * additional_factor
        z = ds.get_parameter('CosmologyCurrentRedshift')


        if cut_field==('gas','H_p0_number_density'):
            column_density_threshold = additional_factor * 1e17 / ds.units.cm / ds.units.cm
            cgm_density_cut = column_density_threshold / np.min(ds.all_data()['gas','dx'].in_units('cm')) * density_cut_factor*10. #in units cm^-3


    return cgm_density_cut




global clump_cell_ids

def _masked_field(field,data):
    '''Only the given clump will be marked as True'''
    return np.isin(data["index","cell_id_2"],clump_cell_ids)


def _pseudo_masked_field(field,data,var_clump_cell_ids):
    '''Only the given clump will be marked as True'''
    return np.isin(data["index","cell_id_2"],var_clump_cell_ids)

def _unmasked_field(field,data):
    '''All but the given clump will be marked as True'''
    return ~np.isin(data["index","cell_id_2"],clump_cell_ids)    

def _pseudo_unmasked_field(field,data,var_clump_cell_ids):
    '''Only the given clump will be marked as True'''
    return ~np.isin(data["index","cell_id_2"],var_clump_cell_ids)

#global max_gid
#global gx_min; global gy_min; global gz_min
#global gx_max; global gy_max; global gz_max

def get_cell_grid_ids(field, data):
    '''Function to assign a unique cell_id to each cell based on it's index on its parent grid'''
    gids = (data['index','grid_indices'] + 1).astype(np.uint64) #These are different in yt and enzo...
    u_id = np.copy(gids)
    
    idx_dx = data['index','dx']

    x_id = np.divide(data['index','x'] - idx_dx/2. , idx_dx)
    y_id = np.divide(data['index','y'] - idx_dx/2. , idx_dx)
    z_id = np.divide(data['index','z'] - idx_dx/2. , idx_dx)
    
    gx = np.subtract(x_id , gx_min[gids-1])
    gy = np.subtract(y_id , gy_min[gids-1])
    gz = np.subtract(z_id , gz_min[gids-1])

    max_x = gx_max[gids-1] - gx_min[gids-1]
    max_y = gy_max[gids-1] - gy_min[gids-1]    

    c_ids = gx + np.multiply(gy , max_x+1) + np.multiply(gz , np.multiply(max_x+1 , max_y+1))

    u_id = np.round(gids + np.multiply(c_ids , max_gid+1)).astype(np.uint64)

    return u_id    
    
def pseudo_get_cell_grid_ids(field, data, max_gid, gx_min, gx_max, gy_min, gy_max, gz_min):
    '''Function to assign a unique cell_id to each cell based on it's index on its parent grid'''
    '''For use as a yt field, must define a partial function resembling get_cell_grid_ids'''
    gids = (data['index','grid_indices'] + 1).astype(np.uint64) #These are different in yt and enzo...
    u_id = np.copy(gids)
    
    idx_dx = data['index','dx']

    x_id = np.divide(data['index','x'] - idx_dx/2. , idx_dx)
    y_id = np.divide(data['index','y'] - idx_dx/2. , idx_dx)
    z_id = np.divide(data['index','z'] - idx_dx/2. , idx_dx)
    
    gx = np.subtract(x_id , gx_min[gids-1])
    gy = np.subtract(y_id , gy_min[gids-1])
    gz = np.subtract(z_id , gz_min[gids-1])

    max_x = gx_max[gids-1] - gx_min[gids-1]
    max_y = gy_max[gids-1] - gy_min[gids-1]    

    c_ids = gx + np.multiply(gy , max_x+1) + np.multiply(gz , np.multiply(max_x+1 , max_y+1))

    u_id = np.round(gids + np.multiply(c_ids , max_gid+1)).astype(np.uint64)

    #The above code is a bit hard to follow, here was the original for loop that was randomly way too slow only for certain calls in yt for certain galaxies for reasons I'm not yt-ie enough to understand.
    '''
    for gid in np.round(np.unique(gids)).astype(int):
        if gid<=0: continue
        grid_mask = (gids==gid)
        if np.size(np.where(grid_mask)[0])<=0: continue

        gx = x_id[grid_mask]
        gy = y_id[grid_mask]
        gz = z_id[grid_mask]

        gx = gx - gx_min[gid-1]
        gy = gy - gy_min[gid-1]
        gz = gz - gz_min[gid-1]


        max_x = gx_max[gid-1]-gx_min[gid-1]
        max_y = gy_max[gid-1]-gy_min[gid-1]

        c_id =  gx+gy*(max_x+1) +gz*(max_x+1)*(max_y+1)

        u_id[grid_mask] = np.round(gid + c_id * (max_gid+1)).astype(np.uint64)
    '''
    return u_id  


def add_cell_id_field(ds):
    '''
        Adds the unique cell id defined by get_cell_grid_ids above to a dataset.
        Will be consistent between reloads of the same snapshot, but not between
        successive snapshots.
    '''
    #global max_gid
    max_gid=-1
    for g,m in ds.all_data().blocks:
        if g.id>max_gid: max_gid=g.id


    #global gx_min; global gy_min; global gz_min
    #global gx_max; global gy_max; global gz_max

    gx_min = np.zeros((max_gid))
    gy_min = np.zeros((max_gid))
    gz_min = np.zeros((max_gid))
    gx_max = np.zeros((max_gid))
    gy_max = np.zeros((max_gid))
    gz_max = np.zeros((max_gid))
    
    for g,m in ds.all_data().blocks:
        g_dx = g['index','dx'].max()

        gx_min[g.id-1] = (g['index','x'].min() - g_dx/2.)  / g_dx
        gy_min[g.id-1] = (g['index','y'].min() - g_dx/2.)  / g_dx
        gz_min[g.id-1] = (g['index','z'].min() - g_dx/2.)  / g_dx

        gx_max[g.id-1] = (g['index','x'].max() - g_dx/2.)  / g_dx
        gy_max[g.id-1] = (g['index','y'].max() - g_dx/2.)  / g_dx
        gz_max[g.id-1] = (g['index','z'].max() - g_dx/2.)  / g_dx 


    F_get_cell_grid_ids = partial(pseudo_get_cell_grid_ids, max_gid = max_gid, gx_min=gx_min, gx_max=gx_max, gy_min=gy_min, gy_max=gy_max, gz_min=gz_min) 


    #ds.add_field(
    #    ('index', 'cell_id_2'),
    #      function=get_cell_grid_ids,
    #      sampling_type='cell',
    #      force_override=True
    #)

    ds.add_field(
        ('index', 'cell_id_2'),
          function=F_get_cell_grid_ids,
          sampling_type='cell',
          force_override=True
    )

    #return ds




def load_clump(ds,clump_file,source_cut = None):
    '''
    Function to load a disk or clump cut region defined by clump_finder.py    
    '''
    global internal_clump_counter
    hf = h5py.File(clump_file,'r')
    clump_cell_ids = np.round(np.array(hf['cell_ids'])).astype(np.uint64)
    hf.close()

    add_cell_id_field(ds)

    F_masked_field = partial(_pseudo_masked_field, var_clump_cell_ids=clump_cell_ids) #Each clump needs its own field and its own function

    ds.add_field(
        ("index","masked_field_"+str(int(internal_clump_counter))),
        function=F_masked_field,
        sampling_type="cell",
        units="",
        force_override=True
    )


    if source_cut is None:
        source_cut = ds.all_data()

    masked_region = source_cut.cut_region(["obj['index','masked_field_"+str(int(internal_clump_counter))+"']"])

    internal_clump_counter+=1

    return masked_region

def mask_clump(ds,clump_file,source_cut = None):
    '''
    Function to return a cut region that excludes
    a disk or clump cut region defined by clump_finder.py    
    '''
    global internal_clump_counter
    hf = h5py.File(clump_file,'r')
    clump_cell_ids = np.round(np.array(hf['cell_ids']).astype(np.uint64))
    hf.close()

    add_cell_id_field(ds)


    F_unmasked_field = partial(_pseudo_unmasked_field, var_clump_cell_ids=clump_cell_ids) #Each clump needs its own field and its own function

    ds.add_field(
        ("index","unmasked_field_"+str(int(internal_clump_counter))),
        function=F_unmasked_field,
        sampling_type="cell",
        units="",
        force_override=True
    )


    if source_cut is None:
        source_cut = ds.all_data()
    masked_region = source_cut.cut_region(["obj['index','unmasked_field_"+str(int(internal_clump_counter))+"']"])

    internal_clump_counter+=1

    return masked_region