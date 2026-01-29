import numpy as np
from foggie.utils.consistency import *
from yt.data_objects.level_sets.api import * #TODO - Not super slow here, but replace with custom clump finding to find disk?
import h5py
from functools import partial
import time
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
    
    

def get_cgm_density_cut(ds,cut_type="comoving_density",additional_factor=2.,run="nref11c_nref9f",code_dir=None,halo=None,snapshot=None, cut_field=('gas','density'), disk_stdv_factor = 100.):
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
            cgm_density_cut += disk_stdv_factor * stdv_density * (1+z)**3



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
    gids_raw = data['index','grid_indices']
    if type(gids_raw) is np.ndarray:
        gids = (gids_raw + 1).astype(np.uint64) #These are different in yt and enzo...
    else:
        gids = (gids_raw.v + 1).astype(np.uint64)
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



def pseudo_get_leaf_ids(field, data, leaf_clump_ids,leaf_cell_id_list,leaf_list_ids):
    '''Function to assign a unique cell_id to each cell based on it's index on its parent grid'''
    '''For use as a yt field, must define a partial function resembling get_cell_grid_ids'''

    cell_ids = data['index','cell_id_2']
    clump_ids = np.zeros_like(cell_ids) - 1.

    for leaf_id in leaf_clump_ids:
        leaf_list_id = leaf_list_ids[leaf_id].astype(int)
        clump_ids[np.isin(cell_ids, leaf_cell_id_list[leaf_list_id])] = leaf_id


    return clump_ids  

def add_leaf_id_field(ds,hierarchy_file,add_cell_ids=False,leaf_id_field_name='leaf_id'):
    
    if add_cell_ids:
        add_cell_id_field(ds)

    hf = h5py.File(hierarchy_file,'r')
    leaf_clump_ids = hf['leaf_clump_ids'][...]

    leaf_cell_id_list = []
    leaf_list_ids = np.zeros(np.max(leaf_clump_ids)+1) - 1
    itr=0
    for leaf_id in leaf_clump_ids:
        leaf_cell_id_list.append(hf[str(leaf_id)]['cell_ids'][...])
        leaf_list_ids[leaf_id] = itr
        itr+=1


    hf.close()
        
    F_get_leaf_ids = partial(pseudo_get_leaf_ids, leaf_clump_ids=leaf_clump_ids,leaf_cell_id_list=leaf_cell_id_list,leaf_list_ids=leaf_list_ids)

    ds.add_field(
        ('gas', leaf_id_field_name),
          function=F_get_leaf_ids,
          sampling_type='cell',
          force_override=True
    )




def flatten_multi_clump_list(clump_cell_id_list):
    '''
    If you have a list of clumps, will flatten to a single list of cell ids.
    Useful for loading in multiply clumps without needing to add the cut regions together.
    '''
    return [x for xs in clump_cell_id_list for x in xs]


def load_clump(ds,clump_file = None, clump_cell_ids = None,source_cut = None, skip_adding_cell_ids=False):
    '''
    Function to load a disk or clump cut region defined by clump_finder.py.
    You can either specificy an individual clump_file saved in clump_finder.py or provide a list of clump_cell_ids.
    Arguments are:
        ds - The yt dataset
        clump_file - An hdf5 file that contains a a 'cell_ids' field
        clump_cell_ids - A list of cell_ids to load (will override clump_file if defined)
        source_cut - A yt cut region. Will only look for the cell ids in this cut region. Defaults to ds.all_data() if None
        skip_adding_cell_ids - If true will not add cell_ids to the dataset. This should only be used if those fields are already in the dataset and you want to speed this up.   
    '''
    t0=time.time()

    

    global internal_clump_counter
    if clump_cell_ids is not None:
        clump_cell_ids = np.round(clump_cell_ids).astype(np.uint64)
    elif clump_file is not None:
        hf = h5py.File(clump_file,'r')
        clump_cell_ids = np.round(np.array(hf['cell_ids'])).astype(np.uint64)
        hf.close()
    
    if not skip_adding_cell_ids:
        add_cell_id_field(ds)

    F_masked_field = partial(_pseudo_masked_field, var_clump_cell_ids=clump_cell_ids) #Each clump needs its own field and its own function
    t1=time.time()
    ds.add_field(
        ("index","masked_field_"+str(int(internal_clump_counter))),
        function=F_masked_field,
        sampling_type="cell",
        units="",
        force_override=True
    )

    t2=time.time()
    if source_cut is None:
        source_cut = ds.all_data()

    masked_region = source_cut.cut_region(["obj['index','masked_field_"+str(int(internal_clump_counter))+"']"])

    internal_clump_counter+=1

    #print("Time to define functions=",t1-t0,": Time to add mask",t2-t1,": Time to define cut region",time.time()-t2)


    return masked_region

def mask_clump(ds,clump_file = None, clump_cell_ids = None,source_cut = None, skip_adding_cell_ids=False):
    '''
    Function to define a cut region that excludes a clump object.
    You can either specificy an individual clump_file saved in clump_finder.py or provide a list of clump_cell_ids.
    Arguments are:
        ds - The yt dataset
        clump_file - An hdf5 file that contains a a 'cell_ids' field
        clump_cell_ids - A list of cell_ids to mask (will override clump_file if defined)
        source_cut - A yt cut region. Will only look for the cell ids in this cut region. Defaults to ds.all_data() if None
        skip_adding_cell_ids - If true will not add cell_ids to the dataset. This should only be used if those fields are already in the dataset and you want to speed this up.   
    '''
    global internal_clump_counter
    if clump_cell_ids is None:
        hf = h5py.File(clump_file,'r')
        clump_cell_ids = np.round(np.array(hf['cell_ids']).astype(np.uint64))
        hf.close()

    if not skip_adding_cell_ids:
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



def add_ion_fields(args,ds):
    '''
    If you're clumping field is not present in the dataset, this will attempt to add the corresponding ion number density with Trident.
    Shouldn't overwrite exisint fields.
    '''
    import trident
    trident_dict = { 'HI':'H I', 'CII':'C II','CIII':'C III',
                'CIV':'C IV','OVI':'O VI','SiII':'Si II','SiIII':'Si III','SiIV':'Si IV','MgII':'Mg II'}

    ions_number_density_dict = {'Lyalpha':'LyAlpha', 'HI':'H_p0_number_density', 'CII':'C_p1_number_density', 'CIII':'C_p2_number_density',
                              'CIV':'C_p3_number_density','OVI':'O_p5_number_density','SiII':'Si_p1_number_density','SiIII':'Si_p2_number_density',
                                 'SiIV':'Si_p3_number_density','MgII':'Mg_p1_number_density'}
    
    field_dict = {v: k for k, v in ions_number_density_dict.items()}

    print("args.clumping_field was",args.clumping_field)
    if args.clumping_field in trident_dict:
        trident.add_ion_fields(ds, ions=[trident_dict[args.clumping_field]])
        args.clumping_field =ions_number_density_dict[args.clumping_field]
    elif args.clumping_field in field_dict:
        trident.add_ion_fields(ds, ions=[trident_dict[field_dict[args.clumping_field]]])
    print("args.clumping_field is",args.clumping_field)

#from foggie.clumps.clump_finder.clump_finder import Clump
def save_clump_hierarchy(args,root_clump):
    '''
    Saves clumps hierarchically in a single hdf5 file.
    Each clump corresponds to a group in the h5py file with following structure:
        -clump_id
            -cell_ids - the cell ids (see add_cell_id_field above) corresponding to the clump
            -parent_id - the clump id (group name in this hdf5 file) of the parent structure
            -child_ids - the clump ids (group names in this hdf5 file) of this clumps immediate children
            -tree_level - the tree level this clump corresponds to (0 being the root)

    Below exists a family of functions to navigate this hiearchy, including loading parents, children, siblings, roots, and leaf clumps
    '''
    print("Saving Clump Hierarchy...")
    output_file = args.output+"_ClumpTree.h5"
    hf = h5py.File(output_file,'w')
    leaf_clump_ids = []
    root_clump_ids = []

    n_levels = root_clump.n_levels
    for l in range(0,n_levels):
        clumps = root_clump.clump_tree[l]

        for clump in clumps:
            clump_id = clump.self_id * (n_levels+1) + l #unique clump_id across all tree levels

            parent_id = clump.parent_id
            child_ids = []
            if parent_id>=0:
                parent_id = parent_id*(n_levels+1) + (l-1)
            if clump.nChildren>0:
                for child_id in clump.child_ids:
                    child_ids.append(child_id * (n_levels+1) + (l+1))
        

            grp = hf.create_group(str(clump_id))
            grp.create_dataset('cell_ids',data=clump.cell_ids)
            grp.create_dataset('parent_id',data=parent_id )
            grp.create_dataset('child_ids',data=child_ids)
            grp.create_dataset('tree_level',data=clump.tree_level)
            if clump.shell_cell_ids is not None:
                grp.create_dataset('shell_cell_ids',data=clump.shell_cell_ids) #list of cell ids corresponding to the shells around each clump defined by n_dilation_iterations and n_cells_per_dilation
            if clump.center_disk_coords is not None: grp.create_dataset('center_disk_coords',data=clump.center_disk_coords)


            if clump.nChildren <= 0:
                leaf_clump_ids.append(clump_id)
                print("Appending leaf...")
            if clump.parent_id <0:
                root_clump_ids.append(clump_id)
                print("Appending root...")
            ###Any other info??

    hf.create_dataset("leaf_clump_ids",data=leaf_clump_ids)
    hf.create_dataset("root_clump_ids",data=root_clump_ids)

    grp = hf.create_group("args")
    grp.create_dataset("clump_min",data=args.clump_min)
    grp.create_dataset("clump_max",data=args.clump_max)
    grp.create_dataset("step",data=args.step)
    grp.create_dataset("min_cells",data=args.min_cells)
    grp.create_dataset("halo",data=args.halo)
    grp.create_dataset("run",data=args.run)
    grp.create_dataset("snapshot",data=args.snapshot)
    grp.create_dataset("clumping_field",data=args.clumping_field)
    grp.create_dataset("n_levels",data=root_clump.n_levels)

    hf.close()




def load_all_roots(ds,hierarchy_file,return_as_list=False):
    '''
    Load all root clumps (clumps with no parents) in a hierarchy file saved with save_clump_hierarchy() as yt cut regions
    Arguments are:
        ds - The yt dataset.
        hierarchy_file - The file saved with save_clump_hierarchy()
        return_as_list - Set true to return as a list of cut regions.
    '''
    from foggie.clumps.clump_finder.clump_finder import TqdmProgressBar
    hf = h5py.File(hierarchy_file,'r')
    root_clump_ids = hf['root_clump_ids'][...]

    if return_as_list: roots=[]
    else: all_cell_ids = []

    itr=0
    pbar = TqdmProgressBar("Loading roots...",len(root_clump_ids),position=0)
    skip_adding_cell_ids = False       

    for root_id in root_clump_ids:
        t0=time.time()
        pbar.update(itr)
        if return_as_list:
            roots.append(load_clump(ds,clump_cell_ids=hf[str(root_id)]['cell_ids'][...], skip_adding_cell_ids=skip_adding_cell_ids))
        else:
            all_cell_ids.append(hf[str(root_id)]['cell_ids'][...])

        itr+=1
        skip_adding_cell_ids=True

    pbar.update(len(root_clump_ids))
    pbar.finish()

    if not return_as_list:
        all_cell_ids = flatten_multi_clump_list(all_cell_ids)
        roots = load_clump(ds,clump_cell_ids=all_cell_ids, skip_adding_cell_ids=False)

    return roots

def load_all_leaves(ds,hierarchy_file,return_as_list=False):
    '''
    Load all leaf clumps (clumps with no children) in a hierarchy file saved with save_clump_hierarchy() as yt cut regions
    Arguments are:
        ds - The yt dataset.
        hierarchy_file - The file saved with save_clump_hierarchy()
        return_as_list - Set true to return as a list of cut regions.
    '''
    from foggie.clumps.clump_finder.clump_finder import TqdmProgressBar
    hf = h5py.File(hierarchy_file,'r')
    leaf_clump_ids = hf['leaf_clump_ids'][...]

    if return_as_list: leaves=[]
    else: all_cell_ids = []

    itr=0
    pbar = TqdmProgressBar("Loading Leaves...",len(leaf_clump_ids),position=0)
    skip_adding_cell_ids = False       

    for leaf_id in leaf_clump_ids:
        #print("leaf_id=",leaf_id)
        t0=time.time()
        pbar.update(itr)
        if return_as_list:
            leaves.append(load_clump(ds,clump_cell_ids=hf[str(leaf_id)]['cell_ids'][...], skip_adding_cell_ids=skip_adding_cell_ids))
        else:
            all_cell_ids.append(hf[str(leaf_id)]['cell_ids'][...])


        itr+=1
        skip_adding_cell_ids=True
        #print("Total time to load leaf=",time.time()-t0)

    if not return_as_list:
        all_cell_ids = flatten_multi_clump_list(all_cell_ids)
        leaves = load_clump(ds,clump_cell_ids=all_cell_ids, skip_adding_cell_ids=False)


    pbar.update(len(leaf_clump_ids))
    pbar.finish()
    return leaves

def load_siblings(ds,hierarchy_file,clump_id,return_as_list=False):
    '''
    Load the siblings (clumps with the same parent) of a given clump in a hierarchy file saved with save_clump_hierarchy() as yt cut regions
    Arguments are:
        ds - The yt dataset.
        hierarchy_file - The file saved with save_clump_hierarchy()
        clump_id - The id of the clump in the hierarchy file you want to find the siblings of
        return_as_list - Set true to return as a list of cut regions.
    '''
    from foggie.clumps.clump_finder.clump_finder import TqdmProgressBar
    hf = h5py.File(hierarchy_file,'r')
    parent_id = hf[str(clump_id)]['parent_id'][...]
    sibling_ids = hf[parent_id]['child_ids'][...]

    if return_as_list: siblings=[]
    else: all_cell_ids=[]

    itr=0
    pbar = TqdmProgressBar("Loading Leaves...",len(sibling_ids),position=0)
    skip_adding_cell_ids = False       

    for sibling_id in sibling_ids:
        #print("leaf_id=",leaf_id)
        t0=time.time()
        pbar.update(itr)
        if sibling_id == clump_id: continue #don't load self
        if return_as_list:
            siblings.append(load_clump(ds,clump_cell_ids=hf[str(sibling_id)]['cell_ids'][...], skip_adding_cell_ids=skip_adding_cell_ids))
        else:
            all_cell_ids.append(hf[str(sibling_id)]['cell_ids'][...])


        itr+=1
        skip_adding_cell_ids=True
        #print("Total time to load leaf=",time.time()-t0)
    
    pbar.update(len(sibling_ids))
    pbar.finish()

    if not return_as_list:
        all_cell_ids = flatten_multi_clump_list(all_cell_ids)
        siblings = load_clump(ds,clump_cell_ids=all_cell_ids, skip_adding_cell_ids=False)

    return siblings

def load_children(ds,hierarchy_file,clump_id,return_as_list=False):
    '''
    Load the children of a given clump in a hierarchy file saved with save_clump_hierarchy() as yt cut regions
    Arguments are:
        ds - The yt dataset.
        hierarchy_file - The file saved with save_clump_hierarchy()
        clump_id - The id of the clump in the hierarchy file you want to find the children of
        return_as_list - Set true to return as a list of cut regions.
    '''
    from foggie.clumps.clump_finder.clump_finder import TqdmProgressBar
    hf = h5py.File(hierarchy_file,'r')
    child_ids = hf[str(clump_id)]['child_ids'][...]

    if return_as_list: children=[]
    else: all_cell_ids=[]

    itr=0
    pbar = TqdmProgressBar("Loading Leaves...",len(child_ids),position=0)
    skip_adding_cell_ids = False       

    for child_id in child_ids:
        #print("leaf_id=",leaf_id)
        t0=time.time()
        pbar.update(itr)
        if child_id == clump_id: continue #don't load self
        if return_as_list:
            children.append(load_clump(ds,clump_cell_ids=hf[str(child_id)]['cell_ids'][...], skip_adding_cell_ids=skip_adding_cell_ids))
        else:
            all_cell_ids.append(hf[str(child_id)]['cell_ids'][...])


        itr+=1
        skip_adding_cell_ids=True
        #print("Total time to load leaf=",time.time()-t0)
    
    pbar.update(len(child_ids))
    pbar.finish()

    if not return_as_list:
        all_cell_ids = flatten_multi_clump_list(all_cell_ids)
        siblings = load_clump(ds,clump_cell_ids=all_cell_ids, skip_adding_cell_ids=False)

    return children

def load_parent(ds,hierarchy_file,clump_id):
    '''
    Load the parent of a given clump in a hierarchy file saved with save_clump_hierarchy() as yt cut regions
    Arguments are:
        ds - The yt dataset.
        hierarchy_file - The file saved with save_clump_hierarchy()
        clump_id - The id of the clump in the hierarchy file you want to load the parent of
    '''
    hf = h5py.File(hierarchy_file,'r')
    parent_id = hf[str(clump_id)]['parent_id'][...]
    return load_clump(ds,clump_cell_ids=hf[str(parent_id)]['cell_ids'][...], skip_adding_cell_ids=False)

def load_root(ds,hierarchy_file,clump_id):
    '''
    Load the root (clump with no parents) of a given clump in a hierarchy file saved with save_clump_hierarchy() as yt cut regions
    Arguments are:
        ds - The yt dataset.
        hierarchy_file - The file saved with save_clump_hierarchy()
        clump_id - The id of the clump in the hierarchy file you want to load the root of
    '''
    hf = h5py.File(hierarchy_file,'r')
    max_clumps = len(hf.keys())
    itr=0
    current_id = clump_id
    while itr<max_clumps:
        itr+=1
        current_id = hf[current_id]['parent_id'][...]
        if hf[current_id]['tree_level']==0:
            return load_clump(ds,clump_cell_ids=hf[str(current_id)]['cell_ids'][...], skip_adding_cell_ids=False)

def load_clump_shell(ds, hierarchy_file,clump_id,shell_number=None):
    '''
    Function to load the shells of a specific clump. Will either load all shells, or a specific shell
    Arguments are:
        ds - The yt dataset.
        hierarchy_file - The file saved with save_clump_hierarchy()
        clump_id - The id of the clump in the hierarchy file you want to load the root of
        shell_number - The index of the shell you want to load (0 is the innermost shell). If None, will load all shells combined
    '''
    hf = h5py.File(hierarchy_file,'r')

    shell_cell_ids = hf[str(clump_id)]['shell_cell_ids'][...]

    if shell_number is not None: shell_cell_ids = shell_cell_ids[shell_number]
    else: shell_cell_ids = flatten_multi_clump_list(shell_cell_ids) #combine all shells into single list

    return load_clump(ds,clump_cell_ids = shell_cell_ids, skip_adding_cell_ids=False)


def load_clump_hierarchy(args,ds,cut_region,hierarchy_file,is_disk=False):
    '''
    Load a given clump hierarchy file into the clump class defined in clump_finder.py
    Arguments are:
        args - The clump_finder arguments defined in clump_finder_argparser.py
        ds - The yt dataset.
        cut_region - The cut region the clump finder was run on
        hierarchy_file - The file saved with save_clump_hierarchy()
        is_disk - Is this a disk clump or not
    '''
    from foggie.clumps.clump_finder.clump_finder import Clump
    hf = h5py.File(hierarchy_file,'r')

    master_clump = Clump(ds, cut_region, args, 0, is_disk)

    for lvl in range(master_clump.n_levels):
        for grpkey in hf.keys():
            try:
                l = hf[grpkey]['tree_level'][...]
                if l != lvl: continue

                clump_id = ( int(grpkey) - l ) / (master_clump.n_levels+1)
                if len(master_clump.clump_tree[l])<=clump_id:
                    while len(master_clump.clump_tree[l])<=clump_id:
                        master_clump.clump_tree[l].append(-1)

                master_clump.clump_tree[l][clump_id] = Clump(ds,cut_region,args,l,is_disk)

                parent_id = (int(hf[grpkey]['parent_id']) - l) / (master_clump.n_levels+1)
                master_clump.clump_tree[l][clump_id].parent_id = parent_id
                master_clump.clump_tree[l][clump_id].parent_index = parent_id

                master_clump.clump_tree[l-1][parent_id].child_ids.append(clump_id)
                master_clump.clump_tree[l-1][parent_id].child_indices.append(clump_id)

            except:
                print("Ignoring key:",grpkey)

    return master_clump


from yt.data_objects.level_sets.clump_handling import Clump as YTClump
class YTClumpLean(YTClump):
    '''
    Overload the initialization of the YT clump object to be faster, as we don't need to do validation checks on clumps.
    '''
    def __init__(
        self,
        data,
        field,
        parent=None,
        clump_info=None,
        validators=None,
        base=None,
        contour_key=None,
        contour_id=None,
    ):
        self.data = data
        self.field = field
        self.parent = parent
        self.quantities = data.quantities

        self.info = {}
        self.children = []

        # is this the parent clump?
        if base is None:
            base = self
            self.total_clumps = 0

        if clump_info is None:
            clump_info = []
        self.clump_info = clump_info


        self.base = base
        self.clump_id = self.base.total_clumps
        self.base.total_clumps += 1
        self.contour_key = contour_key
        self.contour_id = contour_id


        if parent is not None:
            self.data.parent = self.parent.data

        if validators is None:
            validators = []
        self.validators = validators
        # Return value of validity function.
        self.valid = None


def save_as_YTClumpContainer(ds,cut_region,master_clump,clumping_field,args):
    '''
    Load a clump object defined in clump_finder.py into a YT clump class.
    Uses the overloaded class initialization above (YTClumpLean) to speed things up
    '''
    print("Loading into a YTClump object!")

    YTMasterClump = YTClumpLean(cut_region, clumping_field)
    YTClumpList = []

    n_levels = master_clump.n_levels

    for l in range(0,n_levels):
        YTClumpList.append([])


   # YTMasterClump.add_info_item("total_cells")
    for l in range(0,n_levels):
        print("On Level",l)
        for clump in master_clump.clump_tree[l]:
            t0=time.time()
            clump_cut_region = load_clump(ds,clump_cell_ids=clump.cell_ids, skip_adding_cell_ids=True)
            t1=time.time()
            contour_id = clump.self_id * (n_levels+1) + l #unique clump_id across all tree levels

        
            if l==0:
                thisYTClump = YTClumpLean(
                        clump_cut_region,
                        clumping_field,
                        parent = YTMasterClump,
                        validators=None,#Check
                        base=YTMasterClump,
                        clump_info=YTMasterClump.clump_info,
                        contour_key=None,
                        contour_id=contour_id,
                    )
                t2=time.time()
                YTMasterClump.children.append(thisYTClump)
                if len(YTClumpList[l])<=clump.self_id:
                    while len(YTClumpList[l])<=clump.self_id:
                        YTClumpList[l].append(-1)
                YTClumpList[l][clump.self_id] = thisYTClump

            else:
                parent_id = clump.parent_id

                thisYTClump = YTClumpLean(
                        clump_cut_region,
                        clumping_field,
                        parent = YTClumpList[l-1][parent_id],
                        validators=None,#Check
                        base=YTMasterClump,
                        clump_info=YTMasterClump.clump_info,
                        contour_key=None,
                        contour_id=contour_id,
                    )
                
                t2=time.time()
                YTClumpList[l-1][parent_id].children.append(thisYTClump)
                #YTClumpList[l].append(thisYTClump)

                if len(YTClumpList[l])<=clump.self_id:
                    while len(YTClumpList[l])<=clump.self_id:
                        YTClumpList[l].append(-1)
                YTClumpList[l][clump.self_id] = thisYTClump
            
            print("Time to load clump=",t1-t0,"Time define YTClump=",t2-t1,"Time to append clumps=",time.time()-t2)

    fn = YTMasterClump.save_as_dataset(filename=args.output+"YTClumpDataset",fields=[clumping_field])
    return YTMasterClump
            

def GetClumpsInDisk(clump_ids, hierarchy_file, disk_file):
    '''
    Given a list of clump_ids, will identify those within the disk
    Arguments are:
        clump_ids-List of clump ids you want to filter
        hierarchy_file-File saved by save_clump_hierarchy() to search within
    '''
    disk_clump_ids = []
    hf_disk = h5py.File(disk_file,'r')
    disk_cell_ids = hf_disk['cell_ids'][...]

    hf = h5py.File(hierarchy_file,'r')

    for clump_id in clump_ids:
        clump_cell_ids = hf[str(clump_id)]['cell_ids'][...]
        if np.isin(clump_cell_ids,disk_cell_ids).any():
            disk_clump_ids.append(clump_id)

    return disk_clump_ids

def GetClumpDistanceFromDisk(ds, source_cut, clump_ids, hierarchy_file, disk_file):
    '''
    Given a list of clump_ids, will identify the distance from the disk (0 if within disk)
    Arguments are:
        clump_ids-List of clump ids you want to filter
        hierarchy_file-File saved by save_clump_hierarchy() to search within
    '''
    hf_disk = h5py.File(disk_file,'r')
    disk_cell_ids = hf_disk['cell_ids'][...]
    hf_disk.close()

    hf = h5py.File(hierarchy_file,'r')
    distance_dict = {}

    leaf_ids = source_cut['gas','leaf_id']
    gas_mass = source_cut['gas','mass'].in_units('Msun')
    disk_x = source_cut['gas','x_disk']
    disk_y = source_cut['gas','y_disk']
    disk_z = source_cut['gas','z_disk']
    cell_ids = source_cut['index','cell_id_2']

    disk_mask = np.isin(cell_ids,disk_cell_ids)

    for clump_id in clump_ids:
        clump_cell_ids = hf[str(clump_id)]['cell_ids'][...]
        if np.isin(clump_cell_ids,disk_cell_ids).any():
            distance_dict[clump_id] = 0.0
        else:
            mask = (leaf_ids==clump_id)
            x = np.sum(np.multiply(gas_mass[mask],disk_x[mask])) / np.sum(gas_mass[mask])
            y = np.sum(np.multiply(gas_mass[mask],disk_y[mask])) / np.sum(gas_mass[mask])
            z = np.sum(np.multiply(gas_mass[mask],disk_z[mask])) / np.sum(gas_mass[mask])

            

            min_distance = np.min( np.sqrt( (disk_x[disk_mask]-x)**2 + (disk_y[disk_mask]-y)**2 + (disk_z[disk_mask]-z)**2 ) )

            distance_dict[clump_id] = min_distance
    hf.close()
    return distance_dict


def GetClumpDistanceFromClumpIds(ds, source_cut, clump_ids, hierarchy_file, clump_id_list, secondary_hierarchy_file):
    '''
    Given a list of clump_ids, will identify the distance from the disk (0 if within disk)
    Arguments are:
        clump_ids-List of clump ids you want to filter
        hierarchy_file-File saved by save_clump_hierarchy() to search within
    '''
    distance_dict = {}

    cell_id_list = np.array([])
    hf2 = h5py.File(secondary_hierarchy_file,'r')
    for clump_id in clump_id_list:
        cell_id_list = np.append(cell_id_list, hf2[str(clump_id)]['cell_ids'][...])
    hf2.close()

    hf = h5py.File(hierarchy_file,'r')
    leaf_ids = source_cut['gas','leaf_id']
    gas_mass = source_cut['gas','mass'].in_units('Msun')
    disk_x = source_cut['gas','x_disk']
    disk_y = source_cut['gas','y_disk']
    disk_z = source_cut['gas','z_disk']
    cell_ids = source_cut['index','cell_id_2']

    clump_mask = np.isin(cell_ids,cell_id_list)

    for clump_id in clump_ids:
        clump_cell_ids = hf[str(clump_id)]['cell_ids'][...]
        if np.isin(clump_cell_ids,cell_id_list).any():
            distance_dict[clump_id] = 0.0
        else:
            mask = (leaf_ids==clump_id)
            x = np.sum(np.multiply(gas_mass[mask],disk_x[mask])) / np.sum(gas_mass[mask])
            y = np.sum(np.multiply(gas_mass[mask],disk_y[mask])) / np.sum(gas_mass[mask])
            z = np.sum(np.multiply(gas_mass[mask],disk_z[mask])) / np.sum(gas_mass[mask])

            

            min_distance = np.min( np.sqrt( (disk_x[clump_mask]-x)**2 + (disk_y[clump_mask]-y)**2 + (disk_z[clump_mask]-z)**2 ) )

            distance_dict[clump_id] = min_distance

    hf.close()
    return distance_dict


def FindOverlappingClumps(cell_ids, hierarchy_file, return_only_leaves=False, ds=None, return_cut_regions=False):
    '''
    Given a list of cell_ids, will find overlapping clumps (with at least 1 cell id in common) in a hierarchy file.
    Arguments are:
        cell_ids-List of cell ids you want to search for
        hierarchy_file-File saved by save_clump_hierarchy() to search within
        return_only_leaves-If true will return only leaf clumps
        ds-The YT dataset. Only needs to be defined if you actually want to load the cut regions corresponding to the clumps
        return_cut_regions-If true will return the cut regions for the clumps that overlap
    '''

    overlapping_clump_ids = []
    overlapping_clumps = []

    if return_cut_regions and ds is None:
        print("Warning: if you want to return cut regions you must specify ds.")

    hf = h5py.File(hierarchy_file,'r')

    n_levels = hf['args']['n_levels'][...]

    for i in range(0,n_levels):
        overlapping_clump_ids.append([])
        if return_cut_regions: overlapping_clumps.append([])

    skip_adding_cell_ids = False



    for grpkey in hf.keys():
        if not grpkey in ['args','root_clump_ids','leaf_clump_ids']:
            intersection = np.intersect1d( hf[grpkey]['cell_ids'], cell_ids )
            if len(intersection)>0:
                is_leaf = len(hf[grpkey]['child_ids'])==0
                if not return_only_leaves or is_leaf:
                    l = hf[grpkey]['tree_level'][...]
                    overlapping_clump_ids[l].append(grpkey)
                    if return_cut_regions:
                        overlapping_clumps[l].append(load_clump(ds,clump_cell_ids=hf[grpkey]['cell_ids'][...], skip_adding_cell_ids=skip_adding_cell_ids))
                        skip_adding_cell_ids=True


    if return_cut_regions:
        return overlapping_clumps

    return overlapping_clump_ids