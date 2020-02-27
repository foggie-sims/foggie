"""
Code to export from yt to Sunrise


"""


import time
import numpy as np
from numpy import *
import astropy.io.fits as pyfits
import yt
from yt.funcs import get_pbar
from yt.funcs import *
import scipy


class hilbert_state():
	def __init__(self,dim=None,sgn=None,octant=None):
		if dim is None: dim = [0,1,2]
		if sgn is None: sgn = [1,1,1]
		if octant is None: octant = 5
		self.dim = dim
		self.sgn = sgn
		self.octant = octant

	def flip(self,i):
	    self.sgn[i]*=-1
	
	def swap(self,i,j):
	    temp = self.dim[i]
	    self.dim[i]=self.dim[j]
	    self.dim[j]=temp
	    axis = self.sgn[i]
	    self.sgn[i] = self.sgn[j]
	    self.sgn[j] = axis
	
	def reorder(self,i,j,k):
	    ndim = [self.dim[i],self.dim[j],self.dim[k]] 
	    nsgn = [self.sgn[i],self.sgn[j],self.sgn[k]]
	    self.dim = ndim
	    self.sgn = nsgn

	def copy(self):
	    return hilbert_state([self.dim[0],self.dim[1],self.dim[2]],
	                         [self.sgn[0],self.sgn[1],self.sgn[2]],
	                         self.octant)

	def descend(self,o):
		child = self.copy()
		child.octant = o
		if o==0:
		    child.swap(0,2)
		elif o==1:
		    child.swap(1,2)
		elif o==2:
		    pass
		elif o==3:
		    child.flip(0)
		    child.flip(2)
		    child.reorder(2,0,1)
		elif o==4:
		    child.flip(0)
		    child.flip(1)
		    child.reorder(2,0,1)
		elif o==5:
		    pass
		elif o==6:
			child.flip(1)
			child.flip(2)
			child.swap(1,2)
		elif o==7:
			child.flip(0)
			child.flip(2)
			child.swap(0,2)
		return child



	def __iter__(self):
		vertex = np.array([0,0,0]).astype('int32')
		j = 0
		for i in range(3):
		    vertex[self.dim[i]] = 0 if self.sgn[i]>0 else 1
		yield vertex, self.descend(j)
		vertex[self.dim[0]] += self.sgn[0]
		j+=1
		yield vertex, self.descend(j)
		vertex[self.dim[1]] += self.sgn[1] 
		j+=1
		yield vertex, self.descend(j)
		vertex[self.dim[0]] -= self.sgn[0] 
		j+=1
		yield vertex, self.descend(j)
		vertex[self.dim[2]] += self.sgn[2] 
		j+=1
		yield vertex, self.descend(j)
		vertex[self.dim[0]] += self.sgn[0] 
		j+=1
		yield vertex, self.descend(j)
		vertex[self.dim[1]] -= self.sgn[1] 
		j+=1
		yield vertex, self.descend(j)
		vertex[self.dim[0]] -= self.sgn[0] 
		j+=1
		yield vertex, self.descend(j)

class oct_object():
	def __init__(self, is_leaf, fcoords, fwidth, level, oct_id, child_oct_ids, fields = None):
		self.is_leaf = is_leaf	#2 x 2 x 2
		self.fcoords = fcoords  #2 x 2 x 2
		self.octcen = mean(self.fcoords, axis = 0) #3 x 1
		self.fwidth = fwidth #2 x 2 x 2
		self.le = fcoords - 0.5*fwidth #2 x 2 x 2
		self.re = fcoords + 0.5*fwidth #2 x 2 x 2
		self.child_oct_ids = child_oct_ids #2 x 2 x 2
		self.n_refined_visited = 0
		self.n_leaf = len(where(self.is_leaf == True)[0])
		self.n_refined = len(where(self.is_leaf == False)[0])
		self.level = level
		self.child_level = self.level + 1
		self.oct_id = int(oct_id)
		self.fields = fields		

def recursive_generate_oct_list(oct_list, current_oct_id, current_level, mask_arr, fcoords, fwidth, oct_loc, octs_dic):
	current_oct_id = int(current_oct_id)
	mask_i = mask_arr[:,:,:, current_oct_id]
	fcoords_ix, fcoords_iy, fcoords_iz = fcoords[:,:,:, current_oct_id, 0],  fcoords[:,:,:, current_oct_id, 1], fcoords[:,:,:, current_oct_id, 2]
	fwidth_ix, fwidth_iy, fwidth_iz = fwidth[:,:,:, current_oct_id, 0],  fwidth[:,:,:, current_oct_id, 1], fwidth[:,:,:, current_oct_id, 2]

	flat_mask = mask_i.ravel(order = 'F')
	flat_fcoords = array(zip(fcoords_ix.ravel(order = 'F').value[()], fcoords_iy.ravel(order = 'F').value[()], fcoords_iz.ravel(order = 'F').value[()]))
	flat_fwidth = array(zip(fwidth_ix.ravel(order = 'F').value[()], fwidth_iy.ravel(order = 'F').value[()], fwidth_iz.ravel(order = 'F').value[()]))
	child_level	= current_level	+ 1

	refined_locations = where(flat_mask == False)[0]
	nrefined = len(refined_locations)
	child_oct_ids = nan*zeros(8)

	if nrefined > 0:
		child_oct_ids_temp = oct_loc[str(child_level)][1][oct_loc[str(child_level)][0]:oct_loc[str(child_level)][0]+nrefined]		
		oct_loc[str(child_level)][0] += nrefined
		child_oct_ids[refined_locations] = child_oct_ids_temp





	#child_oct_locs = nan*zeros(8)
	#child_oct_locs[refined_locations] = 1+ arange(len(oct_list), len(oct_list)+nrefined)




	fields = octs_dic['Fields'][:,:,:,:, current_oct_id]
	fields_all = zeros((fields.shape[0], 8))
	for field_index in range(fields.shape[0]):
		fields_all[field_index] = fields[field_index,:,:,:].ravel(order = 'F')

	oct_obj = oct_object(flat_mask, flat_fcoords, flat_fwidth, current_level, current_oct_id, child_oct_ids, fields = fields_all)
	oct_list[current_oct_id] = oct_obj
	for n, i in enumerate(refined_locations):
		recursive_generate_oct_list(oct_list, child_oct_ids[i], child_level, mask_arr, fcoords, fwidth, oct_loc, octs_dic)

def add_preamble(oct_list, levels, fwidth, fcoords, LeftEdge, RightEdge, mask_arr):
	i = 0	
	oct_id = -1
	#We are trying to organize the root oct grid into a root grid of parent octs (8 per higher level oct)
	while True:
		#The 8 children for each oct will encode the level
		good = filter(lambda x: x.level == i, oct_list)

		octcens = [[gd.octcen[0], gd.octcen[1], gd.octcen[2], gd.oct_id] for gd in good]
		octcens = array(octcens)

		print(i, len(good))
		if len(good) == 1:
			#We've reached the root single oct (1 x 1 x 1)
			return oct_list
		else:
			i -= 1

		dimens = int( np.ceil( ( (float(len(good)))**(1.0/3.0) )   /2.) )
                
		flat_mask = array([False, False, False, False, False, False, False, False])
		flat_fwidth  = good[0].fwidth*2
		delx = 2*flat_fwidth[0,0]
		oct_list_2 = []
		for ii in arange(dimens):
			print(ii, dimens)
			for jj in arange(dimens):
				for kk in arange(dimens):
					le_oct = array([ii, jj, kk])*delx
					re_oct = le_oct+delx
					good_in = where((octcens[:,0] < re_oct[0]) & (octcens[:,1] < re_oct[1]) & (octcens[:,2] < re_oct[2]) &
									(octcens[:,0] > le_oct[0]) & (octcens[:,1] > le_oct[1]) & (octcens[:,2] > le_oct[2]))[0]
					assert(len(good_in) == 8)
					flat_fcoords = octcens[good_in,0:3]
					child_ids = octcens[good_in, 3]

					oct_obj = oct_object(flat_mask, flat_fcoords, flat_fwidth, i, oct_id, child_ids)
					oct_id-=1
					oct_list_2.append(oct_obj)

		oct_list = concatenate([oct_list_2[::-1], oct_list])

def OctreeDepthFirstHilbert(oct_list, oct_obj, hilbert, grid_structure, output, field_names, debug = False, f  = 'out.out'):
	current_level = oct_obj.level
	child_level = oct_obj.child_level
	fields = oct_obj.fields
	parent_oct_le = array([min(oct_obj.le[:,0]), min(oct_obj.le[:,1]), min(oct_obj.le[:,2])])
	save_to_gridstructure(grid_structure,oct_obj.child_level, np.asarray(oct_obj.octcen-oct_obj.fwidth[0,0]), refined = True, leaf = False)
	#It's the first time visiting this oct, so let's save 
	#the oct information here in our grid structure dictionary
	if debug: f.write('\t'*(current_level+6)+'Entering level %i oct (ID: %i): found %i refined cells and %i leaf cells\n'%(current_level, oct_obj.oct_id, oct_obj.n_refined, oct_obj.n_leaf))
	for (vertex, hilbert_child) in hilbert:
		vertex_new = vertex*oct_obj.fwidth[0]
		next_child_le = parent_oct_le + vertex_new
		i = where((oct_obj.le[:,0] == next_child_le[0]) & (oct_obj.le[:,1] == next_child_le[1]) & (oct_obj.le[:,2] == next_child_le[2]))[0][0]
		if oct_obj.is_leaf[i]:
			#This cell is a leaf, save the grid information and the physical properties
			if debug:  f.write('\t'*(child_level+6)+str(oct_obj.child_level) + '\tFound a leaf in cell %i/%i \t (x,y,z, vol, mass, density) = (%.8f, %.8f, %.8f, %.8f, %.8f, %.8f) \n'%(i, 8, oct_obj.le[i][0], oct_obj.le[i][1], oct_obj.le[i][2], fields[3,i], fields[0,i], fields[0,i]/fields[3,i]))		
			save_to_gridstructure(grid_structure, current_level, np.asarray(oct_obj.le[i]), refined = False, leaf = True)							
			for field_index in range(fields.shape[0]):
				output[field_names[field_index]].append(fields[field_index,i])

		else:
			#This cell is not a leaf, we'll now advance in to this cell
			try:	
				if debug:  f.write('\t'*(child_level+6)+str(child_level) + '\tFound a refinement in cell %i/%i \t (x,y,z) = (%.8f, %.8f, %.8f, %.8f, %.8f, %.8f) \n'%(i, 8, oct_obj.le[i][0], oct_obj.le[i][1], oct_obj.le[i][2], fields[0,i], fields[0,i]/fields[3,i]))
			except:
				if debug:  f.write('\t'*(child_level+6)+str(child_level) + '\tFound a refinement in cell %i/%i \t (x,y,z) = (%.8f, %.8f, %.8f) \n'%(i, 8, oct_obj.le[i][0], oct_obj.le[i][1], oct_obj.le[i][2]))


			child_oct_obj = oct_list[int(oct_obj.child_oct_ids[i]-oct_list[0].oct_id)]
			OctreeDepthFirstHilbert(oct_list, child_oct_obj, hilbert_child, grid_structure, output, field_names, debug, f)

def export_to_sunrise(ds, fn, star_particle_type, fc, fwidth, nocts_wide=None, \
                      debug=False,ad=None,max_level=None, grid_structure_fn = 'grid_structure.npy', no_gas_p = False, form='VELA', **kwargs):

        r"""Convert the contents of a dataset to a FITS file format that Sunrise
        understands.

        This function will accept a dataset, and from that dataset
        construct a depth-first octree containing all of the data in the parameter
        file.  This octree will be written to a FITS file.  It will probably be
        quite big, so use this function with caution!  Sunrise is a tool for
        generating synthetic spectra, available at
        http://sunrise.googlecode.com/ .

        Parameters
        ----------
        ds : `Dataset`
        The dataset to convert.
        fn : string
        The filename of the output FITS file.
        fc : array
        The center of the extraction region
        fwidth  : array  
        Ensure this radius around the center is enclosed
        Array format is (nx,ny,nz) where each element is floating point
        in unitary position units where 0 is leftmost edge and 1
        the rightmost. 

        Notes
        -----

        Note that the process of generating simulated images from Sunrise will
        require substantial user input; see the Sunrise wiki at
        http://sunrise.googlecode.com/ for more information.

        """
        '''
        fc = fc.in_units('code_length').value
        fwidth = fwidth.in_units('code_length').value
        Nocts_root = ds.domain_dimensions/2
        '''


        fc = fc.in_units('code_length').value
        fwidth = fwidth.in_units('code_length').value
        Nocts_root = ds.domain_dimensions/2

        #we must round the dle,dre to the nearest root grid cells
        ile,ire,super_level,nocts_wide = round_nocts_wide(Nocts_root,fc-fwidth,fc+fwidth,nwide=nocts_wide)
        assert np.all((ile-ire)==(ile-ire)[0])
        print("rounding specified region:")
        print("from [%1.5f %1.5f %1.5f]-[%1.5f %1.5f %1.5f]"%(tuple(fc-fwidth)+tuple(fc+fwidth)))
        print("to (integer)   [%07i %07i %07i]-[%07i %07i %07i]"%(tuple(ile)+tuple(ire)))
        assert(len(np.unique(ds.domain_width)) == 1)
        domain_width = ds.domain_width[0]
        fle,fre = ile*domain_width/Nocts_root, ire*domain_width/Nocts_root
        print("to (float)  [%1.5f %1.5f %1.5f]-[%1.5f %1.5f %1.5f]"%(tuple(fle)+tuple(fre)))

        #Create a list of the star particle properties in PARTICLE_DATA
        #Include ID, parent-ID, position, velocity, creation_mass, 
        #formation_time, mass, age_m, age_l, metallicity, L_bol

        if form=='ENZO':
                radkpc=0.05
        elif form=='VELA':
                radkpc=0.01

        particle_data,nstars = prepare_star_particles(ds,star_particle_type,fle=fle,fre=fre, ad=ad,radkpc=radkpc,**kwargs)
        #Create the refinement depth-first hilbert octree structure
        #For every leaf (not-refined) oct we have a column n OCTDATA
        #Include mass_gas, mass_metals, gas_temp_m, gas_teff_m, cell_volume, SFR
        #since our 0-level mesh may have many octs,
        #we must create the octree region sitting 
        #ontop of the first mesh by providing a negative level
        ad = ds.all_data()
        print('Simulation format name:  ',form)
        if form=='ENZO':
                output, grid_structure, nrefined, nleafs = None,None,None,None
                create_simple_fits(ds,fn,particle_data,fle = ds.domain_left_edge,fre = ds.domain_right_edge, no_gas_p = no_gas_p,form=form)
                output_array=None
        elif form=='VELA':
                output, grid_structure, nrefined, nleafs = prepare_octree(ds,ile,fle=fle,fre=fre, ad=ad,start_level=super_level, debug=debug)
                
                output_array = zeros((len(output[output.keys()[0]]), len(output.keys())))
                for i in arange(len(output_array[0])):
                        output_array[:,i] = output[output.keys()[i]]
                #grid_structure['level']+=6
                refined = grid_structure['refined']

                #np.savez('grid_structure.npz',grid_structure)
                np.save(grid_structure_fn,grid_structure)  #way faster to load for some reason?

                create_fits_file(ds,fn,output,refined,particle_data,fle = ds.domain_left_edge,fre = ds.domain_right_edge, no_gas_p = no_gas_p,form=form)

        return fle, fre, ile, ire, nrefined, nleafs, nstars, output, output_array


def create_simple_fits(ds, fn, particle_data, fle, fre, no_gas_p = False,form='VELA', downsample_factor = 50):
        refined=np.asarray([1,0,0,0,0,0,0,0,0])
        
        #first create the grid structure
        structure = pyfits.Column("structure", format="B", array=array(refined).astype("bool"))
        cols = pyfits.ColDefs([structure])
        st_table = pyfits.BinTableHDU.from_columns(cols)
        st_table.name = "GRIDSTRUCTURE"
        st_table.header.set("hierarch lengthunit", "kpc", comment="Length unit for grid")
        fre = ds.arr(fre, 'code_length').in_units('kpc').value
        fle = ds.arr(fle, 'code_length').in_units('kpc').value
        fdx = fre-fle



        for i,a in enumerate('xyz'):
                st_table.header.set("min%s" % a, fle[i])
                st_table.header.set("max%s" % a, fre[i])
                st_table.header.set("n%s" % a, fdx[i])
                st_table.header.set("subdiv%s" % a, 2)
        st_table.header.set("subdivtp", "OCTREE", "Type of grid subdivision")

        #not the hydro grid data
        fields = ["CellMassMsun","TemperatureTimesCellMassMsun", "MetalMassMsun", "CellVolumeKpc", "CellSFRtau","Cellpgascgsx", "Cellpgascgsy", "Cellpgascgsz"]

        fd = {}
        for i,f in enumerate(fields): 
                fd[f]=array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]) #array(output[f][:])

        col_list = []

        col_list.append(pyfits.Column("mass_gas", format='D',
                                      array=fd["CellMassMsun"], unit="Msun"))
        col_list.append(pyfits.Column("mass_metals", format='D',
                                      array=fd['MetalMassMsun'], unit="Msun"))
        col_list.append(pyfits.Column("gas_temp_m", format='D',
                                      array=fd['TemperatureTimesCellMassMsun'], unit="K*Msun"))
        col_list.append(pyfits.Column("gas_teff_m", format='D',
                                      array=fd['TemperatureTimesCellMassMsun'], unit="K*Msun"))
        col_list.append(pyfits.Column("cell_volume", format='D',
                                      array=fd['CellVolumeKpc']+1.0, unit="kpc^3"))
        col_list.append(pyfits.Column("SFR", format='D',
                                      array=fd['CellSFRtau'],  unit = 'Msun'))

        m = 1.0
        if no_gas_p: m = 0.0
        p_gas_zipped = np.ndarray((fd['Cellpgascgsx'].shape[0],3))
        p_gas_zipped[:,0]=fd['Cellpgascgsx']*m
        p_gas_zipped[:,1]=fd['Cellpgascgsy']*m
        p_gas_zipped[:,2]=fd['Cellpgascgsz']*m

        #array(zip(fd['Cellpgascgsx']*m,
        #                   fd['Cellpgascgsy']*m,
        #                   fd['Cellpgascgsz']*m))


        
        col_list.append(pyfits.Column("p_gas", format='3D',
                                      array=p_gas_zipped , unit = 'Msun*kpc/yr'))

        cols = pyfits.ColDefs(col_list)
        mg_table = pyfits.BinTableHDU.from_columns(cols)
        #mg_table = pyfits.new_table(cols)
        mg_table.header.set("M_g_tot", fd["CellMassMsun"].sum())
        mg_table.header.set("timeunit", "yr")
        mg_table.header.set("tempunit", "K")
        mg_table.name = "GRIDDATA"

        # Add a dummy Primary; might be a better way to do this!
        col_list = [pyfits.Column("dummy", format="E", array=np.zeros(1, dtype='float32'))]
        cols = pyfits.ColDefs(col_list)
        md_table = pyfits.BinTableHDU.from_columns(cols, nrows = len(fd['CellSFRtau']))
        #md_table = pyfits.new_table(cols)
        md_table.header.set("snaptime", ds.current_time.in_units('yr').value[()])
        md_table.header.set("redshift",ds.current_redshift)
        md_table.name = "YT"

        phdu = pyfits.PrimaryHDU()
        phdu.header.set('nbodycod','yt')
        hls = [phdu, st_table, mg_table,md_table]
        hls.append(particle_data)
        hdus = pyfits.HDUList(hls)
        hdus.writeto(fn, overwrite=True)
        #Write a compressed version for the grism

        b = hdus.copy()

        b[4].data = b[4].data.compress(condition = b[4].data['ID']%downsample_factor == 0) 
        b[4].data['mass']*=downsample_factor
        b[4].data['creation_mass']*=downsample_factor

        b.writeto(fn.replace('.fits', '_downsampled.fits'), overwrite = True)





def prepare_octree(ds, ile, fle=[0.,0.,0.], fre=[1.,1.,1.], ad=None, start_level=0, debug=True):
        if True: 
                def _MetalMass(field, data):
                        return (data['metal_density']*data['cell_volume']).in_units('Msun')
                ad.ds.add_field('MetalMassMsun', function=_MetalMass, units='Msun')
        
                def _TempTimesMass(field, data):
                        te = data['thermal_energy']
                        hd = data['H_nuclei_density']
                        try:
                                temp = (2.0*te/(3.0*hd*yt.physical_constants.kb)).in_units('K')
                        except:
                                den=data['density']
                                ted=(te*den).in_units('erg/cm**3')
                                temp=(2.0*ted/(3.0*hd*yt.physical_constants.kb)).in_units('K')
                                mass=data["cell_mass"].in_units('Msun')
                
                        return temp*mass
        
                ad.ds.add_field('TemperatureTimesCellMassMsun', function=_TempTimesMass, units='K*Msun')
        

                def _cellMassMsun(field, data):
                        return data["cell_mass"].in_units('Msun')
                ad.ds.add_field('CellMassMsun', function=_cellMassMsun, units='Msun')

                def _cellVolumeKpc(field, data):
                        return data["cell_volume"].in_units('kpc**3')
                ad.ds.add_field('CellVolumeKpc', function=_cellVolumeKpc, units='kpc**3')


                def _pgascgsx(field, data):
                        try:
                                return data['momentum_x'].in_units('Msun/(kpc**2*yr)')*data['cell_volume'].in_units('kpc**3')
                        except:
                                return data['velocity_x'].in_units('kpc/yr')*data['cell_mass'].in_units('Msun')
                ad.ds.add_field('Cellpgascgsx', function=_pgascgsx, units = 'Msun*kpc/yr')


                def _pgascgsy(field, data):
                        try:
                                return data['momentum_y'].in_units('Msun/(kpc**2*yr)')*data['cell_volume'].in_units('kpc**3')
                        except:
                                return data['velocity_y'].in_units('kpc/yr')*data['cell_mass'].in_units('Msun')
                ad.ds.add_field('Cellpgascgsy', function=_pgascgsy, units = 'Msun*kpc/yr')

                def _pgascgsz(field, data):
                        try:
                                return data['momentum_z'].in_units('Msun/(kpc**2*yr)')*data['cell_volume'].in_units('kpc**3')
                        except:
                                return data['velocity_z'].in_units('kpc/yr')*data['cell_mass'].in_units('Msun')
                ad.ds.add_field('Cellpgascgsz', function=_pgascgsz, units = 'Msun*kpc/yr')




                def _cellSFRtau(field, data):
                        min_dens = 0.035 #Msun/pc^3 Ceverino et al. 2009
                        density = data["density"].in_units('Msun/pc**3')
                        temperature = data["temperature"].in_units('K')
                        volume = data["cell_volume"].in_units('pc**3')
                        sfr_times_tau = np.where(np.logical_and(density >= min_dens, temperature <= 1.0e4),density*volume,np.zeros_like(density))
                        return ds.arr(sfr_times_tau,'Msun')
                ad.ds.add_field('CellSFRtau', function=_cellSFRtau,units='Msun')

                #Tau_SFR = 12 Myr for VELA_v2  Ceverino et al. 2015
                #Not sure about VELA_v2.1 or VELA_v1
                #Using this general version should be applicable for any values used across resolutions
                #Must post-process SFR projections by dividing by Tau.
                




                fields = ["CellMassMsun","TemperatureTimesCellMassMsun","MetalMassMsun","CellVolumeKpc", "CellSFRtau", "Cellpgascgsx", "Cellpgascgsy", "Cellpgascgsz"]
                
                #gather the field data from octs 

                print("Retrieving field data")
                field_data = [] 
                for fi,f in enumerate(fields):
                        print(fi, f)
                        field_data = ad[f]
                        
                del field_data


                #Initialize dicitionary with arrays containig the needed
                #properites of all octs
                total_octs = ad.index.total_octs
                print(shape(ad.fcoords))
                mask_arr = np.zeros((2,2,2,total_octs), dtype='bool')
                
                block_iter = ad.blocks.__iter__()  
                
                for i in np.arange(total_octs):
                        octn, mask = block_iter.next()
                        mask_arr[:,:,:,i] = mask

                #added .block_slice to conform to yt 3.3    

                '''levels = octn.block_slice._ires[:,:,:, :]
                icoords = octn.block_slice._icoords[:,:,:, :]
                fcoords = octn.block_slice._fcoords[:,:,:, :]
                fwidth = octn.block_slice._fwidth[:,:,:, :]
                mask_arr = mask_arr[:,:,:,:]'''

                levels = octn._ires[:,:,:, :]
                icoords = octn._icoords[:,:,:, :]
                fcoords = octn._fcoords[:,:,:, :]
                fwidth = octn._fwidth[:,:,:, :]
                mask_arr = mask_arr[:,:,:,:]


                LeftEdge  = (fcoords[0,0,0,:,:]      - fwidth[0,0,0,:,:]*0.5)
                RightEdge = (fcoords[-1,-1,-1,:,:]   + fwidth[-1,-1,-1,:,:]*0.5)



                output = {}
                for field in fields:
                        output[field] = []




                #RCS commented out fill_octree_arrays, replaced with the code above
                octs_dic = {}
                total_octs = ad.index.total_octs
                octs_dic['LeftEdge'] = LeftEdge[:,:]
                octs_dic['dx']       = fwidth[0,0,0,:,0]
                octs_dic['Level']    = levels[0,0,0,:]

                octs_dic['Fields']    = np.array([ad[f] for f in fields])


                #Location of all octrees, at a given level, and a counter

                oct_loc = {}
                for i in np.arange(max(levels[0,0,0,:])+1):
                        oct_loc[str(i)] = [0,where(levels[0,0,0,:] == i)[0]]
                        
                oct_list = [None for i in arange (total_octs)]


                for i in arange(len(oct_loc['0'][1])):
                        if i%10000 == 0: print(i, len(oct_loc['0'][1]))
                        current_oct_id = oct_loc['0'][1][oct_loc['0'][0]]
                        current_level = 0
                        recursive_generate_oct_list(oct_list, current_oct_id, current_level, mask_arr, fcoords, fwidth, oct_loc, octs_dic)
                        oct_loc['0'][0]+=1

                #np.save('oct_list_orig.npy', oct_list)

                oct_list = array(oct_list)
                oct_list_new = add_preamble(oct_list, levels, fwidth, fcoords, LeftEdge, RightEdge, mask_arr)

                #np.save('oct_list.npy', oct_list_new)

                #oct_list_new = np.load('oct_list.npy')





                grid_structure                   = {}
                grid_structure['level']       = []
                grid_structure['refined']       = []
                grid_structure['coords']       = []
                grid_structure['level_index'] = []
                grid_structure['nleafs']      = 0.
                grid_structure['nrefined']    = 0.


                hs = hilbert_state()
                oct_obj_init = oct_list_new[0]
    
                debug = False    

                outfile = open('debug_hilbert.out', 'w+')
                a = time.time()

                OctreeDepthFirstHilbert(oct_list_new, oct_obj_init, hs, grid_structure, output, field_names = fields, debug = debug, f = outfile)
                b = time.time()


                print('DFH: ', int(b-a), 'seconds')


                if debug: outfile.close()


                return output, grid_structure, grid_structure['nrefined'], grid_structure['nleafs']

def create_fits_file(ds, fn, output, refined, particle_data, fle, fre, no_gas_p = False,form='VELA'):
    #first create the grid structure
    structure = pyfits.Column("structure", format="B", array=array(refined).astype("bool"))
    cols = pyfits.ColDefs([structure])
    st_table = pyfits.BinTableHDU.from_columns(cols)
    st_table.name = "GRIDSTRUCTURE"
    st_table.header.set("hierarch lengthunit", "kpc", comment="Length unit for grid")
    fre = ds.arr(fre, 'code_length').in_units('kpc').value
    fle = ds.arr(fle, 'code_length').in_units('kpc').value
    fdx = fre-fle



    for i,a in enumerate('xyz'):
        st_table.header.set("min%s" % a, fle[i])
        st_table.header.set("max%s" % a, fre[i])
        st_table.header.set("n%s" % a, fdx[i])
        st_table.header.set("subdiv%s" % a, 2)
    st_table.header.set("subdivtp", "OCTREE", "Type of grid subdivision")

    #not the hydro grid data
    fields = ["CellMassMsun","TemperatureTimesCellMassMsun", "MetalMassMsun", "CellVolumeKpc", "CellSFRtau","Cellpgascgsx", "Cellpgascgsy", "Cellpgascgsz"]

    fd = {}
    for i,f in enumerate(fields): 
        fd[f]=array(output[f][:])
    del output

    col_list = []

    col_list.append(pyfits.Column("mass_gas", format='D',
                    array=fd["CellMassMsun"], unit="Msun"))
    col_list.append(pyfits.Column("mass_metals", format='D',
                    array=fd['MetalMassMsun'], unit="Msun"))
    col_list.append(pyfits.Column("gas_temp_m", format='D',
                    array=fd['TemperatureTimesCellMassMsun'], unit="K*Msun"))
    col_list.append(pyfits.Column("gas_teff_m", format='D',
                    array=fd['TemperatureTimesCellMassMsun'], unit="K*Msun"))
    col_list.append(pyfits.Column("cell_volume", format='D',
                    array=fd['CellVolumeKpc'], unit="kpc^3"))
    col_list.append(pyfits.Column("SFR", format='D',
                    array=fd['CellSFRtau'],  unit = 'Msun'))

    m = 1
    if no_gas_p: m = 0
    p_gas_zipped = zip(fd['Cellpgascgsx']*m,
                       fd['Cellpgascgsy']*m,
                       fd['Cellpgascgsz']*m)

    col_list.append(pyfits.Column("p_gas", format='3D',
                    array=p_gas_zipped , unit = 'Msun*kpc/yr'))


    cols = pyfits.ColDefs(col_list)
    mg_table = pyfits.BinTableHDU.from_columns(cols)
    #mg_table = pyfits.new_table(cols)
    mg_table.header.set("M_g_tot", fd["CellMassMsun"].sum())
    mg_table.header.set("timeunit", "yr")
    mg_table.header.set("tempunit", "K")
    mg_table.name = "GRIDDATA"

    # Add a dummy Primary; might be a better way to do this!
    col_list = [pyfits.Column("dummy", format="E", array=np.zeros(1, dtype='float32'))]
    cols = pyfits.ColDefs(col_list)
    md_table = pyfits.BinTableHDU.from_columns(cols, nrows = len(fd['CellSFRtau']))
    #md_table = pyfits.new_table(cols)
    md_table.header.set("snaptime", ds.current_time.in_units('yr').value[()])
    md_table.name = "YT"

    phdu = pyfits.PrimaryHDU()
    phdu.header.set('nbodycod','yt')
    hls = [phdu, st_table, mg_table,md_table]
    hls.append(particle_data)
    hdus = pyfits.HDUList(hls)
    hdus.writeto(fn, clobber=True)





def round_nocts_wide(dds,fle,fre,nwide=None):
    fc = (fle+fre)/2.0

    assert np.all(fle < fc)
    assert np.all(fre > fc)
    ic = np.rint(fc*dds) #nearest vertex to the center
    ile,ire = ic.astype('int32'),ic.astype('int32')
    cfle,cfre = fc.copy(),fc.copy()
    idx = np.array([0,0,0]) #just a random non-equal array
    width = 0.0
    if nwide is None:
        #expand until borders are included and
        #we have an equaly-sized, non-zero box
        idxq,out=False,True
        while not out or not idxq:
            cfle,cfre = fc-width, fc+width
            #These .ceil and floors were rints (commented by rcs)
            ile = np.floor(cfle*dds).astype('int32')
            ire = np.ceil(cfre*dds).astype('int32')
            idx = ire-ile
            width += 0.1/dds
            #quit if idxq is true:
            idxq = idx[0]>0 and np.all(idx==idx[0])
            out  = np.all(fle>cfle) and np.all(fre<cfre) 
            out &= abs(np.log2(idx[0])-np.rint(np.log2(idx[0])))<1e-5 #nwide should be a power of 2
            assert width[0] < 1.1 #can't go larger than the simulation volume
        nwide = idx[0]
    else:
        #expand until we are nwide cells span
        while not np.all(idx==nwide):
            assert np.any(idx<=nwide)
            cfle,cfre = fc-width, fc+width
            #These .ceil and floors were rints (commented by rcs)
            ile = np.floor(cfle*dds).astype('int32')
            ire = np.ceil(cfre*dds).astype('int32')
            idx = ire-ile
            width += 1e-2*1.0/dds
    assert np.all(idx==nwide)
    assert idx[0]>0
    maxlevel = -np.rint(np.log2(nwide)).astype('int32')
    assert abs(np.log2(nwide)-np.rint(np.log2(nwide)))<1e-5 #nwide should be a power of 2
    return ile,ire,maxlevel,nwide

def prepare_star_particles(ds,star_type,pos=None,vel=None, age=None, creation_time=None,
    initial_mass=None, current_mass=None,metallicity=None, radius = None, 
                           fle=[0.,0.,0.],fre=[1.,1.,1.], ad=None, radkpc=0.01):

    if ad is None:
        ad = ds.all_data()

    nump = ad[star_type,"particle_ones"]
    assert nump.sum()>1 #make sure we select more than a single particle
    
    if pos is None:
        pos = yt.YTArray([ad[star_type,"particle_position_%s" % ax]
                        for ax in 'xyz']).transpose()

    idx = np.all(pos > fle, axis=1) & np.all(pos < fre, axis=1)
    assert np.sum(idx)>0 #make sure we select more than a single particle
    pos = pos[idx].in_units('kpc') #unitary units -> kpc

    if creation_time is None:
        try:
                formation_time = ad[star_type,"particle_creation_time"][idx].in_units('yr')
        except:
                formation_time = ad[star_type,'creation_time'][idx].in_units('yr')
                
    if age is None:
        age = (ds.current_time - formation_time).in_units('yr')

    if vel is None:
        vel = yt.YTArray([ad[star_type,"particle_velocity_%s" % ax]
                        for ax in 'xyz']).transpose()
        # Velocity is cm/s, we want it to be kpc/yr
        #vel *= (ds["kpc"]/ds["cm"]) / (365*24*3600.)
        vel = vel[idx].in_units('kpc/yr')
    
    if initial_mass is None:
        #in solar masses
        try:
                initial_mass = ad[star_type,"particle_mass_initial"][idx].in_units('Msun')
        except:
                initial_mass = ad[star_type,"particle_mass"][idx].in_units('Msun')

    if current_mass is None:
        #in solar masses
        current_mass = ad[star_type,"particle_mass"][idx].in_units('Msun')
    
    if metallicity is None:
        #this should be in dimensionless units, metals mass / particle mass
        try:
                metallicity = ad[star_type,"particle_metallicity1"][idx]
        except:
                metallicity = ad[star_type,"metallicity_fraction"][idx]
                
                
    if radius is None:
        radius = ds.arr(metallicity*0.0 + radkpc, 'kpc') #10pc radius
    
    #create every column
    col_list = []
    col_list.append(pyfits.Column("ID", format="J", array=np.arange(current_mass.size).astype('int32')))
    col_list.append(pyfits.Column("parent_ID", format="J", array=np.arange(current_mass.size).astype('int32')))
    col_list.append(pyfits.Column("position", format="3D", array=pos, unit="kpc"))
    col_list.append(pyfits.Column("velocity", format="3D", array=vel, unit="kpc/yr"))
    col_list.append(pyfits.Column("creation_mass", format="D", array=initial_mass, unit="Msun"))
    col_list.append(pyfits.Column("formation_time", format="D", array=formation_time, unit="yr"))
    col_list.append(pyfits.Column("radius", format="D", array=radius, unit="kpc"))
    col_list.append(pyfits.Column("mass", format="D", array=current_mass, unit="Msun"))
    col_list.append(pyfits.Column("age", format="D", array=age,unit='yr'))
    #For particles, Sunrise takes 
    #the dimensionless metallicity, not the mass of the metals
    col_list.append(pyfits.Column("metallicity", format="D",
        array=metallicity,unit="dimensionless")) 
    
    #make the table
    cols = pyfits.ColDefs(col_list)
    pd_table = pyfits.BinTableHDU.from_columns(cols)
    #pd_table = pyfits.new_table(cols)
    pd_table.name = "PARTICLEDATA"
    
    #make sure we have nonzero particle number
    assert pd_table.data.shape[0]>0
    return pd_table, np.sum(idx)

def save_to_gridstructure(grid_structure, level, fcoords, refined, leaf):
    '''
    Function to save grid information
    '''
    grid_structure['level'].append(level)
    grid_structure['refined'].append(refined)
    grid_structure['coords'].append(fcoords)
    if leaf:
        grid_structure['nleafs']+=1
    if refined:
        grid_structure['nrefined']+=1
    return













































