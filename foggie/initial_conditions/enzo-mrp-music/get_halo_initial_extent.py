import os
import sys
import yt
import glob
import h5py as h5
import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.rank
    my_size = comm.size
    parallel = True
    yt.enable_parallelism()
except:
    my_rank = 0
    my_size = 1
    parallel = False

axes = ['x', 'y', 'z']

Msun_to_g = 1.988921e+33
Mpc_to_cm = 3.08568025e24
rho_crit_now = 1.8788e-29

def read_from_catalog(my_halo, halo_catalog, fields=None):
    if fields is None:
        fields = {'id': 0, 'mass': 1, 'center': [7, 8, 9]}
    if 'id' not in fields:
        print ("id column not given.")
        return None
    halo_data = np.loadtxt(halo_catalog)
    this_halo = halo_data[:, fields['id']] == my_halo['id']
    my_data = {}
    for field, column in fields.items():
        if isinstance(column, list):
            my_data[field] = [halo_data[this_halo, my_column][0] for my_column in column]
        else:
            my_data[field] = halo_data[this_halo, column][0]
    return my_data

def get_halo_sphere_particles(my_halo, par_file, radius_factor=5):
    pf = yt.load(par_file)
    halo_catalog = os.path.join(pf.fullpath, 'MergerHalos.out')
    if 'center' in my_halo:
        my_halo_data = my_halo
        my_halo_data["center"] = pf.arr(my_halo['center'][0], my_halo['center'][1])
    else:
        my_halo_data = read_from_catalog(my_halo, halo_catalog)
    rho_crit = pf.quan(rho_crit_now, "g/cm**3") * pf.hubble_constant**2 * \
        (1 + pf.current_redshift)**3
    if 'mass' in my_halo:
        if isinstance(my_halo_data['mass'], tuple):
            units = my_halo_data['mass'][1]
        else:
            units = "Msun"
        hmass = pf.quan(my_halo_data['mass'][0], units)
    else:
        hmass = None

    if 'radius' in my_halo:
        r_200 = pf.quan(my_halo['radius'][0], my_halo['radius'][1])
    else:
        r_200 = (((3. * hmass) /
                  (4. * np.pi * rho_crit * 200.))**(1./3.)).to("Mpc")

    if yt.is_root():
        print ("Reading particles for a sphere surrounding halo %d." % my_halo['id'])
        print ("Halo %d, pos: %f, %f, %f, mass: %s, r_200: %s." % \
            (my_halo_data['id'], my_halo_data['center'][0], my_halo_data['center'][1],
             my_halo_data['center'][2], hmass, r_200))

    my_sphere = pf.sphere(my_halo_data['center'], radius_factor * r_200)
    return (my_halo_data['center'],
            my_sphere['particle_index'], 
            my_sphere['particle_mass'].in_units('Msun'),
            np.array([my_sphere['particle_position_x'],
                      my_sphere['particle_position_y'],
                      my_sphere['particle_position_z']]))

def get_halo_particles(my_halo, par_file):
    pf = yt.load(par_file)

    if yt.is_root():
        print ("Reading in particles for halo %d." % my_halo['id'])

    halo_files = glob.glob(os.path.join(pf.fullpath, 'MergerHalos_*.h5'))

    particle_indices = np.array([])
    particle_masses = np.array([])
    pos_x = np.array([])
    pos_y = np.array([])
    pos_z = np.array([])
    halo_name = 'Halo%08d' % my_halo['id']
    for halo_file in halo_files:
        input = h5.File(halo_file, 'r')
        if halo_name in input.keys():
            particle_indices = np.concatenate([particle_indices, input[halo_name]['particle_index'].value])
            particle_masses = np.concatenate([particle_masses, input[halo_name]['ParticleMassMsun'].value])
            pos_x = np.concatenate([pos_x, input[halo_name]['particle_position_x'].value])
            pos_y = np.concatenate([pos_y, input[halo_name]['particle_position_y'].value])
            pos_z = np.concatenate([pos_z, input[halo_name]['particle_position_z'].value])
        input.close()
    particle_positions = np.array([pos_x, pos_y, pos_z])
    if particle_indices is None:
        if yt.is_root():
            print ("Error: could not locate halo %d." % my_halo['id'])
        return None
    return (particle_indices, particle_masses, particle_positions)

def get_halo_indices(my_halo, dataset, method='sphere', radius_factor=5.0):
    shifted = np.zeros(3, dtype=bool)

    if method == 'halo':
        halo_indices, particle_masses, particle_positions = \
            get_halo_particles(my_halo, dataset)
    elif method == 'sphere':
        halo_com, halo_indices, particle_masses, particle_positions = \
            get_halo_sphere_particles(my_halo, dataset, radius_factor=radius_factor)

    unitary_1 = halo_com.to("unitary").uq
    for i, axis in enumerate(axes):
        if particle_positions[i].max() - particle_positions[i].min() > 0.5:
            if yt.is_root():
                print ("Halo periodic in %s." % axis)
            particle_positions[i] -= 0.5
            particle_positions[i][particle_positions[i] < 0.0] += 1.0
            halo_com[i] -= 0.5 * unitary_1
            if halo_com[i] < 0.0:
                halo_com[i] += 1.0 * unitary_1
            shifted[i] = True
    if method == 'halo':
        halo_com = (particle_positions * particle_masses).sum(axis=1) / particle_masses.sum()
    return (halo_indices, halo_com, shifted)

def get_center_and_extent(my_halo, 
                          initial_dataset,
                          final_dataset,
                          round_size=None,
                          radius_factor=5.0,
                          output_format=None):
    if output_format not in ["hdf5", "txt", None]:
        raise RuntimeError("output_format = %s not known.  Valid choices: hdf5, txt, None" %
                           (output_format))
    if "id" not in my_halo:
        my_halo['id'] = 0
    halo_indices, halo_com, shifted = get_halo_indices(my_halo, final_dataset,
                                                       radius_factor=radius_factor)
    halo_size = halo_indices.size
    if halo_indices is None: sys.exit(0)
    if yt.is_root():
        print ("Halo %d has %d particles." % (my_halo['id'], halo_size))
        print ("Halo center of mass: %f, %f, %f." % \
            (halo_com[0], halo_com[1], halo_com[2]))
        print ("Comparing datasets:\n" \
              "\t Original: %s\n" \
              "\t Final:    %s" % (initial_dataset, final_dataset))
        

    pf = yt.load(initial_dataset)

    num_stars = (halo_indices >= pf.parameters['NumberOfParticles']).sum()
    if yt.is_root():
        print ("Removing %d star particles." % num_stars)
    halo_indices = halo_indices[halo_indices < pf.parameters['NumberOfParticles']]

    axis_min = [None for axis in axes]
    axis_max = [None for axis in axes]

    # Save positions for writing to file
    save_pos = [[], [], []]

    my_work = slice(my_rank, None, my_size)
    if yt.is_root():
        print ("Reading in initial particle positions.")
    for grid in pf.index.grids[my_work]:
        if halo_indices.size <= 0: break

        grid_p_indices = grid['particle_index']
        my_indices = np.in1d(grid_p_indices, halo_indices)

        if my_indices.sum() == 0: continue

        print ("PROC %04d - %s: %d matching particles." % (my_rank, grid, my_indices.sum()))

        for i, axis in enumerate(axes):
            particle_position = grid['particle_position_%s' % axis][my_indices].value
            save_pos[i] += particle_position.tolist()
            if shifted[i]:
                particle_position -= 0.5
                particle_position[particle_position < 0.0] += 1.0
            particle_position -= halo_com[i].v
            particle_position[particle_position > 0.5] -= 1.0
            particle_position[particle_position < -0.5] += 1.0
            my_min = particle_position.min()
            my_max = particle_position.max()

            if axis_min[i] is None:
                axis_min[i] = my_min
            else:
                axis_min[i] = min(axis_min[i], my_min)

            if axis_max[i] is None:
                axis_max[i] = my_max
            else:
                axis_max[i] = max(axis_max[i], my_max)

            halo_indices = np.setdiff1d(halo_indices,
                                        halo_indices[np.in1d(halo_indices,
                                                             grid_p_indices)])

    for i in range(len(axes)):
        if axis_min[i] is None: axis_min[i] = 2.0
        if axis_max[i] is None: axis_max[i] = -2.0
        if parallel:
            axis_min[i] = comm.allreduce(axis_min[i], op=MPI.MIN)
            axis_max[i] = comm.allreduce(axis_max[i], op=MPI.MAX)

    axis_min = np.array(axis_min)
    axis_max = np.array(axis_max)
    halo_com = np.array(halo_com)

    # Gather all positions for output
    save_pos = np.array(save_pos)
    if parallel:
        temp_array = comm.gather(save_pos)
        if my_rank == 0:
            all_save_pos = np.empty((3, halo_size))
            p = 0
            for a in temp_array:
                s = a.shape[1]
                all_save_pos[:,p:(p+s)] = a
                p += s
    else:
        all_save_pos = save_pos

    if my_rank == 0:
        print ("Halo %d has %d particles (%d dark matter)." % (my_halo['id'], halo_size,
                                                              (halo_size-num_stars)))
        print ("Halo center of mass: %f, %f, %f." % \
            (halo_com[0], halo_com[1], halo_com[2]))
        print ("Initial particle extend for halo %d [in domain center]." % my_halo['id'])
        for i, axis in enumerate(axes):
            print ("%s: %+f to %+f [%f %f]." % (axis, axis_min[i], axis_max[i],
                                               (axis_min[i] + 0.5),
                                               (axis_max[i] + 0.5)))


        best_center = 0.5 * (axis_min + axis_max) + halo_com
        print ("")
        print ("Ideal center: %.12f %.12f %.12f." % \
            (best_center[0], best_center[1], best_center[2]))
        my_region = axis_max - axis_min
        print ("Region size: %.12f %.12f %.12f." % \
            (my_region[0], my_region[1], my_region[2]))
        min_size = np.max(my_region)
        print ("Minimum cubical region size: %f." % min_size)
        if round_size is not None:
            my_region = np.ceil(round_size * my_region) / round_size
            print ("Region size (to nearest 1/%d): %.12f %.12f %.12f." % \
                (round_size, my_region[0], my_region[1], my_region[2]))
            min_size = np.ceil(round_size * min_size) / round_size
            print ("Minimum cubical region size (to nearest 1/%d): %f." % \
              (round_size, min_size))
        output_fn = "initial_particle_positions-%d-%s" % (my_halo['id'], pf)
        if output_format == "hdf5":
            output_fn += ".h5"
            fp = h5.File(output_fn, "w")
            fp["pos"] = all_save_pos.T
            fp.close()
        elif output_format == "txt":
            output_fn += ".dat"
            np.savetxt(output_fn, all_save_pos.T)
        elif output_format != None:
            raise RuntimeError("output_format = %s not known.  Valid choices: hdf5, txt, None" % (output_format))
        else:
            output_fn = None
    else:
        best_center = None
        min_size = None
        my_region = None
        output_fn = None

    if parallel:
        best_center = comm.bcast(best_center)
        output_fn = comm.bcast(output_fn)
        my_region = comm.bcast(my_region)

    return (best_center, my_region, output_fn)
        
if __name__ == '__main__':
    my_halo = {'id': 0,
               'center': [ 0.46977694,  0.51334109,  0.48830934],
               'mass': 3.45e8, 
               #'radius': 23.4,
               'radius_units': 'kpc'}

    get_center_and_extent(my_halo, 'DD0000/output_0000', 'DD0020/output_0020',
                          round_size=32, radius_factor=5.0)
#    from copy import deepcopy
#    for i in range(7,15):
#        fn = "DD%4.4d/output_%4.4d" % (i,i)
#        my_halo_copy = deepcopy(my_halo)
#        get_center_and_extent(my_halo_copy, fn, 'DD0015/output_0015',
#                              round_size=32)
