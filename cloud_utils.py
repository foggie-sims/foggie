"""
useful stuff for cloud analysis JT090618
"""
import copy
import numpy as np

def reduce_ion_vector(vx, ion):
    """ this function takes in two vectors for velocity and ionization
        fraction and chunks the ionization fraction into a uniform velocity
        grid. JT 082018"""
    v = np.arange(3001) - 1500
    ion_hist = v * 0.
    index = np.clip(np.around(vx) + 1500, 0, 2999)
    for i in np.arange(np.size(ion)):
        ion_hist[int(index[i])] = ion_hist[int(
            index[i])] + ion[int(i)]

    return v, ion_hist

def get_fion_threshold(ion_to_use, coldens_fraction):
    cut = 0.999
    total = np.sum(ion_to_use)
    ratio = 0.001
    while ratio < coldens_fraction:
        part = np.sum(
            ion_to_use[ion_to_use > cut * np.max(ion_to_use)])
        ratio = part / total
        cut = cut - 0.001

    threshold = cut * np.max(ion_to_use)
    number_of_cells_above_threshold = np.size(
        np.where(ion_to_use > threshold))

    return threshold, number_of_cells_above_threshold

def get_sizes(ray_df, species, x, axis_to_use, ion_to_use, cell_mass, coldens_threshold):

    threshold, number_of_cells = get_fion_threshold(
        ion_to_use, coldens_threshold)

    dx = np.array(ray_df['dx'])
    ion_density = copy.deepcopy(ion_to_use)

    axis_velocity = np.array(ray_df[axis_to_use+'-velocity'])
    print('V'+axis_to_use, axis_velocity)
    print(axis_to_use, x)

    indexsizes = []
    kpcsizes = []
    column_densities = []
    masses = []
    centers = []
    velocities = []
    indices = []
    xs = []
    for m in np.arange(100):  # can find up to 100 peaks
        i = np.squeeze(np.where(np.array(ion_to_use) > threshold))

        if np.size(i) >= 1:
            startindex = np.min(i)
            f = ion_to_use[startindex]
            index = startindex
            ion_to_use[startindex] = 0.0
            sum_mass = cell_mass[startindex]
            sum_coldens = ion_density[startindex] * dx[index]
            count = 0
            velsum = 0.
            while (f > threshold) and (index < np.size(x)-1):
                count += 1
                if (count > 10000):
                    os.sys.exit('stuck in the size finding loop')
                index += 1
                if index == np.size(x):  # this means we're at the edge
                    index = np.size(x)-1
                    f = 0.0
                else:
                    f = ion_to_use[index]
                    ion_to_use[index] = 0.0
                    sum_mass = sum_mass + cell_mass[index]
                    velsum = velsum + \
                        cell_mass[index] * axis_velocity[index]
                    sum_coldens = sum_coldens + \
                        ion_density[index] * dx[index]

            x_coord = x[startindex:index]
            ion_d = ion_density[startindex:index]
            ion_center = np.sum(x_coord * ion_d) / np.sum(ion_d)

            indexsizes.append(index - startindex)
            kpcsizes.append(x[startindex]-x[index])
            column_densities.append(sum_coldens)
            masses.append(sum_mass)
            # should end up with mass-weighted velocity along LOS
            velocities.append(velsum / sum_mass)
            centers.append(ion_center)
            indices.append(index)
            xs.append(x[index])

    size_dict = {'coldens_threshold': coldens_threshold}
    size_dict[species+'_xs'] = xs
    size_dict[species+'_indices'] = indices
    size_dict[species+'_kpcsizes'] = kpcsizes
    size_dict[species+'_indexsizes'] = indexsizes
    size_dict[species+'_coldens'] = column_densities
    size_dict[species+'_n_cells'] = number_of_cells
    size_dict[species+'_cell_masses'] = masses
    size_dict[species+'_centers'] = centers
    size_dict[species+'_velocities'] = velocities

    return size_dict

