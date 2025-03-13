import numpy as np
from scipy.ndimage import label
from scipy.ndimage import map_coordinates
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_closing
from joblib import Parallel, delayed
import multiprocessing
from functools import partial
import time

def fill_voids(mask,max_hole_size=None,structure=None):
    '''
    Fill topologically closed regions within a matrix (2d or 3d). Recommended for disk finding only.
    '''

    mask = mask.astype(bool)
    holes = ~mask
    if structure is not None:
       labeled_holes, num_features = label(holes, structure=structure)
    else:
       labeled_holes, num_features = label(holes)
    
    '''
    # Create a mask to track components connected to the edge of the domain
    boundary_connected = np.zeros_like(mask, dtype=bool)
    
    # Check all edges of the domain for boundary-connected holes
    slices = [slice(None)] * mask.ndim
    for axis in range(mask.ndim):
        for side in [0, -1]:  # Check the start and end along each axis
            edge_slices = slices.copy()
            edge_slices[axis] = side
            boundary_connected |= labeled_holes == labeled_holes[tuple(edge_slices)]
    '''
    
    # Create a mask to fill only small holes
    for hole_label in range(1, num_features + 1):
        hole_mask = labeled_holes == hole_label
        #if not np.any(boundary_connected[hole_mask]):
        if True:
            if max_hole_size is None:
                mask[hole_mask] = True
            elif np.sum(hole_mask) <= max_hole_size:
                # Fill the small hole
                mask[hole_mask] = True
    #print("Filled!")
    return mask



def slice_along_arbitrary_axis(matrix, axis_vector, slice_position):
    """
    Extract a slice from a 3D matrix along an arbitrary axis and return the slice 
    along with the corresponding indices in the original matrix.

    Parameters:
        matrix (np.ndarray): The input 3D matrix.
        axis_vector (np.ndarray): A 3-element array defining the slicing axis (arbitrary vector).
        slice_position (float): Position along the axis to extract the slice.

    Returns:
        tuple: 
            - np.ndarray: A 2D slice of the matrix.
            - np.ndarray: Indices in the original matrix corresponding to the slice.
    """
    # Normalize the axis vector
    t0=time.time()
    axis_vector = axis_vector / np.linalg.norm(axis_vector)

    # Get the shape of the matrix
    shape = np.array(matrix.shape)

    # Define a grid of points for the original matrix
    z, y, x = np.meshgrid(
        np.arange(shape[2]),
        np.arange(shape[1]),
        np.arange(shape[0]),
        indexing="ij"
    )

    # Compute the coordinates of the slice plane
    # Start with the center of the grid
    center = shape / 2

    # Define a grid in the slice plane (orthogonal to the axis vector)
    plane_size = max(shape)
    u = np.linspace(-plane_size / 2, plane_size / 2, plane_size)
    v = np.linspace(-plane_size / 2, plane_size / 2, plane_size)
    U, V = np.meshgrid(u, v)

    # Compute the normal to the slice plane
    # The slice plane passes through (slice_position * axis_vector) relative to the center
    slice_origin = center + slice_position * axis_vector

    # Define two orthogonal vectors to the axis_vector
    if np.abs(axis_vector[0]) < 1e-6:
        ortho1 = np.cross(axis_vector, [1, 0, 0])
    else:
        ortho1 = np.cross(axis_vector, [0, 1, 0])
    ortho1 /= np.linalg.norm(ortho1)

    ortho2 = np.cross(axis_vector, ortho1)
    ortho2 /= np.linalg.norm(ortho2)

    # Compute the coordinates of the slice plane in the original grid
    slice_coords = (
        slice_origin[0] + U * ortho1[0] + V * ortho2[0],
        slice_origin[1] + U * ortho1[1] + V * ortho2[1],
        slice_origin[2] + U * ortho1[2] + V * ortho2[2],
    )

    t1=time.time()

    # Interpolate the matrix values at the slice plane coordinates
    slice_data = map_coordinates(matrix, slice_coords, order=1, mode='constant', cval=0)

    # Convert the floating-point coordinates to integer indices (rounded to nearest grid points)
    indices = np.stack(slice_coords, axis=-1).astype(int)
    t2=time.time()
    print("Time to slice:",t1-t0,"Time to interpolate:",t2-t1)
    return slice_data, indices


def generate_connectivity_matrix(size,use_cylindrical_connectivity_matrix = False,return_full_connectivity=False):
    if return_full_connectivity: return np.ones((size,size,size))
    if use_cylindrical_connectivity_matrix: return generate_cylindrical_connectivity(size)
    # Create coordinate grids
    x,y,z = np.ogrid[:size, :size, :size]

    # Compute the distance from the center
    radius = np.floor(size/2)
    distance = np.sqrt((x - radius) ** 2 + (y - radius) ** 2 + (z - radius) ** 2)

    return (distance <= radius)


def generate_cylindrical_connectivity(size):
    # Create coordinate grids
    x,y,z = np.ogrid[:size, :size, :size]

    # Compute the distance from the cylinder axis (along Z)
    radius = np.floor(size/2)
    distance = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)

    # Define mask: points inside the cylinder are 1, outside are 0
    return ((distance <= radius) & (z <= size))
    





def iterate_fill_holes(i,unit_vector, mask, max_size, structure):
    nx0,ny0,nz0=np.shape(mask)
    filled_mask = np.copy(mask)
    t0=time.time()
    slice_mask,indices = slice_along_arbitrary_axis(np.array(mask).astype(float),unit_vector, i)
    t1=time.time()
    slice_mask = (slice_mask > 0.5)
    if np.size(np.where(slice_mask))<=0: return filled_mask

    filled_slice_mask = fill_voids(slice_mask, max_size, structure=structure)
    t2=time.time()
    indices2=np.zeros((np.size(filled_slice_mask),3)).astype(int)
    indices2[:,0] = indices[:,:,0].flatten()
    indices2[:,1] = indices[:,:,1].flatten()
    indices2[:,2] = indices[:,:,2].flatten()
    
    grid_mask = np.where(( (indices2[:,0]>=0) & (indices2[:,1]>=0) & (indices2[:,2]>=0) & (indices2[:,0]<nx0) & (indices2[:,1]<ny0) & (indices2[:,2]<nz0)))

    filled_mask[indices2[grid_mask,0],indices2[grid_mask,1],indices2[grid_mask,2]] = filled_slice_mask.flatten()[grid_mask] &  ~slice_mask.flatten()[grid_mask] #only get indices that were filled by the hole finder
    t3=time.time()
   # print("Thread",i,"-Time to slice:",t1-t0,"Time to fill:",t2-t1,"Time to update:",t3-t2)
    return filled_mask

def fill_holes_parallel(unit_vector, mask, max_size, nz, n_jobs=None, structure = None):
    """
    Slices the binary mask along the plane defined by the unit vector.
    Fills holes that are smaller than the given size along each size,
    updates the original mask.

    Parameters:
        mask (np.ndarray): The input binary mask.
        max_size (int): The maximum size of the holes to fill.

    Returns:
        np.ndarray: The mask with the holes filled
    """

    _iterate_fill_holes = partial(iterate_fill_holes, unit_vector=unit_vector, mask=mask, max_size=max_size, structure=structure)
    num_cores = multiprocessing.cpu_count()-1
    if n_jobs is None:
        n_jobs =num_cores
    elif n_jobs > num_cores:
        n_jobs = num_cores

    if n_jobs > nz:
        n_jobs = nz

    print("Filling holes with",n_jobs,"threads...")

    filled_mask_list = Parallel(n_jobs=n_jobs)(delayed(_iterate_fill_holes)(i) for i in np.linspace(-nz/2,nz/2,nz))

    print("Combining filled masks...")
    filled_mask = np.copy(mask)
    for filled_mask_i in filled_mask_list:
        filled_mask = filled_mask | filled_mask_i

    return filled_mask       

def fill_holes(unit_vector, mask, max_size, nz, structure = None):
    """
    Slices the binary mask along the plane defined by the unit vector.
    Fills holes that are smaller than the given size along each size,
    updates the original mask.

    Parameters:
        mask (np.ndarray): The input binary mask.
        max_size (int): The maximum size of the holes to fill.

    Returns:
        np.ndarray: The mask with the holes filled
    """
    filled_mask = np.copy(mask)
    nx0,ny0,nz0=np.shape(filled_mask)
    nFailed=0
    for i in np.linspace(-nz/2,nz/2,nz):
        try:
            slice_mask,indices = slice_along_arbitrary_axis(np.array(mask).astype(float),unit_vector, i)
            slice_mask = (slice_mask > 0.5)
            if np.size(np.where(slice_mask))<=0: continue

            filled_slice_mask = fill_voids(slice_mask, max_size, structure=structure)
            indices2=np.zeros((np.size(filled_slice_mask),3)).astype(int)
            indices2[:,0] = indices[:,:,0].flatten()
            indices2[:,1] = indices[:,:,1].flatten()
            indices2[:,2] = indices[:,:,2].flatten()
    
            grid_mask = np.where(( (indices2[:,0]>=0) & (indices2[:,1]>=0) & (indices2[:,2]>=0) & (indices2[:,0]<nx0) & (indices2[:,1]<ny0) & (indices2[:,2]<nz0)))

            filled_mask[indices2[grid_mask,0],indices2[grid_mask,1],indices2[grid_mask,2]] = filled_slice_mask.flatten()[grid_mask] &  ~slice_mask.flatten()[grid_mask] #only get indices that were filled by the hole finder

        except:
            nFailed+=1
    if nFailed>0: print("Warning: ",nFailed,"/",nz," slices failed to fill.")
    return (filled_mask | mask)


def expand_slice(slice_obj, max_dilation, max_size):
    '''
    Expands slice object so that dilation does not overflow the minimal parallelepiped returned by scipy.ndimage.find_objects.
    Will not go past the bounds of the original cut region.
    '''
    new_start = max(slice_obj.start - max_dilation, 0)
    new_stop = min(slice_obj.stop + max_dilation, max_size)
    return slice(new_start, new_stop)


from scipy.ndimage import binary_dilation
def get_dilated_shells(mask, n_iterations, cells_per_iteration, structure=None):
    """
    Get the dilated shells of a binary mask. The thickness of each shell is defined by cells_per_iteration.
    The total dilated mask is be the union of the original mask and all shells.

    Parameters:
        mask (np.ndarray): The input binary mask.
        n_iterations (int): The number of dilation iterations.
        structure (np.ndarray): The structure for the binary dilation.

    Returns:
        list of np.ndarrays: The dilated shells of the mask.
    """

    dilated_mask = np.copy(mask)
    shells = []
    for i in range(0,n_iterations):
        dilated_mask_2 = binary_dilation(dilated_mask, iterations = cells_per_iteration, structure=structure)
        shells.append(dilated_mask_2 & ~dilated_mask)
        dilated_mask = np.copy(dilated_mask_2)

    return shells



from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
def fill_convex_hull(mask,max_length=1.0):
    """
    Get the convex hull of a binary mask and return a filled binary mask.

    Parameters:
        mask (np.ndarray): The input binary mask.

    Returns:
        np.ndarray: The convex hull of the mask.
    """
    
    points = np.argwhere(mask).astype(np.int16)
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices]) #tessellate hull into triangles

    idx_2d = np.indices(mask.shape[1:],np.int16)
    idx_2d = np.moveaxis(idx_2d,0,-1)

    idx_3d = np.zeros((*mask.shape[1:],mask.ndim))
    idx_3d[:,:,1:] = idx_2d
    
    filled_mask = np.zeros_like(mask,dtype=bool)
    for z in range(len(mask)):
        idx_3d[:,:,0] = z
        s = deln.find_simplex(idx_3d)

    

        filled_mask[z, (s!=-1)] = True



    return mask | filled_mask


