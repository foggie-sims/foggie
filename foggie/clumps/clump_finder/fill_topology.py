import numpy as np
from scipy.ndimage import label
from scipy.ndimage import map_coordinates

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
    
    # Create a mask to fill only small holes
    for hole_label in range(1, num_features + 1):
        hole_mask = labeled_holes == hole_label
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

    # Interpolate the matrix values at the slice plane coordinates
    slice_data = map_coordinates(matrix, slice_coords, order=1, mode='constant', cval=0)

    # Convert the floating-point coordinates to integer indices (rounded to nearest grid points)
    indices = np.stack(slice_coords, axis=-1).astype(int)

    return slice_data, indices


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