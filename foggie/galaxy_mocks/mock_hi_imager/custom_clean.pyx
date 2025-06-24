import numpy as np
cimport numpy as np

# Disable bounds checking and wraparound for speed
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def custom_clean(np.ndarray[np.float64_t, ndim=2] dirty_slice, 
                 np.ndarray[np.float64_t, ndim=2] beam,
                 np.ndarray[np.float64_t, ndim=2] clean_beam, 
                 float max_residual_limit, 
                 int tid,
                 float clean_gain,
                 int max_itr):

    cdef int itr = 0
    cdef int nx = dirty_slice.shape[0]
    cdef int ny = dirty_slice.shape[1]

    # Allocate memory for residual and clean map
    cdef double[:,:] residual = np.copy(dirty_slice)
    cdef double[:,:] clean_map = np.zeros_like(dirty_slice)

    cdef double[:,:] beam_mv = np.copy(beam)
    cdef double[:,:] clean_beam_mv = np.copy(clean_beam)

    cdef double max_residual
    cdef int max_x, max_y, xshift, yshift
    cdef int b_min_x, b_max_x, b_min_y, b_max_y
    cdef int i,j

    #cdef int max_iterations = nx*ny
    cdef int max_iterations = max_itr


    while itr < max_iterations:
        # Compute max residual and it's index
        max_residual = 0
        #if tid==0: print('In Cython: Finding max...')
        for i in range(0,nx):
            for j in range(0,ny):
                if residual[i,j] > max_residual:
                    max_residual = residual[i,j]
                    max_x = i
                    max_y = j

        if max_residual <= max_residual_limit:
            break

        # Compute shifts
        xshift = max_x - nx//2
        yshift = max_y - ny//2

        b_min_x = nx//2-xshift
        b_max_x = 3*nx//2-xshift
        b_min_y = ny//2-yshift
        b_max_y = 3*ny//2-yshift


        for i in range(0,nx):
            for j in range(0,ny):
                residual[i,j]  = residual[i,j]  - clean_gain * max_residual * beam_mv[b_min_x+i,b_min_y+j]
                clean_map[i,j] = clean_map[i,j] + clean_gain * max_residual * clean_beam_mv[b_min_x+i,b_min_y+j]

        ####clean_map[max_x,max_y] = clean_map[max_x,max_y] + clean_gain * max_residual

        if tid==0: print('In Cython: max_residual=',f"{max_residual:e}","at itr=",itr,"for limit",f"{max_residual_limit:e}")

        itr+=1


    return np.asarray(clean_map), np.asarray(residual)

from libc.math cimport sqrt, atan2, cos
def complex_clean(np.ndarray[np.complex128_t, ndim=2] dirty_slice, 
                 np.ndarray[np.complex128_t, ndim=2] beam,
                 np.ndarray[np.float64_t, ndim=2] clean_beam, 
                 float max_residual_limit, 
                 int tid,
                 float clean_gain,
                 int max_itr):

    cdef int itr = 0
    cdef int nx = dirty_slice.shape[0]
    cdef int ny = dirty_slice.shape[1]

    # Allocate memory for residual and clean map
    cdef double complex[:,:] residual = np.copy(dirty_slice)
    cdef double[:,:] clean_map = np.zeros((nx,ny))

    cdef double complex[:,:] beam_mv = np.copy(beam)
    cdef double[:,:] clean_beam_mv = np.copy(clean_beam)

    cdef double max_residual
    cdef int max_x, max_y, xshift, yshift
    cdef int b_min_x, b_max_x, b_min_y, b_max_y
    cdef int i,j

    #cdef int max_iterations = nx*ny
    cdef int max_iterations = max_itr

    cdef double magnitude,phase,real_residual


    while itr < max_iterations:
        # Compute max residual and it's index
        max_residual = 0
        #if tid==0: print('In Cython: Finding max...')
        for i in range(0,nx):
            for j in range(0,ny):
                magnitude = sqrt(residual[i,j].real * residual[i,j].real + residual[i,j].imag * residual[i,j].imag)
                phase = atan2(residual[i,j].imag, residual[i,j].real)
                real_residual = magnitude * cos(phase)  # Projects onto real axis
                if real_residual > max_residual:
                    max_residual = real_residual
                    max_x = i
                    max_y = j

        if abs(max_residual) <= max_residual_limit:
            break

        # Compute shifts
        xshift = max_x - nx//2
        yshift = max_y - ny//2

        b_min_x = nx//2-xshift
        #b_max_x = 3*nx//2-xshift
        b_min_y = ny//2-yshift
        #b_max_y = 3*ny//2-yshift

        
        for i in range(0,nx):
            for j in range(0,ny):
                residual[i,j]  = residual[i,j]  - clean_gain * max_residual * beam_mv[b_min_x+i,b_min_y+j]
                clean_map[i,j] = clean_map[i,j] + clean_gain * max_residual * clean_beam_mv[b_min_x+i,b_min_y+j]

        ####clean_map[max_x,max_y] = clean_map[max_x,max_y] + clean_gain * max_residual

        if tid==0: print('In Cython: max_residual=',f"{max_residual:e}","at itr=",itr,"for limit",f"{max_residual_limit:e}")

        itr+=1

    print("Slice",tid,"cleaned in",itr,"iterations")
    return np.asarray(clean_map), np.asarray(residual)


def masked_clean(np.ndarray[np.float64_t, ndim=2] dirty_slice, 
                 np.ndarray[np.float64_t, ndim=2] beam,
                 np.ndarray[np.float64_t, ndim=2] clean_beam, 
                 np.ndarray[np.int64_t, ndim=2] _clean_mask_indices,
                 int n_masked_pixels, 
                 float max_residual_limit, 
                 int tid,
                 float clean_gain,
                 int max_itr):

    cdef int itr = 0
    cdef int nx = dirty_slice.shape[0]
    cdef int ny = dirty_slice.shape[1]

    # Allocate memory for residual and clean map
    cdef double[:,:] residual = np.copy(dirty_slice)
    cdef long long[:,:] clean_mask_indices = np.copy(_clean_mask_indices)
    cdef double[:,:] clean_map = np.zeros_like(dirty_slice)

    cdef double[:,:] beam_mv = np.copy(beam)
    cdef double[:,:] clean_beam_mv = np.copy(clean_beam)

    cdef double max_residual
    cdef int max_x, max_y, xshift, yshift
    cdef int b_min_x, b_max_x, b_min_y, b_max_y
    cdef int i,j,k

    #cdef int max_iterations = nx*ny
    cdef int max_iterations = max_itr


    while itr < max_iterations:
        # Compute max residual and it's index
        max_residual = 0
        #if tid==0: print('In Cython: Finding max...')

        for k in range(0,n_masked_pixels):
            i = <int>clean_mask_indices[0, k]
            j = <int>clean_mask_indices[1, k]
            if residual[i,j] > max_residual:
                max_residual = residual[i,j]
                max_x = i
                max_y = j

        if max_residual <= max_residual_limit:
            break

        # Compute shifts
        xshift = max_x - nx//2
        yshift = max_y - ny//2

        b_min_x = nx//2-xshift
        b_max_x = 3*nx//2-xshift
        b_min_y = ny//2-yshift
        b_max_y = 3*ny//2-yshift


        for i in range(0,nx):
            for j in range(0,ny):
                residual[i,j]  = residual[i,j]  - clean_gain * max_residual * beam_mv[b_min_x+i,b_min_y+j]
                clean_map[i,j] = clean_map[i,j] + clean_gain * max_residual * clean_beam_mv[b_min_x+i,b_min_y+j]

        ####clean_map[max_x,max_y] = clean_map[max_x,max_y] + clean_gain * max_residual

        if tid==0: print('In Cython: max_residual=',f"{max_residual:e}","at itr=",itr,"for limit",f"{max_residual_limit:e}")

        itr+=1

    print("Slice",tid,"cleaned in",itr,"iterations")

    return np.asarray(clean_map), np.asarray(residual)


def complex_masked_clean(np.ndarray[np.complex128_t, ndim=2] dirty_slice, 
                 np.ndarray[np.complex128_t, ndim=2] beam,
                 np.ndarray[np.float64_t, ndim=2] clean_beam, 
                 np.ndarray[np.int64_t, ndim=2] _clean_mask_indices,
                 int n_masked_pixels, 
                 float max_residual_limit, 
                 int tid,
                 float clean_gain,
                 int max_itr):

    cdef int itr = 0
    cdef int nx = dirty_slice.shape[0]
    cdef int ny = dirty_slice.shape[1]

    # Allocate memory for residual and clean map
    cdef double complex[:,:] residual = np.copy(dirty_slice)
    cdef double[:,:] clean_map = np.zeros((nx,ny))
    cdef long long[:,:] clean_mask_indices = np.copy(_clean_mask_indices)

    cdef double complex[:,:] beam_mv = np.copy(beam)
    cdef double[:,:] clean_beam_mv = np.copy(clean_beam)

    cdef double max_residual
    cdef int max_x, max_y, xshift, yshift
    cdef int b_min_x, b_max_x, b_min_y, b_max_y
    cdef int i,j

    #cdef int max_iterations = nx*ny
    cdef int max_iterations = max_itr

    cdef double magnitude,phase,real_residual


    while itr < max_iterations:
        # Compute max residual and it's index
        max_residual = 0
        #if tid==0: print('In Cython: Finding max...')
        for k in range(0,n_masked_pixels):
            i = <int>clean_mask_indices[0, k]
            j = <int>clean_mask_indices[1, k]
            magnitude = sqrt(residual[i,j].real * residual[i,j].real + residual[i,j].imag * residual[i,j].imag)
            phase = atan2(residual[i,j].imag, residual[i,j].real)
            real_residual = magnitude * cos(phase)  # Projects onto real axis
            if real_residual > max_residual:
                max_residual = real_residual
                max_x = i
                max_y = j


        if abs(max_residual) <= max_residual_limit:
            break

        # Compute shifts
        xshift = max_x - nx//2
        yshift = max_y - ny//2

        b_min_x = nx//2-xshift
        #b_max_x = 3*nx//2-xshift
        b_min_y = ny//2-yshift
        #b_max_y = 3*ny//2-yshift

        
        for i in range(0,nx):
            for j in range(0,ny):
                residual[i,j]  = residual[i,j]  - clean_gain * max_residual * beam_mv[b_min_x+i,b_min_y+j]
                clean_map[i,j] = clean_map[i,j] + clean_gain * max_residual * clean_beam_mv[b_min_x+i,b_min_y+j]

        ####clean_map[max_x,max_y] = clean_map[max_x,max_y] + clean_gain * max_residual

        if tid==0: print('In Cython: max_residual=',f"{max_residual:e}","at itr=",itr,"for limit",f"{max_residual_limit:e}")

        itr+=1

    print("Slice",tid,"cleaned in",itr,"iterations")
    return np.asarray(clean_map), np.asarray(residual)
