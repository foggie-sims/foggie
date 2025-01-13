import numpy as np
cimport numpy as np

def merge_clumps(np.ndarray[np.int32_t, ndim=3] clump_id_subarray, list clump_merge_map):
    cdef int nSlices = len(clump_merge_map)
    cdef int nPairs, i, m, n
    cdef np.ndarray[np.int32_t, ndim=2] mSlice
    cdef int cid0, cid1

    cdef int x,y,z,u


    for n in range(0,nSlices):
        mSlice = clump_merge_map[n]
        if mSlice.size == 0:
            continue
        nPairs = mSlice.shape[0]
        for i in range(0,nPairs): 
            cid0 = mSlice[i,0]
            cid1 = mSlice[i,1]
            for x in range(0,clump_id_subarray.shape[0]):
                for y in range(0,clump_id_subarray.shape[1]):
                    for z in range(0,clump_id_subarray.shape[2]):
                        if clump_id_subarray[x,y,z] == cid1:
                            clump_id_subarray[x,y,z] = cid0
            #clump_id_subarray[clump_id_subarray==cid1] = cid0
            for m in range(0,nSlices):
                for u in range(0,clump_merge_map[m].shape[0]):
                    if clump_merge_map[m][u,0] == cid1:
                        clump_merge_map[m][u,0] = cid0
                    if clump_merge_map[m][u,1] == cid1:
                        clump_merge_map[m][u,1] = cid0
                #clump_merge_map[m][clump_merge_map[m]==cid1] = cid0
                

    return clump_id_subarray
    

def gen_merge_map(np.ndarray[np.int32_t,ndim=3] boundary_slice, np.ndarray[np.int32_t,ndim=2] merge_map):
    cdef int nx,ny,nz
    cdef int i,j
    cdef int itr=0

    nx=boundary_slice.shape[0]
    ny=boundary_slice.shape[1]
    nz=boundary_slice.shape[2]


    if nx==2:
        for i in range(0,ny):
            for j in range(0,nz):
                val0 = boundary_slice[0,i,j]
                val1 = boundary_slice[1,i,j]
                if val0>0 and val1>0:
                    merge_map[itr,0] = val0
                    merge_map[itr,1] = val1
                    itr+=1
    if ny==2:
        for i in range(0,nx):
            for j in range(0,nz):
                val0 = boundary_slice[i,0,j]
                val1 = boundary_slice[i,1,j]
                if val0>0 and val1>0:
                    merge_map[itr,0] = val0
                    merge_map[itr,1] = val1
                    itr+=1
    if nz==2:
        for i in range(0,nx):
            for j in range(0,ny):
                val0 = boundary_slice[i,j,0]
                val1 = boundary_slice[i,j,1]
                if val0>0 and val1>0:
                    merge_map[itr,0] = val0
                    merge_map[itr,1] = val1
                    itr+=1

    if itr==0: itr=1

    return merge_map[0:itr,:]
