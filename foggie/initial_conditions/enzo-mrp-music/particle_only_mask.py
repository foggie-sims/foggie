import os, sys, shutil
import configparser
import numpy as np
import h5py as h5

#original_run_levelmax = 8
#smooth_edges = True  # smooth the edges of the exact Lagrangian region
#backup = True   # make a backup of the original RefinementMask, just in case!

def particle_only_mask(music_config, smooth_edges=True, backup=True):

    # Error check
    if not os.path.exists(music_config):
        raise RuntimeError("Config file %s not found." % (music_config))
    try:
        from scipy import signal
    except:
        if smooth_edges == False:
            print ("scipy not installed. smooth_edges = True requires scipy. " \
                       "Turning off.")
            smooth_edges = True

    print ("Modifying RefinementMask: reading parameters...")

    # Read necessary parameters from config file
    cp = configparser.ConfigParser()
    music_parms = cp.read(music_config)
    pt_file = cp.get("setup", "region_point_file")
    #pt_shift = map(int, cp.get("setup", "region_point_shift").split(","))
    pt_shift = [int(rpt) for rpt in  cp.get("setup", "region_point_shift").split(",")]
    pt_level = cp.getint("setup", "region_point_levelmin")
    data_dir = cp.get("output", "filename")
    levelmin = cp.getint("setup", "levelmin")
    levelmax = cp.getint("setup", "levelmax")
    finest_level = levelmax - levelmin

    # Read origin of the innermost refine region from Enzo skeleton parameter file
    origin_parameter = "CosmologySimulationGridLeftEdge[%d]" % (finest_level)
    upper_parameter = "CosmologySimulationGridRightEdge[%d]" % (finest_level)
    with open("%s/parameter_file.txt" % (data_dir)) as fh:
        for l in fh:
            if l.startswith(origin_parameter):
                values = l.split("=")[1]
                origin = [float(val) for val in values.split()]
            if l.startswith(upper_parameter):
                values = l.split("=")[1]
                upper = [float(val) for val in values.split()]

    # Get domain shift from the log file
    log_file = "%s_log.txt" % (music_config)
    box_shift = [0]*3
    with open(log_file) as fh:
        for l in fh:
            if l.find("setup/shift_x") >= 0:
                box_shift[0] = int(l.split("=")[1])
            elif l.find("setup/shift_y") >= 0:
                box_shift[1] = int(l.split("=")[1])
            elif l.find("setup/shift_z") >= 0:
                box_shift[2] = int(l.split("=")[1])

    # Load points and convert to relative position inside box in units of
    # the finest cell width
    shift = np.array(box_shift) / 2.0**levelmin - np.array(pt_shift) / 2.0**pt_level
    centered = np.loadtxt(pt_file) + shift
    centered[centered < 0.0] += 1.0
    pts = ((centered - origin) * 2.0**levelmax).astype('int32')

    mask_name = "RefinementMask.%d" % (finest_level)
    mask_fn = "%s/%s" % (data_dir, mask_name)

    if backup and not os.path.exists(mask_fn+".bak"):
        print ("Modifying RefinementMask: Backing up original file...")
        shutil.copyfile(mask_fn, mask_fn+".bak")

    # Calculate mask from particles only
    print ("Modifying RefinementMask: Calculating the new particle mask...")
    h5p = h5.File(mask_fn, "a")
    mask_shape = h5p[mask_name].shape[:0:-1]

    # Deposit particles in a mask with 2*dx which is the finest
    # initial grid of the previous simulation.
    dx = 2
    newmask_shape = np.ceil(np.array(mask_shape) / float(dx)).astype('int32')
    newmask = -np.ones(newmask_shape, dtype='int32')
    H, edges = np.histogramdd(pts/float(dx), bins=newmask_shape, 
                              range=[[0, newmask_shape[0]], [0, newmask_shape[1]], 
                                     [0, newmask_shape[2]]])
    #import pdb; pdb.set_trace()
    # Smooth edges with Gaussian filter
    if smooth_edges:
        print ("Modifying RefinementMask: Smoothing particle mask...")
        from scipy import signal
        window = signal.gaussian(3,0.5)
        fil3d = np.outer(np.outer(window, window), window).reshape((3,3,3))
        H = signal.fftconvolve(H, fil3d, mode='same')
        limit = window[0]
    else:
        limit = 0
    newmask[H > limit] = 0
    if dx > 1:
        h5p[mask_name][0,:,:,:] = np.kron(newmask, np.ones((dx,dx,dx)))\
                                  [:mask_shape[0], :mask_shape[1], :mask_shape[2]].T
    else:
        h5p[mask_name][0,:,:,:] = newmask.T
    h5p.close()
    print ("Modifying RefinementMask: Complete.")


if __name__ == "__main__":
    fn = sys.argv[-1]
    particle_only_mask(fn)
