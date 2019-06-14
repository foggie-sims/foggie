import numpy as np
import yt

def get_refine_box(ds, zsnap, track, **kwargs):
    ## find closest output
    old = kwargs.get("old",False) # modulo not updating before printout -- old runs
    print("inside get_refine_box", zsnap)
    diff = np.abs(track['col1'] - zsnap)

    if old:
        this_loc = track[np.where(diff == np.min(diff[np.where(diff > 1.e-6)]))]
    else:
        this_loc = track[np.where(diff == np.min(diff))]
    print("using this loc:", this_loc)
    x_left = this_loc['col2'][0]
    y_left = this_loc['col3'][0]
    z_left = this_loc['col4'][0]
    x_right = this_loc['col5'][0]
    y_right = this_loc['col6'][0]
    z_right = this_loc['col7'][0]

    refine_box_center = [0.5*(x_left+x_right), 0.5*(y_left+y_right), 0.5*(z_left+z_right)]
    refine_box = ds.r[x_left:x_right, y_left:y_right, z_left:z_right]
    refine_width = np.abs(x_right - x_left)

    return refine_box, refine_box_center, refine_width
