import yt
import numpy as np
from yt.fields.api import ValidateParameter
from yt.fields.field_detector import FieldDetector

###########################################################################
def _los_velocity_mw(field, data):
    """
    Calculate line-of-sight velcity of gas wrt to an observer, taking
    into account the observer's bulk motion (i.e., peculiar velocity), so
    that the los_velocity_mw is w.r.t. the frame of the observer

    How to use:

    from yt.fields.api import ValidateParameter
    from mocky_way_fields import _los_velocity_mw
    ds.add_field(("gas", "los_velocity_mw"),
                 function=_los_velocity_mw, units="km/s",
                 validators=[ValidateParameter("observer_location"),
                             ValidateParameter("observer_bulkvel")])

    History:
    09/30/2019, Yong Zheng added a how to use part.
    10/09/2019, adapted from _los_velocity_bk in mocky_way.derived_fields_mw, Yong Zheng.

    """
    # first some dumb tests to tell yt these are vectors.
    # the location of the observer in the disk
    if isinstance(data, FieldDetector):
        observer_location = np.array([0.5, 0.5, 0.5])  # supposed to be code length
    elif data.has_field_parameter("observer_location"):
        observer_location = data.get_field_parameter("observer_location")
        observer_location = observer_location.value
    else:
        observer_location = np.array([0.5, 0.5, 0.5])

    # motion of observer, taken to be bulk vel of gas with 1 kpc for DD0946
    if isinstance(data, FieldDetector):
        observer_bulkvel = np.array([0, 0, 0]) # suppposed to be km/s
    elif data.has_field_parameter("observer_bulkvel"):
        observer_bulkvel = data.get_field_parameter("observer_bulkvel")
        observer_bulkvel = observer_bulkvel.value
    else:
        observer_bulkvel = np.array([0, 0, 0]) # suppposed to be km/s

    # position and position vector of each cell
    x = data["gas", "x"].in_units("code_length").flatten()
    y = data["gas", "y"].in_units("code_length").flatten()
    z = data["gas", "z"].in_units("code_length").flatten()
    los_x = x.value - observer_location[0] # shape of (N, )
    los_y = y.value - observer_location[1]
    los_z = z.value - observer_location[2]

    los_xyz = np.array([los_x, los_y, los_z]) # shape of (3, N)
    los_r = np.sqrt(los_x**2 + los_y**2 + los_z**2) # shape of (N, )

    # velocity and velocity vector of each cell
    # Note that here observer_bulkvel and all vx, vy, vz are in simulation
    # rest frame. Ideally we need to shift all velocity to the galaxy
    # rest frame (wrt to disk_bulkvel), but since we want los_vx, the equation
    # cancel out each other the rule of galaxy frame. So this is still correct
    # to calculate velocity with respect to obserer's rest frame. 
    vx = data["gas", "velocity_x"].in_units('km/s').flatten() # shape of (N, )
    vy = data["gas", "velocity_y"].in_units('km/s').flatten()
    vz = data["gas", "velocity_z"].in_units('km/s').flatten()

    los_vx = vx.value - observer_bulkvel[0] # shape of (N, )
    los_vy = vy.value - observer_bulkvel[1]
    los_vz = vz.value - observer_bulkvel[2]
    los_vxyz = np.array([los_vx, los_vy, los_vz]) # shape of (3, N)
    los_vr = np.sqrt(los_vx**2 + los_vy**2 + los_vz**2) # shape of (N, )

    #los_vr is not the velocity along los, we need a projection.
    # angle (theta) between los position vector and velocity vector
    # kinda slow, need to figure out a way to speed it up
    cos_theta = np.zeros(los_vr.size)
    for i in np.arange(los_vr.size):
        cos_theta[i] = np.dot(los_xyz[:, i], los_vxyz[:, i])/los_r[i]/los_vr[i]
    los_vr_proj = los_vr*cos_theta

    vx_ori = data['gas', 'velocity_x']
    los_vr_proj = np.reshape(los_vr_proj, vx_ori.shape)

    return yt.YTArray(los_vr_proj, 'km/s')

yt.add_field(("gas", "los_velocity_mw"),
             function=_los_velocity_mw,
             units="km/s",
             validators=[ValidateParameter("observer_location"),
                         ValidateParameter("observer_bulkvel")])
