import yt
import numpy as np
from yt.fields.api import ValidateParameter

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
    from yt.fields.field_detector import FieldDetector
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


################# add galactic coordinates #########################

def _l_Galactic_Longitude(field, data):
    """
    Calculate the longitude of gas cell as viewed from a local mock observer
    in a coordinate setting similar to the MW.

    How to use:

    from yt.fields.api import ValidateParameter
    from mocky_way.derived_fields_mw import _l_Galactic_Longitude
    ds.add_field(("gas", "l"), function=_l_Galactic_Longitude,
                 units="degree", take_log=False,
                 validators=[ValidateParameter("observer_location"),
                             ValidateParameter("L_vec"),    # disk ang mom vector
                             ValidateParameter("sun_vec")]) # GC/sun direction

    09/30/2019, YZ. added how to use.
    """

    # first some dumn tests to tell yt that these are vectors ...
    # the location of the observer in the disk
    if isinstance(data, FieldDetector):
        observer_location = np.array([0.5, 0.5, 0.5])
    elif data.has_field_parameter("observer_location"):
        observer_location = data.get_field_parameter("observer_location")
    else:
        observer_location = np.array([0.5, 0.5, 0.5])

    # the disk normal/ang mom vector
    if isinstance(data, FieldDetector):
        L_vec = np.array([0, 0, 1])
    elif data.has_field_parameter("L_vec"):
        L_vec = data.get_field_parameter("L_vec")
    else:
        L_vec = np.array([0, 0, 1])

    # the direction from galactic center to the observer/sun
    if isinstance(data, FieldDetector):
        sun_vec = np.array([0, 1, 0])
    elif data.has_field_parameter("sun_vec"):
        sun_vec = data.get_field_parameter("sun_vec")
    else:
        sun_vec = np.array([0, 1, 0])

    # flatten variables into [3, x] shape before performing vector calculation
    gas_x = (data[("gas", "x")].in_units("code_length").value).flatten()
    gas_y = (data[("gas", "y")].in_units("code_length").value).flatten()
    gas_z = (data[("gas", "z")].in_units("code_length").value).flatten()
    gas_xyz = yt.YTArray([gas_x, gas_y, gas_z])

    gas_pos_vector = gas_xyz - np.reshape(observer_location, (3, 1)) # dimensionless
    vector_length = np.sqrt(np.sum(gas_pos_vector**2, axis=0)).value
    vector_length[np.where(vector_length==0.)[0]] = 1. # observer location
    los_vector = gas_pos_vector/vector_length  # dimensionless

    # get longitude
    dsize = data[("gas", "x")].size
    cos_theta = np.dot(L_vec, los_vector)
    cos_theta = np.reshape(cos_theta, dsize)
    # projection of the LOS vector along the ang mon direction
    proj_vec = np.array([L_vec]*dsize).T*cos_theta   # dimensionless

    # now projection of the LOS vector along the UV/Galactic plane
    radial_vec = los_vector - proj_vec
    radial_vec = radial_vec/np.sqrt((radial_vec**2).sum(axis=0))

    # decide if radial vector is in l=[180, 360] or [0, 180] part
    temp = (np.cross(radial_vec, sun_vec, axisa=0)).T # no unit
    temp = temp/np.sqrt((temp**2).sum(axis=0))        # no unit

    # calculate l and divide into [0, 180] and [180, 360] section
    l = np.zeros(dsize)
    cos_phi = np.dot(sun_vec, radial_vec).value #

    # [0, 180] and [180, 360]
    ind = np.dot(L_vec, temp) == 1. # the dot product is code length
    l[ind] = np.around(180.-np.arccos(cos_phi[ind])/np.pi*180.) % 360.
    noind = np.logical_not(ind)
    l[noind] = np.around(180.+np.arccos(cos_phi[noind])/np.pi*180.) % 360.
    l = yt.YTArray(np.reshape(l, data[("gas", "x")].shape), "degree")

    return l

yt.add_field(("gas", "l"), function=_l_Galactic_Longitude,
                 units="degree", take_log=False,
                 validators=[ValidateParameter("observer_location"),
                             ValidateParameter("L_vec"),    # disk ang mom vector
                             ValidateParameter("sun_vec")]) # GC/sun direction

############################################################################
def _b_Galactic_Latitude(field, data):
    """
    Calculate the latitude of gas cell as viewed from a local mock observer
    in a coordinate setting similar to the MW.

    Note: this field was written for the stand alone mocky way, has not been
          tested when Yong merged mocky_way into foggie. Should do some test
          before using.

    How to use:

    from yt.fields.api import ValidateParameter
    from mocky_way.derived_fields_mw import _b_Galactic_Latitude
    ds.add_field(("gas", "b"), function=_b_Galactic_Latitude,
                 units="degree", take_log=False,
                 validators=[ValidateParameter("observer_location"), # loc of observer
                             ValidateParameter("L_vec")]) # disk ang mom vector

    History:
    Sometime in 2014-2015, first written for Zheng et al. 2015
    09/30/2019, Yong Zheng added a how to use part.
    11/01/2019, Yong Zheng merged into foggie. have not tested the compatibility
                of the code since merging. Need to do some test if used.
    """

    if isinstance(data, FieldDetector):
        L_vec = np.array([0, 0, 1])
    elif data.has_field_parameter("L_vec"):
        L_vec = data.get_field_parameter("L_vec")
    else:
        L_vec = np.array([0, 0, 1])

    # the location of the observer in the disk
    if isinstance(data, FieldDetector):
        observer_location = np.array([0.5, 0.5, 0.5])
    elif data.has_field_parameter("observer_location"):
        observer_location = data.get_field_parameter("observer_location")
    else:
        observer_location = np.array([0.5, 0.5, 0.5])


    # flatten variables into [3, x] shape before performing vector calculation
    gas_x = (data[("gas", "x")].in_units("code_length").value).flatten()
    gas_y = (data[("gas", "y")].in_units("code_length").value).flatten()
    gas_z = (data[("gas", "z")].in_units("code_length").value).flatten()
    gas_xyz = yt.YTArray([gas_x, gas_y, gas_z])

    gas_pos_vector = gas_xyz - np.reshape(observer_location, (3, 1)) # code_length
    vector_length = np.sqrt(np.sum(gas_pos_vector**2, axis=0)).value

    # the point at the observer location
    vector_length[np.where(vector_length==0.)[0]] = 1.
    los_vector = gas_pos_vector/vector_length  # dimensionless

    # theta: angle between disk ang mom vector and the LOS vector
    cos_theta = np.dot(L_vec, los_vector).value   # no unit
    b = yt.YTArray(np.around(90. - np.arccos(cos_theta)/np.pi*180.), "degree")
    b = np.reshape(b, data[("gas", "x")].shape)
    return b

#yt.add_field(("gas", "b"), function=_b_Galactic_Latitude,
#                 units="degree", take_log=False,
#                 validators=[ValidateParameter("observer_location"), # loc of observer
#                             ValidateParameter("L_vec")]) # disk ang mom vector


def _l_Galactic_Longitude(field, data):
    """
    Calculate the longitude of gas cell as viewed from a local mock observer
    in a coordinate setting similar to the MW.

    Note: this field was written for the stand alone mocky way, has not been
          tested when Yong merged mocky_way into foggie. Should do some test
          before using.

    How to use:

    from yt.fields.api import ValidateParameter
    from mocky_way.derived_fields_mw import _l_Galactic_Longitude
    ds.add_field(("gas", "l"), function=_l_Galactic_Longitude,
                 units="degree", take_log=False,
                 validators=[ValidateParameter("observer_location"),
                             ValidateParameter("L_vec"),    # disk ang mom vector
                             ValidateParameter("sun_vec")]) # GC/sun direction

    History:
    Sometime in 2014-2015, first written for Zheng et al. 2015
    09/30/2019, Yong Zheng added a how to use part.
    11/01/2019, Yong Zheng merged into foggie. have not tested the compatibility
                of the code since merging. Need to do some test if used.
    """

    # first some dumn tests to tell yt that these are vectors ...
    # the location of the observer in the disk
    if isinstance(data, FieldDetector):
        observer_location = np.array([0.5, 0.5, 0.5])
    elif data.has_field_parameter("observer_location"):
        observer_location = data.get_field_parameter("observer_location")
    else:
        observer_location = np.array([0.5, 0.5, 0.5])

    # the disk normal/ang mom vector
    if isinstance(data, FieldDetector):
        L_vec = np.array([0, 0, 1])
    elif data.has_field_parameter("L_vec"):
        L_vec = data.get_field_parameter("L_vec")
    else:
        L_vec = np.array([0, 0, 1])

    # the direction from galactic center to the observer/sun
    if isinstance(data, FieldDetector):
        sun_vec = np.array([0, 1, 0])
    elif data.has_field_parameter("sun_vec"):
        sun_vec = data.get_field_parameter("sun_vec")
    else:
        sun_vec = np.array([0, 1, 0])

    # flatten variables into [3, x] shape before performing vector calculation
    gas_x = (data[("gas", "x")].in_units("code_length").value).flatten()
    gas_y = (data[("gas", "y")].in_units("code_length").value).flatten()
    gas_z = (data[("gas", "z")].in_units("code_length").value).flatten()
    gas_xyz = yt.YTArray([gas_x, gas_y, gas_z])

    gas_pos_vector = gas_xyz - np.reshape(observer_location, (3, 1)) # dimensionless
    vector_length = np.sqrt(np.sum(gas_pos_vector**2, axis=0)).value
    vector_length[np.where(vector_length==0.)[0]] = 1. # observer location
    los_vector = gas_pos_vector/vector_length  # dimensionless

    # get longitude
    dsize = data[("gas", "x")].size
    cos_theta = np.dot(L_vec, los_vector)
    cos_theta = np.reshape(cos_theta, dsize)
    # projection of the LOS vector along the ang mon direction
    proj_vec = np.array([L_vec]*dsize).T*cos_theta   # dimensionless

    # now projection of the LOS vector along the UV/Galactic plane
    radial_vec = los_vector - proj_vec
    radial_vec = radial_vec/np.sqrt((radial_vec**2).sum(axis=0))

    # decide if radial vector is in l=[180, 360] or [0, 180] part
    temp = (np.cross(radial_vec, sun_vec, axisa=0)).T # no unit
    temp = temp/np.sqrt((temp**2).sum(axis=0))        # no unit

    # calculate l and divide into [0, 180] and [180, 360] section
    l = np.zeros(dsize)
    cos_phi = np.dot(sun_vec, radial_vec).value #

    # [0, 180] and [180, 360]
    ind = np.dot(L_vec, temp) == 1. # the dot product is code length
    l[ind] = np.around(180.-np.arccos(cos_phi[ind])/np.pi*180.) % 360.
    noind = np.logical_not(ind)
    l[noind] = np.around(180.+np.arccos(cos_phi[noind])/np.pi*180.) % 360.
    l = yt.YTArray(np.reshape(l, data[("gas", "x")].shape), "degree")

    return l

#yt.add_field(("gas", "l"), function=_l_Galactic_Longitude,
#                 units="degree", take_log=False,
#                 validators=[ValidateParameter("observer_location"),
#                             ValidateParameter("L_vec"),    # disk ang mom vector
#                             ValidateParameter("sun_vec")]) # GC/sun direction
