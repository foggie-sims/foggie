
def _static_average_rampressure(field, data):
    bulk_velocity = data.get_field_parameter("bulk_velocity").in_units('km/s')
    velx = data['enzo', 'x-velocity'].to('km/s') - bulk_velocity[0]
    vely = data['enzo', 'y-velocity'].to('km/s') - bulk_velocity[1]
    velz = data['enzo', 'z-velocity'].to('km/s') - bulk_velocity[2]
    vel = np.sqrt(velx**2. + vely**2. + velz**2.)/np.sqrt(3)
    rp = data['density'] * vel**2.
    return np.log10(rp.to('dyne/cm**2').value)

def _static_radial_rampressure(field, data):
    vel = data['gas', 'radial_velocity']
    vel[vel<0] = 0.
    rp = data['density'] * vel**2.
    return np.log10(rp.to('dyne/cm**2').value)


def _radial_rampressure(field, data):

    vel = data['gas', 'circular_velocity'] + data['gas', 'radial_velocity']
    vel[vel<0] = 0.
    rp = data['density'] * vel**2.
    return np.log10(rp.to('dyne/cm**2').value)





### Filter Particles ###
def _stars(pfilter, data):
    """Filter star particles
    To use: yt.add_particle_filter("stars",function=_stars, filtered_type='all',requires=["particle_type"])"""

    return data[(pfilter.filtered_type, "particle_type")] == 2


def _darkmatter(pfilter, data):
    """Filter dark matter particles
    To use: yt.add_particle_filter("darkmatter",function=_darkmatter, filtered_type='all',requires=["particle_type"])"""
    return data[(pfilter.filtered_type, "particle_type")] == 4

def filter_particles():
    """Run the particle filter"""
    yt.add_particle_filter("stars",function=_stars, filtered_type='all',requires=["particle_type"])
    yt.add_particle_filter("darkmatter",function=_darkmatter, filtered_type='all',requires=["particle_type"])
    return
















def _cooling_criteria(field,data):
    """adds cooling criteria field
    to use: yt.add_field(("gas","cooling_criteria"),function=_cooling_criteria,units=None)"""
    return -1*data['cooling_time'] / ((data['dx']/data['sound_speed']).in_units('s'))

