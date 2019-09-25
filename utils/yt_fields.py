
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





