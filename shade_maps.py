""" a module for datashader renders of phase diagrams"""
from functools import partial
import datashader as dshader
from datashader.utils import export_image
import datashader.transfer_functions as tf
import pandas as pd
import yt
import trident
import numpy as np
from astropy.table import Table
from get_refine_box import get_refine_box as grb
from consistency import ion_frac_color_key, phase_color_key, metal_color_key



def prep_dataset(fname, trackfile, ion_list=['H I'], region='trackbox'):
    """prepares the dataset for rendering by extracting box or sphere"""
    data_set = yt.load(fname)

    trident.add_ion_fields(data_set, ions=ion_list)
    for ion in ion_list:
        print("Added ion "+ion+" into the dataset.")

    track = Table.read(trackfile, format='ascii')
    track.sort('col1')
    refine_box, refine_box_center, refine_width = \
            grb(data_set, data_set.current_redshift, track)
    print('Refine box corners: ', refine_box)
    print('           center : ', refine_box_center)

    if region == 'trackbox':
        all_data = refine_box
    elif region == 'sphere':
        sph = data_set.sphere(center=refine_box_center, radius=(500, 'kpc'))
        all_data = sph
    else:
        print("your region is invalid!")

    return all_data, refine_box, refine_width


def categorize_by_temp(temp):
    """ define the temp category strings"""
    phase = np.chararray(np.size(temp), 4)
    phase[temp < 9.] = 'hot'
    phase[temp < 6.] = 'warm'
    phase[temp < 5.] = 'cool'
    phase[temp < 4.] = 'cold'
    return phase


def categorize_by_fraction(f_ion):
    """ define the ionization category strings"""
    frac = np.chararray(np.size(f_ion), 4)
    frac[f_ion > -10.] = 'all'
    frac[f_ion > 0.01] = 'low' # yellow
    frac[f_ion > 0.1] = 'med'  # orange
    frac[f_ion > 0.2] = 'high' # red
    return frac

def categorize_by_metallicity(metallicity):
    """ define the metallicity category strings"""
    metal_label = np.chararray(np.size(metal), 5)
    metal_label[metallicity < 10.] = 'high'
    metal_label[metallicity < 0.005] = 'solar'
    metal_label[metallicity < 0.000001] = 'low'
    metal_label[metallicity < 0.0000001] = 'poor'
    return metal

def scale_lvec(lvec):
    lvec[lvec > 0.] = (np.log10(lvec[lvec > 0.]) - 25.) / 8.
    lvec[lvec < 0.] = (-1. * np.log10(-1.*lvec[lvec < 0.]) + 25.) / 8.
    return lvec


def prep_dataframe(all_data, refine_box, refine_width):
    """ add fields to the dataset, create dataframe for rendering"""

    density = all_data['density']
    temperature = all_data['temperature']
    cell_vol = all_data["cell_volume"]
    cell_mass = cell_vol.in_units('kpc**3') * density.in_units('Msun / kpc**3')
    cell_size = np.array(cell_vol)**(1./3.)

    x_particles = all_data['x'].ndarray_view() + \
                cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1.)
    y_particles = all_data['y'].ndarray_view() + \
                cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1.)
    z_particles = all_data['z'].ndarray_view() + \
                cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1.)
    x_particles = (x_particles - refine_box.center[0].ndarray_view()) / (refine_width/2.)
    y_particles = (y_particles - refine_box.center[1].ndarray_view()) / (refine_width/2.)
    z_particles = (z_particles - refine_box.center[2].ndarray_view()) / (refine_width/2.)

    lx = scale_lvec(all_data['specific_angular_momentum_x'])
    ly = scale_lvec(all_data['specific_angular_momentum_y'])
    lz = scale_lvec(all_data['specific_angular_momentum_z'])

    f_o6 = all_data['O_p5_ion_fraction']
    f_c4 = all_data['C_p3_ion_fraction']
    f_si4 = all_data['Si_p3_ion_fraction']
    phase = categorize_by_temp(np.log10(temperature))

    dens = (np.log10(density) + 25.) / 6.
    temp = (np.log10(all_data['temperature']) - 5.0) / 3.
    mass = (np.log10(cell_mass) - 3.0) / 5.

    data_frame = pd.DataFrame({'x':x_particles, 'y':y_particles, \
                               'z':z_particles, 'temp':temp, 'dens':dens, \
                               'mass': mass, \
                               'phase':phase, \
                               'lx':lx, 'ly':ly, 'lz':lz, \
                               'o6frac':categorize_by_fraction(f_o6),\
                               'c4frac':categorize_by_fraction(f_c4),\
                               'si4frac':categorize_by_fraction(f_si4)})
    data_frame.o6frac = data_frame.o6frac.astype('category')
    data_frame.c4frac = data_frame.c4frac.astype('category')
    data_frame.si4frac = data_frame.si4frac.astype('category')
    data_frame.phase = data_frame.phase.astype('category')

    return data_frame


def render_image(frame, field1, field2, count_cat, x_range, y_range, filename):
    """ renders density and temperature 'Phase' with linear aggregation"""

    export = partial(export_image, background='white', export_path="export")
    cvs = dshader.Canvas(plot_width=1080, plot_height=1080,
                         x_range=x_range, y_range=y_range)
    agg = cvs.points(frame, field1, field2, dshader.count_cat(count_cat))

    if 'frac' in count_cat:
        color_key = ion_frac_color_key
    elif 'phase' in count_cat:
        color_key = phase_color_key

    img = tf.shade(agg, color_key=color_key, how='linear')
    export(img, filename)




def drive(fname, trackfile, ion_list=['H I', 'C IV', 'Si IV', 'O VI']):
    """this function drives datashaded phase plots"""

    all_data, refine_box, refine_width = \
        prep_dataset(fname, trackfile, ion_list=ion_list, region='sphere')

    data_frame = prep_dataframe(all_data, refine_box, refine_width)

    phase = ((-1.1, 1.1), (-1.1, 1.1))
    proj = ((-3.1, 3.1), (-3.1, 3.1))

    for ion in ['o6', 'c4', 'si4']:
        render_image(data_frame, 'dens', 'temp', ion+'frac', *phase,
                     'RD0020_phase_'+ion)
        render_image(data_frame, 'x', 'y', ion+'frac', *proj,
                     'RD0020_proj_'+ion)

    render_image(data_frame, 'dens', 'temp', 'phase', *phase, 'RD0020_phase')
    render_image(data_frame, 'x', 'y', 'phase', *proj, 'RD0020_proj')
    render_image(data_frame, 'x', 'mass', 'phase', *proj, 'RD0020_mass')
    render_image(data_frame, 'x', 'lz', 'phase', *phase, 'RD0020_lz')





def cart2pol(x, y):
    return np.sqrt(x**2 + y**2), np.arctan2(y, x)

def pol2cart(rho, phi):
    return rho * np.cos(phi), rho * np.sin(phi)

def rotate_box(fname, trackfile, x1, x2, y1, y2):
    """ not yet functional"""

    all_data, refine_box, refine_width = \
        prep_dataset(fname, trackfile, ion_list=['H I', 'C IV', 'Si IV', 'O VI'],
                     region='sphere')

    data_frame = prep_dataframe(all_data, refine_box, refine_width)

    phase = ((-1.1, 1.1), (-1.1, 1.1))
    proj = ((-3.1, 3.1), (-3.1, 3.1))#take in four fields, x1-->x2, y1-->y2

    # this function rotates from x/y plane to density / y
    for ii in np.arange(100):
        x_center, d_center = 0.5, 0.5
        rr, phi = cart2pol(data_frame['x'] - x_center, data_frame['dens'] - d_center)
        xxxx, yyyy = pol2cart(rr, phi - np.pi / 2. / 100.)
        data_frame.x = xxxx+x_center
        data_frame.dens = yyyy+d_center
        render_image(data_frame, 'x', 'y', 'phase', *phase, 'RD0020_phase'+str(1000+ii))
        print(ii)

    # now start with dens / y and gradually turn y into temp
    for ii in np.arange(100):
        y_center, t_center = 0.5, 0.5
        rr, phi = cart2pol(data_frame['y'] - y_center, data_frame['temp'] - t_center)
        xxxx, yyyy = pol2cart(rr, phi - np.pi / 2. / 100.)
        data_frame.y = xxxx+y_center
        data_frame.temp = yyyy+t_center
        render_image(data_frame, 'x', 'y', 'phase', *phase, 'RD0020_phase'+str(2000+ii))
        print(ii)
