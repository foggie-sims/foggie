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
from consistency import ion_frac_color_key


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
    phase = np.chararray(np.size(tmep), 4)
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
                               'o6frac':categorize_by_fraction(f_o6),
                               'c4frac':categorize_by_fraction(f_c4),
                               'si4frac':categorize_by_fraction(f_si4)})
    data_frame.o6frac = data_frame.o6frac.astype('category')
    data_frame.c4frac = data_frame.c4frac.astype('category')
    data_frame.si4frac = data_frame.si4frac.astype('category')

    return data_frame


def render_image(frame, field1, field2, count_cat, x_range, y_range, filename):

    """ renders density and temperature 'Phase' with linear aggregation"""
    export = partial(export_image, background='white', export_path="export")
    cvs = dshader.Canvas(plot_width=1080, plot_height=1080,
                         x_range=x_range, y_range=y_range)
    agg = cvs.points(frame, field1, field2, dshader.count_cat(count_cat))
    img = tf.shade(agg, color_key=ion_frac_color_key, how='linear')
    print('filename:', filename)
    export(img, filename)



def drive(fname, trackfile, ion_list=['H I', 'C IV', 'Si IV', 'O VI']):
    """this function drives datashaded phase plots"""

    all_data, refine_box, refine_width = \
        prep_dataset(fname, trackfile, ion_list=ion_list, region='sphere')

    data_frame = prep_dataframe(all_data, refine_box, refine_width)

    phase = ((-1.1, 1.1), (-1.1, 1.1))
    proj = ((-3.1, 3.1), (-3.1, 3.1))

    render_image(data_frame, 'dens', 'temp', 'o6frac', *phase, 'RD0020_o6_phase')
    render_image(data_frame, 'x', 'y', 'o6frac', *proj, 'RD0020_o6_proj')

    render_image(data_frame, 'dens', 'temp', 'c4frac', *phase, 'RD0020_c4_phase')
    render_image(data_frame, 'x', 'y', 'c4frac', *proj, 'RD0020_c4_proj')

    render_image(data_frame, 'dens', 'temp', 'si4frac', *phase, 'RD0020_si4_phase')
    render_image(data_frame, 'x', 'y', 'si4frac', *proj, 'RD0020_si4_proj')

    render_image(data_frame, 'x', 'y', 'phase', *proj, 'RD0020_temp_proj')


def rotate_box():
    """ not yet functional"""
    print("This doesn't work yet!")
