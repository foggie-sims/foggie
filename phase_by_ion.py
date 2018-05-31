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
#from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno
from get_refine_box import get_refine_box as grb
from consistency import ion_frac_color_key

def prep_dataset(fname, trackfile, ion_list=['H I'], region='trackbox'):
    """prepares the dataset for rendering by extracting box or sphere"""
    data_set = yt.load(fname)

    trident.add_ion_fields(data_set, ions=ion_list)

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


def prep_dataframe(all_data, refine_box, refine_width):
    """ add fields to the dataset, create dataframe for rendering"""

    cell_vol = all_data["cell_volume"]
    cell_mass = all_data["cell_volume"].in_units('kpc**3') * \
                    all_data["density"].in_units('Msun / kpc**3')
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
    dens = np.log10(all_data['density'])
    temp = np.log10(all_data['temperature'])
    mass = np.log10(cell_mass) # log of the cell mass in Msun

    f_o6 = all_data['O_p5_ion_fraction']
    #f_c4 = ad['O_c3_ion_fraction']
    #f_si4 = ad['O_si3_ion_fraction']

    #categorize by ion fraction
    frac = np.chararray(np.size(dens), 4)
    frac[f_o6 > -10.] = 'all'
    frac[f_o6 > 0.01] = 'low' # yellow
    frac[f_o6 > 0.1] = 'med'  # orange
    frac[f_o6 > 0.2] = 'high' # red

    dens = (dens + 25.) / 6.
    temp = (temp - 5.0) / 3.
    mass = (mass - 3.0) / 5.

    data_frame = pd.DataFrame({'x':x_particles, 'y':y_particles, \
                               'z':z_particles, 'temp':temp, 'dens':dens, \
                               'mass': mass, 'frac':frac})
    data_frame.frac = data_frame.frac.astype('category')

    return data_frame


def drive(fname, trackfile, ion_list=['H I', 'C IV', 'Si IV', 'O VI']):
    """this function drives datashaded phase plots"""

    all_data, refine_box, refine_width = \
        prep_dataset(fname, trackfile, ion_list=ion_list, region='sphere')

    data_frame = prep_dataframe(all_data, refine_box, refine_width)

    phase = ((-1.1, 1.1), (-1.0, 1.0))
    galaxy = ((-3.1, 3.1), (-3.1, 3.1))

    def create_image(frame, x_range, y_range):
        """ renders density and temperature 'Phase' with linear aggregation"""
        cvs = dshader.Canvas(plot_width=1080, plot_height=1080,
                             x_range=x_range, y_range=y_range)
        agg = cvs.points(frame, 'dens', 'temp', dshader.count_cat('frac'))
        img = tf.shade(agg, color_key=ion_frac_color_key, how='linear')
        return img

    def create_image2(frame, x_range, y_range):
        """ renders x and y 'Proj' with linear aggregation"""
        cvs = dshader.Canvas(plot_width=1080, plot_height=1080,
                             x_range=x_range, y_range=y_range)
        agg = cvs.points(frame, 'x', 'y', dshader.count_cat('frac'))
        img = tf.shade(agg, color_key=ion_frac_color_key, how='linear')
        return img

    #this part renders the images
    export = partial(export_image, background='white', export_path="export")

    # all cells
    export(create_image(data_frame, *phase), fname[0:6]+"_O6frac_Phase_All")
    export(create_image2(data_frame, *galaxy), fname[0:6]+"_O6frac_Proj_All")

    for part in [b'high', b'med', b'low']:
        frame = data_frame[data_frame['frac'] == part]
        export(create_image(frame, *phase), fname[0:6]+"_O6frac_Phase_"+part)
        export(create_image2(frame, *galaxy), fname[0:6]+"_O6frac_Proj_"+part)

