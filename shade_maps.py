
""" a module for datashader renders of phase diagrams"""
from functools import partial
import datashader as dshader
from datashader.utils import export_image
import datashader.transfer_functions as tf
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
mpl.rcParams['font.family'] = 'stixgeneral'

import yt
import trident
import numpy as np
from astropy.table import Table

import os
os.sys.path.insert(0, os.environ['FOGGIE_REPO'])
import cmap_utils as cmaps

from get_refine_box import get_refine_box as grb
from get_halo_center import get_halo_center
from consistency import ion_frac_color_key, new_phase_color_key, \
        new_metals_color_key, axes_label_dict, logfields, new_categorize_by_temp, \
        new_categorize_by_metals, categorize_by_fraction

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
    print('            center: ', refine_box_center)

    if region == 'trackbox':
        all_data = refine_box
    elif region == 'sphere':
        sph = data_set.sphere(center=refine_box_center, radius=(500, 'kpc'))
        all_data = sph
    else:
        print("your region is invalid!")

    halo_center, halo_vcenter = get_halo_center(data_set, refine_box_center, \
                                        units = 'physical')

    return all_data, refine_box, refine_width, halo_center, halo_vcenter

def scale_lvec(lvec):
    lvec[lvec > 0.] = (np.log10(lvec[lvec > 0.]) - 25.) / 8.
    lvec[lvec < 0.] = (-1. * np.log10(-1.*lvec[lvec < 0.]) + 25.) / 8.
    return lvec

def prep_dataframe(all_data, refine_box, refine_width, field1, field2, \
                        halo_center, halo_vcenter):
    """ add fields to the dataset, create dataframe for rendering
        The enzo fields x, y, z, temperature, density, cell_vol, cell_mass,
        and metallicity will always be included, others will be included
        if they are requested as fields. """

    # obtain fields that we'll use no matter what the input fields.
    density = all_data['density']
    cell_mass = all_data['cell_volume'].in_units('kpc**3') * density.in_units('Msun / kpc**3')
    cell_size = np.array(all_data["cell_volume"].in_units('kpc**3'))**(1./3.)

    x = (all_data['x'].in_units('kpc')).ndarray_view()
    y = (all_data['y'].in_units('kpc')).ndarray_view()
    z = (all_data['z'].in_units('kpc')).ndarray_view()

    x = x + cell_size * (np.random.rand(np.size(cell_size)) * 2. - 1.)
    y = y + cell_size * (np.random.rand(np.size(cell_size)) * 2. - 1.)
    z = z + cell_size * (np.random.rand(np.size(cell_size)) * 2. - 1.)
    halo_center = np.array(halo_center) - np.array([np.min(x), np.min(y), np.min(z)])
    x = x - np.min(x)
    y = y - np.min(y)
    z = z - np.min(z)
    radius = ((x-halo_center[0])**2 + (y-halo_center[1])**2 + (z-halo_center[2])**2 )**0.5

    density = np.log10(density)
    temperature = np.log10(all_data['temperature'])
    mass = np.log10(cell_mass)
    phase = new_categorize_by_temp(temperature)
    metal = new_categorize_by_metals(all_data['metallicity'])
    frac = categorize_by_fraction(all_data['O_p5_ion_fraction'], all_data['temperature'])

    # build data_frame with mandatory fields
    data_frame = pd.DataFrame({'x':x, 'y':y, 'z':z, 'temperature':temperature, \
                               'density':density, 'cell_mass': mass, 'radius':radius, \
                               'phase':phase, 'metal':metal, 'frac':frac})
    data_frame.phase = data_frame.phase.astype('category')
    data_frame.metal = data_frame.metal.astype('category')
    data_frame.frac  = data_frame.frac.astype('category')

    relative_velocity = ( (all_data['x-velocity'].in_units('km/s')-halo_vcenter[0])**2 \
                        + (all_data['y-velocity'].in_units('km/s')-halo_vcenter[1])**2 \
                        + (all_data['z-velocity'].in_units('km/s')-halo_vcenter[2])**2 )**0.5
    data_frame['relative_velocity'] = relative_velocity

    print("you have requested fields ", field1, field2)

    if field1 not in data_frame.columns:    #  add those two fields
        print("Did not find field 1 = "+field1+" in the dataframe, will add it.")
        if field1 in logfields:
            print("Field 1, "+field1+" is a log field.")
            data_frame[field1] = np.log10(all_data[field1])
        else:
            data_frame[field1] = all_data[field1]
            if ('vel' in field1): data_frame[field1] = all_data[field1].in_units('km/s')
    if field2 not in data_frame.columns:
        print("Did not find field 2 = "+field2+" in the dataframe, will add it.")
        if field2 in logfields:
            print("Field 2, "+field2+" is a log field.")
            data_frame[field2] = np.log10(all_data[field2])
        else:
            data_frame[field2] = all_data[field2]
            if ('vel' in field2): data_frame[field2] = all_data[field2].in_units('km/s')

    return data_frame


def wrap_axes(filename, field1, field2, colorcode, ranges):
    """intended to be run after render_image, take the image and wraps it in
        axes using matplotlib and so offering full customization."""

    img = mpimg.imread(filename+'.png')
    print('IMG', np.shape(img[:,:,0:3]))
    fig = plt.figure(figsize=(8,8))

    ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])
    ax.imshow(np.flip(img[:,:,0:3],0), alpha=1.)

    xstep = 1
    x_max = ranges[0][1]
    x_min = ranges[0][0]
    if (x_max > 10.): xstep = 10
    if (x_max > 100.): xstep = 100
    xtext = ax.set_xlabel(axes_label_dict[field1], fontsize=20)
    ax.set_xticks(np.arange((x_max - x_min) + 1., step=xstep) * 1000. / (x_max - x_min))
    ax.set_xticklabels([ str(int(s)) for s in \
        np.arange((x_max - x_min) + 1., step=xstep) +  x_min ], fontsize=20)

    ystep = 1
    y_max = ranges[1][1]
    y_min = ranges[1][0]
    if (y_max > 10.): ystep = 10
    if (y_max > 100.): ystep = 100
    ytext = ax.set_ylabel(axes_label_dict[field2], fontsize=20)
    ax.set_yticks(np.arange((y_max - y_min) + 1., step=ystep) * 1000. / (y_max - y_min))
    ax.set_yticklabels([ str(int(s)) for s in \
        np.arange((y_max - y_min) + 1., step=ystep) + y_min], fontsize=20)

    ax2 = fig.add_axes([0.7, 0.91, 0.25, 0.06])
    phase_cmap, metal_cmap = cmaps.create_foggie_cmap()

    if 'phase' in colorcode:
        ax2.imshow(np.flip(phase_cmap.to_pil(), 1))
        ax2.set_xticks([100,350,600])
        ax2.set_xticklabels(['4','5','6',' '])
        ax2.set_xlabel('log T [K]')
    elif 'metal' in colorcode:
        ax2.imshow(np.flip(metal_cmap.to_pil(), 1))
        ax2.set_xticks([0, 400, 800])
        ax2.set_xticklabels(['-4', '-2', '0'])
        ax2.set_xlabel('log Z')

    ax2.spines["top"].set_color('white')
    ax2.spines["bottom"].set_color('white')
    ax2.spines["left"].set_color('white')
    ax2.spines["right"].set_color('white')
    ax2.set_ylim(60, 180)
    ax2.set_xlim(-10, 800)
    ax2.set_yticklabels([])
    ax2.set_yticks([])

    plt.savefig(filename)




def render_image(frame, field1, field2, count_cat, x_range, y_range, filename):
    """ renders density and temperature 'Phase' with linear aggregation"""

    export = partial(export_image, background='white', export_path="./")
    cvs = dshader.Canvas(plot_width=1000, plot_height=1000,
                         x_range=x_range, y_range=y_range)
    print("count_cat: ", count_cat)
    agg = cvs.points(frame, field1, field2, dshader.count_cat(count_cat))

    if 'frac' in count_cat:
        color_key = ion_frac_color_key
    elif 'phase' in count_cat:
        color_key = new_phase_color_key
    elif 'metal' in count_cat:
        color_key = new_metals_color_key

    img = tf.shade(agg, color_key=color_key, how='log')
    export(img, filename)
    return img


def drive(fname, trackfile, field1, field2, ion_list=['H I', 'C IV', 'Si IV', 'O VI']):
    """this function drives datashaded phase plots"""

    all_data, refine_box, refine_width = \
        prep_dataset(fname, trackfile, ion_list=ion_list, region='sphere')

    data_frame = prep_dataframe(all_data, refine_box, refine_width, field1, field2)

    for ion in ['o6', 'c4', 'si4']:
        render_image(data_frame, 'density', 'temperature', ion+'frac',
                     (-31, -20), (2,8), 'RD0020_phase_'+ion)
        render_image(data_frame, 'x', 'y', ion+'frac',
                     (-3,3), (-3,3), 'RD0020_proj_'+ion)

def simple_plot(fname, trackfile, field1, field2, colorcode, ranges, outfile):
    """This function makes a simple plot with two fields plotted against
        one another. The color coding is given by variable 'colorcode'
        which can be phase, metal, or an ionization fraction"""

    all_data, refine_box, refine_width, halo_center, halo_vcenter = \
        prep_dataset(fname, trackfile,
            ion_list=['H I', 'C IV', 'Si IV', 'O VI'], region='trackbox')

    prof = yt.Profile1D(all_data, "pressure", 100, 1e-16, 1e-9, True, weight_field="cell_mass")

    data_frame = prep_dataframe(all_data, refine_box, refine_width, \
                                field1, field2, halo_center, halo_vcenter)

    image = render_image(data_frame, field1, field2, colorcode, *ranges, outfile)
    wrap_axes(outfile, field1, field2, colorcode, ranges)
    return data_frame









# code below here is in storage for later use in building rorating movies.

def cart2pol(x, y):
    return np.sqrt(x**2 + y**2), np.arctan2(y, x)

def pol2cart(rho, phi):
    return rho * np.cos(phi), rho * np.sin(phi)

def rotate_box(fname, trackfile, x1, y1, x2, y2):
    """ not yet functional"""

    print("NEED TO DO VARIABLE NORMALIZATION HERE SINCE IT IS NOT DONE ANYWEHRE ELSE NOW")
    all_data, refine_box, refine_width = \
        prep_dataset(fname, trackfile, ion_list=['H I', 'C IV', 'Si IV', 'O VI'],
                     region='sphere')

    data_frame = prep_dataframe(all_data, refine_box, refine_width)

    phase = ((-1.1, 1.1), (-1.1, 1.1))
    proj = ((-3.1, 3.1), (-3.1, 3.1))

    # this function rotates from x/y plane to density / y
    for ii in np.arange(100):
        x_center, d_center = 0.5, 0.5
        rr, phi = cart2pol(data_frame['x'] - x_center, data_frame['dens'] - d_center)
        xxxx, yyyy = pol2cart(rr, phi - np.pi / 2. / 100.)
        data_frame.x = xxxx+x_center
        data_frame.dens = yyyy+d_center
        render_image(data_frame, 'x', 'y', 'phase', *phase, 'RD0020_phase'+str(1000+ii))
        print(ii)

    # now start with dens / y and gradually turn y into temperature
    for ii in np.arange(100):
        y_center, t_center = 0.5, 0.5
        rr, phi = cart2pol(data_frame['y'] - y_center, data_frame['temperature'] - t_center)
        xxxx, yyyy = pol2cart(rr, phi - np.pi / 2. / 100.)
        data_frame.y = xxxx+y_center
        data_frame.temperature = yyyy+t_center
        render_image(data_frame, 'x', 'y', 'phase', *phase, 'RD0020_phase'+str(2000+ii))
        print(ii)
