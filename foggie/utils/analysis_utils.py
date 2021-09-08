"""
Filename: analysis_utils.py
This file contains functions that are used across all of Cassi's code, which includes:
-flux_tracking/flux_tracking.py
-radial_quantities/stats_in_shells.py
-radial_quantities/totals_in_shells.py
-plots/plot_fluxes.py
-plots/plot_1Dhistograms.py
-plots/plot_2Dhistograms.py
-plots/plot_outputs.py
-segmenting_regions/find_shape_for_region.py
-segmenting_regions/stack_FRBs.py
-paper_plots/mod_vir_temp/mod_vir_temp_paper_plots.py
Please let Cassi know if you make changes to this file!!!!!
"""

# Import everything as needed
from __future__ import print_function

import numpy as np
import yt
from yt.units import *
from yt import YTArray
import argparse
import os
import glob
import sys
from astropy.table import Table
from astropy.io import ascii
import multiprocessing as multi
import datetime
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.optimize import curve_fit
import shutil
import ast

# These imports are FOGGIE-specific files
from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *

# These imports for datashader plots
import datashader as dshader
from datashader.utils import export_image
import datashader.transfer_functions as tf
import pandas as pd
import matplotlib as mpl


def identify_shape(shape_args, halo, run, units_kpc=False, units_rvir=False):
    '''Returns an organized list of shape arguments from the input shape args.'''

    try:
        shape_args = ast.literal_eval(shape_args)
    except ValueError:
        sys.exit("Something's wrong with your shape arguments. Make sure to include both the outer " + \
        "quotes and the inner quotes around the shape type, like so:\n" + \
        '"[\'sphere\', 0.05, 2., 200.]"')
    if (type(shape_args[0])==str):
        shape_args = [shape_args]
    shapes = []
    for i in range(len(shape_args)):
        if (shape_args[i][0]=='sphere'):
            shapes.append([shape_args[i][0],shape_args[i][1],shape_args[i][2],shape_args[i][3]])
            print('Sphere arguments: inner_radius - %.3f outer_radius - %.3f num_radius - %d' % \
              (shapes[i][1], shapes[i][2], shapes[i][3]))
        elif (shape_args[i][0]=='frustum') or (shape_args[i][0]=='cylinder'):
            if (shape_args[i][1][0]=='-'):
                flip = True
                if (shape_args[i][1][1:]=='minor'):
                    axis = 'disk minor axis'
                else:
                    axis = shape_args[i][1][1]
            elif (shape_args[i][1]=='minor'):
                flip = False
                axis = 'disk minor axis'
            else:
                flip = False
                axis = shape_args[i][1]
            if (shape_args[i][0]=='frustum'):
                shapes.append([shape_args[i][0], shape_args[i][2], shape_args[i][3], shape_args[i][4], axis, flip, shape_args[i][5]])
                if (flip):
                    print('Frustum arguments: axis - flipped %s inner_radius - %.3f outer_radius - %.3f num_steps - %d opening_angle - %d' % \
                      (axis, shapes[i][1], shapes[i][2], shapes[i][3], shapes[i][6]))
                else:
                    print('Frustum arguments: axis - %s inner_radius - %.3f outer_radius - %.3f num_steps - %d opening_angle - %d' % \
                      (str(axis), shapes[i][1], shapes[i][2], shapes[i][3], shapes[i][6]))
            if (shape_args[i][0]=='cylinder'):
                if (shape_args[i][5]!='height') and (shape_args[i][5]!='radius'):
                    sys.exit("I don't understand which way you want to calculate fluxes. Specify 'height' or 'radius'.")
                shapes.append([shape_args[i][0], shape_args[i][2], shape_args[i][3], shape_args[i][6], axis, flip, shape_args[i][4], shape_args[i][5]])
                if (flip):
                    print('Cylinder arguments: axis - flipped %s bottom_edge - %.3f top_edge - %.3f radius - %.3f step_direction - %s num_steps - %d' % \
                      (axis, shapes[i][1], shapes[i][2], shapes[i][6], shapes[i][7], shapes[i][3]))
                else:
                    print('Cylinder arguments: axis - %s bottom_edge - %.3f top_edge - %.3f radius - %.3f step_direction - %s num_steps - %d' % \
                      (str(axis), shapes[i][1], shapes[i][2], shapes[i][6], shapes[i][7], shapes[i][3]))
        elif (shape_args[i][0]=='ellipse'):
            filename = output_dir + 'ellipse_regions_halo_00' + halo + '/' + run + '/' + shape_args[i][1]
            shapes.append([shape_args[i][0], shape_args[i][2], shape_args[i][3], shape_args[i][4], filename])
            print('Ellipse arguments:\nfile name - %s\ninner_radius - %.3f outer_radius %.3f num_steps - %d' % \
              (shapes[i][4], shapes[i][1], shapes[i][2], shapes[i][3]))
        else:
            sys.exit("That shape has not been implemented. Ask Cassi to add it.")

    # Check to make sure if anything is a cylinder, then there are no more shapes
    cyls = 0
    for i in range(len(shapes)):
        if (shapes[i][0]=='cylinder'):
            cyls += 1
    if (cyls > 0):
        if (cyls!=len(shapes)) or (cyls > 1):
            sys.exit("You can't have more than one cylinder or mix cylinders with other shapes! Calculate them separately.")

    # Check to make sure if multiple shapes are specified that they have the same inner_radius, outer_radius, and num_steps
    inner = shapes[0][1]
    outer = shapes[0][2]
    numsteps = shapes[0][3]
    for i in range(len(shapes)):
        if (shapes[i][1]!=inner) or (shapes[i][2]!=outer) or (shapes[i][3]!=numsteps):
            sys.exit('When specifying multiple shapes, you must give the same inner_radius,\n' + \
            'outer_radius, and num_steps for all shapes!')

    if (units_kpc):
        print('Shape arguments are in units of kpc.')
    elif (units_rvir):
        print('Shape arguments are in units of Rvir.')
    else:
        print('Shape arguments are fractions of refine_width.')

    return shapes

def make_output_list(output_args, output_step=1):
    '''Returns a list of outputs from the output arguments.'''

    if (',' in output_args):
        outs = output_args.split(',')
        for i in range(len(outs)):
            if ('-' in outs[i]):
                ind = outs[i].find('-')
                first = outs[i][2:ind]
                last = outs[i][ind+3:]
                output_type = outs[i][:2]
                outs_sub = []
                for j in range(int(first), int(last)+1, output_step):
                    if (j < 10):
                        pad = '000'
                    elif (j >= 10) and (j < 100):
                        pad = '00'
                    elif (j >= 100) and (j < 1000):
                        pad = '0'
                    elif (j >= 1000):
                        pad = ''
                    outs_sub.append(output_type + pad + str(j))
                outs[i] = outs_sub
        flat_outs = []
        for i in outs:
            if (type(i)==list):
                for j in i:
                    flat_outs.append(j)
            else:
                flat_outs.append(i)
        outs = flat_outs
    elif ('-' in output_args):
        ind = output_args.find('-')
        first = output_args[2:ind]
        last = output_args[ind+3:]
        output_type = output_args[:2]
        outs = []
        for i in range(int(first), int(last)+1, output_step):
            if (i < 10):
                pad = '000'
            elif (i >= 10) and (i < 100):
                pad = '00'
            elif (i >= 100) and (i < 1000):
                pad = '0'
            elif (i >= 1000):
                pad = ''
            outs.append(output_type + pad + str(i))
    else: outs = [output_args]

    return outs

def ellipse(center_x, center_y, a, b, rot_angle, x, y):
    '''This function returns True if the point (x, y) is within the ellipse defined by
    (center_x,center_y), the horizontal axis a, the vertical axis b, and the rotation from the horizontal axis rot_angle,
    and returns False otherwise.'''

    A = a**2. * np.sin(rot_angle)**2. + b**2. * np.cos(rot_angle)**2.
    B = 2. * (b**2. - a**2.) * np.sin(rot_angle) * np.cos(rot_angle)
    C = a**2. * np.cos(rot_angle)**2. + b**2. * np.sin(rot_angle)**2.
    D = -2.*A*center_x - B*center_y
    E = -B*center_x - 2.*C*center_y
    F = A*center_x**2. + B*center_x*center_y + C*center_y**2. - a**2.*b**2.

    return A*x**2. + B*x*y + C*y**2. + D*x + E*y + F < 0.

def segment_region(x, y, z, theta, phi, radius, shapes, refine_width_kpc, x_disk=False, y_disk=False, z_disk=False, Rvir=100., units_kpc=False, units_rvir=False):
    '''This function reads in arrays of x, y, z, theta_pos, phi_pos, and radius values and returns a
    boolean list of the same size that is True if a cell is contained within a shape in the list of
    shapes given by 'shapes' and is False otherwise. If disk-relative coordinates are needed for some
    shapes, they can be passed in with the optional x_disk, y_disk, z_disk.'''

    bool_inshape = np.zeros(len(x), dtype=bool)

    for i in range(len(shapes)):
        if (shapes[i][0]=='sphere'):
            if (units_kpc):
                inner_radius = shapes[i][1]
                outer_radius = shapes[i][2]
            elif (units_rvir):
                inner_radius = shapes[i][1]*Rvir
                outer_radius = shapes[i][2]*Rvir
            else:
                inner_radius = shapes[i][1]*refine_width_kpc
                outer_radius = shapes[i][2]*refine_width_kpc
            bool_insphere = (radius > inner_radius) & (radius < outer_radius)
            bool_inshape = bool_inshape | bool_insphere
        elif (shapes[i][0]=='frustum'):
            if (units_kpc):
                inner_radius = shapes[i][1]
                outer_radius = shapes[i][2]
            elif (units_rvir):
                inner_radius = shapes[i][1]*Rvir
                outer_radius = shapes[i][2]*Rvir
            else:
                inner_radius = shapes[i][1]*refine_width_kpc
                outer_radius = shapes[i][2]*refine_width_kpc
            op_angle = shapes[i][6]
            axis = shapes[i][4]
            flip = shapes[i][5]
            if (flip):
                min_theta = np.pi-op_angle*np.pi/180.
                max_theta = np.pi
            else:
                min_theta = 0.
                max_theta = op_angle*np.pi/180.
            if (axis=='x'):
                theta_frus = np.arccos(np.sin(theta)*np.cos(phi))
                phi_frus = np.arctan2(np.cos(theta), np.sin(theta)*np.sin(phi))
            if (axis=='y'):
                theta_frus = np.arccos(np.sin(theta)*np.sin(phi))
                phi_frus = np.arctan2(np.sin(theta)*np.cos(phi), np.cos(theta))
            if (axis=='disk minor axis'):
                theta_frus = np.arccos(z_disk/radius)
                phi_frus = np.arctan2(y_disk, x_disk)
            if (type(axis)==tuple) or (type(axis)==list):
                axis = np.array(axis)
                norm_axis = axis / np.sqrt((axis**2.).sum())
                # Define other unit vectors orthagonal to the angular momentum vector
                np.random.seed(99)
                x_axis = np.random.randn(3)            # take a random vector
                x_axis -= x_axis.dot(norm_axis) * norm_axis       # make it orthogonal to L
                x_axis /= np.linalg.norm(x_axis)            # normalize it
                y_axis = np.cross(norm_axis, x_axis)           # cross product with L
                x_vec = np.array(x_axis)
                y_vec = np.array(y_axis)
                z_vec = np.array(norm_axis)
                # Calculate the rotation matrix for converting from original coordinate system
                # into this new basis
                xhat = np.array([1,0,0])
                yhat = np.array([0,1,0])
                zhat = np.array([0,0,1])
                transArr0 = np.array([[xhat.dot(x_vec), xhat.dot(y_vec), xhat.dot(z_vec)],
                                     [yhat.dot(x_vec), yhat.dot(y_vec), yhat.dot(z_vec)],
                                     [zhat.dot(x_vec), zhat.dot(y_vec), zhat.dot(z_vec)]])
                rotationArr = np.linalg.inv(transArr0)
                x_rot = rotationArr[0][0]*np.sin(theta)*np.cos(phi) + rotationArr[0][1]*np.sin(theta)*np.sin(phi) + rotationArr[0][2]*np.cos(theta)
                y_rot = rotationArr[1][0]*np.sin(theta)*np.cos(phi) + rotationArr[1][1]*np.sin(theta)*np.sin(phi) + rotationArr[1][2]*np.cos(theta)
                z_rot = rotationArr[2][0]*np.sin(theta)*np.cos(phi) + rotationArr[2][1]*np.sin(theta)*np.sin(phi) + rotationArr[2][2]*np.cos(theta)
                theta_frus = np.arccos(z_rot)
                phi_frus = np.arctan2(y_rot, x_rot)
            bool_infrus = (theta_frus >= min_theta) & (theta_frus <= max_theta) & (radius >= inner_radius) & (radius <= outer_radius)
            bool_inshape = bool_inshape | bool_infrus
        elif (shapes[i][0]=='cylinder'):
            if (units_kpc):
                bottom_edge = shapes[i][1]
                top_edge = shapes[i][2]
                cyl_radius = shapes[i][6]
            elif (units_rvir):
                bottom_edge = shapes[i][1]*Rvir
                top_edge = shapes[i][2]*Rvir
                cyl_radius = shapes[i][6]*Rvir
            else:
                bottom_edge = shapes[i][1]*refine_width_kpc
                top_edge = shapes[i][2]*refine_width_kpc
                cyl_radius = shapes[i][6]*refine_width_kpc
            axis = shapes[i][4]
            flip = shapes[i][5]
            if (flip): mult = -1.
            else: mult = 1.
            if (axis=='z'):
                norm_coord = mult*z
                rad_coord = np.sqrt(x**2. + y**2.)
            if (axis=='x'):
                norm_coord = mult*x
                rad_coord = np.sqrt(y**2. + z**2.)
            if (axis=='y'):
                norm_coord = mult*y
                rad_coord = np.sqrt(x**2. + z**2.)
            if (axis=='disk minor axis'):
                norm_coord = mult*z_disk
                rad_coord = np.sqrt(x_disk**2. + y_disk**2.)
            if (type(axis)==tuple) or (type(axis)==list):
                axis = np.array(axis)
                norm_axis = axis / np.sqrt((axis**2.).sum())
                # Define other unit vectors orthagonal to the angular momentum vector
                np.random.seed(99)
                x_axis = np.random.randn(3)            # take a random vector
                x_axis -= x_axis.dot(norm_axis) * norm_axis       # make it orthogonal to L
                x_axis /= np.linalg.norm(x_axis)            # normalize it
                y_axis = np.cross(norm_axis, x_axis)           # cross product with L
                x_vec = np.array(x_axis)
                y_vec = np.array(y_axis)
                z_vec = np.array(norm_axis)
                # Calculate the rotation matrix for converting from original coordinate system
                # into this new basis
                xhat = np.array([1,0,0])
                yhat = np.array([0,1,0])
                zhat = np.array([0,0,1])
                transArr0 = np.array([[xhat.dot(x_vec), xhat.dot(y_vec), xhat.dot(z_vec)],
                                     [yhat.dot(x_vec), yhat.dot(y_vec), yhat.dot(z_vec)],
                                     [zhat.dot(x_vec), zhat.dot(y_vec), zhat.dot(z_vec)]])
                rotationArr = np.linalg.inv(transArr0)
                x_rot = rotationArr[0][0]*x + rotationArr[0][1]*y + rotationArr[0][2]*z
                y_rot = rotationArr[1][0]*x + rotationArr[1][1]*y + rotationArr[1][2]*z
                z_rot = rotationArr[2][0]*x + rotationArr[2][1]*y + rotationArr[2][2]*z
                norm_coord = mult*z_rot
                rad_coord = np.sqrt(x_rot**2. + y_rot**2.)
            bool_incyl = (norm_coord >= bottom_edge) & (norm_coord <= top_edge) & (rad_coord <= cyl_radius)
            bool_inshape = bool_inshape | bool_incyl
        elif (shapes[i][0]=='ellipse'):
            filename = shapes[i][4]
            if (units_kpc):
                inner_radius = shapes[i][1]
                outer_radius = shapes[i][2]
            elif (units_rvir):
                inner_radius = shapes[i][1]*Rvir
                outer_radius = shapes[i][2]*Rvir
            else:
                inner_radius = shapes[i][1]*refine_width_kpc
                outer_radius = shapes[i][2]*refine_width_kpc
            r_inner, r_outer = np.loadtxt(filename, unpack=True, usecols=[0,1])
            ellipse_params = np.loadtxt(filename, usecols=[2,3,4,5,6])
            betw_radii = (radius > inner_radius) & (radius < outer_radius)
            for r in range(len(r_inner)):
                if False in (ellipse_params[r]==[0,0,0,0,0]):
                    in_ellipse = ellipse(ellipse_params[r][0], ellipse_params[r][1], ellipse_params[r][2], \
                      ellipse_params[r][3], ellipse_params[r][4], theta, phi)
                    radbins = (radius > r_inner[r]) & (radius < r_outer[r])
                    in_ellipse = in_ellipse & radbins & betw_radii
                    bool_inshape = bool_inshape | in_ellipse

    if (shapes[0][0]=='cylinder') and (shapes[0][7]=='radius'):
        return bool_inshape, rad_coord
    elif (shapes[0][0]=='cylinder') and (shapes[0][7]=='height'):
        return bool_inshape, norm_coord
    else:
        return bool_inshape

def filter_ds(box, x_data, y_data, weight_data):
    '''This function filters the yt data object passed in as 'box' into inflow and outflow regions, based on temperature
    and radial velocity, and returns the x_data, y_data, and weight_data filtered into these regions.'''

    '''bool_inflow = (box['radial_velocity_corrected'].in_units('km/s').flatten().v < -100.) & \
                  (box['temperature'].flatten().v < 10**4.7)
    bool_outflow = (box['radial_velocity_corrected'].in_units('km/s').flatten().v > 200.) & \
                   (box['temperature'].flatten().v > 10**6.)
    bool_neither = (not bool_inflow) & (not bool_outflow)
    box_inflow = box.include_below('radial_velocity_corrected', -100., 'km/s')
    box_inflow = box_inflow.include_below('temperature', 10**4.7, 'K')
    box_outflow = box.include_above('radial_velocity_corrected', 200., 'km/s')
    box_outflow = box_outflow.include_above('temperature', 10**6., 'K')
    box_neither = box.cut_region("((obj['temperature'] < 10**6) & (obj['temperature']>10**4.7)) " + \
      "| ((obj['radial_velocity_corrected'].in_units('km/s') > -100) & (obj['radial_velocity_corrected'].in_units('km/s') < 200)) " + \
      "| ((obj['temperature']>10**6) & (obj['radial_velocity_corrected'].in_units('km/s') < 200)) " + \
      "| ((obj['temperature']<10**4.7) & (obj['radial_velocity_corrected'].in_units('km/s') > -100)) " + \
      "| ((obj['radial_velocity_corrected'].in_units('km/s') > 200) & (obj['temperature'] < 10**6)) " + \
      "| ((obj['radial_velocity_corrected'].in_units('km/s') < -100) & (obj['temperature'] > 10**4.7))")'''

    bool_inflow = box['metallicity'] < 0.01
    bool_outflow = box['metallicity'] > 1.
    bool_neither = (~bool_inflow) & (~bool_outflow)
    box_inflow = box.include_below('metallicity', 0.01, 'Zsun')
    box_outflow = box.include_above('metallicity', 1., 'Zsun')
    box_neither = box.include_above('metallicity', 0.01, 'Zsun')
    box_neither = box_neither.include_below('metallicity', 1., 'Zsun')

    x_data_inflow = x_data[bool_inflow]
    y_data_inflow = y_data[bool_inflow]
    weight_data_inflow = weight_data[bool_inflow]
    x_data_outflow = x_data[bool_outflow]
    y_data_outflow = y_data[bool_outflow]
    weight_data_outflow = weight_data[bool_outflow]
    x_data_neither = x_data[bool_neither]
    y_data_neither = y_data[bool_neither]
    weight_data_neither = weight_data[bool_neither]

    return box_inflow, box_outflow, box_neither, x_data_inflow, y_data_inflow, weight_data_inflow, \
      x_data_outflow, y_data_outflow, weight_data_outflow, x_data_neither, y_data_neither, weight_data_neither

def filter_FRB(FRB, save_file=False, save_dir='', file_name='', save_suffix=''):
    '''This function filters the FRB passed in into inflow, outflow, and neither regions, based on temperature
    and radial velocity, and returns the inflow FRB, outflow FRB, and neither FRB. The fields 'radial_velocity_corrected'
    and 'temperature' must exist within the FRB. If save_FRB=True, it also saves these filtered FRBs to file,
    in the directory 'save_dir' with the name 'file_name'.'''

    #bool_inflow = (FRB['radial_velocity_corrected'] < -100.) & (FRB['temperature'] < 10**4.7)
    #bool_outflow = (FRB['radial_velocity_corrected'] > 200.) & (FRB['temperature'] > 10**6.)
    #bool_inflow = FRB['radial_velocity_corrected'] < -100.
    #bool_outflow = FRB['radial_velocity_corrected'] > 200.
    bool_inflow = FRB['metallicity'] < 0.01
    bool_outflow = FRB['metallicity'] > 1.
    bool_neither = (~bool_inflow) & (~bool_outflow)

    FRB_inflow = Table()
    FRB_outflow = Table()
    FRB_neither = Table()
    for j in range(len(FRB.columns)):
        FRB_inflow.add_column(FRB.columns[j][bool_inflow], name=FRB.columns[j].name)
        FRB_inflow[FRB.columns[j].name].unit = FRB[FRB.columns[j].name].unit
        FRB_outflow.add_column(FRB.columns[j][bool_outflow], name=FRB.columns[j].name)
        FRB_outflow[FRB.columns[j].name].unit = FRB[FRB.columns[j].name].unit
        FRB_neither.add_column(FRB.columns[j][bool_neither], name=FRB.columns[j].name)
        FRB_neither[FRB.columns[j].name].unit = FRB[FRB.columns[j].name].unit
    if (save_file):
        print('Writing inflow, outflow, and neither FRBs to file')
        FRB_inflow.write(save_dir + file_name + '_inflow_rv-only.hdf5', path='all_data', serialize_meta=True, overwrite=True)
        FRB_outflow.write(save_dir + file_name + '_outflow_rv-only.hdf5', path='all_data', serialize_meta=True, overwrite=True)
        FRB_neither.write(save_dir + file_name + '_neither_rv-only.hdf5', path='all_data', serialize_meta=True, overwrite=True)

    return FRB_inflow, FRB_outflow, FRB_neither

def create_foggie_cmap(cmin, cmax, cfunc, color_key, log=False):
    '''This function makes the image for the little colorbar that can be put on the datashader main
    image. It takes the minimum and maximum values of the field that is being turned into a colorbar,
    'cmin' and 'cmax', the name of the color-categorization function (in consistency.py), 'cfunc',
    and the name of the color key (also in consistency.py), 'color_key', and returns the color bar.'''

    x = np.random.rand(100000)
    y = np.random.rand(100000)
    if (log): rand = np.random.rand(100000) * (np.log10(cmax)-np.log10(cmin)) + np.log10(cmin)
    else: rand = np.random.rand(100000) * (cmax-cmin) + cmin

    df = pd.DataFrame({})
    df['x'] = x
    df['y'] = y
    df['rand'] = rand
    n_labels = np.size(list(color_key))
    sightline_length = np.max(df['x']) - np.min(df['x'])
    value = np.max(df['x'])

    cat = cfunc(rand)
    for index in np.flip(np.arange(n_labels), 0):
        cat[x > value - sightline_length*(1.*index+1)/n_labels] = \
          list(color_key)[index]
    df['cat'] = cat
    df.cat = df.cat.astype('category')

    cvs = dshader.Canvas(plot_width=750, plot_height=100,
                         x_range=(np.min(df['x']),
                                  np.max(df['x'])),
                         y_range=(np.min(df['y']),
                                  np.max(df['y'])))
    agg = cvs.points(df, 'x', 'y', dshader.count_cat('cat'))
    cmap = tf.spread(tf.shade(agg, color_key=color_key), px=2, shape='square')
    return cmap
