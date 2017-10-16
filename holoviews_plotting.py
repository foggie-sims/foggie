import holoviews as hv
import holoviews.util
hv.extension('bokeh')
#hv.extension('matplotlib')

from holoviews import Store
import holoviews.plotting.mpl

import numpy as np
import yt
import pandas as pd
import datashader as dshade
import matplotlib.cm as cm
import datashader as dshader
import trident
import cPickle

from radial_data_nozeros import *
from astropy.table import Table
from holoviews.operation.datashader import aggregate, datashade, dynspread, shade
from holoviews.operation import decimate
from holoviews.operation import histogram

track_name = '/Users/dalek/data/Jason/symmetric_box_tracking/complete_track_symmetric_50kpc'

def _cooling_criteria(field,data):
    return data['cooling_time'] / ((data['dx']/data['sound_speed']).in_units('s'))

yt.add_field(("gas","cooling_criteria"),function=_cooling_criteria,units=None)

def sym_refine_box(ds,halo_center):
    dx = ds.arr(20.,'kpc').in_units('code_length').value
    dy = ds.arr(50.,'kpc').in_units('code_length').value
    box_left  = [halo_center[0]-dx, halo_center[1]-dy, halo_center[2]-dx]
    box_right = [halo_center[0]+dx, halo_center[1]+dy, halo_center[2]+dx]
    refine_box = ds.r[box_left[0]:box_right[0],
                      box_left[1]:box_right[1],
                      box_left[2]:box_right[2]]
    return refine_box

def get_halo_center(ds, center_guess):
    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') / ds.get_parameter('CosmologyHubbleConstantNow') * 1000. # in kpc
    ad = ds.sphere(center_guess, (200., 'kpc'))
    x,y,z = np.array(ad["x"]), np.array(ad["y"]), np.array(ad["z"])
    dm_density =  ad['Dark_Matter_Density']
    imax = (np.where(dm_density > 0.9999 * np.max(dm_density)))[0]
    halo_center = [x[imax[0]], y[imax[0]], z[imax[0]]]
    #print 'We have located the main halo at :', halo_center
    return halo_center

def initial_center_guess(ds,track_name):
    track = Table.read(track_name, format='ascii')
    track.sort('col1')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    centerx = np.interp(zsnap, track['col1'], track['col2'])
    centery = np.interp(zsnap, track['col1'], track['col3'])
    centerz = 0.5 * ( np.interp(zsnap, track['col1'], track['col4']) +
                      np.interp(zsnap, track['col1'], track['col7']))
    center = [centerx, centery+20. / 143886., centerz]
    return center

def plot_radial_profiles(filenames,fileout):
    for i in range(len(filenames)):
        print 'i: ',i
        ds = yt.load(filenames[i])
        center_guess = initial_center_guess(ds,track_name)
        halo_center = get_halo_center(ds,center_guess)
        rb = sym_refine_box(ds,halo_center)

        args = filenames[i].split('/')
        sim_label = args[-3]

        dens = np.log10(rb['H_nuclei_density'])
        temp = np.log10(rb['Temperature'])
        Zgas = np.log10(rb['metallicity'])
        x = rb['x']
        y = rb['y']
        z = rb['z']

        halo_center = ds.arr(halo_center,'code_length')
        dist = np.sqrt((halo_center[0]-rb['x'])**2.+(halo_center[1]-rb['y'])**2.+(halo_center[2]-rb['z'])**2.).in_units('kpc')

        df = pd.DataFrame({'temp':temp, 'dens':dens, 'Zgas':Zgas,
                            'x':x,'y':y,'z':z,'dist':dist})

        temp_dist = hv.Scatter(df,kdims=['dist'],vdims=['temp'],label="Temperature "+sim_label)
        dens_dist = hv.Scatter(df,kdims=['dist'],vdims=['dens'],label='Hydrogen Number Density')
        metal_dist = hv.Scatter(df,kdims=['dist'],vdims=['Zgas'],label='Metallicity')

        if i == 0:
            dist_plots = (datashade(temp_dist,cmap=cm.Reds, dynamic=False,x_range=(0,60),y_range=(2,8.4))
		          + datashade(dens_dist,cmap=cm.Blues, dynamic=False,x_range=(0,60),y_range=(-8,2))
		          + datashade(metal_dist,cmap=cm.BuGn, dynamic=False,x_range=(0,60),y_range=(-5,1.4)))
        else:
            dist_plots2 = (datashade(temp_dist,cmap=cm.Reds, dynamic=False,x_range=(0,60),y_range=(2,8.4))
		          + datashade(dens_dist,cmap=cm.Blues, dynamic=False,x_range=(0,60),y_range=(-8,2))
		          + datashade(metal_dist,cmap=cm.BuGn, dynamic=False,x_range=(0,60),y_range=(-5,1.4)))
            dist_plots = dist_plots + dist_plots2

    renderer = hv.renderer('bokeh').instance(fig='html')
    renderer.save(dist_plots.cols(3), fileout)
    return

def plot_phase_diagrams(filenames,fileout):
    for i in range(len(filenames)):
        print 'i: ',i
        ds = yt.load(filenames[i])
        center_guess = initial_center_guess(ds,track_name)
        halo_center = get_halo_center(ds,center_guess)
        rb = sym_refine_box(ds,halo_center)

        args = filenames[i].split('/')
        sim_label = args[-3]

        dens = np.log10(rb['H_nuclei_density'])
        temp = np.log10(rb['Temperature'])

        df = pd.DataFrame({'temp':temp, 'dens':dens})
        phase_scatter = hv.Scatter(df,kdims=['dens'],vdims=['temp'],label=sim_label)

        #phase_data = np.zeros((len(rb['H_nuclei_density']),2))
        #phase_data[:,0] = np.log10(rb['H_nuclei_density'])
        #phase_data[:,1] = np.log10(rb['Temperature'])

        #points = hv.Points(phase_data,kdims=['nH','temp'],label=sim_label)
        hv.opts({'Histogram': {'style': {'alpha':0.3, 'fill_color':'k'}}})
        xhist = (histogram(phase_scatter, bin_range=(-7.5, 1), dimension='dens',normed=True)) #,alpha=0.3, fill_color='k'))
        yhist = (histogram(phase_scatter, bin_range=(3, 8.5), dimension='temp',normed=True)) #,alpha=0.3, fill_color='k'))

        if i == 0:
            phase_plot = (datashade(phase_scatter,cmap=cm.plasma, dynamic=False,x_range=(-7.5,1),y_range=(3,8.5))) << yhist(plot=dict(width=125)) << xhist(plot=dict(height=125))
        else:
            plot2 = (datashade(phase_scatter,cmap=cm.plasma, dynamic=False,x_range=(-7.5,1),y_range=(3,8.5))) << yhist(plot=dict(width=125)) << xhist(plot=dict(height=125))
            phase_plot = phase_plot + plot2

        renderer = hv.renderer('bokeh').instance(fig='html')
        renderer.save(phase_plot, fileout)
    return

#def plot_cooling_criteria(filenames,fileout):
#    return

filenames = ['/Users/dalek/data/Jason/symmetric_box_tracking/nref10f_sym50kpc/DD0165/DD0165',
             '/Users/dalek/data/Jason/symmetric_box_tracking/nref11f_sym50kpc/DD0165/DD0165']
fileout = 'nref10_nf_basic_rp_props'
fileout = 'nref10_phase_test'

#plot_radial_profiles(filenames,fileout)
plot_phase_diagrams(filenames,fileout)
