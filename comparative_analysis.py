#import matplotlib
#matplotlib.use('Agg')

import holoviews as hv
import holoviews.util 
#hv.extension('bokeh')


import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from yt.analysis_modules.star_analysis.api import StarFormationRate
from yt.data_objects.particle_filters import add_particle_filter
import os
#from radial_data_nozeros import *
import trident
import cPickle
from astropy.table import Table
import builtins
import pandas as pd
import matplotlib.cm as cm

import datashader as dshader
from holoviews.operation.datashader import aggregate, datashade, dynspread, shade
from holoviews.operation import decimate
from holoviews.operation import histogram

## Many of these functions were envisioned to enable the comparison
## of different simulation outputs. Using too many simulations at the same
## time will result in plots that are too busy.


def _cooling_criteria(field,data):
    """
    Calculates criteria used in enzo CellFlaggingMethod = 7
    """
    return -1*data['cooling_time'] / ((data['dx']/data['sound_speed']).in_units('s'))

# Add field to any ds that gets loaded
yt.add_field(("gas","cooling_criteria"),function=_cooling_criteria,units=None)


def fdbk_refine_box(ds,halo_center):
    """
    Find the refinement box in the asymmetric feedback runs

    Parameters
    ----------

    :ds: enzo dataoutput
        Enzo data output to be analyzed

    :halo_center: array
        Location of halo center of interest in code units

    Returns
    -------
    enzo region encompassing the must refine region
    """
    box_center = np.copy(halo_center)
    box_center[1] = box_center[1]+ds.arr(60.,'kpc').in_units('code_length').value

    dx = ds.arr(40.,'kpc').in_units('code_length').value
    dy = ds.arr(80.,'kpc').in_units('code_length').value
    box_left  = [box_center[0]-dx, box_center[1]-dy, box_center[2]-dx]
    box_right = [box_center[0]+dx, box_center[1]+dy, box_center[2]+dx]

    refine_box = ds.r[box_left[0]:box_right[0], box_left[1]:box_right[1], box_left[2]:box_right[2]]
    return refine_box

def sym_refine_box(ds,halo_center):
    """
    Find the refinement box in the symmetric resolution runs

    Parameters
    ----------

    :ds: enzo dataoutput
        Enzo data output to be analyzed

    :halo_center: array
        Location of halo center of interest in code units

    Returns
    -------
    enzo region encompassing the must refine region
    """

    dx = ds.arr(20.,'kpccm/h').in_units('code_length').value
    dy = ds.arr(50.,'kpccm/h').in_units('code_length').value
    box_left  = [halo_center[0]-dx, halo_center[1]-dy, halo_center[2]-dx]
    box_right = [halo_center[0]+dx, halo_center[1]+dy, halo_center[2]+dx]
    refine_box = ds.r[box_left[0]:box_right[0],
                      box_left[1]:box_right[1],
                      box_left[2]:box_right[2]]
    return refine_box


def formed_star(pfilter, data):
    """
    yt filter for identifying star particles in a data region
    """
    filter = data["all", "creation_time"] > 0
    return filter

## kwargs to set styles for most lines in upcoming plots
plot_kwargs = {
    'nref10_track_2'         : {'ls':'-','color':'#e7298a'}, #,'marker':'o','markersize':'0.25'},
    'nref10_track_lowfdbk_1' : {'ls':'-','color':'#d95f02'}, #,'marker':'s','markersize':'0.25'},
    'nref10_track_lowfdbk_2' : {'ls':'-','color':'#1b9e77'}, #,'marker':'*','markersize':'0.25'},
    'nref10_track_lowfdbk_3' : {'ls':'-','color':'#7570b3'}, #,'marker':'^','markersize':'0.25'},
    'nref10_track_lowfdbk_4' : {'ls':'-','color':'#e6ab02'}, #,'marker':'p','markersize':'0.25'},
    'nref10_z1_0.5_natural'           : {'ls':'--','color':'#e7298a'}, #,'marker':'o','markersize':'0.25'},
    'nref10_z1_0.5_natural_lowfdbk_1' : {'ls':'--','color':'#d95f02'}, #,'marker':'s','markersize':'0.25'},
    'nref10_z1_0.5_natural_lowfdbk_2' : {'ls':'--','color':'#1b9e77'}, #,'marker':'*','markersize':'0.25'},
    'nref10_z1_0.5_natural_lowfdbk_3' : {'ls':'--','color':'#7570b3'}, #,'marker':'^','markersize':'0.25'},
    'nref10_z1_0.5_natural_lowfdbk_4' : {'ls':'--','color':'#e6ab02'}, #,'marker':'p','markersize':'0.25'}
    'nref11f_sym50kpc' : {'ls':'-','color':'#d95f02'},
    'nref10f_sym50kpc' : {'ls':'-','color':'#e7298a'},
    'nref10' : {'ls':'--','color':'#e7298a'},
    'nref11f_50kpc' : {'ls':'-','color':'#d95f02'},
    'nref10f_50kpc' : {'ls':'-','color':'#e7298a'},
    'nref11' : {'ls':'--','color':'#d95f02'}
}

### utility functions ###
def get_halo_center(ds, center_guess):
    """
    Returns halo center bby finding center of mass of densest dark matter points

    Parameters
    ----------
    ds : enzo data source
        Datasource of interest

    center_guess : array
        Initial center guess for the calculation in code units

    Returns
    -------
    halo_center: array of final, calculated halo center
    """
    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') / ds.get_parameter('CosmologyHubbleConstantNow') * 1000. # in kpc
    ad = ds.sphere(center_guess, (200., 'kpc'))
    x,y,z = np.array(ad["x"]), np.array(ad["y"]), np.array(ad["z"])
    dm_density =  ad['Dark_Matter_Density']
    imax = (np.where(dm_density > 0.9999 * np.max(dm_density)))[0]
    halo_center = [x[imax[0]], y[imax[0]], z[imax[0]]]
    #print 'We have located the main halo at :', halo_center
    return halo_center

def initial_center_guess(ds,track_name):
    """
    Calculation initial center guess using track file generated for enzo

    Parameters
    ----------
    ds : enzo data source
        Datasource of interest

    Returns
    -------
    <returned value> : array containing center guess in code units
    """
    track = Table.read(builtins.track_name, format='ascii')
    track.sort('col1')
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    centerx = np.interp(zsnap, track['col1'], track['col2'])
    centery = np.interp(zsnap, track['col1'], track['col3'])
    centerz = 0.5 * ( np.interp(zsnap, track['col1'], track['col4']) +
                      np.interp(zsnap, track['col1'], track['col7']))
    center = [centerx, centery+20. / 143886., centerz]
    return center

def diskVectors(ds, center):
    """
    Finds the angular momentum vector and the perpendicular unit vector
    for a disk in ds with the given center.

    Parameters
    ----------

    ds : enzo data source
        Datasource of interest

    Returns
    -------
    <returned value> : tuple
        Tuple containing the angular momentum vector and perpendicular unit vector
    """
    sphere = ds.sphere(center,(5.,'kpc'))
    angular_momentum = sphere.quantities['AngularMomentumVector']()
    x = np.cross(angular_momentum,[1.0,0.0,0.0])
    x /= np.linalg.norm(x)
    return (angular_momentum,x)

def compute_cell_distance(halo_center,x,y,z):
    """
    Compute distance from point halo_center from array of x/y/z values
    Units must be the same but otherwise unspecified.

    Parameters
    ----------

    halo_center: array
        x/y/z of center point of interest

    x/y/z : point/array
        Either single values or array of x/y/z positions to calculation distance

    Returns
    -------
    distance: array
        Distance between halo_center and x/y/z positions
    """
    return np.sqrt((halo_center[0]-x)**2.0+
                    (halo_center[1]-y)**2.0+
                    (halo_center[2]-z)**2.0)

def compute_disk_masses(basenames,RDnums,prefix):
    """
    Compute how the disk mass evolves with time.

    Parameters
    ----------
    basename : string array
        Array of base directory names for the simulations

    RDnums : float array
        Array containing the numbers of directory outputs to be included

    prefix : string
        Prefix of the directories - i.e. "RD" or "DD"

    Returns
    -------
    gas_masses : array
        Array of disk gas masses for each time step in each simulations: [RD,sim]

    stellar_masses : array
        Array of disk stellar masses for each time step in each sim: [RD,sim]

    timesteps : array
        Array of time step redshift for which the masses are calculated.
    """

    timesteps = np.zeros(len(RDnums))
    gas_masses = np.zeros((len(RDnumss),len(filenames)))
    stellar_masses = np.zeros((len(RDnums),len(filenames)))

    add_particle_filter("formed_star", function=formed_star, filtered_type='all',
                        requires=["creation_time"])

    for j in range(len(filenames)):
        for i in range(len(RDnums)):
            if RDnums[i] < 100:
                zero = '00'
            else:
                zero = '0'
            ds = yt.load(basename+('/'+prefix+zero+str(RDnums[i]))*2)
            ds.add_particle_filter('formed_star')
            center_guess = initial_center_guess(ds,builtins.track_name)
            halo_center = get_halo_center(ds,center_guess)
            center_guess = halo_center
            (Lx,x) = diskVectors(ds,halo_center)
            disk = ds.disk(halo_center,Lx,(40.,'kpc'),(20.,'kpc'))
            ## let's look for cold/dense gas
            ## ISM conditions from review paper:
            ## n > 0.1, T < 4
            idx = np.where((disk['H_p0_number_density'] > 0.1) &
                           (disk['Temperature'] < 1e4))[0]
            gas_masses[i,j] = np.sum(disk['cell_mass'][idx].in_units('Msun'))
            stellar_masses[i,j] = np.sum(disk['formed_star', 'particle_mass'].in_units('Msun'))
            timesteps[i] = ds.current_redshift
    #data = {'gas_masses':gas_masses,'stellar_masses':stellar_masses,'timesteps':timesteps}
    #cPickle.dump(data,open('disk_mass_evol.cpkl','wb'),protocol=-1)
    return gas_masses,stellar_masses,timesteps

def make_frbs(filename,fields,ions,asym=False):
    """
    Compute column densities for the given ions for the must refine regions
    in either the asymmetric or symmetric runs.

    Parameters
    ----------

    filename : string
        Filename of enzo output to load as ds

    fields : string array
        Array of strings of fields for the column density projections
        e.g. "O_p5_number_density" for OVI

    ions : string array
        Array of ion names to be passed to trident for the new coldens fields
        e.g. "O VI"

    asym : boolean
        Does the shape of the frb match the asymmetric or symmetric runs.
        Default : false for symmetric

    Returns
    -------
    Nothing but does print save the frbs as cpkl files to be read in later.

    """

    ds = yt.load(filename)
    args = filename.split('/')
    trident.add_ion_fields(ds,ions=ions)
    center_guess = initial_center_guess(ds,builtins.track_name)
    halo_center = get_halo_center(ds,center_guess)

    if asym == True:
        #print 'in fdbk'
        refine_box = fdbk_refine_box(ds,halo_center)
        width = [(160,'kpc'),(80.,'kpc')]
        resolution = (320,160)
        for field in fields:
            fileout = args[-3]+'_'+args[-2]+'_x_'+field+'.cpkl'
            obj = ds.proj(field,axis,data_source=refine_box)
            frb = obj.to_frb(width,resolution,center=box_center)
            cPickle.dump(frb[field],open(fileout,'wb'),protocol=-1)
    else:
        #print 'in else'
        refine_box = sym_refine_box(ds,halo_center)
        width = [(40,'kpc'),(100.,'kpc')]
        resolution = (80,200)

        for field in fields:
            fileout = args[-3]+'_'+args[-2]+'_x_'+field+'.cpkl'
            obj = ds.proj(field,axis,data_source=refine_box)
            frb = obj.to_frb(width,resolution,center=halo_center)
            cPickle.dump(frb[field],open(fileout,'wb'),protocol=-1)
    return

def fdbk_coldens_profile(frb):
    """
    Plots radial profiles of the column densities made above.
    NOTE: HARD CODED FOR THE ASYMMETRIC RUNS

    Parameters
    ----------

    frb : fixed resolution buffer array
        Already loaded output of make_frbs - NOT THE filenames

    Returns
    -------

    rp : radial profile object
        Radial profile object with quantities stored
    """

    whole_box = np.zeros((560,560))
    dx = 80 #40/0.5
    dy1 = 40 #20/0.5
    dy2 = 280 #140/0.5

    #frb = frb.T
    whole_box[280-80:280+80,280-40:280+280] = np.log10(frb)
    mask = np.zeros((560,560),dtype=bool)
    mask[280-80:280+80,280-40:280+280] = True

    xL = np.linspace(-140,140,560)
    xL,yL = np.meshgrid(xL,xL)
    rp = radial_data(whole_box,working_mask=mask,x=xL,y=yL)
    return rp

def sym_coldens_profile(frb):
    """
    Plots radial profiles of the column densities made above.
    NOTE: HARD CODED FOR THE SYMMETRIC RUNS

    Parameters
    ----------

    frb : fixed resolution buffer array
        Already loaded output of make_frbs - NOT THE filenames

    Returns
    -------

    rp : radial profile object
        Radial profile object with quantities stored
    """

    whole_box = np.zeros((200,200))
    dx = 40 #20/0.5
    dy = 100 #50/0.5

    #frb = frb.T
    whole_box[100-dx:100+dx,:] = np.log10(frb)
    mask = np.zeros((200,200),dtype=bool)
    mask[100-dx:100+dx,:] = True

    xL = np.linspace(-50,50,200)
    xL,yL = np.meshgrid(xL,xL)
    rp = radial_data(whole_box,working_mask=mask,x=xL,y=yL)
    return rp

def get_evolultion_Lx(filenames,center_guess):
    """
    UNFINISHED METHOD FOR TRACKING HOW LX CHANGES WITH TIME
    NOT READY FOR USE!!!!!
    """
    rds = np.arange(27,43)
    rds = rds[::-1]
    timesteps = np.zeros(len(rds))
    lvectors = np.zeros((len(rds),len(filenames)))

    for j in range(len(filenames)):
        for i in range(len(rds)):
            filein = filenames[i]+'/RD00'+str(rds[i])+'/RD00'+str(rds[i])
            ds = yt.load(filein)
            halo_center = get_halo_center(ds,center_guess)
            (Lx,x) = diskVectors(ds,halo_center)
            lvectors[j,i] = Lx

            if (j==0) & (i == 0):
                timesteps[i] = ds.current_redshift

    return

class SphericalRadialDat:
    """Empty object container for holding the outputs of the radial profile calculations
    """
    def __init__(self):
        self.q75 = None
        self.q25 = None
        self.mean = None
        self.std = None
        self.median = None
        self.numel = None
        self.max = None
        self.min = None
        self.r = None
        self.fractionAbove = None
        self.fractionAboveidx = None

def spherical_radial_profile(r,y,dr,r_max=None):
    """
    Calculates radial profile values for y points with radii r
    annulus size dr and maximum radius r_max.

    Parameters
    ----------
    r: array
        Array of radii for data_source

    y: array
        Array of data_source

    dr: float
        Size of give radial annulus

    r_max: float
        Maximum radius
        Default: None

    Returns
    -------
    SphericalRadialDat object which contains the outputs as a function of r
    """
    if r_max is None:
        r_max = r.max()
    #dr = np.abs(r.max()-r.min()) * dr_width
    radial = np.arange(r_max/dr)*dr + dr/2.
    nrad = len(radial)
    radialdata = SphericalRadialDat()
    radialdata.q25 = np.zeros(nrad)
    radialdata.q75 = np.zeros(nrad)
    radialdata.mean = np.zeros(nrad)
    radialdata.std = np.zeros(nrad)
    radialdata.median = np.zeros(nrad)
    radialdata.numel = np.zeros(nrad)
    radialdata.max = np.zeros(nrad)
    radialdata.min = np.zeros(nrad)
    radialdata.r = radial
    radialdata.fractionAboveidx = []
    radialdata.fractionAbove = np.zeros(nrad)
    for irad in range(nrad): #= 1:numel(radial)
      minrad = irad*dr
      maxrad = minrad + dr
      thisindex = (r>=minrad) * (r<maxrad) #* working_mask
      datanow = y[thisindex]
      radialdata.q25[irad] = np.percentile(datanow,25)
      radialdata.q75[irad] = np.percentile(datanow,75)
      radialdata.mean[irad] = datanow.mean()
      radialdata.std[irad]  = datanow.std()
      radialdata.median[irad] = np.median(datanow)
      radialdata.numel[irad] = datanow.size
      radialdata.max[irad] = datanow.max()
      radialdata.min[irad] = datanow.min()
      radialdata.fractionAbove[irad] = (len(np.where(datanow > datanow.mean())[0])/float(len(datanow)))
    return radialdata

def gas_mass_phase_evolution(basename,RDnums,prefix):
    """
    For a given run, calculates the gas mass per phase for each RD/DD prefix given

    Parameters
    ----------
    basename : string
        Base directory name for the simulations

    RDnums : float array
        Array containing the numbers of directory outputs to be included

    prefix : string
        Prefix of the directories - i.e. "RD" or "DD"

    Returns
    -------
    gas_mass : array
        Array with the mass per phase for each sim output [phase,sim]
    """
    gas_mass = np.zeros((5,len(RDnums)))
    #set up correct filename prefix
    for i in range(len(RDnums)):
        if RDnums[i] < 100:
            zero = '00'
        else:
            zero = '0'
        ds = yt.load(basename+('/'+prefix+zero+str(RDnums[i]))*2)

        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)
        rb = sym_refine_box(ds,halo_center)

        temp = np.log10(rb['Temperature'])
        cold = np.where(temp < 4.)[0]
        cool = np.where((temp >= 4.) & (temp < 5.))[0]
        warm = np.where((temp >= 5.) & (temp < 6.))[0]
        hot = np.where(temp >= 6.)[0]

        gas_mass[0,i] = ds.current_redshift
        gas_mass[1,i] = np.log10(np.sum(rb['cell_mass'][cold].in_units('Msun')))
        gas_mass[2,i] = np.log10(np.sum(rb['cell_mass'][cool].in_units('Msun')))
        gas_mass[3,i] = np.log10(np.sum(rb['cell_mass'][warm].in_units('Msun')))
        gas_mass[4,i] = np.log10(np.sum(rb['cell_mass'][hot].in_units('Msun')))
    return gas_mass

##########################
##########################
### PLOTTING FUNCTIONS ###
##########################
##########################

def plot_SFHs(filenames,pltname,redshift_limits=None):
    """Plot SFHs for whatever filenames are provided with given redshift limits.
       Uses default line styles/colors set above.

       ** Parameters **

        :filenames: array of strings
            Array of enzo output filenames.

        :fileout: strings
            Name of output file to be saved

        :redshift_limits: array
            Array with left and right redshift limits.
            Default : 'None' means set automatically
    """

    i = 0
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])
        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)

        sp = ds.sphere(halo_center,(50.,'kpc'))
        sfr = StarFormationRate(ds, data_source=sp)
        args = filenames[i].split('/')[-3]
        kwargs = plot_kwargs[args]
        plt.plot(sfr.redshift,sfr.Msol_yr,**kwargs)

    if redshift_limits is not None:
        plt.xlim(redshift_limits[0],redshift_limits[1])

    plt.ylim(0,25)
    plt.xlabel('Redshift')
    plt.ylabel('SFR [Msun/yr]')
    plt.savefig(pltname)
    plt.close()
    return

def confirm_halo_centers(filenames):
    """
    Makes plots indicating the center found by get_halo_center to visually confirm_halo_centers

    Parameters
    ----------

    filenames : array
        Filenames to be loaded as yt data sources
    """
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])
        args = filenames[i].split('/')[-3]
        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)
        sl = yt.SlicePlot(ds,'x','Density',center=halo_center,width=(200,'kpc'))
        sl.annotate_text(center,'c')
        sl.save(args)
    return

def check_cooling_criteria(filenames):
    """
    Plots the calculated cooling criteria using the mip method in
    ProjectionPlot to see where along the projection the criteria is met

    Parameters
    ----------

    filenames : array
        Filenames to be loaded as yt data sources

    """
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])
        args = filenames[i].split('/')[-3]

        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)
        rb = sym_refine_box(ds,halo_center)
        proj = yt.ProjectionPlot(ds,'x',('gas','cooling_criteria'),
                                 center=halo_center,width=(100,'kpc'),
                                 method='mip',data_source=rb)
        proj.set_zlim(('gas','cooling_criteria'),0.001,1.)
        #proj.annotate_text(center,'c')
        proj.save(args+'_refinebox')
    plt.close()
    return

def confirm_disks(filenames):
    """
    Makes plots with only a disk region to visually confirm vectors in diskVectors

    Parameters
    ----------

    filenames : array
        Filenames to be loaded as yt data sources

    """

    for i in range(len(filenames)):
        ds = yt.load(filenames[i])
        args = filenames[i].split('/')[-3]
        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)
        (Lx,x) = diskVectors(ds, halo_center)
        disk = ds.disk(halo_center,Lx,(100.,'kpc'),(20.,'kpc'))
        sl = yt.ProjectionPlot(ds,'y','Density',center=halo_center,width=(200,'kpc'),
                          data_source=disk)
        sl.annotate_text(center,'c')
        sl.save(args+'_disk')
    plt.close()
    return

def plot_point_radialprofiles(filenames,field,fileout,plt_log=True):
    """
    Make radial profile by plotting each point in a region. If the region is big,
    look at holoviews code for a much better alternative.

    Parameters
    ----------

    filenames : array
        Filenames to be loaded as yt data sources

    field : string
        Field for which the radial profile is plotted

    fileout : string
        Name of output file

    plt_log : boolean
        Set y-axis to log or linear. Set to log by default.
        Default : True
    """
    fig,ax = plt.subplots(1,len(filenames),sharex=True)
    ax = ax.flatten()

    ### this may have to be tweaked since I had originally hard-coded this
    fig.set_size_inches(len(filenames)*3,6)
    fig.subplots_adjust(hspace=0.1,wspace=0.1)

    for i in range(len(filenames)):
        kwargs = plot_kwargs[filenames[i].split('/')[-3]]
        ds = yt.load(filenames[i])
        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)
        refine_box = sym_refine_box(ds,halo_center)
        halo_center = ds.arr(halo_center,'code_length')
        dists = compute_cell_distance(halo_center,refine_box['x'],
                                      refine_box['y'],refine_box['z'])
        dists = dists.in_units('kpc')
        if plt_log:
            ax[i].plot(dists,np.log10(refine_box[field]),alpha=0.1,color=kwargs['color'])
        else:
            ax[i].plot(dists,refine_box[field],'.',alpha=0.1,color=kwargs['color'])

        ax[-1].set_ylabel(field)
        ax[-1].set_xlabel('Distance [kpc]')
        plt.savefig(fileout)
        plt.close()
    return

## Required to run compute_disk_masses first to pass in here
def plot_disk_gas_masses(filenames,timesteps,gas_masses,fileout,redshift_limits=None):
    """
    Plot the disk gas masses as a function of time

    Parameters
    ----------

    filenames : array
        Filenames to be loaded as yt data sources
        NOTE: THESE NAMES SHOULD MATCH WHAT WAS PASSED TO
              COMPUTE_DISK_MASSES

    timesteps : array
        Result of compute_disk_masses defined above

    gas_masses : array
        Result of compute_disk_masses defined above

    fileout : string
        Name of resulting plot

    redshift_limits : array
        Array with left and right redshift limits.
        Default : 'None' means set automatically
    """
    fig,ax = plt.subplots(len(filenames),1,sharex=True)
    fig.set_size_inches(6,len(filenames)*2)
    fig.subplots_adjust(hspace=0.1,wspace=0.1)

    i = 0
    for i in range(len(filenames)):
        args = filenames[i].split('/')[-3]
        kwargs = plot_kwargs[args]

        ax[i].plot(timesteps,gas_masses[:,i],**kwargs)
        i = i + 1

    if redshift_limits is not None:
        ax[0].set_xlim(redshift_limits[0],redshift_limits[1])

    ax[-1].set_xlabel('Redshift')
    ax[-1].set_ylabel('ISM Mass [log(Msun)]')
    plt.savefig(fileout)
    plt.close()
    return

def plot_phase_diagrams(filenames,fileout):
    """
    Plot the phase diagrams of two runs side by side for comparison

    Parameters
    ----------

    filenames : array
        Filenames to be loaded as yt data sources. Meant to compare two specific runs.

    fileout : string
        Name of resulting plot

    """

    fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
    ax = ax.flatten()
    fig.set_size_inches(14,8)
    fig.subplots_adjust(hspace=0.1,wspace=0.1)
    fig.subplots_adjust(right=0.8)
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])
        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)
        refine_box = sym_refine_box(ds,halo_center)
        cellmass = refine_box['cell_mass'].in_units('Msun')
        H, xedges, yedges = np.histogram2d(np.log10(refine_box[('gas','H_nuclei_density')]),
                                           np.log10(refine_box['Temperature']),
                                           bins=200.,range=[[-6,0],[3, 8]],weights=cellmass) #,normed=True)

        im = ax[i].imshow(np.log10(H.T),extent=[-6,0,3,8],interpolation='nearest',
                     origin='lower',cmap='plasma',vmin=4,vmax=8)

        ax[i].set_title(filenames[i].split('/')[-3])
        ax[i].grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.87')
        ax[i].grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.87',alpha=0.2)
        ax[i].grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.87')
        ax[i].grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.87',alpha=0.2)
        ax[i].tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='on')
        ax[i].tick_params(axis='y',which='both',bottom='off',top='off',labelbottom='off')
        ax[i].set_xlim([-6,0])
        ax[i].set_ylim([3,8])


    ax[0].set_ylabel('Temperature [log(K)]')
    fig.text(0.5, 0.04, 'nH [log(cm^-3)]', ha='center')
    #ax[0].set_xlabel('nH [log(cm^-3)]')
    #ax[0].set_xbound(lower=-6,upper=0)
    #ax[0].set_ybound(lower=3,upper=8)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(fileout)
    plt.close()
    return

def plot_compare_basic_radial_profiles(filenames,fileout):
    """
    Plot radial profiles of hden, temperature, and metallicity showing the
    median and shading between the 25th and 75th quartiles

    NOTE: Accepts many file names but they'll all be shaded onto the same subplots.

    Parameters
    ----------

    filenames : array
        Filenames to be loaded as yt data sources.

    fileout : string
        Name of resulting plot
    """
    fig,ax = plt.subplots(1,3)
    fig.set_size_inches(12,4)
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])

        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)
        rb = sym_refine_box(ds,halo_center)

        dens = np.log10(rb['H_nuclei_density'])
        temp = np.log10(rb['Temperature'])
        Zgas = np.log10(rb['metallicity'])
        y_dists = [dens,temp,Zgas]
        x = rb['x']
        y = rb['y']
        z = rb['z']

        halo_center = ds.arr(halo_center,'code_length')
        dist = np.sqrt((halo_center[0]-rb['x'])**2.+(halo_center[1]-rb['y'])**2.+(halo_center[2]-rb['z'])**2.).in_units('kpc')

        sim_label = filenames[i].split('/')[-3]
        kwargs = plot_kwargs[sim_label]
        for j in range(len(y_dists)):
            rp = spherical_radial_profile(dist,y_dists[j],0.5)
            ax[j].plot(rp.r,rp.median,label=sim_label,**kwargs)
            ax[j].plot(rp.r,rp.q25,**kwargs)
            ax[j].plot(rp.r,rp.q75,**kwargs)
            ax[j].fill_between(rp.r,rp.q25,rp.q75,alpha=0.3,color=kwargs['color'])
            ax[j].set_xlabel('Radius [kpc]')
            #if j == (len(y_dists)-1)

        ax[0].set_title('Hydrogen Number Density')
        ax[1].set_title('Temperature')
        ax[2].set_title('Metallicity')
        ax[2].legend()
        plt.savefig(fileout)
        plt.close()
    return

def plot_cooling_time_histogram(filenames,fileout):
    """
    Plot histograms of cell mass weighted cooling times within the symmetric refine region.

    Parameters
    ----------

    filenames : array
        Filenames to be loaded as yt data sources

    fileout : string
        Name given to resulting plot

    """
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])

        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)
        sim_label = filenames[i].split('/')[-3]
        kwargs = plot_kwargs[sim_label]

        rb = sym_refine_box(ds,halo_center)
        cooling_time = rb['cooling_time'].in_units('Myr')
        cooling_time = np.log10(cooling_time)
        cell_mass = rb['cell_mass'].in_units('Msun')
        plt.hist(cooling_time,normed=True,bins=100,alpha=0.3,range=(0,6),
                 color=kwargs['color'],label=sim_label,weights=cell_mass)
    hubble_time = np.log10(13e9/1.e6)
    plt.axvline(hubble_time,ls='--',color='k')
    plt.legend()
    plt.xlabel('Cooling Time [log(Myr)]')
    plt.savefig(fileout)
    plt.close()
    return

def plot_cooling_length_histogram(filenames,fileout):
    """
    Plot histograms of cell mass weighted cooling lengths within the symmetric refine region.
    Creates two panels - one for the whole region and one excluding dense, disk gas.

    Parameters
    ----------

    filenames : array
        Filenames to be loaded as yt data sources

    fileout : string
        Name given to resulting plot

    """
    fig,ax = plt.subplots(1,2,sharey=True)
    fig.set_size_inches(10,5)
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])

        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)
        sim_label = filenames[i].split('/')[-3]
        kwargs = plot_kwargs[sim_label]

        rb = sym_refine_box(ds,halo_center)
        cooling_length = rb['cooling_time']*rb['sound_speed']
        cooling_length = np.log10(cooling_length.in_units('kpc'))
        cell_mass = rb['cell_mass'].in_units('Msun')
        ax[0].hist(cooling_length,normed=True,bins=100,alpha=0.3,range=(-6,4),
                 color=kwargs['color'],label=sim_label,weights=cell_mass)

        idx = np.where(rb['H_nuclei_density'] < 0.1)[0]
        ax[1].hist(cooling_length[idx],normed=True,bins=100,alpha=0.3,range=(-6,4),
                 color=kwargs['color'],label=sim_label,weights=cell_mass[idx])
    #line for nref11
    ax[0].axvline(np.log10(0.176622518811),ls='--',color='k')
    ax[1].axvline(np.log10(0.176622518811),ls='--',color='k',label='size10')
    #line for nref10
    ax[0].axvline(np.log10(0.353245037622),ls='--',color='k')
    ax[1].axvline(np.log10(0.353245037622),ls='--',color='k',label='size11')

    ax[1].legend()
    ax[0].set_title('All Cells')
    ax[1].set_title('Excluding Dense Disk')
    fig.text(0.5, 0.04, 'Cooling Length [log(kpc)]', ha='center')
    plt.savefig(fileout)
    plt.close()
    return

def plot_cell_mass_histogram(filenames,fileout):
    """
    Plot histograms of cell mass within the symmetric refine region.
    Overplots values of well-known SPH simulations.

    Parameters
    ----------

    filenames : array
        Filenames to be loaded as yt data sources

    fileout : string
        Name given to resulting plot

    """
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])

        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)
        sim_label = filenames[i].split('/')[-3]
        kwargs = plot_kwargs[sim_label]

        rb = sym_refine_box(ds,halo_center)
        cell_mass = rb['cell_mass'].in_units('Msun')
        cell_mass = np.log10(cell_mass)
        plt.hist(cell_mass,normed=True,bins=100,alpha=0.6,range=(-2,6.5),
                 color=kwargs['color'],label=sim_label)
    #hubble_time = 13e9/1.e6
    #plt.axvline(hubble_time,ls='--',color='k')
    plt.axvline(np.log10(2.2e5),ls='--',color='k') ## EAGLE https://arxiv.org/abs/1709.07577
    plt.text(np.log10(2.2e5),1.0,'EAGLE')
    plt.axvline(np.log10(7.1e3),ls='--',color='k') ## FIRE https://arxiv.org/abs/1606.09252
    plt.text(np.log10(7.1e3),1.0,'FIRE')
    plt.axvline(np.log10(1e6),ls='--',color='k') ## IllustrisTNG https://arxiv.org/abs/1703.02970
    plt.text(np.log10(1e6)-0.5,0.9,'IllustrisTNG')
    plt.legend()
    plt.xlabel('Cell Mass [log(Msun)]')
    plt.savefig(fileout)
    plt.close()
    return

def plot_mass_in_phase(filenames,fileout):
    """
    Plot stacked histograms of gas mass in a given phase for the sims provided.

    Parameters
    ----------

    filenames : array
        Filenames to be loaded as yt data sources

    fileout : string
        Name given to resulting plot

    """

    total_masses,cold_masses,cool_masses,warm_masses,hot_masses = [],[],[],[],[]
    tick_labels = []
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])

        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)
        sim_label = filenames[i].split('/')[-3]
        kwargs = plot_kwargs[sim_label]
        tick_labels = np.append(tick_labels,sim_label)

        rb = sym_refine_box(ds,halo_center)
        temp = np.log10(rb['Temperature'])
        cold = np.where(temp < 4.)[0]
        cool = np.where((temp >= 4.) & (temp < 5.))[0]
        warm = np.where((temp >= 5.) & (temp < 6.))[0]
        hot = np.where(temp >= 6.)[0]

        total_masses = np.append(total_masses,np.log10(np.sum(rb['cell_mass'].in_units('Msun'))))
        cold_masses = np.append(cold_masses,np.log10(np.sum(rb['cell_mass'][cold].in_units('Msun'))))
        cool_masses = np.append(cool_masses,np.log10(np.sum(rb['cell_mass'][cool].in_units('Msun'))))
        warm_masses = np.append(warm_masses,np.log10(np.sum(rb['cell_mass'][warm].in_units('Msun'))))
        hot_masses = np.append(hot_masses,np.log10(np.sum(rb['cell_mass'][hot].in_units('Msun'))))

    width = 0.35
    ind = np.arange(len(filenames))
    p1 = plt.bar(ind,cold_masses,width,color='salmon')
    p2 = plt.bar(ind,cool_masses,width,color='purple')
    p3 = plt.bar(ind,warm_masses,width,color='green')
    p4 = plt.bar(ind,hot_masses,width,color='yellow')

    plt.xlabel('Simulation')
    plt.ylabel('Gas Mass in Each Phase [log(Msun)]')
    plt.legend((p1[0],p2[0],p3[0],p4[0]),('Cold','Cool','Warm','Hot'))


    plt.xticks(ind,tick_labels)
    plt.savefig(fileout)
    plt.close()
    return

def plot_mass_in_phase_evolution(basename,RDnums,prefix,fileout):
    """
    For a given run, calculates the gas mass per phase for each RD/DD prefix given
    NOTE: Only works for a single run.

    Parameters
    ----------
    basename : string
        Base directory name for the simulations

    RDnums : float array
        Array containing the numbers of directory outputs to be included

    prefix : string
        Prefix of the directories - i.e. "RD" or "DD"

    Returns
    -------
    gas_mass : array
        Array with the mass per phase for each sim output [phase,sim]
    """
    ## the actual calcuation of mass in a given phase
    gas_mass = gas_mass_phase_evolution(basename,RDnums,prefix)
    colors = ['salmon','purple','green','yellow']
    labels = ['cold','cool','warm','hot']
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False,
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   figsize=(7, 5))
    for i in range(4):
        #print i
        ax[0].plot(gas_mass[0,:],gas_mass[i+1,:],color=colors[i],label=labels[i])
    zero = '00'
    if RDnums[-1] > 100 : zero='0'
    ds = yt.load(basename+('/'+prefix+zero+str(RDnums[-1]))*2)
    center_guess = initial_center_guess(ds,builtins.track_name)
    halo_center = get_halo_center(ds,center_guess)
    sp = ds.sphere(halo_center,(50.,'kpc'))
    sfr = StarFormationRate(ds, data_source=sp)
    ax[1].plot(sfr.redshift,sfr.Msol_yr,'k')

    ax[0].set_xlim(gas_mass[0,:].max(),gas_mass[0,:].min())
    ax[1].set_xlabel('Redshift')
    ax[0].set_ylabel('log(Gas Mass) [Msun]')
    ax[1].set_ylabel('SFR [Msun/yr]')
    ax[0].legend()
    ax[0].set_ylim(6,11)
    ax[1].set_ylim(0,8)

    plt.tight_layout()
    plt.savefig(fileout)
    plt.close()
    return

def plot_field_profile_evolution(basename,RDnums,prefix,field,fileout,plt_log=True):
    """
    For a given run, plots the yt-generated entropy profile for each
    timestep given with the prefix+RDnums

    Parameters
    ----------
    basename : string
        Base directory name for the simulations

    RDnums : float array
        Array containing the numbers of directory outputs to be included

    prefix : string
        Prefix of the directories - i.e. "RD" or "DD"

    field : string
        Field that can be passed to yt to create radial profile

    fileout : string
        Name of resulting plot

    plt_log : boolean
        Determine is log or linear radial profile is generated
        Default : True == log

        """

    n = len(RDnums)
    colors = pl.cm.viridis(np.linspace(0,1,n))
    #fig,ax = plt.subplots(2,1)
    for i in range(len(RDnums)):
        if RDnums[i] < 100:
            zero = '00'
        else:
            zero = '0'
        ds = yt.load(basename+('/'+prefix+zero+str(RDnums[i]))*2)
        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)
        rb = sym_refine_box(ds,halo_center)
        rp = yt.create_profile(rb,'radius',[field],
                                units = {'radius':'kpc'},logs = {'radius':False})
        zhere = "%.2f" % ds.current_redshift

        if plt_log:
            plt.plot(rp.x.value,np.log10(rp[field]),color=colors[i],lw=2.0,label=zhere)
        else:
            plt.plot(rp.x.value,rp[field],color=colors[i],lw=2.0,label=zhere)

    plt.legend()
    plt.xlabel('Radius [kpc]')
    plt.ylabel(field)
    plt.savefig(fileout)
    plt.close()

    return
###################################################################################################
#####################
## HOLOVIEWS PLOTS ##
#####################

def plot_holoviews_radial_profiles(filenames,fileout):
    """
    Plot radial profiles of the H number density, temperature, and metallicity
    for the simulations provided using holoviews and datashader.

    Parameters
    ----------

    filenames : string array
        Array of names of the simulation to be analyzed

    fileout : string
        Name of resulting plot
        NOTE: CURRENTLY SAVES AS HTML FILE

    """
    for i in range(len(filenames)):
        #print 'i: ',i
        ds = yt.load(filenames[i])
        center_guess = initial_center_guess(ds,builtins.track_name)
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

def plot_holoviews_phase_diagrams(filenames,fileout):
    """
    Plot phase diagrams for the simulations provided using holoviews and datashader.

    Parameters
    ----------

    filenames : string array
        Array of names of the simulation to be analyzed

    fileout : string
        Name of resulting plot
        NOTE: CURRENTLY SAVES AS HTML FILE

    """

    for i in range(len(filenames)):
        #print 'i: ',i
        ds = yt.load(filenames[i])
        center_guess = initial_center_guess(ds,builtins.track_name)
        halo_center = get_halo_center(ds,center_guess)
        rb = sym_refine_box(ds,halo_center)

        args = filenames[i].split('/')
        sim_label = args[-3]

        dens = np.log10(rb['H_nuclei_density'])
        temp = np.log10(rb['Temperature'])

        df = pd.DataFrame({'temp':temp, 'dens':dens})
        phase_scatter = hv.Scatter(df,kdims=['dens'],vdims=['temp'],label=sim_label)

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

