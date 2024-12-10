'''
AUTHOR: Melissa Morris
DATE: 08/18/2017
NAME: siman.py
DESCRIPTION: includes simulation analysis functions that can be called by other scripts and notebooks 
'''

import numpy as np
import matplotlib.pyplot as plt
import yt
from glob import glob
from astropy.table import Table
from yt.units import kpc,km,s,cm
import astropy


# General Functions


'''
Finds the center of the galaxy

- Uses the function found in t.py to find an approximate center of the galaxy.
- Searches a 50 kpc sphere around this galaxy and makes the center the point at
    which the dark matter density is at its maximum.
  
INPUT:
filename: name of the snapshot file

OUTPUT:
ds: loaded dataset
center: calculated center of galaxy
'''

def galaxy_center(filename):
    track = Table.read('/astro/simulations/FOGGIE/'+filename[26:-14]+'/halo_track', format='ascii') 
    track.sort('col1')

    # load the snapshot
    ds = yt.load(filename)
    zsnap = ds.get_parameter('CosmologyCurrentRedshift')
    
    # interpolate the center from the track
    centerx = np.interp(zsnap, track['col1'], track['col2']) 
    centery = np.interp(zsnap, track['col1'], track['col3']) 
    centerz = 0.5 * ( np.interp(zsnap, track['col1'], track['col4']) + np.interp(zsnap, track['col1'], track['col7'])) 

    # approximation of center using jason's method
    nearcen = [centerx, centery+20. / 143886., centerz]
    
    # searches sphere w/ 50 kpc radius and sets center at point with highest density
    sph = ds.sphere(nearcen, (50,'kpc'))
    best = np.where(np.max(sph['dark_matter_density'])==sph['dark_matter_density'])[0][0]
    center = [sph['x'][best],sph['y'][best],sph['z'][best]]

    return ds,center


'''
Defines a gaussian curve for fitting purposes.
'''
def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


'''
Calculates the angular momentum vector and a vector orthogonal to the angular momentum vector

INPUT:
sph: data sphere

OUTPUT:
l: angular momentum vector
lx: vector orthogonal to angular momentum vector
'''

def angular_momentum(sph):
    # finds angular momentum vector
    l = sph.quantities.angular_momentum_vector() # At larger radii, this takes much longer
    l /= np.linalg.norm(l)
    # finds vector orthogonal to angular momentum vector
    lx = np.cross(l,[1.,0.,0.])
    lx /= np.linalg.norm(lx)
    # ensures that neither vector has units
    l = np.array(l)
    lx = np.array(lx)
    return l,lx # Angular momentum vector, Orthogonal vector


'''
Calculates the star formation history of a galaxy within a sphere of the specified radius

INPUT:
ds: data set
cen: center of the galaxy
rad: radius within which the star formation rate will be calculated

OUTPUT:
sfr.redshift: a list of redshifts over which the sfr is calculated
sfr.Msol_yr: a list of star formation rates
'''

def make_sfh(ds,cen,rad):
    # Creates a data sphere of specified radius
    sp = ds.sphere(cen,(rad,'kpc'))
    ct = sp['creation_time'].in_units('Gyr')
    sm = sp['particle_mass']
    # creates a star formation history using only star particles with a creation time
    #   and stellar mass greater than 0 (excludes dark matter)
    sfr = StarFormationRate(ds,data_source=sp,volume=sp.volume(),star_creation_time=ct[ct>0],star_mass=sm[ct>0])
    return sfr.redshift,sfr.Msol_yr


'''
Finds the amount of feedback in specified simulation

INPUT:
filename: name of snapshot file or list of snapshot files

OUTPUT:
fdbk: a string that specifies what % feedback the current snapshot is part of
'''

def find_feedback(filename):
    # searches for strings that should be in the file name and matches those
    #    to the corresponding feedback percentage
    if type(filename) == str:
        if '.1' in filename:
            fdbk = '10%'
        elif '.3' in filename:
            fdbk = '30%'
        elif '1.' in filename:
            fdbk = '100%'
    # if a list is given, only looks at first file name
    elif type(filename) == list:
        if '.1' in filename[0]:
            fdbk = '10%'
        elif '.3' in filename[0]:
            fdbk = '30%'
        elif '1.' in filename[0]:
            fdbk = '100%'
    return fdbk


# YT Derived Functions

'''
Calculates the velocity component parallel to the angular momentum vector

'''

def _vflux(field,data):
    x = data['x-velocity'].in_units('km/s')
    y = data['y-velocity'].in_units('km/s')
    z = data['z-velocity'].in_units('km/s')
    # Take dot product of bulk velocity and angular momentum vector
    bx = np.multiply(bulk_v[0],L[0])
    by = np.multiply(bulk_v[1],L[1])
    bz = np.multiply(bulk_v[2],L[2])
    leng = bx+by+bz
    nx = x-leng
    ny = y-leng
    nz = z-leng
    Lxx = np.multiply(nx,L[0])
    Ly = np.multiply(ny,L[1])
    Lz = np.multiply(nz,L[2])
    return Lxx+Ly+Lz


# Calculates the velocity component orthogonal to the angular momentum vector
def _rflux(field,data):
    x = data['x-velocity'].in_units('km/s')
    y = data['y-velocity'].in_units('km/s')
    z = data['z-velocity'].in_units('km/s')
    # Take dot product of bulk velocity and Lr vector
    Lr = np.cross(L,Lx)
    bx = np.multiply(bulk_v[0],Lr[0])
    by = np.multiply(bulk_v[1],Lr[1])
    bz = np.multiply(bulk_v[2],Lr[2])
    leng = bx+by+bz
    nx = x-leng
    ny = y-leng
    nz = z-leng
    Lxx = np.multiply(nx,Lr[0])
    Ly = np.multiply(ny,Lr[1])
    Lz = np.multiply(nz,Lr[2])
    return Lxx+Ly+Lz


def _centpos(field,data):
    xpos = data['x'] - center[0]
    ypos = data['y'] - center[1]
    zpos = data['z'] - center[2]
    pos = np.array([xpos,ypos,zpos])
    return pos


# Making Galaxies Edge-On
 
# Finding the surface density of a fixed resolution buffer

'''
Calculates the distance of an element from the center of a 2D array.

INPUT:
i: x index of element
j: y index of element
rad: radius that subarray spans in kpc (equal ro wid/2 if not a subarray)
wid: whole width of initial projection in kpc
res: resolution of initial projection

OUTPUT:
fulldist: distance of an element from the 
'''

def distance_from_center(i,j,rad,wid,res):
    w=wid/res    # x & y length of each element (in kpc)
    xdist = j*w-rad    # x distance of element from center
    ydist = i*w-rad    # y distance of element from center
    fulldist = (xdist**2+ydist**2)**.5 # full distance from center
    return fulldist


'''
Calculates the radial surface density of a fixed resolution buffer

INPUT:
im: image
ubin: upper radius
wid: width of projection, in kpc
res: resolution of projection

OUTPUT:
imval: list of surface densities of elements that fall within the given radii
farout: list of element indices that fall within radii range
'''
def surface_density(im,lbin,ubin,wid,res):
    farout=[]
    dist=[]
    imval=[]
    w=wid/res    # x & y length of each element (in kpc)
    plubin = int(ubin/w+res/2+1)     # highest element that should be looped over
    minbin = int(-ubin/w+res/2)      # lowets element that should be looped over
    # loops through every pixel that falls within this range in order to
    #    minimize the number of pixels being searched over
    for i,row in enumerate(im[minbin:plubin,minbin:plubin]):
        for j,elem in enumerate(im[minbin:plubin,minbin:plubin]):
            fulldist = distance_from_center(i,j,ubin,wid,res) # full distance from center
            # if distance from center falls within specified range, adds
            #   the value of that element to the list imval
            if fulldist <= ubin and fulldist >= lbin:
                imval.append(im[int(i+res/2-ubin/w),int(j+res/2-ubin/w)])
                farout.append([int(i+res/2-ubin/w),int(j+res/2-ubin/w)])
    return imval,farout


# Surface Density Completeness

# In[6]:

'''
Finds the completion limit of a cumulative plot

INPUT:
v: % completeness limit to be found
xlist: list of x-values
ylist: list of y-values

OUTPUT:
rad: radius at which the plot reaches v% completeness
'''
def find_completeness(v,xlist,ylist):
    for i in range(len(xlist)):
        if ylist[i] <= v:
            x1 = xlist[i]
            x2 = xlist[i+1]
            y1 = ylist[i]
            y2 = ylist[i+1]
    m = (y2-y1)/(x2-x1)
    rad = (v-y1)/m+x1
    return rad


def surface_density_completeness(image,max_radius,width,resolution):
    radius_list = np.linspace(0,max_radius,20)
    density_profile = []
    for i in range(len(radius_list))[1:]:
        density_list,index_list = surface_density(image,radius_list[i-1],radius_list[i],width,resolution)
        density_profile.append(np.sum(density_list))
    radius_list = radius_list[1:]
    cumulative_profile=[np.sum(density_profile[:i+1]) for i in range(len(density_profile))]
    log_profile=np.log10(cumulative_profile)
    normalized_profile = [log_profile[i]/log_profile[-1] for i in range(len(log_profile))]
    return find_completeness(.95,radius_list,normalized_profile)


# Super duper long method of finding the thinnest edge of the galaxy

def thinnest_edge(ds,r,cen,wid,res):
    # r is the current radius around which the edge-on galaxy will be searched for
    rs = np.arange(r-5,r+6)
    rad = []
    sig = []
    for current_rad in rs:
        # Defines data sphere at current radius
        sp = ds.sphere(cen, (current_rad,'kpc'))

        # Calculates angular momentum vector and orthogonal vector at the current radius
        L=sp.quantities.angular_momentum_vector()
        L /= np.linalg.norm(L)

        Lx = np.cross(L,[1.,0.,0.])
        Lx /= np.linalg.norm(Lx)
        
        image = yt.off_axis_projection(ds,cen,Lx,wid*kpc,res,'density',north_vector=L)

        pix = [range(res)[i]*wid/res-wid/2 for i in range(res)]
        
        # Take average density of each row of pixels
        avg_rho = [np.mean(image[:,i]) for i in range(len(pix))]

        # Fits a gaussian to the average density versus column number
        fit,er = opt.curve_fit(gaus,pix,avg_rho)

        # adds the current radius and sigma value to a list for comparison
        sig.append(fit[2])
        rad.append(current_rad)
    return rad[sig.index(min(sig))]


# Velocity Flux Calculating and Plot Making

'''
Calculates the mean, median, 25th, and 75th percentile of the velocity
flux profile, can be weighted by mass

INPUT:
cyl: data cylinder
lower: lowest distance from galaxy at which calculations are done
upper: largest distance from galaxy at which calculations are done
step: the thickness of each shell over which calculations are done
weight = True/False : choose whether or not to weigh the velocity by mass
flow = 'all'/'in'/'out' : choose what set of flows to look at; differentiates by sign of material velocity

OUTPUT:
results: A dictionary that contains the following keywords:
     'height','mean_flux_profile','median_flux_profile','lowperlist','highperlist', and 'weighted_profile' (if specified)
'''

def calculate_vflux_profile(cyl,lower,upper,step,weight=False,flow='all'):
    # List of heights over which the calculations are performed
    height_list = np.arange(lower,upper+2*step,step)*kpc
    # Height of each element in the data cylinder
    height = cyl['height'].in_units('kpc')

    # Creates an output dictionary, which can or cannot include the weighted_profile
    #     keyword, depending on the specified weight argument.
    if weight == False:
        results = {'height':height_list[:len(height_list)-1],'mean_flux_profile':[],'median_flux_profile':[],'lowperlist':[],'highperlist':[]}
    elif weight == True:
        results = {'height':height_list[:len(height_list)-1],'mean_flux_profile':[],'median_flux_profile':[],'lowperlist':[],'highperlist':[],'weighted_profile':[]}

    # Differentiates outflows and inflows by comparing the sign of the velocity flux.
    # It is important to note that you must reverse the 'out' and 'in' keywords when
    #     looking below the disk of the galaxy, due to the fact that the signs will be
    #     reversed.

    # Inflows: v < 0 above the disk, v > 0 below the disk
    # Outflows: v > 0 above the disk, v < 0 below the disk
    if flow =='out':
        f = cyl['velocity_flux'] > 0
    elif flow == 'in':
        f = cyl['velocity_flux'] < 0
    elif flow == 'all':
        f = cyl['velocity_flux']
    else:
        print "error: flow not specified"
        return None    
    
    # Loops through list of heights over which calculations are performed
    for i in range(len(height_list)-1):
        # Selects the indices where elements fall within the height range
        x = np.where((height > height_list[i])&
                     (height < height_list[i+1]))

        # Calculates the mean velocity flux within the specified height range
        mean_flux = np.mean(cyl['cyl_flux'][x].in_units('km/s'))
        results['mean_flux_profile'].append(mean_flux)

        # Calculates the median velocity flux within the specified height range
        median_flux = np.median(cyl['cyl_flux'][x].in_units('km/s'))
        results['median_flux_profile'].append(median_flux)

        if flow == 'all':
            # Calculates the 25th percentile for the specified height range
            lowper = np.percentile(cyl['cyl_flux'][x].in_units('km/s'),25)
            results['lowperlist'].append(lowper)

            # Calculates the 75th percentile for the specified height range
            highper = np.percentile(cyl['cyl_flux'][x].in_units('km/s'),75)
            results['highperlist'].append(highper)

        # If the weight parameter was set to True, calculates the mass-weighted velocity flux in
        #    the specified height range
        if weight == True:
            weighted_flux = np.sum(np.multiply(cyl['velocity_flux'][x],cyl['cell_mass'][x]))/np.sum(cyl['cell_mass'][x])
            results['weighted_profile'].append(weighted_flux)
    return results


# Outflow and Inflow Calculations

# Usage of these functions can be found in the flowstrength notebook.

'''
Calculates the mass outflow with given radial flux and mass

INPUT:
flux: velocity flux of material
mass: mass of material
step: size of shell over which mass flow is being calculated

OUTPUT:
flow: the flow rate of material within a shell of step size step, in units Msun/yr
'''

def calculate_mass_flow(flux,mass,step):
    flow = np.sum(np.multiply(flux,mass)/(step*kpc))
    return flow.in_units('Msun/yr')


'''
Calculates the mass flow rate at various distances from the galaxy

NEEDS:
L: angular momentum vector

INPUT:
sp: data sphere
lower: lowest distance from galaxy at which calculations are done
upper: largest distance from galaxy at which calculations are done
step: the thickness of each shell over which calculations are done
pos = 'above'/'below'/'all' : position relative to the galactic disk where calculations are done
flow = 'in'/'out'/'all' : looks at material with negative/positive velocity only
vmeth = 'cyl'/'rad' : method used to calculate the velocity flux

OUTPUT:
height_list: a list of heights over which the calculations are performed
outflow: the flow rate of material in shells specified by height_list
'''

# Calculates the mass outflow
def calculate_mass_flow_profile(spcyl,L,lower=0,upper=100,step=5,pos='above',flow='out',vmeth='cyl'):
    if vmeth == 'cyl':
        # Allows us to call the cylindrically derived flux
        met = 'cyl_flux'
        # cyl method: radius refers to the distance above the disk of the galaxy
        radius = spcyl['height'].in_units('kpc')
    elif vmeth == 'rad':
        # Allows us to call the radially derived flux
        met='radial_flux'
        # sph method: radius refers to the distance from the center of the galaxy
        radius = spcyl['radius'].in_units('kpc')
    
    # The list of heights over which the calculations are done
    height_list = np.arange(lower,upper+2*step,step)*kpc
    # Loads the position of each point with respect to the galactic center
    position = spcyl['center_position']
    # Loads the velocity flux using the method specified
    vel_flux = spcyl[met].in_units('km/s')
    # Loads the cell mass of each element
    mass = spcyl['cell_mass'].in_units('Msun')
    
    # Differentiates outflows and inflows by comparing the sign of the velocity flux.
    # It is important to note that, when using the cyl method, you must reverse the 
    #     'out' and 'in' keywords when looking below the disk of the galaxy, due to
    #     the fact that the signs will be reversed.
    #
    # For the CYL method:
    # Inflows: v < 0 above the disk, v > 0 below the disk
    # Outflows: v > 0 above the disk, v < 0 below the disk
    #
    # For the RAD method:
    # Inflows: v < 0 everywhere
    # Outflows: v > 0 everywhere
    
    if flow =='out':
        f = vel_flux > 0
    elif flow == 'in':
        f = vel_flux < 0
    elif flow == 'all':
        f = all
    else:
        print "error: flow not specified"
        return None

    # Specifies where calculations are performed with respect to the disk.
    # Again, it is important to note that, when using the cyl method, you will
    #     need to reverse the signs for inflows and outflows.
    if pos == 'above':
        p = np.dot(L,position)>0
    elif pos == 'below':
        p = np.dot(L,position)<0
    elif pos == 'all':
        p = all
    else:
        print "error: position not specified"
        return None

    outflow = []
    for i in range(len(height_list)-1):
        # Includes only points within a specified height range that follow
        #     the flow and position criteria specified.
        x = np.where((radius > height_list[i])&
                     (radius < height_list[i+1])&
                     (f)&
                     (p))[0]
        # Adds the calculated outflow rate for the current shell to outflow
        outflow.append(calculate_mass_flow(spcyl[met][x],spcyl['cell_mass'][x],step))
    return height_list[:-1],outflow


# Movie Making

'''
Makes a 4-panel plot of each snapshot that shows the star formation rate as a function of
  time, up to the point of the snapshot, as well as the following 3 views of the galaxy
  from an edge-on perspective:
  
- A density projection plot
- A slice plot of the metallicity, overlaid with arrows that show the direction in which material moves
- A slice plot of the temperature

INPUT:
file_name: the full name of the snapshot file
ds: the loaded data set
sph: a data sphere with a radius large enough to accomodate how much the simulation is zoomed in
center: the center of the galaxy
L: angular momentum vector
Lx: vector orthogonal to angular momentum vector
z: redshift of snapshot
sfr: star formation rate of snapshot
zoom: the full width of the density, metallicity, and temperature plots, in kpc

OUTPUT:
N/A, does not return anything, but saves the 4-panel plot
'''

def make_two_plots(file_name,ds,sph,center,L,Lx,z,sfr,zoom=70):
    fig = plt.figure()
    
    # creates grid on which the figure will be plotted
    grid = AxesGrid(fig, (.5,.5,1.5,1.5),
                    nrows_ncols = (1, 3),
                    axes_pad = .8,
                    label_mode = "L",
                    cbar_location="right",
                    cbar_mode="each",
                    cbar_set_cax = False,
                    cbar_size="3%",
                    cbar_pad="0%",
                    direction='column')

    sns.set_style("whitegrid", {'axes.grid' : False})

    # creates density projection plot
    pro = yt.OffAxisProjectionPlot(ds,Lx,('gas','density'),center=center,width=(zoom,'kpc'),north_vector=L)
    pro.set_font({'size':10})
    pro.hide_axes()
    pro.set_unit(('gas','density'),'Msun/pc**2')
    pro.set_zlim(('gas','density'),2,500)
    cmap = sns.blend_palette(("black","#984ea3","#d73027","darkorange","#ffe34d","#4daf4a","white"), n_colors=60, as_cmap=True)
    pro.set_cmap(("gas","density"), cmap)
    pro.annotate_scale(size_bar_args={'color':'white'})

    # draws density axes onto main figure
    pro.plots[('gas','density')].figure = fig
    pro.plots[('gas','density')].axes = grid[0].axes
    pro.plots[('gas','density')].cax = grid.cbar_axes[0]
    pro._setup_plots()

    # creates metallicity plot
    oap=yt.SlicePlot(ds,Lx,("gas", "metallicity"),center=center,width=(zoom,'kpc'),north_vector=L,data_source=sph)
    oap.annotate_cquiver('r_flux','cyl_flux',16)
    oap.set_font({'size':10})
    oap.hide_axes()
    oap.set_zlim(("gas", "metallicity"),2e-4,10.5)
    cmap = sns.blend_palette(("black","#984ea3","#4575b4","#4daf4a","#ffe34d","darkorange"), as_cmap=True)
    oap.set_cmap(("gas", "metallicity"),cmap)

    # draws metallicity axes onto main figure
    oap.plots[("gas", "metallicity")].figure = fig
    oap.plots[("gas", "metallicity")].axes = grid[1].axes
    oap.plots[("gas", "metallicity")].cax = grid.cbar_axes[1]
    oap._setup_plots()
    oap.annotate_scale()

    # creates temperature plot
    tp=yt.SlicePlot(ds,Lx,("gas", "temperature"),center=center,width=(zoom,'kpc'),north_vector=L,data_source=sph)
    tp.set_font({'size':10})
    tp.hide_axes()
    tp.set_zlim(("gas", "temperature"),400,8e7)
    cmap = sns.blend_palette(("black","#d73027","darkorange","#ffe34d"), n_colors=50, as_cmap=True)
    tp.set_cmap(("gas", "temperature"),cmap)

    # draws axes onto main figure
    tp.plots[('gas','temperature')].figure = fig
    tp.plots[('gas','temperature')].axes = grid[2].axes
    tp.plots[('gas','temperature')].cax = grid.cbar_axes[2]
    tp._setup_plots()
    tp.annotate_scale()

    # creates SFH plot below the first three panels on main figure
    ax3 = fig.add_subplot(211,position=[.5,.8,1.5,.17])

    ax3.plot(z,sfr)
    ax3.set_xlim(1.05,.45)
    ax3.set_ylim(-2,33)
    ax3.set_xlabel('Redshift')
    ax3.set_ylabel(r'SFR (M$_\odot$/yr)')

    # saves the figure
    plt.savefig(plot_dir+file_name[-6:]+'_plot.png',bbox_inches='tight')
