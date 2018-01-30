'''
AUTHOR: Melissa Morris
DATE: 08/18/2017
NAME: autosiman.py
DESCRIPTION: analyzes a set of simulations snapshots at a time, can calculate flow strengths, velocity flux profiles, and create movie frames
'''

import numpy as np
import yt
from astropy.table import Table
import matplotlib.pyplot as plt
from yt import derived_field
import multiprocessing as mp
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib
from yt.units import kpc,km,s,cm
from yt.analysis_modules.star_analysis.api import StarFormationRate
import seaborn as sns
import argparse
from glob import glob
import os
plt.switch_backend('agg')

'''
Includes each of the following arguments:

--movie, -m: makes movie frame plots for each snapshot iterated over
--vprovile, -v: calculates a velocity profile for each galaxy and saves it to a file
--outflowrate, -o: calculates the outflow rate for each galaxy and saves it to a file
--snapshot, -s: allows you to specify a range of snapshots over which you would like to perform calculations

'''

# Defines all available arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('fdbk',type=float,choices=[2.,1.,.3,.1,.03,.01],help='choose which set of simulations to perform calculation on by specifying the feedback percentage (1, 0.3, 0.1, etc.)')
    parser.add_argument("--movie","-m",dest='movie',action='store_true',help='creates movie frames of each snapshot')
    parser.add_argument("--zoommovie","-z",dest='zoommovie',type=float,default=70.,help='creates movie frames of each snapshot zoomed out')
    parser.add_argument("--vprofile","-v",dest='vprofile',action='store_true',help='calculates velocity profile of galaxy')
    parser.add_argument("--vout","-p",dest='voprofile',action='store_true',help='calculates velocity profile of galaxy')
    parser.add_argument("--wvprofile","-w",dest='wvprofile',action='store_true',help='calculates mass weighted velocity profile of outflowing material only')
    parser.add_argument("--outflowrate","-o",dest='outflowrate',action='store_true',help='calculates outflow rate profile')
    parser.add_argument("--inflowrate","-i",dest='inflowrate',action='store_true',help='calculates inflow rate profile')
    parser.add_argument("--snapshot","-s",dest='snap',type=int,nargs=2,help='specify the snapshots over which to perform the calculation')
    parser.add_argument("--check",'-c',dest='check',action='store_true',help='checks plot directory for files already created and omits those snapshots')
    parser.add_argument("--energy","-e",dest='energy',action='store_true',help='calculates thermal and kinetic energy profile for each snapshot')
    
    args = parser.parse_args()
    return args


'''
General Functions
'''

# Finds center of galaxy using combination of Jason's Method and BS method
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

# Calculates the star formation history
def make_sfh(ds,sp,cen):
    ct = sp['creation_time'].in_units('Gyr')
    sm = sp['particle_mass']
    sfr = StarFormationRate(ds,data_source=sp,volume=sp.volume(),star_creation_time=ct[ct>0],star_mass=sm[ct>0])
    return sfr.redshift,sfr.Msol_yr

# Calculates velocity flux profile
def calculate_vflux_profile(cyl,lower,upper,step,weight=False): #cyl = disk object, lower = lower bound for profile, upper = upper bound for profile
    height_list = np.arange(lower,upper+2*step,step)*kpc
    height = cyl['height'].in_units('kpc')

    if weight == False:
        results = {'height':height_list[:len(height_list)-1],'mean_flux_profile':[],'median_flux_profile':[],'lowperlist':[],'highperlist':[]}
    elif weight == True:
        results = {'height':height_list[:len(height_list)-1],'mean_flux_profile':[],'median_flux_profile':[],'lowperlist':[],'highperlist':[],'weighted_profile':[]}

    for i in range(len(height_list)-1):
        x = np.where((height > height_list[i]) & (height < height_list[i+1])) # the indices where you care about everything

        mean_flux = np.mean(cyl['cyl_flux'][x].in_units('km/s'))
        results['mean_flux_profile'].append(mean_flux)

        median_flux = np.median(cyl['cyl_flux'][x].in_units('km/s'))
        results['median_flux_profile'].append(median_flux)

        lowper = np.percentile(cyl['cyl_flux'][x].in_units('km/s'),25)
        results['lowperlist'].append(lowper)

        highper = np.percentile(cyl['cyl_flux'][x].in_units('km/s'),75)
        results['highperlist'].append(highper)

        if weight == True:
            weighted_flux = np.sum(np.multiply(cyl['velocity_flux'][x],cyl['cell_mass'][x]))/np.sum(cyl['cell_mass'][x])   #cyl['mvelocity_flux'][x]#
            results['weighted_profile'].append(weighted_flux)
    return results

# Calculates the mass outflow with given radial flux and mass
def calculate_mass_flow(radial_flux,mass,step):
    happy = np.sum(np.multiply(radial_flux,mass)/(step*kpc))
    return happy.in_units('Msun/yr')

# Calculates the mass outflow
def calculate_mass_flow_profile(sp,lower=0,upper=100,step=5,pos='above',flow='out',vmeth='cyl'):
    if vmeth == 'cyl':
        met = 'cyl_flux'
    elif vmeth == 'rad':
        met='radial_flux'
    height_list = np.arange(lower,upper+2*step,step)*kpc
    radius = sp['radius'].in_units('kpc')
    position = sp['center_position']
    vel_flux = sp[met].in_units('km/s')
    mass = sp['cell_mass'].in_units('Msun')
    outflow = []
    if flow =='out':
        f = vel_flux > 0
    elif flow == 'in':
        f = vel_flux < 0
    elif flow == 'all':
        f = all
    else:
        print "error: flow not specified"
        return None

    if pos == 'above':
        p = np.dot(L,position)>0
    elif pos == 'below':
        p = np.dot(L,position)<0
    elif pos == 'all':
        p = all
    else:
        print "error: position not specified"
        return None

    for i in range(len(height_list)-1):
        x = np.where((radius > height_list[i])&
                     (radius < height_list[i+1])&
                     (f)&
                     (p))[0]
        outflow.append(calculate_mass_flow(sp[met][x],sp['cell_mass'][x],step))
    return height_list[:-1],outflow


def calculate_energy_profile(sp,lower=0,upper=150,step=5):
    radius = sp['radius'].in_units('kpc')
    thermal_energy = np.multiply(sp['thermal_energy'],sp['cell_mass']).in_units('erg')
    kinetic_energy = np.multiply(sp['kinetic_energy'],sp['cell_volume']).in_units('erg')

    thermlist = []
    kinlist = []
    x = np.where((radius > lower)&
                 (radius < upper))[0]
    return np.sum(thermal_energy[x]),np.sum(kinetic_energy[x])


def final_sfh(pack):
    sfile,j,L,Lx,args = pack
    plorbus = []

    # creates star formation history from current file
    ds,center = galaxy_center(sfile)
    bsp = ds.sphere(center, (12,'kpc'))
    z,sfr = make_sfh(ds,bsp,center)
    current_z = z[-1]
    current_sfr = sfr[-1]

    file = open(plot_dir+'movie_sfr.dat','a')
    file.write(str(current_z)+'\t'+str(current_sfr)+'\t'+str(np.array(center))+'\t'+sfile[-6:]+'\n')
    file.close()

def calculate_vout_flux_profile(cyl,lower,upper,step,weight=True,flow='out'): #cyl = disk object, lower = lower bound for profile, upper = upper bound for profile
    height_list = np.arange(lower,upper+2*step,step)*kpc
    height = cyl['height'].in_units('kpc')

    results = []

    if flow =='out':
        f = cyl['velocity_flux'] > 0
    elif flow == 'in':
        f = cyl['velocity_flux'] < 0
    elif flow == 'all':
        f = all
    else:
        print "error: flow not specified"
        return None
    
    for i in range(len(height_list)-1):
        x = np.where((height > height_list[i])&
                     (height < height_list[i+1])&
                     (f)) # the indices where you care about everything

        if weight == False:
            mean_flux = np.mean(cyl['velocity_flux'][x])
            results.append(mean_flux)

        elif weight == True:
            weighted_flux = np.sum(np.multiply(cyl['velocity_flux'][x],cyl['cell_mass'][x]))/np.sum(cyl['cell_mass'][x])
            results.append(weighted_flux)

    return height_list[:len(height_list)-1],results

'''
Added YT calculations
'''

# Calculates the velocity component aligned with the angular momentum vector
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

# Calculates the velocity component perpendicular to the angular momentum vector
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

'''''''''
MAIN FUNCTIONS
'''''''''

# Defines angular momentum vector and creates list of snapshot file names
def setup(args):
    global plot_dir
    global L
    global Lx

    # Directory under which the simulations are saved
    base_dir = '/astro/simulations/FOGGIE/halo_008508/'
    
    # Specifying the snapshot that you would like to choose from and loads the matching angular momentum vector
    if args.fdbk == 1.:
        isim = 'nref10_track_2'
        L = [-0.37645994,  0.50723191,  0.77523784]
        Lx = [ 0.,          0.83679869, -0.54751068]
    elif args.fdbk == .3:
        isim = 'nref10_track_lowfdbk_1'
        L = [-0.56197719,  0.56376017,  0.60527358]
        Lx = [ 0.,          0.73175552, -0.6815672 ]
    elif args.fdbk == .1:
        isim = 'nref10_track_lowfdbk_2'
        L = [-0.48868346,  0.6016812,   0.63179761]
        Lx = [ 0., 0.72415556, -0.68963666]
    elif args.fdbk == .03:
        isim = 'nref10_track_lowfdbk_3'
        L = [-0.28229104, 0.68215233, 0.67452203]
        Lx = [ 0., 0.7031187, -0.71107249]
    elif args.fdbk == .01:
        isim = 'nref10_track_lowfdbk_4'
        L = [-0.37819875, 0.57817821, 0.72296312]
        Lx = [ 0., 0.78097012, -0.62456839]
    elif args.fdbk == 2.:
        isim = 'nref10_z1_0.5_natural'
        L = [-0.30149469,  0.59755023,  0.74299036]
        Lx = [ 0., 0.77925056, -0.62671251]

    sim = isim+'/'
    sim_dir = base_dir+sim
    
    # Where the created files will be saved to
    plot_dir = './'+isim+"_plots/"

    # Creates directory if it does not already exist
    if isim+"_plots" not in glob('*'):
        os.mkdir(plot_dir)
    
    # Makes a list of all sequential snapshots
    file_names = sorted(glob(sim_dir+'/DD????/DD????'))
    for i,filename in enumerate(file_names):
        if int(file_names[i][-4:]) == int(filename[-4:]):
            fin = i
        else:
            break
    file_names = file_names[:fin+1]

    # Creates list of arguments to be read by do_the_thing
    package = []
    for i,name in enumerate(file_names):
        package.append([name,i,L,Lx,args])

    # Only looks at the specified snapshot if argument is called
    if args.snap:
        hi = args.snap[1]
        lo = args.snap[0]
        dirs=np.array([int(i[0][-4:]) for i in package])
        yas = np.array(np.where((dirs>=lo)&(dirs<=hi))[0])
        package = np.array([package[i] for i in yas])
    # Checks for snapshots that are already in the plots directory and does not calculate anything for them
    elif args.check:
        bad = glob(plot_dir+'DD*')
        bad = np.array([int(i[len(plot_dir)+2:len(plot_dir)+6]) for i in bad])
        every = np.array([int(i[-4:]) for i in file_names])
        h = []
        for j in bad:
            if j in every:
                h.append(np.where(j == every)[0][0])
        package = np.delete(package,h,axis=0)
    return package

# Makes a 4-panel plot of each snapshot, keyword = movie
def make_two_plots(file_name,ds,sph,center,L,Lx,z,sfr,zoom=70):
    fig = plt.figure()
    
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

    # creates density plot
    pro = yt.OffAxisProjectionPlot(ds,Lx,('gas','density'),center=center,width=(zoom,'kpc'),north_vector=L)
    pro.set_font({'size':10})
    pro.hide_axes()
    pro.set_unit(('gas','density'),'Msun/pc**2')
    pro.set_zlim(('gas','density'),2,500)
    cmap = sns.blend_palette(("black","#984ea3","#d73027","darkorange","#ffe34d","#4daf4a","white"), n_colors=60, as_cmap=True)
    pro.set_cmap(("gas","density"), cmap)
    pro.annotate_scale(size_bar_args={'color':'white'})

    # draws density axes onto plot
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

    # draws metallicity axes onto plot
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

    # draws axes onto plot
    tp.plots[('gas','temperature')].figure = fig
    tp.plots[('gas','temperature')].axes = grid[2].axes
    tp.plots[('gas','temperature')].cax = grid.cbar_axes[2]
    tp._setup_plots()
    tp.annotate_scale()

    # creates SFH plot below the first three panels
    ax3 = fig.add_subplot(211,position=[.5,.8,1.5,.17])

    ax3.plot(z,sfr)
    ax3.set_xlim(1.05,.45)
    ax3.set_ylim(-2,33)
    ax3.set_xlabel('Redshift')
    ax3.set_ylabel(r'SFR (M$_\odot$/yr)')

    plt.savefig(plot_dir+file_name[-6:]+'_plot.png',bbox_inches='tight')

# Calculates the velocity flux profile and saves it to a file, keyword = vprofile
def save_vflux_profile(file_name,ds,center,L,Lx,weight = False):
    cyl = ds.disk(center,L,(15.,'kpc'),(200.,'kpc'))
    results = calculate_vflux_profile(cyl,-100,100,5,weight)
    
    if weight == False:
        v_flux = open(plot_dir+file_name[-6:]+'_velocity_profile.dat','a')
        v_flux.write('SFR = '+str(current_sfr)+',\t Current Redshift = '+str(current_z))
        v_flux.write('\n height \t mean_vprof \t median_vprof \t loper \t hiper')
        for i in range(len(results['height'])):
           v_flux.write('\n'+str(results['height'][i])+'\t'+str(results['mean_flux_profile'][i])
                        +'\t'+str(results['median_flux_profile'][i])+'\t'+str(results['lowperlist'][i])
                        +'\t'+str(results['highperlist'][i]))
        v_flux.flush()
    elif weight == True:
        v_flux = open(plot_dir+file_name[-6:]+'_weighted_velocity_profile.dat','a')
        v_flux.write('SFR = '+str(current_sfr)+',\t Current Redshift = '+str(current_z))
        v_flux.write('\nheight\tmean_vprof\tmedian_vprof\tloper\thiper\tweighted_vprof')
        for i in range(len(results['height'])):
           v_flux.write('\n'+str(results['height'][i])+'\t'+str(results['mean_flux_profile'][i])
                        +'\t'+str(results['median_flux_profile'][i])+'\t'+str(results['lowperlist'][i])
                        +'\t'+str(results['highperlist'][i])+'\t'+str(results['weighted_profile'][i]))
        v_flux.flush()
    v_flux.close()

# Calculates the velocity flux profile and saves it to a file, keyword = vprofile
def save_ovflux_profile(file_name,ds,center,L,Lx,weight = False):
    cyl = ds.disk(center,L,(15.,'kpc'),(200.,'kpc'))
    height,mean_flux_profile = calculate_vout_flux_profile(cyl,-100,100,5,weight=weight,flow='out')
    
    if weight == False:
        v_flux = open(plot_dir+file_name[-6:]+'_outvelocity_profile.dat','a')
        v_flux.write('SFR = '+str(current_sfr)+',\t Current Redshift = '+str(current_z))
        v_flux.write('\n height \t mean_vprof \t median_vprof \t loper \t hiper')
        for i in range(len(height)):
           v_flux.write('\n'+str(height[i])+'\t'+str(mean_flux_profile[i]))
        v_flux.flush()
    elif weight == True:
        v_flux = open(plot_dir+file_name[-6:]+'_weighted_outvelocity_profile.dat','a')
        v_flux.write('SFR = '+str(current_sfr)+',\t Current Redshift = '+str(current_z))
        v_flux.write('\nheight\tmean_vprof\tmedian_vprof\tloper\thiper\tweighted_vprof')
        for i in range(len(height)):
           v_flux.write('\n'+str(height[i])+'\t'+str(mean_flux_profile[i])
                        +'\t'+str(mean_flux_profile[i]))
        v_flux.flush()
    v_flux.close()

# Calculates the outflow rate and saves it to a file, keyword = outflowrate
def save_flow_rate(file_name,sp,f='out',p='above'):
    height,flow = calculate_mass_flow_profile(sp,flow=f,pos=p)
    outrate = open(plot_dir+file_name[-6:]+'_'+f+'flow_'+p+'_profile.dat','a')
    outrate.write('SFR = '+str(current_sfr)+',\t Current Redshift = '+str(current_z))
    for i in range(len(height)):
       outrate.write('\n'+str(height[i])+'\t'+str(flow[i]))
    outrate.flush()
    outrate.close()

# Calculates the thermal and kinetic energy flux and saves to file
def save_energy_profile(sp):
    thermal_energy,kinetic_energy = calculate_energy_profile(sp)
    if plot_dir+'energy.dat' not in glob(plot_dir+'*'):
        energyfil =  open(plot_dir+'energy.dat','a')
        energyfil.write('\nz\tSFR\tThermalEn\tKineticEn')
    else:
        energyfil =  open(plot_dir+'energy.dat','a')
    energyfil.write('\n'+str(current_z)+'\t'+str(current_sfr)+'\t'+str(thermal_energy)+'\t'+str(kinetic_energy))
    energyfil.flush()
    energyfil.close()


def do_the_thing(pack):
    sfile,_,L,Lx,args = pack

    global bulk_v
    global center
    global current_z
    global current_sfr
    global z
    global sfr

    if plot_dir+'movie_sfr.dat' not in glob(plot_dir+'/*'):
        ds,center = galaxy_center(sfile)
        sp = ds.sphere(center, (150,'kpc'))
        _,sfrlist = make_sfh(ds,sp,center)
        current_sfr = sfrlist[-1]
    else:
        ds = yt.load(sfile)
        finallist = open(plot_dir+'movie_sfr.dat','r')
        data = np.array([i.split('\t') for i in finallist])
        zlist = np.array([float(i) for i in data[:,0]])
        sfrlist = np.array([float(i) for i in data[:,1]])
        fillist = np.array([i[-1][:-1] for i in data])
        finallist.close()
        fin = np.where(fillist == sfile[-6:])[0]
        center = data[fin][0][2]
        center = center.split()[1:]
        if center[-1] == ']':
            center = center[:-1]
        else:
            center[2] = center[2][:-1]
        center = ds.arr([float(i) for i in center], 'code_length')
        z = zlist[:fin+1]
        sfr = sfrlist[:fin+1]
        current_sfr = sfrlist[fin][0]
        sp = ds.sphere(center, (150,'kpc'))

    bsp = ds.sphere(center, (12,'kpc'))
    bulk_v = bsp.quantities.bulk_velocity().in_units('km/s')
    current_z = ds.current_redshift
    
    ds.add_field('r_flux',function=_rflux,units='km/s')
    ds.add_field('cyl_flux',function=_vflux,units='km/s',display_name='Velocity Flux')
    ds.add_field('velocity_flux',function=_vflux,units='km/s',display_name='Velocity Flux')
    if args.movie:
        make_two_plots(sfile,ds,sp,center,L,Lx,z,sfr,zoom=args.zoommovie)
        if args.energy:
        ds.add_field('center_position',force_override=True, function=_centpos,units='', take_log=False,display_name='Position from Galaxy')
        save_energy_profile(sp)
    if args.wvprofile:
        save_vflux_profile(sfile,ds,center,L,Lx,weight=True)
    elif args.vprofile:
        save_vflux_profile(sfile,ds,center,L,Lx)
    if args.voprofile:
        save_ovflux_profile(sfile,ds,center,L,Lx)
        save_ovflux_profile(sfile,ds,center,L,Lx,weight=True)
    if args.outflowrate and args.inflowrate:
        ds.add_field('center_position',force_override=True, function=_centpos,units='', take_log=False,display_name='Position from Galaxy')
        save_flow_rate(sfile,sp)
        save_flow_rate(sfile,sp,p='below')
        save_flow_rate(sfile,sp,f='in')
        save_flow_rate(sfile,sp,f='in',p='below')
    elif args.outflowrate:
        ds.add_field('center_position',force_override=True, function=_centpos,units='', take_log=False,display_name='Position from Galaxy')
        save_flow_rate(sfile,sp)
        save_flow_rate(sfile,sp,p='below')
    elif args.inflowrate:
        ds.add_field('center_position',force_override=True, function=_centpos,units='', take_log=False,display_name='Position from Galaxy')
        save_flow_rate(sfile,sp,f='in')
        save_flow_rate(sfile,sp,f='in',p='below')

def main():
    # takes a look at each of the arguments called
    args = parse_args()
    # sets up the filenames and the angular momentum vectors associated with them

    package = setup(args)

    if args.movie and plot_dir+'movie_sfr.dat' not in glob(plot_dir+'/*'):
        pool = mp.Pool(processes = 5)
        r = pool.map_async(final_sfh,package)
        pool.close()
        pool.join()

        #for i in package:
        #    final_sfh(i)

        file = open(plot_dir+'movie_sfr.dat','r')
        data = np.array([i.split('\t') for i in file])
        cows = [[float(data[i][0]),float(data[i][1][:-8]),data[i][2],data[i][3]] for i in range(len(data))]
        cows = sorted(cows,reverse=True)
        file.close()
        file = open(plot_dir+'movie_sfr.dat','w')
        for i in cows:
            file.write(str(i[0])+'\t'+str(i[1])+'\t'+str(i[2])+'\t'+i[3])
        file.close()

    ####### only do this stuff if you don't want to do multiprocessing
    #for i in package:
    #    do_the_thing(i)
    
    pool = mp.Pool(processes = 5)
    r = pool.map_async(do_the_thing,package)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
