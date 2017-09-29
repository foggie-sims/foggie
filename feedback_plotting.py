import matplotlib
matplotlib.use('Agg')

import yt
import numpy as np
import matplotlib.pyplot as plt
from yt.analysis_modules.star_analysis.api import StarFormationRate
from yt.data_objects.particle_filters import add_particle_filter
import os
from radial_data_nozeros import *
import trident
import cPickle
from astropy.table import Table

def fdbk_refine_box(ds,halo_center):
    box_center = np.copy(halo_center)
    box_center[1] = box_center[1]+ds.arr(60.,'kpc').in_units('code_length').value

    dx = ds.arr(40.,'kpc').in_units('code_length').value
    dy = ds.arr(80.,'kpc').in_units('code_length').value
    box_left  = [box_center[0]-dx, box_center[1]-dy, box_center[2]-dx]
    box_right = [box_center[0]+dx, box_center[1]+dy, box_center[2]+dx]

    refine_box = ds.r[box_left[0]:box_right[0], box_left[1]:box_right[1], box_left[2]:box_right[2]]
    return refine_box

def sym_refine_box(ds,halo_center):
    dx = ds.arr(20.,'kpc').in_units('code_length').value
    dy = ds.arr(50.,'kpc').in_units('code_length').value
    box_left  = [halo_center[0]-dx, halo_center[1]-dy, halo_center[2]-dx]
    box_right = [halo_center[0]+dx, halo_center[1]+dy, halo_center[2]+dx]
    refine_box = ds.r[box_left[0]:box_right[0],
                      box_left[1]:box_right[1],
                      box_left[2]:box_right[2]]
    return refine_box


def formed_star(pfilter, data):
    filter = data["all", "creation_time"] > 0
    return filter

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
    'nref10f_sym50kpc' : {'ls':'-','color':'#e7298a'}
}

### utility functions ###
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

def diskVectors(ds, center):
    sphere = ds.sphere(center,(5.,'kpc'))
    angular_momentum = sphere.quantities['AngularMomentumVector']()
    x = np.cross(angular_momentum,[1.0,0.0,0.0])
    x /= np.linalg.norm(x)
    return (angular_momentum,x)

def compute_cell_distance(halo_center,x,y,z):
    return np.sqrt((halo_center[0]-x)**2.0+
                    (halo_center[1]-y)**2.0+
                    (halo_center[2]-z)**2.0)

def compute_disk_masses(filenames):
    rds = np.arange(28,43)
    rds = rds[::-1]
    timesteps = np.zeros(len(rds))
    gas_masses = np.zeros((len(rds),len(filenames)))
    stellar_masses = np.zeros((len(rds),len(filenames)))

    add_particle_filter("formed_star", function=formed_star, filtered_type='all',
                        requires=["creation_time"])

    for j in range(len(filenames)):
        center_guess = [0.48988587,0.47121728,0.50938220]
        for i in range(len(rds)):
            filein = filenames[j]+'/RD00'+str(rds[i])+'/RD00'+str(rds[i])
            ds = yt.load(filein)
            ds.add_particle_filter('formed_star')
            halo_center = get_halo_center(ds,center_guess)
            center_guess = halo_center
            (Lx,x) = diskVectors(ds,halo_center)
            disk = ds.disk(halo_center,Lx,(40.,'kpc'),(20.,'kpc'))
            ## let's look for cold/dense gas
            ## ISM conditions from review paper:
            ## n > 0.1, T < 4
            idx = np.where((disk['H_p0_number_density'] < 0.1) &
                           (disk['Temperature'] < 1e4))[0]
            gas_masses[i,j] = np.sum(disk['cell_mass'][idx].in_units('Msun'))
            stellar_masses[i,j] = np.sum(disk['formed_star', 'particle_mass'].in_units('Msun'))
            timesteps[i] = ds.current_redshift
    data = {'gas_masses':gas_masses,'stellar_masses':stellar_masses,'timesteps':timesteps}
    cPickle.dump(data,open('disk_mass_evol.cpkl','wb'),protocol=-1)
    return gas_masses,stellar_masses,timesteps

def make_frbs(filename,center,fields,ions,fdbk=False):
    ds = yt.load(filename)
    args = filename.split('/')

    trident.add_ion_fields(ds,ions=ions)

    if fdbk== True:
        print 'in fdbk'
        halo_center = get_halo_center(ds,center)

        refine_box = fdbk_refine_box(ds,halo_center)

        width = [(160,'kpc'),(80.,'kpc')]
        resolution = (320,160)
        for field in fields:
            fileout = args[-3]+'_'+args[-2]+'_x_'+field+'.cpkl'
            obj = ds.proj(field,'x',data_source=refine_box)
            frb = obj.to_frb(width,resolution,center=box_center)
            cPickle.dump(frb[field],open(fileout,'wb'),protocol=-1)

    else:
        print 'in else'
        halo_center = get_halo_center(ds,center)
        refine_box = sym_refine_box(ds,halo_center)
        width = [(40,'kpc'),(100.,'kpc')]
        resolution = (80,200)

        for field in fields:
            fileout = args[-3]+'_'+args[-2]+'_x_'+field+'.cpkl'
            obj = ds.proj(field,'x',data_source=refine_box)
            frb = obj.to_frb(width,resolution,center=halo_center)
            cPickle.dump(frb[field],open(fileout,'wb'),protocol=-1)
    return

def fdbk_coldens_profile(frb):
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
    """Empty object container.
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

### plotting functions ###
def plot_SFHS(filenames,center,radius,pltname):
    fig,ax = plt.subplots(6,1,sharex=True,sharey=True)
    fig.set_size_inches(6,14)
    fig.subplots_adjust(hspace=0.1,wspace=0.1)
    i = 0
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])
        sp = ds.sphere(center,radius)
        sfr = StarFormationRate(ds, data_source=sp)
        args = filenames[i].split('/')[-3]
        kwargs = plot_kwargs[args]
        if i < 5:
            ax[0].plot(sfr.redshift,sfr.Msol_yr,**kwargs)
            ax[i+1].plot(sfr.redshift,sfr.Msol_yr,**kwargs)
        if i > 4:
            ax[i-4].plot(sfr.redshift,sfr.Msol_yr,**kwargs)
            ax[i-4].set_xlim(1,0)
            ax[i-4].set_ylim(0,25)
            #ax[i-3].annotate('')
        i = i + 1
    ax[0].set_xlim(1,0)
    ax[0].set_ylim(0,25)
    ax[5].set_xlabel('Redshift')
    ax[2].set_ylabel('SFR [Msun/yr]')
    plt.savefig(pltname)
    return

def confirm_halo_centers(filenames,center):
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])
        args = filenames[i].split('/')[-3]
        halo_center = get_halo_center(ds,center)
        sl = yt.SlicePlot(ds,'x','Density',center=halo_center,width=(200,'kpc'))
        sl.annotate_text(center,'c')
        sl.save(args)
    return

def confirm_disks(filenames,center):
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])
        args = filenames[i].split('/')[-3]
        halo_center = get_halo_center(ds,center)
        (Lx,x) = diskVectors(ds, halo_center)
        disk = ds.disk(halo_center,Lx,(100.,'kpc'),(20.,'kpc'))
        sl = yt.ProjectionPlot(ds,'y','Density',center=halo_center,width=(200,'kpc'),
                          data_source=disk)
        sl.annotate_text(center,'c')
        sl.save(args+'_disk')
    return

def plot_point_radialprofiles(filenames,center,field,fileout,plt_log=True):
    fig,ax = plt.subplots(2,5,sharex=True,sharey=True)
    ax = ax.flatten()
    fig.set_size_inches(14,8)
    fig.subplots_adjust(hspace=0.1,wspace=0.1)
    for i in range(len(filenames)):
        kwargs = plot_kwargs[filenames[i].split('/')[-3]]
        ds = yt.load(filenames[i])
        halo_center = get_halo_center(ds,center)
        refine_box = fdbk_refine_box(ds,halo_center)
        halo_center = ds.arr(halo_center,'code_length')
        dists = compute_cell_distance(halo_center,refine_box['x'],
                                      refine_box['y'],refine_box['z'])
        dists = dists.in_units('kpc')
        if plt_log:
            ax[i].plot(dists,np.log10(refine_box[field]),alpha=0.1,color=kwargs['color'])
        else:
            ax[i].plot(dists,refine_box[field],'.',alpha=0.1,color=kwargs['color'])

        ax[0].set_ylabel(field)
        ax[7].set_xlabel('Distance [kpc]')
        plt.savefig(fileout)
    return

def plot_coldens_radialprofiles(filenames,fields,xlen,ylen,fileout,fdbk=False):
    if fdbk == True:
        fig,ax = plt.subplots(6,len(fields),sharex=True) #,sharey=True)
        fig.set_size_inches(xlen,ylen)
        fig.subplots_adjust(hspace=0.1,wspace=0.1)
        i = 0
        for i in range(len(filenames)):
            args = filenames[i].split('/')
            kwargs = plot_kwargs[args[-3]]
            for j in range(len(fields)):
                filein = 'coldens_cpkl/'+args[-3]+'_'+args[-2]+'_x_'+fields[j]+'.cpkl'
                #print filein
                frb = cPickle.load(open(filein,'rb'))
                rp = fdbk_coldens_profile(frb)
                if i < 5:
                    if len(fields) == 1:
			ax[0].plot(rp.r,rp.mean,**kwargs)
		    	ax[i+1].plot(rp.r,rp.mean,**kwargs)
		    else:
		    	ax[0,j].plot(rp.r,rp.mean,**kwargs)
                    	ax[i+1,j].plot(rp.r,rp.mean,**kwargs)
                if i > 4:
		    if len(fields) == 1:
		    	ax[0].set_title(fields[j].split('_')[0:2])
		        ax[i-4].plot(rp.r,rp.mean,**kwargs)
			ax[5].set_xlabel('Radius [kpc]')
		    else:
                    	ax[0,j].set_title(fields[j].split('_')[0:2])
                    	ax[i-4,j].plot(rp.r,rp.mean,**kwargs)
        		ax[5,j].set_xlabel('Radius [kpc]')
                    #ax[i-4].set_xlim(1,0)
                    #ax[i-4].set_ylim(0,25)
        plt.savefig(fileout)
    return

def plot_disk_gas_masses(filenames,timesteps,gas_masses,fileout):
    fig,ax = plt.subplots(6,1,sharex=True,sharey=True)
    fig.set_size_inches(6,14)
    fig.subplots_adjust(hspace=0.1,wspace=0.1)

    i = 0
    for i in range(len(filenames)):
        args = filenames[i].split('/')[-3]
        kwargs = plot_kwargs[args]
        if i < 5:
            ax[0].plot(timesteps,gas_masses[:,i],**kwargs)
            ax[i+1].plot(timesteps,gas_masses[:,i],**kwargs)
        if i > 4:
            ax[i-4].plot(timesteps,gas_masses[:,i],**kwargs)
            ax[i-4].set_xlim(1,0)
            #ax[i-4].set_ylim(0,25)
            #ax[i-3].annotate('')
        i = i + 1
    ax[0].set_xlim(1,0)
    #ax[0].set_ylim(0,25)
    ax[5].set_xlabel('Redshift')
    ax[2].set_ylabel('ISM Mass [log(Msun)]')
    plt.savefig(fileout)
    return

def plot_phase_diagrams(filenames,center,fileout):
    fig,ax = plt.subplots(2,5,sharex=True,sharey=True)
    ax = ax.flatten()
    fig.set_size_inches(14,8)
    fig.subplots_adjust(hspace=0.1,wspace=0.1)
    fig.subplots_adjust(right=0.8)
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])
        halo_center = get_halo_center(ds,center)
        refine_box = fdbk_refine_box(ds,halo_center)
        cellmass = refine_box['cell_mass'].in_units('Msun')
        H, xedges, yedges = np.histogram2d(np.log10(refine_box[('gas','H_nuclei_density')]),
                                           np.log10(refine_box['Temperature']),
                                           bins=200.,range=[[-6,0],[3, 8]],weights=cellmass) #,normed=True)

        im = ax[i].imshow(np.log10(H.T),extent=[-6,0,3,8],interpolation='nearest',
                     origin='lower',cmap='plasma',vmin=4,vmax=8)

        ax[i].set_xlim([-6,0])
        ax[i].set_ylim([3,8])
        ax[i].set_title(filenames[i].split('/')[-3])
        ax[i].grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.87')
        ax[i].grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.87',alpha=0.2)
        ax[i].grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.87')
        ax[i].grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.87',alpha=0.2)
        ax[i].tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='on')
        ax[i].tick_params(axis='y',which='both',bottom='off',top='off',labelbottom='off')

    ax[0].set_ylabel('Temperature [log(K)]')
    ax[7].set_xlabel('nH [log(cm^-3)]')
    #ax[0].set_xbound(lower=-6,upper=0)
    #ax[0].set_ybound(lower=3,upper=8)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(fileout)
    return

def plot_compare_basic_radial_profiles(filenames,fileout):
    fig,ax = plt.subplots(1,3)
    fig.set_size_inches(12,4)
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])
        track_name = '/Users/dalek/data/Jason/symmetric_box_tracking/nref11f_sym50kpc/complete_track_symmetric_100kpc'
        center_guess = initial_center_guess(ds,track_name)
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
    return

def plot_cooling_time_histogram(filenames,fileout):
    for i in range(len(filenames)):
        ds = yt.load(filenames[i])
        track_name = '/Users/dalek/data/Jason/symmetric_box_tracking/nref11f_sym50kpc/complete_track_symmetric_100kpc'
        center_guess = initial_center_guess(ds,track_name)
        halo_center = get_halo_center(ds,center_guess)
        sim_label = filenames[i].split('/')[-3]
        kwargs = plot_kwargs[sim_label]

        rb = sym_refine_box(ds,halo_center)
        cooling_time = rb['cooling_time'].in_units('Myr')
        cooling_time = np.log10(cooling_time)
        plt.hist(cooling_time,normed=True,bins=100,alpha=0.3,range=(0,6),
                 color=kwargs['color'],label=sim_label)
    hubble_time = 13e9/1.e6
    plt.axvline(hubble_time,ls='--',color='k')
    plt.legend()
    plt.xlabel('Cooling Time [log(Myr)]')
    plt.savefig(fileout)
    return

###################################################################################################

filenames = ['/astro/simulations/FOGGIE/halo_008508/nref10_track_2/RD0042/RD0042',
             '/astro/simulations/FOGGIE/halo_008508/nref10_track_lowfdbk_1/RD0042/RD0042',
             '/astro/simulations/FOGGIE/halo_008508/nref10_track_lowfdbk_2/RD0042/RD0042',
             '/astro/simulations/FOGGIE/halo_008508/nref10_track_lowfdbk_3/RD0042/RD0042',
             '/astro/simulations/FOGGIE/halo_008508/nref10_track_lowfdbk_4/RD0042/RD0042',
             '/astro/simulations/FOGGIE/halo_008508/nref10_z1_0.5_natural/RD0042/RD0042',
             '/astro/simulations/FOGGIE/halo_008508/nref10_z1_0.5_natural_lowfdbk_1/RD0042/RD0042',
             '/astro/simulations/FOGGIE/halo_008508/nref10_z1_0.5_natural_lowfdbk_2/RD0042/RD0042',
             '/astro/simulations/FOGGIE/halo_008508/nref10_z1_0.5_natural_lowfdbk_3/RD0042/RD0042',
             '/astro/simulations/FOGGIE/halo_008508/nref10_z1_0.5_natural_lowfdbk_4/RD0042/RD0042']

filenames_ts = ['/astro/simulations/FOGGIE/halo_008508/nref10_track_2',
             '/astro/simulations/FOGGIE/halo_008508/nref10_track_lowfdbk_1',
             '/astro/simulations/FOGGIE/halo_008508/nref10_track_lowfdbk_2',
             '/astro/simulations/FOGGIE/halo_008508/nref10_track_lowfdbk_3',
             '/astro/simulations/FOGGIE/halo_008508/nref10_track_lowfdbk_4',
             '/astro/simulations/FOGGIE/halo_008508/nref10_z1_0.5_natural',
             '/astro/simulations/FOGGIE/halo_008508/nref10_z1_0.5_natural_lowfdbk_1',
             '/astro/simulations/FOGGIE/halo_008508/nref10_z1_0.5_natural_lowfdbk_2',
             '/astro/simulations/FOGGIE/halo_008508/nref10_z1_0.5_natural_lowfdbk_3',
             '/astro/simulations/FOGGIE/halo_008508/nref10_z1_0.5_natural_lowfdbk_4']

filenames = ['/Users/dalek/data/Jason/symmetric_box_tracking/nref11f_sym50kpc/DD0165/DD0165',
             '/Users/dalek/data/Jason/symmetric_box_tracking/nref10f_sym50kpc/DD0165/DD0165']

plot_compare_basic_radial_profiles(filenames,'basic_dists_quartile_nref1011.pdf')

#filenames = ['/Users/dalek/data/Jason/nref10_track_lowfdbk_1/RD0042/RD0042']
#ds = yt.load(filenames[0])
#center = [0.48988587,0.47121728,0.50938220]
#halo_center = get_halo_center(ds,center)
#halo_center = np.array([0.48984, 0.47133, 0.50956])

#plot_point_radialprofiles(filenames,halo_center,'Density','dens_pt_profile.png',plt_log=True)
#plot_point_radialprofiles(filenames,halo_center,'Temperature','temp_pt_profile.png',plt_log=True)
#plot_point_radialprofiles(filenames,halo_center,'metallicity','metal_pt_profile.png',plt_log=False)

#gas_masses, stellar_masses, timesteps = compute_disk_masses(filenames_ts)
#plot_disk_gas_masses(filenames,timesteps,gas_masses,'fdbk_diskgasmass.pdf')

#plot_phase_diagrams(filenames,halo_center,'fdbk_phase.pdf')

#fields = ['H_p0_number_density','O_p5_number_density','C_p3_number_density','C_p2_number_density',
#          'Si_p2_number_density','Si_p3_number_density']

#plot_coldens_radialprofiles(filenames,fields,fdbk=True)

#ions = ['O VI','C IV','C III','Si III','Si IV']
#for filename in filenames:
#    make_frbs(filename,halo_center,fields,ions,fdbk=True)


#Cfields  = ['C_p3_number_density','C_p2_number_density']
#SiFields = ['Si_p2_number_density','Si_p3_number_density']
#OFields  = ['O_p5_number_density']

#plot_coldens_radialprofiles(filenames,['H_p0_number_density'],6,14,
#			    'coldensH.pdf',fdbk=True)
#plot_coldens_radialprofiles(filenames,Cfields,12,14,'coldensC.pdf',fdbk=True)
#plot_coldens_radialprofiles(filenames,SiFields,12,14,'coldensSi.pdf',fdbk=True)
#plot_coldens_radialprofiles(filenames,OFields,6,14,'coldensO.pdf',fdbk=True)
#plot_coldens_radialprofiles(filenames,fields,24,14,'coldensALL.pdf',fdbk=True)

#plot_SFHS(filenames,halo_center,(300.,'kpc'),'feedback_SFHs.pdf')
#confirm_halo_centers(filenames,halo_center)
