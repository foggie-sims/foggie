'''
    Filename: rockstar_plots.py
    Author: Anna
    Created: 6-12-24
    Last modified: 6-12-24 by Anna
    This file works with fogghorn_analysis.py to make a set of plots for investigating the properties of non-central galaxies.
'''

from foggie.fogghorn.header import *

# --------------------------------------------------------------------------------------------------------------------
def read_dataset_file(ds, args):
    '''
    Reads the dataset.txt file created when yt runs rockstar to match a snapshot with its corresponding rockstar output.
    Adapted from the version Anna wrote for tangos.
    '''

    rockstar_index = -1
    if os.path.exists(os.path.join(args.rockstar_directory, "datasets.txt")):
        with open(os.path.join(args.rockstar_directory, "datasets.txt")) as f:
            for l in f:
                if l.split()[0].endswith(ds.basename):
                    rockstar_index = int(l.split()[1])
    else:
        raise AssertionError("Unable to open datasets.txt")
    
    if rockstar_index<0:
        raise AssertionError("Rockstar does not appear to have been run on "+ds.basename)
    else:
        return rockstar_index
    
# --------------------------------------------------------------------------------------------------------------------
def get_cosmology(args,rockstar_index):
    '''
    Fetches the values of h and a from the rockstar output file.
    Adapted from the version Anna wrote for tangos.
    '''

    if os.path.exists(os.path.join(args.rockstar_directory, "out_"+str(rockstar_index)+".list")):
        with open(os.path.join(args.rockstar_directory, "out_"+str(rockstar_index)+".list"),'r') as f:
            for l in f:
                if l.startswith("#a"):
                    cosmo_a = float(l.split('=')[-1])
                if l.startswith("#O"):
                    cosmo_h = float(l.split(';')[-1].split('=')[-1])
        if cosmo_h<0 or cosmo_a<0:
            raise AssertionError("Cosmological values not found in rockstar output file")
            exit()
    else:
        raise AssertionError("Unable to open rockstar output file")
    return cosmo_h,cosmo_a

# --------------------------------------------------------------------------------------------------------------------
def halo_occupation_distribution(ds, args):
    '''
    Plots the fraction of halos in a given virial mass bin that are occupied by a galaxy (i.e., have formed stars)
    '''

    rockstar_index = read_dataset_file(ds,args)
    cosmo_h,cosmo_a = get_cosmology(args,rockstar_index)

    # Open rockstar output file and read in halo parameters
    hid,mv,rv,xc,yc,zc = np.loadtxt(os.path.join(args.rockstar_directory, "out_"+str(rockstar_index)+".list"),usecols=(0,2,5,8,9,10),unpack=True,comments='#')
    if len(hid)<1:
        raise AssertionError("No halos found in this output")
    
    # Identify the star particles in the simulation
    def Stars(pfilter, data):
        filter = data[("all", "particle_type")] == 2 # DM = 1, Stars = 2
        return filter
    add_particle_filter("stars", function=Stars, filtered_type='all', requires=["particle_type"])
    ds.add_particle_filter("stars")

    # Calculate stellar mass of each galaxy contained within 0.1Rvir
    sm_disk = []
    for m,r,x,y,z in zip(mv,rv,xc,yc,zc):
        r2 = max((YTArray([0.1*r*cosmo_a/cosmo_h],'kpc')).value[0],ds.index.get_smallest_dx().in_units('kpc').value)
        disk = ds.sphere(center=[x/100,y/100,z/100],radius=(r2,'kpc'))
        sm_disk.append(np.sum(disk['stars','particle_mass'].in_units('Msun')))
    
    # Calculate occupation fraction for virial mass bins
    mhbins = np.arange(7,13,0.25)
    totinbin = scipy.stats.binned_statistic(np.log10(mv),np.log10(sm_disk),bins=mhbins,statistic='count')
    totsminbin = scipy.stats.binned_statistic(np.log10(mv[np.array(sm_disk)>0]),np.log10(np.array(sm_disk)[np.array(sm_disk)>0]),bins=mhbins,statistic='count')
    occfrac = totsminbin.statistic/totinbin.statistic
    occfrac[np.isnan(occfrac)]=-1

    # Create HOD plot
    # 0 means there are halos in this bin, but nothing is occupied;
    # -1 means there are no halos in this bin
    fig = plt.figure(figsize=(9,9))
    plt.gca().stairs(occfrac,totinbin.bin_edges,color='cadetblue',linewidth=2)
    plt.ylim(-0.02,1.01)
    plt.ylabel('Occupation Fraction',fontsize=16)
    plt.xlabel('log$_{10}$(M$_\mathrm{vir,rockstar}$/M$_\odot$)',fontsize=16)
    plt.savefig(os.path.join(args.save_directory, ds.basename+"_HOD.png"),dpi=300,bbox_inches='tight')
    plt.close()

    return

# --------------------------------------------------------------------------------------------------------------------
def dwarf_SMHM(ds, args):
    '''
    Plots the stellar mass-halo mass relation for all halos in a given snapshot. Note that comparison relation (Read+17) 
    is calibrated at z=0.
    '''

    rockstar_index = read_dataset_file(ds,args)
    cosmo_h,cosmo_a = get_cosmology(args,rockstar_index)

    # Open rockstar output file and read in halo parameters
    hid,mv,rv,xc,yc,zc = np.loadtxt(os.path.join(args.rockstar_directory, "out_"+str(rockstar_index)+".list"),usecols=(0,2,5,8,9,10),unpack=True,comments='#')
    if len(hid)<1:
        raise AssertionError("No halos found in this output")
    
    # Identify the star particles in the simulation
    def Stars(pfilter, data):
        filter = data[("all", "particle_type")] == 2 # DM = 1, Stars = 2
        return filter
    add_particle_filter("stars", function=Stars, filtered_type='all', requires=["particle_type"])
    ds.add_particle_filter("stars")

    # Calculate stellar mass of each galaxy contained within 0.1Rvir
    sm_disk = []
    for m,r,x,y,z in zip(mv,rv,xc,yc,zc):
        r2 = max((YTArray([0.1*r*cosmo_a/cosmo_h],'kpc')).value[0],ds.index.get_smallest_dx().in_units('kpc').value)
        disk = ds.sphere(center=[x/100,y/100,z/100],radius=(r2,'kpc'))
        sm_disk.append(np.sum(disk['stars','particle_mass'].in_units('Msun')))
    
    # Read in Read+17 data
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    rh,rs1,rs2 = np.loadtxt(os.path.join(__location__,"Read17.txt"),unpack=True,skiprows=1)

    # Plot SMHM
    # Note division by 0.8 to correct for abundance matching being DMO
    fig = plt.figure(figsize=(9,9))
    plt.plot(np.log10(mv/0.8),np.log10(sm_disk),'k.',linestyle='none',alpha=0.5,markersize=15)
    plt.xlabel(r'log$_{10}$(M$_\mathrm{vir,rockstar}$/M$_\odot$)',fontsize=16)
    plt.ylabel(r'log$_{10}$(M$_\mathrm{\bigstar,disk}$/M$_\odot$)',fontsize=16)
    plt.plot([12.5],[12.5],c='pink',linewidth=2,linestyle='-',label='Read+17')
    plt.plot(np.log10(rh),np.log10(rs1),c='pink',linewidth=2,linestyle='-',alpha=0.9)
    plt.plot(np.log10(rh),np.log10(rs2),c='pink',linewidth=2,linestyle='-',alpha=0.9)
    plt.gca().fill_between(np.log10(rh),np.log10(rs1),np.log10(rs2),facecolor='pink',alpha=0.3)
    plt.legend(loc=2,fontsize=12,fancybox=True,framealpha=0.5)
    plt.xlim(7,12.5)
    plt.ylim(4,11)
    plt.savefig(os.path.join(args.save_directory, ds.basename+"_dwarfSMHM.png"),dpi=300,bbox_inches='tight')
    plt.close()

    return
