import yt
from yt.data_objects.level_sets.api import *
from foggie.utils.foggie_load import foggie_load as fl
import os
import matplotlib.pyplot as plt
import numpy as np
from foggie.utils.get_run_loc_etc import get_run_loc_etc
import argparse
from astropy.io import fits

def parse_args():
    '''Parse command line arguments. Returns args object.
    NOTE: Need to move command-line argument parsing to separate file.'''

    parser = argparse.ArgumentParser()
    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='8508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f. Alternative: nref11n_nref10f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--output', metavar='output', type=str, action='store', \
                        help='Which output? Default is RD0027 = redshift 1')
    parser.set_defaults(output='RD0027')

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is ramona_pleiades')
    parser.set_defaults(system='ramona_pleiades')

    parser.add_argument('--pwd', dest='pwd', action='store_true', \
                        help='Just use the working directory? Default is no')
    parser.set_defaults(pwd=False)

    args = parser.parse_args()
    return args

args = parse_args()
foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
output_dir = output_dir+"clumps/"
if not (os.path.exists(output_dir)): os.system('mkdir -p ' + output_dir)
os.chdir(output_dir)
halo = args.halo
sim = args.run
snap = args.output

filename = foggie_dir+'halo_00'+halo+'/'+sim+'/'+snap+'/'+snap
track_name = trackname
ds, region = fl(filename,trackname)
halocenter = region.center
[halocenter_x,halocenter_y,halocenter_z]=region.center



#output_dir = "/Users/raugustin/"
#os.chdir(output_dir)

clumpmasses = []
clumpvolumes = []
elongations = []
radialvelocities = []
SiIImasses = []
SiIIImasses = []
SiIVmasses = []
CIImasses = []
CIVmasses = []
OVImasses = []
MgImasses = []
MgIImasses = []
HImasses = []
IDs = []
radialdistances=[]
com=[]
coordinates=[]
xc=[]
yc=[]
zc=[]
distancefromhalocenter=[]
numberofcells = []

for boxi in ['box1','box2','box3','box4','box5','box6','box7','box8','box9','box10','box11','box12','box13','box14','box15','box16','box17','box18','box19','box20','box21','box22','box23','box24','box25','box26','box27']:
    clumpmasses = []
    clumpvolumes = []
    elongations = []
    radialvelocities = []
    SiIImasses = []
    SiIIImasses = []
    SiIVmasses = []
    CIImasses = []
    CIVmasses = []
    OVImasses = []
    MgImasses = []
    MgIImasses = []
    HImasses = []
    IDs = []
    radialdistances=[]
    com=[]
    clumpcenter=[]

    for i in range(4000):
        i=i+1
        #clumpfile=str(i)+"_single_clump.h5"
        clumpfile="/nobackupp13/raugust4/WORK/Outputs/plots_halo_008508/nref11c_nref9f/clumps/"+boxi+"/individual_clumps/"+str(i)+"_single_clump.h5"
        clumptree="/nobackupp13/raugust4/WORK/Outputs/plots_halo_008508/nref11c_nref9f/clumps/"+boxi+"/halo_008508_nref11c_nref9f_RD0027_RD0027_clumps_tree.h5"
        if (os.path.exists(clumpfile)):
            IDs.append(i)
            clump1 = yt.load(clumpfile)
            f = yt.load(clumptree)
            for myclump in f.leaves:
                if myclump['clump','clump_id'] == i:
                    raddist=myclump['clump','distance_to_main_clump']
                    radialdistances.append(raddist)
                    comi=myclump['clump','center_of_mass']
                    com.append(comi)
            ad = clump1.all_data()
            clumpmass = ad["gas", "cell_mass"].sum().in_units("Msun")
            clumpvolume = ad["gas", "cell_volume"].sum().in_units("kpc**3")

            radvel = ad["gas", "radial_velocity_corrected"].mean().in_units('km/s')
            radialvelocities.append(radvel)
            #radvel=newclump["gas", "radial_velocity_corrected"]
            #radvel
            #radvel=radvel.exclude_outside(ad["gas", "x"])
            #radvel=radvel.include_equal('x',data_source['x'])
            #radvel=radvel.include_equal('x',data_source['x'])
            #print(ad["gas", "radial_velocity_corrected"])

            clumpmasses.append(clumpmass)
            clumpvolumes.append(clumpvolume)
            xi=ad["grid", "x"].mean()
            xc.append(xi)
            yi=ad["grid", "y"].mean()
            yc.append(yi)
            zi=ad["grid", "z"].mean()
            zc.append(zi)
            coordis = (xi,yi,zi)
            coordinates.append(coordis)

            distancefromhalocenteri= np.sqrt(((xi-halocenter_x))**2 + ((yi-halocenter_y))**2 + ((zi-halocenter_z))**2)
            distancefromhalocenteri=distancefromhalocenteri.in_units("kpc")
            distancefromhalocenter.append(distancefromhalocenteri)
            x_extend = (ad["grid", "x"].max().in_units("kpc") - ad["gas", "x"].min().in_units("kpc"))
            y_extend = (ad["grid", "y"].max().in_units("kpc") - ad["gas", "y"].min().in_units("kpc"))
            z_extend = (ad["grid", "z"].max().in_units("kpc") - ad["gas", "z"].min().in_units("kpc"))
            #print(x_extend)
            #print(y_extend)
            #print(z_extend)
            maxex=max([x_extend,y_extend,z_extend]).value + ad["grid", "dx"].mean().in_units("kpc").value #add cell size because otherwise minimum extend can be 0 but should always be at least  the length of the cell
            minex=min([x_extend,y_extend,z_extend]).value + ad["grid", "dx"].mean().in_units("kpc").value
            #print(maxex)
            #print(minex)
            #elo=(np.max([x_extend.value,y_extend.value,z_extend.value])-np.min([x_extend.value,y_extend.value,z_extend.value]))/(np.max([x_extend.value,y_extend.value,z_extend.value])+np.min([x_extend.value,y_extend.value,z_extend.value]))
            elo = (maxex-minex)/(maxex+minex)
            #elo = minex/maxex
            #print(elo)
            elongations.append(elo)

            SiIImass = ad["gas", 'Si_p1_mass'].sum().in_units("Msun")
            SiIIImass = ad["gas", 'Si_p2_mass'].sum().in_units("Msun")
            SiIVmass = ad["gas", 'Si_p3_mass'].sum().in_units("Msun")
            CIImass = ad["gas", 'C_p1_mass'].sum().in_units("Msun")
            CIVmass = ad["gas", 'C_p3_mass'].sum().in_units("Msun")
            OVImass = ad["gas", 'O_p5_mass'].sum().in_units("Msun")
            MgImass = ad["gas", 'Mg_p0_mass'].sum().in_units("Msun")
            MgIImass = ad["gas", 'Mg_p1_mass'].sum().in_units("Msun")
            HImass = ad["gas", 'H_p0_mass'].sum().in_units("Msun")

            SiIImasses.append(SiIImass)
            SiIIImasses.append(SiIIImass)
            SiIVmasses.append(SiIVmass)
            CIImasses.append(CIImass)
            CIVmasses.append(CIVmass)
            OVImasses.append(OVImass)
            MgImasses.append(MgImass)
            MgIImasses.append(MgIImass)
            HImasses.append(HImass)

            numberofcellsi = len(np.array(ad["grid", "x"]))
            numberofcells.append(numberofcellsi)

    IDs=np.array(IDs)
    clumpmasses=np.array(clumpmasses)
    clumpvolumes=np.array(clumpvolumes)
    elongations=np.array(elongations)
    radialvelocities=np.array(radialvelocities)
    SiIImasses=np.array(SiIImasses)
    SiIIImasses=np.array(SiIIImasses)
    SiIVmasses=np.array(SiIVmasses)
    CIImasses=np.array(CIImasses)
    CIVmasses=np.array(CIVmasses)
    OVImasses=np.array(OVImasses)
    MgImasses=np.array(MgImasses)
    MgIImasses=np.array(MgIImasses)
    HImasses=np.array(HImasses)
    radialdistances=np.array(radialdistances)
    com=np.array(com)
    coordinates=np.array(coordinates)
    distancefromhalocenter=np.array(distancefromhalocenter)
    numberofcells=np.array(numberofcells)
    clumpradii = (3/4/np.pi * clumpvolumes)**(1/3)



    col1 = fits.Column(name='clump_ID', format='J', unit='None', array=IDs)
    col2 = fits.Column(name='clumpmasses', format='E', unit='Msun', array=clumpmasses)
    col3 = fits.Column(name='clumpvolumes', format='E', unit='kpc3', array=clumpvolumes)
    col4 = fits.Column(name='clumpradii', format='E', unit='kpc', array=clumpradii)
    col5 = fits.Column(name='elongations', format='E', unit='None', array=elongations)
    col6 = fits.Column(name='SiIImasses', format='E', unit='Msun', array=SiIImasses)
    col7 = fits.Column(name='SiIIImasses', format='E', unit='Msun', array=SiIIImasses)
    col8 = fits.Column(name='SiIVmasses', format='E', unit='Msun', array=SiIVmasses)
    col9 = fits.Column(name='CIImasses', format='E', unit='Msun', array=CIImasses)
    col10 = fits.Column(name='CIVmasses', format='E', unit='Msun', array=CIVmasses)
    col11 = fits.Column(name='OVImasses', format='E', unit='Msun', array=OVImasses)
    col12 = fits.Column(name='MgImasses', format='E', unit='Msun', array=MgImasses)
    col13 = fits.Column(name='MgIImasses', format='E', unit='Msun', array=MgIImasses)
    col14 = fits.Column(name='HImasses', format='E', unit='Msun', array=HImasses)
    col15 = fits.Column(name='radialvelocities', format='E', unit='km/s', array=radialvelocities)
    #col16 = fits.Column(name='radialdistances', format='E', unit='pc', array=radialdistances)
    #col17 = fits.Column(name='com', format='3E', unit='code_units', array=com)
    #col18 = fits.Column(name='coordinates', format='3E', unit='code_units', array=coordinates)
    col19 = fits.Column(name='centerx', format='E', unit='code_units', array=xc)
    col20 = fits.Column(name='centery', format='E', unit='code_units', array=yc)
    col21 = fits.Column(name='centerz', format='E', unit='code_units', array=zc)
    col22 = fits.Column(name='distancefromhalocenter', format='E', unit='kpc', array=distancefromhalocenter)
    col23 = fits.Column(name='numberofcells', format='E', unit='None', array=numberofcells)

    #coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22])
    coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col19, col20, col21, col22, col23])
    hdu = fits.BinTableHDU.from_columns(coldefs)
    outfile = 'halo_00'+halo+'_'+sim+'_'+snap+'_'+snap+'_'+boxi+'_clump_measurements.fits'
    oldoutfile = 'halo_00'+halo+'_'+sim+'_'+snap+'_'+snap+'_'+boxi+'_clump_measurements_old.fits'
    if (os.path.exists(outfile)):
        if (os.path.exists(oldoutfile)):
            os.remove(oldoutfile)
        os.rename(outfile, oldoutfile)
    hdu.writeto(outfile)
"""
plt.figure()
plt.hist(clumpvolumes,bins=50)
plt.ylabel('counts')
plt.xlabel('clump volume [kpc^3]')
plt.savefig('clumpvolumes.png')

clumpmasses = np.log10(clumpmasses[clumpmasses>0.])
SiIImasses = np.log10(SiIImasses[SiIImasses>0.])
SiIIImasses = np.log10(SiIIImasses[SiIIImasses>0.])
SiIVmasses = np.log10(SiIVmasses[SiIVmasses>0.])
CIImasses = np.log10(CIImasses[CIImasses>0.])
CIVmasses = np.log10(CIVmasses[CIVmasses>0.])
OVImasses = np.log10(OVImasses[OVImasses>0.])
MgImasses = np.log10(MgImasses[MgImasses>0.])
MgIImasses = np.log10(MgIImasses[MgIImasses>0.])
HImasses = np.log10(HImasses[HImasses>0.])

plt.figure()
plt.hist(clumpmasses,bins=50)
plt.ylabel('counts')
plt.xlabel('log clump mass [Msol]')
plt.savefig('clumpmasses.png')


#clumpradii = (3/4/np.pi * clumpvolumes)**(1/3)

plt.figure()
plt.hist(clumpradii,bins=50)
plt.ylabel('counts')
plt.xlabel('clump radius [kpc]')
plt.savefig('clumpradii.png')


plt.figure()
plt.hist(elongations,bins=50)
plt.ylabel('counts')
plt.xlabel('spherical <- elongation -> filamentary')
plt.savefig('elongations.png')


plt.figure()
plt.hist(radialvelocities,bins=50)
plt.ylabel('counts')
plt.xlabel('radial velocities')
plt.savefig('radvel.png')

plt.figure()
plt.hist(SiIImasses,bins=50)
plt.ylabel('counts')
plt.xlabel('SiIImasses')
plt.savefig('SiIImasses.png')

plt.figure()
plt.hist(SiIIImasses,bins=50)
plt.ylabel('counts')
plt.xlabel('SiIIImasses')
plt.savefig('SiIIImasses.png')
plt.figure()
plt.hist(SiIVmasses,bins=50)
plt.ylabel('counts')
plt.xlabel('SiIVmasses')
plt.savefig('SiIVmasses.png')
plt.figure()
plt.hist(CIImasses,bins=50)
plt.ylabel('counts')
plt.xlabel('CIImasses')
plt.savefig('CIImasses.png')
plt.figure()
plt.hist(CIVmasses,bins=50)
plt.ylabel('counts')
plt.xlabel('OVImasses')
plt.savefig('OVImasses.png')
plt.figure()
plt.hist(CIVmasses,bins=50)
plt.ylabel('counts')
plt.xlabel('CIVmasses')
plt.savefig('CIVmasses.png')
plt.figure()
plt.hist(MgImasses,bins=50)
plt.ylabel('counts')
plt.xlabel('MgImasses')
plt.savefig('MgImasses.png')
plt.figure()
plt.hist(MgIImasses,bins=50)
plt.ylabel('counts')
plt.xlabel('MgIImasses')
plt.savefig('MgIImasses.png')
plt.figure()
plt.hist(HImasses,bins=50)
plt.ylabel('counts')
plt.xlabel('HImasses')
plt.savefig('HImasses.png')
"""
