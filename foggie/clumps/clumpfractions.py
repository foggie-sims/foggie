import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas
from functools import partial
import matplotlib.image as mpimg
from matplotlib import cm
from scipy import stats
plt.rcParams['figure.figsize'] = [6, 5]
import matplotlib as mpl
mpl.rc("savefig", dpi=150)
from foggie.utils.foggie_load import foggie_load as fl
import argparse
from foggie.utils.get_run_loc_etc import get_run_loc_etc
import yt
import trident
from scipy.ndimage import gaussian_filter
import pandas as pd




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
                                            help='Which output? Default is RD0027 = redshift 2')
    parser.set_defaults(output='RD0020')
                        
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
output_dir = output_dir+"fractionplots/"
if not (os.path.exists(output_dir)): os.system('mkdir -p ' + output_dir)
os.chdir(output_dir)
#halo = args.halo
sim = args.run
snap = args.output


halolist=[]
levellist=[]
halogasmasslist=[]
halovolumelist=[]
halohimasslist=[]
halometalmasslist=[]
halodensitylist=[]
dmin=6.628487544938248e-31
dmax=2.10952437540689e-22
#dmax=6.628487544938248e-31*2

for halo in ['2392','2878','4123','5016','5036','8508']:
    #for halo in ['5016']:
    # load simfile
    print('loading halo %s ' %halo)
    #filename = '/Users/raugustin/WORK/SIMULATIONS/halo_00'+halo+'/nref11c_nref9f/RD0027/RD0027'
    #trackname = '/Users/raugustin/foggie/foggie/halo_tracks/00'+halo+'/nref11n_selfshield_15/halo_track_200kpc_nref9'
    filename = foggie_dir+'halo_00'+halo+'/'+sim+'/'+snap+'/'+snap
    track_name = trackname
    ds, region = fl(filename,trackname)
    regall = ds.sphere(ds.halo_center_kpc, (80, 'kpc'))
    radiusarray=np.array(regall['gas','radius_corrected'])
    densityarray=np.array(regall['gas','density'])
    massarray=np.array(regall['gas','cell_mass'])
    volumearray = np.array(regall['gas', 'volume'].in_units('kpc**3'))
    himassarray=np.array(regall['gas','H_p0_mass'])
    metalmassarray=np.array(regall['gas','metal_mass'])
    
    for lv in [1,2,3,4]:
        print('starting at lv %s of halo %s' %(lv,halo))
        if lv==1: regfltr = radiusarray < lv * 20
        else: regfltr = (radiusarray > (lv-1) * 20) & (radiusarray < lv * 20)
        d=dmin
        while d <= dmax:
            print('searching within density %s and %s of lv %s of halo %s' %(d,2*d,lv,halo))
            if d == dmax: filtr = (densityarray>=d)
            else: filtr = (densityarray>=d) & (densityarray<d*2)
            
            
            halogasmass=np.sum(massarray[regfltr&filtr])
            halogasvolume=np.sum(volumearray[regfltr&filtr])
            halohimass=np.sum(himassarray[regfltr&filtr])
            halometalmass=np.sum(metalmassarray[regfltr&filtr])
            
            halolist.append(halo)
            levellist.append(lv)
            halogasmasslist.append(halogasmass)
            halovolumelist.append(halogasvolume)
            halohimasslist.append(halohimass)
            halometalmasslist.append(halometalmass)
            halodensitylist.append(d)
            
            
            d=d*2.0



halolist2=[]
levellist2=[]
clumpgasmasslist=[]
clumpgasvolumelist=[]
clumphimasslist=[]
clumpmetalmasslist=[]
clumpdensitylist=[]

dmin=6.628487544938248e-31
dmax=2.10952437540689e-22
#dmax=6.628487544938248e-31*2

d=dmin





#import the clumpfiles under consideration as trees
for halo in ['2392','2878','4123','5016','5036','8508']:
    #for halo in ['5016']:
    for lv in [1,2,3,4]:
        d=dmin
        #clumpfilename = '/Users/raugustin/WORK/Outputs/00'+halo+'/shell_level'+str(lv)+'_20cells_20.0kpc/halo_00'+halo+'_nref11c_nref9f_RD0027_RD0027_clumps_tree.h5'
        clumpfilename = '/nobackupp13/raugust4/WORK/Outputs/plots_halo_00'+halo+'/nref11c_nref9f/clumps_boxes/shell_level'+str(lv)+'_20cells_20.0kpc/halo_00'+halo+'_nref11c_nref9f_RD0027_RD0027_clumps_tree.h5'
        if not (os.path.exists(clumpfilename)):
            while d <= dmax:
                print('searching within density %s and %s of lv %s of halo %s' %(d,2*d,lv,halo))
                
                halolist2.append(halo)
                levellist2.append(lv)
                clumpgasmasslist.append('N/A')
                clumpgasvolumelist.append('N/A')
                clumphimasslist.append('N/A')
                clumpmetalmasslist.append('N/A')
                clumpdensitylist.append(d)
                d=d*2.0
    
        elif (os.path.exists(clumpfilename)):
            clumptree = yt.load(clumpfilename)
            my_clump = clumptree.leaves[0]
            #zerototalmass = sum(my_clump["grid", "cell_mass"]) - sum(my_clump["grid", "cell_mass"])
            #zerototalvolume = np.nanmean(my_clump["grid", "dx"].in_units('kpc'))**3 - np.nanmean(my_clump["grid", "dx"].in_units('kpc'))**3
            #zerohiclumpmass = sum(my_clump["grid", "H_p0_mass"]) - sum(my_clump["grid", "H_p0_mass"])
            #zerometalclumpmass = sum(my_clump["grid", "metal_mass"]) - sum(my_clump["grid", "metal_mass"])
            
            while d <= dmax:
                print('searching within density %s and %s of lv %s of halo %s' %(d,2*d,lv,halo))
                
                totalmass = ds.quan(0, "g")
                totalvolume = ds.quan(0, "kpc**3")
                hiclumpmass = ds.quan(0, "g")
                metalclumpmass = ds.quan(0, "g")
                #print(totalmass)
                #print(totalvolume)
                for clump in clumptree.leaves:
                    #condtn = (np.min(clump["grid", "density"]) >= d) & (np.min(clump["grid", "density"]) < d*2.0)
                    condtn = (clump["grid", "density"] >= d) & (clump["grid", "density"] < d*2.0)
                    #if condtn == True:
                    #totalmass += sum(clump["grid", "cell_mass"])
                    #totalvolume += len(clump["grid", "dx"])*np.mean(clump["grid", "dx"])**3
                    #hiclumpmass += sum(clump["grid", "H_p0_mass"])
                    #metalclumpmass += sum(clump["grid", "metal_mass"])
                    totalmass += np.sum(clump["grid", "cell_mass"][condtn])
                    totalvolume += np.sum(clump["grid", "dx"][condtn].in_units('kpc')**3)
                    #if len(clump["grid", "dx"][condtn]) > 0:
                    #    totalvolume += len(clump["grid", "dx"][condtn])*np.nanmean(clump["grid", "dx"][condtn].in_units('kpc'))**3
                    hiclumpmass += np.sum(clump["grid", "H_p0_mass"][condtn])
                    metalclumpmass += np.sum(clump["grid", "metal_mass"][condtn])
                #print(totalmass)
                #print(totalvolume)
                
                halolist2.append(halo)
                levellist2.append(lv)
                clumpgasmasslist.append(totalmass)
                clumpgasvolumelist.append(totalvolume)
                clumphimasslist.append(hiclumpmass)
                clumpmetalmasslist.append(metalclumpmass)
                clumpdensitylist.append(d)
                #print(totalmass)
                #print(totalvolume)
                d=d*2.0


densityseparatedvaluesdf = pd.DataFrame(list(zip(halolist,
                                                 levellist,
                                                 halogasmasslist,
                                                 halovolumelist,
                                                 halohimasslist,
                                                 halometalmasslist,
                                                 halodensitylist,
                                                 halolist2,
                                                 levellist2,
                                                 clumpgasmasslist,
                                                 clumpgasvolumelist,
                                                 clumphimasslist,
                                                 clumpmetalmasslist,
                                                 clumpdensitylist)),
                                        columns =['halohalo',
                                                  'halolv',
                                                  'halogasmass',
                                                  'halovolume',
                                                  'halohimass',
                                                  'halometalmass',
                                                  'halodensity',
                                                  'clumphalo', 
                                                  'clumplv',
                                                  'clumpgasmass',
                                                  'clumpvolume',
                                                  'clumphimass',
                                                  'clumpmetalmass',
                                                  'clumpdensity'
                                                  ])

densityseparatedvaluesdf['clumpvolume_mod'] = densityseparatedvaluesdf['clumpvolume'][densityseparatedvaluesdf['clumpvolume']!= 'N/A'] * (0.54887732/7.62939453e-06)**3


for lv in [1,2,3,4]:
    fig=plt.figure(dpi=200,facecolor='white')
    #plt.title(r'shell from  %s to %s kpc' %(i,a),fontsize=10)
    if lv == 1:
        i=0
        a=20
    elif lv == 2:
        i=20
        a=40
    elif lv == 3:
        i=40
        a=60
    elif lv == 4:
        i=60
        a=80
    
    listofvariables=['gasmass','volume','himass','metalmass']
    for index in range(len(listofvariables)):
        ax = fig.add_subplot(2, 2, index+1,facecolor='white')
        for halo in ['2392','2878','4123','5016','5036','8508']:
            fltr = (densityseparatedvaluesdf['clump'+listofvariables[index]] != 'N/A') \
                & (densityseparatedvaluesdf['halohalo'] == halo) \
                    & (densityseparatedvaluesdf['halolv'] == lv) \
                        & (densityseparatedvaluesdf['halogasmass'] > 0.1)
            df=densityseparatedvaluesdf[fltr]
            
            ax.plot(np.log10(df['halodensity']),df['clump'+listofvariables[index]].values/df['halo'+listofvariables[index]].values, 'o',label=halo,alpha=0.5)
            #ax.set_xticks([1,2,3,4],[10,30,50,70])
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.set_ylabel(listofvariables[index]+' fraction',fontsize=10)
            ax.set_xlabel('log density [g/cm3]',fontsize=10)
            #ax.set_ylim([0,1])
            ax.set_xlim([-30,-22])
            if index == 1: ax.legend(fontsize=10)
            if index == 0:  ax.set_title(r'shell from  %s to %s kpc' %(i,a),fontsize=10)


#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=10)
"""
    """    #plt.title(r'shell from  %s to %s kpc' %(i,a),fontsize=10)

plt.tight_layout()
    plt.savefig(output_dir+'densityseparatedfractions_'+str(lv)+'.png')
    plt.show()
