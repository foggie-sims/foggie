import yt
from yt.data_objects.level_sets.api import *
from foggie.utils.foggie_load import foggie_load as fl
from foggie.utils.foggie_load import load_sim
import os
import matplotlib.pyplot as plt
import numpy as np
from foggie.utils.get_run_loc_etc import get_run_loc_etc
import argparse
import trident
from datetime import datetime

"""
    halo='8508'
    min 3.625314185909559e-31 g/cm**3
    max 2.10952437540689e-22 g/cm**3
    
    halo='2392'
    min 1.2543553490817013e-30 g/cm**3
    max 5.211153035675982e-22 g/cm**3
    
    halo='2878'
    min 2.9556980036367175e-31 g/cm**3
    max 5.453768672501874e-22 g/cm**3
    
    halo='4123'
    min 6.628487544938248e-31 g/cm**3
    max 1.0107420908932098e-21 g/cm**3
    
    halo='5016'
    min 7.59760762075589e-32 g/cm**3
    max 6.123170753736257e-22 g/cm**3
    
    halo='5036'
    min 4.021896838122221e-31 g/cm**3
    max 6.218920603802716e-22 g/cm**3
    
    
    
"""



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

    parser.add_argument('--individual', dest='individual', action='store_true', \
                        help='Want outputs of the individual clumps? Default is no')
    parser.set_defaults(individual=False)
    parser.add_argument('--width', metavar='width', type=float, action='store', \
                        help='Width of the box around the halo center in kpc. default = 30')
    parser.set_defaults(width=30.)
        
    parser.add_argument('--shape', metavar='shape', type=str, action='store', \
                        help='What shape of region do you want to run the clump finder on? options are box or shell. default is shell')
    parser.set_defaults(patchname='shell')
                        
    parser.add_argument('--mincells', metavar='mincells', type=int, action='store', \
                        help='Minimum cells for a clump. default = 20')
    parser.set_defaults(mincells=20)
    
    parser.add_argument('--level', metavar='level', type=int, action='store', \
                        help='level of shells, if shells chosen. default is 1, which is just a sphere around the center.')
    parser.set_defaults(level=1)
    
    parser.add_argument('--step', metavar='step', type=float, action='store', \
                        help='clumpfinder step parameter. default = 2. ')
    parser.set_defaults(step=2.)
    
    parser.add_argument('--patchname', metavar='patchname', type=str, action='store', \
                        help='Name  for the patch to find clumps? Default is central_30kpc')
    parser.set_defaults(patchname='box1')
    
    parser.add_argument('--center', metavar='center', type=str, action='store', \
                        help='Center of the box in the halo center in code units. default = center1')
    parser.set_defaults(center='center1')
    
    args = parser.parse_args()
    return args

args = parse_args()
foggie_dir, output_dir, run_loc, code_path, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
patchname = args.patchname
bwdth = str(args.width)
sim = args.run
snap = args.output

for halo in ['2392','2878','4123','5016','5036','8508']:
    for lv in [1,2,3,4]:
        if args.shape=='box':
            output_dir = output_dir+"clumps_boxes/"+patchname+'_'+str(args.mincells)+'cells_'+bwdth+'kpc/'
        elif args.shape == 'shell':
            output_dir = output_dir+"clumps_boxes/"+'shell'+"_level"+str(lv)+'_'+str(args.mincells)+'cells_'+bwdth+'kpc/'
        if not (os.path.exists(output_dir)): os.system('mkdir -p ' + output_dir)
        os.chdir(output_dir)
        
        treefile =output_dir+'halo_00'+halo+'_'+sim+'_'+snap+'_'+snap+'_clumps_tree'
        if not (os.path.exists(treefile)):


            filename = foggie_dir+'halo_00'+halo+'/'+sim+'/'+snap+'/'+snap
            if args.system == 'ramona_pleiades':
                trackname = '/nobackupp13/raugust4/foggie/foggie/halo_tracks/00'+halo+'/nref11n_selfshield_15/halo_track_200kpc_nref9'
            elif args.system == 'ramona':
                trackname = '/Users/raugustin/foggie/foggie/halo_tracks/00'+halo+'/nref11n_selfshield_15/halo_track_200kpc_nref9'
            else: print('idk where to find the trackfile lol')

            now = datetime.now()
            print(now)
            print("~~~~~~~loading halo")

            #ds, region = fl(filename,trackname)
            ds, region = fl(filename, trackname, \
                            particle_type_for_angmom=False, do_filter_particles=False, \
                            gravity=True, \
                            masses_dir = code_path+'halo_infos/00'+halo+'/'+sim+'/' , \
                            region='refine_box') # this *SHOULD* work better, I just hope I'm not losing anything important
            # if halo_c_v file does not include the halo center, foggie_load will try to calculate it (which doesnt work without problems in yt4 so here is the workaround from Ayan using load_sim)


            #ds, refine_box = load_sim(args, region='refine_box')
            #args.halo_center = ds.halo_center_kpc
            #args.halo_velocity = ds.halo_velocity_kms
            #[centerx,centery,centerz] = ds.halo_center
            #args.halo_velocity = ds.halo_velocity_kms



            for chosenion in ['O VI','C II','C IV','Si II','Si III','Si IV', 'Mg I', 'Mg II', 'H I', 'H II']:
                trident.add_ion_fields(ds, ions=[chosenion])

            chosenwidth = args.width

            dx= ds.quan(chosenwidth,'kpc').in_units('code_length')

            dy= ds.quan(chosenwidth,'kpc').in_units('code_length')

            dz= ds.quan(chosenwidth,'kpc').in_units('code_length')


            [centerx,centery,centerz]=ds.halo_center_kpc

            center1 = [centerx+dx,centery,centerz]
            center2 = [centerx+dx,centery+dy,centerz]
            center3 = [centerx+dx,centery+dy,centerz+dz]
            center4 = [centerx+dx,centery+dy,centerz-dz]
            center5 = [centerx+dx,centery,centerz+dz]
            center6 = [centerx+dx,centery,centerz-dz]
            center7 = [centerx+dx,centery-dy,centerz]
            center8 = [centerx+dx,centery-dy,centerz+dz]
            center9 = [centerx+dx,centery-dy,centerz-dz]
            center10 = [centerx,centery,centerz]
            center11 = [centerx,centery+dy,centerz]
            center12 = [centerx,centery+dy,centerz+dz]
            center13 = [centerx,centery+dy,centerz-dz]
            center14 = [centerx,centery,centerz+dz]
            center15 = [centerx,centery,centerz-dz]
            center16 = [centerx,centery-dy,centerz]
            center17 = [centerx,centery-dy,centerz+dz]
            center18 = [centerx,centery-dy,centerz-dz]
            center19 = [centerx-dx,centery,centerz]
            center20 = [centerx-dx,centery+dy,centerz]
            center21 = [centerx-dx,centery+dy,centerz+dz]
            center22 = [centerx-dx,centery+dy,centerz-dz]
            center23 = [centerx-dx,centery,centerz+dz]
            center24 = [centerx-dx,centery,centerz-dz]
            center25 = [centerx-dx,centery-dy,centerz]
            center26 = [centerx-dx,centery-dy,centerz+dz]
            center27 = [centerx-dx,centery-dy,centerz-dz]


            if args.center == 'center1':
                chosencenter =  center1
            if args.center == 'center2':
                chosencenter =  center2
            if args.center == 'center3':
                chosencenter =  center3
            if args.center == 'center4':
                chosencenter =  center4
            if args.center == 'center5':
                chosencenter =  center5
            if args.center == 'center6':
                chosencenter =  center6
            if args.center == 'center7':
                chosencenter =  center7
            if args.center == 'center8':
                chosencenter =  center8
            if args.center == 'center9':
                chosencenter =  center9
            if args.center == 'center10':
                chosencenter =  center10
            if args.center == 'center11':
                chosencenter =  center11
            if args.center == 'center12':
                chosencenter =  center12
            if args.center == 'center13':
                chosencenter =  center13
            if args.center == 'center14':
                chosencenter =  center14
            if args.center == 'center15':
                chosencenter =  center15
            if args.center == 'center16':
                chosencenter =  center16
            if args.center == 'center17':
                chosencenter =  center17
            if args.center == 'center18':
                chosencenter =  center18
            if args.center == 'center19':
                chosencenter =  center19
            if args.center == 'center20':
                chosencenter =  center20
            if args.center == 'center21':
                chosencenter =  center21
            if args.center == 'center22':
                chosencenter =  center22
            if args.center == 'center23':
                chosencenter =  center23
            if args.center == 'center24':
                chosencenter =  center24
            if args.center == 'center25':
                chosencenter =  center25
            if args.center == 'center26':
                chosencenter =  center26
            if args.center == 'center27':
                chosencenter =  center27

            #print(center)
            #data_source = ds.sphere(chosencenter, (chosenwidth, 'kpc'))

            leftedge = chosencenter - ds.quan(chosenwidth/2., 'kpc').in_units('code_length')
            rightedge = chosencenter + ds.quan(chosenwidth/2., 'kpc').in_units('code_length')

            if args.shape=='box':
                
                data_source = ds.box(leftedge, rightedge)

            elif args.shape=='shell':
                i = args.level
                if i == 1:
                    data_source = ds.sphere(ds.halo_center_kpc, (chosenwidth, 'kpc'))
                else:
                    data_source = ds.sphere(ds.halo_center_kpc, (i*chosenwidth, 'kpc')) - ds.sphere(ds.halo_center_kpc, ((i-1)*chosenwidth, 'kpc'))

            #yt.ProjectionPlot(ds, 2, ("gas", "density"), center=chosencenter, width=(chosenwidth,'kpc'),data_source=data_source, weight_field=("gas", "density")).show()

            #yt.ProjectionPlot(ds, 2, ("gas", "temperature"), center=chosencenter, width=(chosenwidth,'kpc'),data_source=data_source, weight_field=("gas", "density")).show()

            #yt.ProjectionPlot(ds, 2, ("gas", "metallicity"), center=chosencenter, width=(chosenwidth,'kpc'),data_source=data_source, weight_field=("gas", "density")).show()

            #### change min to min of all 6 halos

            master_clump = Clump(data_source, ("gas", "density"))
            master_clump.add_validator("min_cells", args.mincells)
            #c_min = data_source["gas", "density"].min()
            #c_max = data_source["gas", "density"].max()
            c_min = ds.quan(6.628487544938248e-31, "g/cm**3")
            c_max = ds.quan(2.10952437540689e-22, "g/cm**3")
            step = args.step #100. #2.0

            now = datetime.now()
            print(now)
            print("~~~~~~~running clumpfinder now - this will take a while")
            find_clumps(master_clump, c_min, c_max, step)
            now = datetime.now()
            print(now)
            print("~~~~~~~finished running clump finder")


            leaf_clumps = master_clump.leaves
            now = datetime.now()
            print(now)
            print("~~~~~~~calculating clump properties and adding fields of interest - this will take a while")

            master_clump.add_info_item("total_cells")
            master_clump.add_info_item("cell_mass")
            master_clump.add_info_item("mass_weighted_jeans_mass")
            master_clump.add_info_item("volume_weighted_jeans_mass")
            master_clump.add_info_item("max_grid_level")
            master_clump.add_info_item("min_number_density")
            master_clump.add_info_item("max_number_density")
            master_clump.add_info_item("center_of_mass")
            master_clump.add_info_item("distance_to_main_clump")

            halocenter = ds.halo_center_kpc
            [halocenter_x,halocenter_y,halocenter_z]=ds.halo_center_kpc

            fitsfile =output_dir+'halo_00'+halo+'_'+sim+'_'+snap+'_'+snap+'_'+patchname+'_clump_measurements.fits'
            if not (os.path.exists(fitsfile)):

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
                coordinates=[]
                xc=[]
                yc=[]
                zc=[]
                distancefromhalocenter=[]
                numberofcells = []
                metallicity=[]
                metalmasses = []
                vffs = []
                rvovervffs = []
                pressures = []
                densities = []
                temperatures = []
                envvels = []
                envpressures = []
                envdensities = []
                envtemps = []
                pressureratios = []


                for clump in leaf_clumps:
                    raddist=clump.info['distance_to_main_clump']
                    radialdistances.append(raddist)
                    comi=clump.info['center_of_mass']
                    com.append(comi)
                    ad = clump.data
                    clumpmass = ad["gas", "cell_mass"].sum().in_units("Msun")
                    clumpvolume = ad["gas", "cell_volume"].sum().in_units("kpc**3")
                    clumpmetallicity = ad["gas", "metallicity"].mean().in_units("Zsun")
                    metallicity.append(clumpmetallicity)
                    metalmass = ad["gas","metal_mass"].sum().in_units("Msun")
                    metalmasses.append(metalmass)
                    radvel = ad["gas", "radial_velocity_corrected"].mean().in_units('km/s')
                    radialvelocities.append(radvel)
                    vff = ad["gas", "vff"].mean().in_units('km/s')
                    vffs.append(vff)
                    rvovervff = ad["gas", "radial_velocity_corrected"]/ad["gas", "vff"]
                    rvovervff = rvovervff.mean()
                    rvovervffs.append(rvovervff)
                    
                    #radvel=newclump["gas", "radial_velocity_corrected"]
                    #radvel
                    #radvel=radvel.exclude_outside(ad["gas", "x"])
                    #radvel=radvel.include_equal('x',data_source['x'])
                    #radvel=radvel.include_equal('x',data_source['x'])
                    #print(ad["gas", "radial_velocity_corrected"])
                    
                    clumpmasses.append(clumpmass)
                    clumpvolumes.append(clumpvolume)
                    xi=ad["gas", "x"].mean()
                    xc.append(xi)
                    yi=ad["gas", "y"].mean()
                    yc.append(yi)
                    zi=ad["gas", "z"].mean()
                    zc.append(zi)
                    coordis = (xi,yi,zi)
                    coordinates.append(coordis)
                    
                    #distancefromhalocenteri= np.sqrt(((xi-halocenter_x))**2 + ((yi-halocenter_y))**2 + ((zi-halocenter_z))**2)
                    #distancefromhalocenteri=distancefromhalocenteri.in_units("kpc")
                    distancefromhalocenteri=ad["gas", "radius_corrected"].mean().in_units("kpc")
                    distancefromhalocenter.append(distancefromhalocenteri)
                    x_extend = (ad["gas", "x"].max().in_units("kpc") - ad["gas", "x"].min().in_units("kpc"))
                    y_extend = (ad["gas", "y"].max().in_units("kpc") - ad["gas", "y"].min().in_units("kpc"))
                    z_extend = (ad["gas", "z"].max().in_units("kpc") - ad["gas", "z"].min().in_units("kpc"))
                    #print(x_extend)
                    #print(y_extend)
                    #print(z_extend)
                    maxex=max([x_extend,y_extend,z_extend]).value + ad["gas", "dx"].mean().in_units("kpc").value #add cell size because otherwise minimum extend can be 0 but should always be at least  the length of the cell
                    minex=min([x_extend,y_extend,z_extend]).value + ad["gas", "dx"].mean().in_units("kpc").value
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
                    
                    numberofcellsi = len(np.array(ad["gas", "x"]))
                    numberofcells.append(numberofcellsi)
                    
                    # pressure within vs outside of clump
                    clumppressure = ad["gas", "pressure"].mean()
                    pressures.append(clumppressure)
                    if maxex > 0.275:
                        environment = ds.sphere(coordis, (maxex/2., "kpc")) - clump.data
                    else: environment = ds.sphere(coordis, (maxex+0.14, "kpc")) - clump.data
                    
                    envpressure = environment["gas", "pressure"].mean()
                    envpressures.append(envpressure)
                    
                    clumpdensity = ad["gas", "density"].mean()
                    densities.append(clumpdensity)
                    
                    envdensity = environment["gas", "density"].mean()
                    envdensities.append(envdensity)
                    
                    clumptemp = ad["gas", "temperature"].mean()
                    temperatures.append(clumptemp)
                    
                    envtemp = environment["gas", "temperature"].mean()
                    envtemps.append(envtemp)
                    
                    envvel = environment["gas", "radial_velocity_corrected"].mean().in_units('km/s')
                    envvels.append(envvel)


                    pressureratios.append(clumppressure/envpressure)


                print('finished calculations - preparing output')
                print(patchname)
                if IDs == []: print('halo '+halo+' box '+patchname+' appears to be empty - check and rerun!')
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
                metallicity=np.array(metallicity)
                metalmasses=np.array(metalmasses)
                com=np.array(com)
                coordinates=np.array(coordinates)
                distancefromhalocenter=np.array(distancefromhalocenter)
                numberofcells=np.array(numberofcells)
                clumpradii = (3/4/np.pi * clumpvolumes)**(1/3)
                vffs = np.array(vffs)
                rvovervffs = np.array(rvovervffs)
                pressures = np.array(pressures)
                envpressures = np.array(envpressures)
                pressureratios = np.array(pressureratios)
                envdensities = np.array(envdensities)
                envtemps = np.array(envtemps)


                from astropy.io import fits
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
                col24 = fits.Column(name='metallicity', format='E', unit='Zsun', array=metallicity)
                col25 = fits.Column(name='metalmasses', format='E', unit='Msun', array=metalmasses)
                col26 = fits.Column(name='vffs', format='E', unit='km/s', array=vffs)
                col27 = fits.Column(name='rvovervffs', format='E', unit='None', array=rvovervffs)
                col28 = fits.Column(name='pressures', format='E', unit='dyn/cm**2', array=pressures)
                col29 = fits.Column(name='envpressures', format='E', unit='dyn/cm**2', array=envpressures)
                col30 = fits.Column(name='pressureratios', format='E', unit='None', array=pressureratios)
                col31 = fits.Column(name='envdensities', format='E', unit='None', array=envdensities)
                col32 = fits.Column(name='envtemps', format='E', unit='None', array=envtemps)
                col33 = fits.Column(name='envvels', format='E', unit='None', array=envvels)

                coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, \
                    col11, col12, col13, col14, col15, col19, col20, \
                    col21, col22, col23, col24, col25, col26, col27, col28, col29, col30, \
                    col31, col32, col33])
                hdu = fits.BinTableHDU.from_columns(coldefs)
                outfile = 'halo_00'+halo+'_'+sim+'_'+snap+'_'+snap+'_'+patchname+'_clump_measurements.fits'
                oldoutfile = 'halo_00'+halo+'_'+sim+'_'+snap+'_'+snap+'_'+patchname+'_clump_measurements_old.fits'
                if (os.path.exists(outfile)):
                    if (os.path.exists(oldoutfile)):
                        os.remove(oldoutfile)
                    os.rename(outfile, oldoutfile)
                hdu.writeto(outfile)
                print('successfuly wrote fits output file for:')
                print(patchname)

            fields_of_interest = [("gas", "density"),("gas", "temperature"), ("gas", "metallicity"),("gas", "metal_mass"),"particle_mass",'particle_position',("gas", 'cell_mass'),#("gas", "cell_volume"), \
                                  ("gas", 'radial_velocity_corrected'), \
                                  ("gas", 'density_gradient_magnitude'), \
                                  ("gas", 'velocity_magnitude'), \
                                  ("gas", 'velocity_x'), \
                                  ("gas", 'velocity_y'), \
                                  ("gas", 'velocity_z'), \
                                  ("gas", 'Si_p1_number_density'), ("gas", 'Si_p2_number_density'), ("gas", 'Si_p3_number_density'), ("gas", 'C_p1_number_density'), ("gas", 'C_p3_number_density'), ("gas", 'O_p5_number_density'), ("gas", 'Mg_p0_number_density'),("gas", 'Mg_p1_number_density'),("gas", 'H_p0_number_density'),("gas", 'H_p1_number_density'), \
                                  ("gas", 'Si_p1_mass'), ("gas", 'Si_p2_mass'), ("gas", 'Si_p3_mass'), ("gas", 'C_p1_mass'), ("gas", 'C_p3_mass'), ("gas", 'O_p5_mass'), ("gas", 'Mg_p0_mass'),("gas", 'Mg_p1_mass'),("gas", 'H_p0_mass'),("gas", 'H_p1_mass') \
                                  ]
                                  
            now = datetime.now()
            print(now)
            print("~~~~~~~saving clumptree")
              
            fn ='halo_00'+halo+'_'+sim+'_'+snap+'_'+snap+'_clumps_tree'
            os.system("mv " + fn + ".h5 " + fn + "_old.h5")
            master_clump.save_as_dataset(fn,fields=fields_of_interest)
            now = datetime.now()
            print(now)
            print("~~~~~~~saved clumptree")
              
            if args.individual==True:
              indclumpdir = output_dir +'individual_clumps'
              now = datetime.now()
              print(now)
              print("~~~~~~~removing old individual clump files before saving new ones")
              #os.rmdir(indclumpdir)
              os.system("rm -r " + indclumpdir)
              if not (os.path.exists(indclumpdir)): os.system('mkdir -p ' + indclumpdir)
              
              leaf_clumps = master_clump.leaves
              for clump in leaf_clumps:
                  clumpfn=str(clump.clump_id)+'_single_clump'
                  #clump.save_as_dataset(filename=clumpfn,fields=["density", "particle_mass",'particle_position'])
                  clump.data.save_as_dataset(filename=indclumpdir+'/'+clumpfn,fields=fields_of_interest)
                  now = datetime.now()
                  print(now)
                  print("~~~~~~~saved individual files")
                  
                  
                  filename = 'halo_00'+halo+'_'+sim+'_'+snap+'_'+snap+'_clumps_cut_region'
                  os.system("mv " + filename + ".h5 " + filename + "_old.h5")
                  master_clump.data.save_as_dataset(filename=filename,fields=fields_of_interest)
                  now = datetime.now()
                  print(now)
                  print("~~~~~~~saved clumps cut region")
