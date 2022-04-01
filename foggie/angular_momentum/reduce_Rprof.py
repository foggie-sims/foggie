import os
from glob import glob
from joblib import Parallel, delayed
import argparse
import numpy as np
from numpy import *


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-profdir', '--profdir', metavar='profdir', type=str, action='store',
                        default = "/nobackupp2/rcsimons/foggie/angular_momentum/profiles")

    parser.add_argument('-halo', '--halo', metavar='halo', type=str, action='store',
                        default = "8508")

    parser.add_argument('-cores', '--cores', metavar='cores', type=int, action='store',
                        default=1)

    args = parser.parse_args()
    return args




def reduce_rprof(DD_fl, args, situations):
    fl = DD_fl.replace(DD_fl.split('/')[-1], 'rdist/'+ DD_fl.split('/')[-1].replace('.npz', '_rdist.npy'))
    if os.path.exists(fl): return
    print (fl)
    try: Lprof = np.load(DD_fl, allow_pickle = True)['a'][()]
    except: return
    c = {}
  
    mass_types = ['cold', 'warm', 'warmhot', 'hot', 'stars', 'young_stars', 'dm']
    c['props'] = Lprof['props']
    for mtype in mass_types: 
        c[mtype] = {}
        c[mtype]['rprof'] = Lprof[mtype]['rprof']
        for situation in situations:
            for map_type in ['L', 'M']:
                print (mtype, situation)
                '''
                gas variables
                'variables': [('index', 'cylindrical_radius'),
                ('index', 'cylindrical_z'),
                ('gas', 'radial_velocity'),
                ('gas', 'metallicity'),
                'thel',
                'phil']}}
		particle variables
  		'variables': [('stars', 'particle_position_cylindrical_radius'),
   		('stars', 'particle_position_cylindrical_z'),
  		('stars', 'particle_radial_velocity'),
   		'thel',
   		'phil']}}

                '''
                if situation == 'galaxy_soutflow':
                    hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][:20,3:4]
                    
                    if ('stars' in mtype) | ('dm' in mtype): hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][:20,3:]

                if situation == 'galaxy_foutflow':
                    hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][:20,4:]
                    if ('stars' in mtype) | ('dm' in mtype): continue



                if situation == 'galaxy_sinflow':
                    ###vr < 0 km/s
                    hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][:20,2:3]
                    if ('stars' in mtype) | ('dm' in mtype): hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][:20,:3]                    
                if situation == 'galaxy_finflow':
                    ###vr < -100 km/s
                    hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][:20,:2]
                    if ('stars' in mtype) | ('dm' in mtype): continue


                if situation == 'cgm_soutflow':
                    hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][20:,3:4]
                    if ('stars' in mtype) | ('dm' in mtype): hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][20:,3:]
                
                if situation == 'cgm_foutflow':
                    hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][20:,4:]
                    if ('stars' in mtype) | ('dm' in mtype): continue
                    


                if situation == 'cgm_sinflow':
                    ###vr < 0 km/s
                    hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][20:,2:3]
                    if ('stars' in mtype) | ('dm' in mtype): hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][20:,:3]
                if situation == 'cgm_finflow':
                    ###vr < -100 km/s
                    hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][20:,:2]
                    if ('stars' in mtype) | ('dm' in mtype): continue

                    



                if situation == 'cgm_zpoor_inflow':
                    ###metal-poor inflow, Z < 0.02 Zsun & vr < -100 km/s
                    hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][20:,:2,:1]
                    if (mtype != 'cold') & (mtype != 'warm'): continue
                    if ('stars' in mtype) | ('dm' in mtype): continue
                if situation == 'galaxy_zpoor_inflow':
                    ###metal-poor inflow, Z < 0.02 Zsun & vr < -100 km/s
                    hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][:20,:2,:1]
                    if (mtype != 'cold') & (mtype != 'warm'): continue
                    if ('stars' in mtype) | ('dm' in mtype): continue

                if situation == 'galaxy':
                    hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][:20]
                
                if situation == 'cgm':
                    hst_full = Lprof[mtype]['r_dist']['%s_hst'%map_type][20:]


                dim_tuple = tuple(np.arange(hst_full.ndim-2))
                hst_center  = np.rot90(np.nansum(hst_full, axis = dim_tuple))
                
                if ('stars' in mtype) | ('dm' in mtype):
                    situation_use = situation.replace('sinflow', 'inflow').replace('soutflow', 'outflow')
                else:  situation_use = situation
                if not situation_use in c[mtype]: c[mtype][situation_use] = {}
                c[mtype][situation_use][map_type] = hst_center


 
    np.save(fl, c)



if __name__ == '__main__':
    args = parse_args()
    DD_fls = glob(args.profdir + '/' + args.halo + '/*npz')
    #situations = ['galaxy_sinflow', 'galaxy_minflow', 'galaxy_finflow', \
    #              'galaxy_soutflow', 'galaxy_moutflow', 'galaxy_foutflow', \
    #              'cgm_sinflow', 'cgm_minflow', 'cgm_finflow', \
    #              'cgm_soutflow', 'cgm_moutflow', 'cgm_foutflow'\
    #              'cgm_zpoor_inflow', 'galaxy_zpoor_inflow']             
    situations = ['galaxy_sinflow',  'galaxy_finflow',  \
                  'galaxy_soutflow', 'galaxy_foutflow',  \
                  'cgm_sinflow',     'cgm_finflow', \
                  'cgm_soutflow',    'cgm_foutflow',\
                  'cgm_zpoor_inflow', 'galaxy_zpoor_inflow']





    #for DD_fl in DD_fls: reduce_rprof(DD_fl, args, situations)
    Parallel(n_jobs = args.cores, backend='multiprocessing')(delayed(reduce_rprof)(DD_fl, args, situations) for DD_fl in DD_fls)
