import os, sys, argparse
import yt
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.foggie_load import *
from collections import OrderedDict

def parse_args():
    '''
    Parse command line arguments
    ''' 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='''\
                                Generate the cameras to use in Sunrise and make projection plots
                                of the data for some of these cameras. Then export the data within
                                the fov to a FITS file in a format that Sunrise understands.
                                ''')

    parser.add_argument('-system', '--system', metavar='system', type=str, action='store', \
                        help='Which system are you on? Default is Jase')
    parser.set_defaults(system="jase")

    parser.add_argument('--full_camera_suite', dest='full_camera_suite', action='store_true',
                        help='use the full suite of cameras?')
    parser.set_defaults(full_camera_suite=False)


    parser.add_argument('--use_halo_c_v', dest='use_halo_c_v', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(use_halo_c_v=False)


    parser.add_argument('--do_export', dest='do_export', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(do_export=False)


    parser.add_argument('--do_cameras', dest='do_cameras', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(do_cameras=False)




    parser.add_argument('--run', metavar='run', type=str, action='store',
                        help='which run? default is natural')
    parser.set_defaults(run="nref11c_nref9f")

    parser.add_argument('--halo', metavar='halo', type=str, action='store',
                        help='which halo? default is 8508 (Tempest)')
    parser.set_defaults(halo="8508")

    parser.add_argument('--pwd', dest='pwd', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(pwd=False)


    parser.add_argument('--output', metavar='output', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(output="RD0020")

    parser.add_argument('--cam_fov', dest='cam_fov', type=float, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(cam_fov=50.)

    parser.add_argument('--cam_dist', dest='cam_dist', type=float, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(cam_dist=100000.)


    parser.add_argument('--seed', dest='seed', type=float, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(seed=0.)




    args = parser.parse_args()
    return args



def generate_cameras(normal_vector, seed = 0, distance=100.0, fov=50.0, segments_random=7, segments_fixed=4, full_camera_suite = False):
    '''
    Set camera positions and orientations
    '''
    from yt.utilities.orientation import Orientation

    print( "\nGenerating cameras")
    
    north = np.array([0.,1.,0.])
    orient = Orientation(normal_vector=normal_vector, north_vector=north)
    R=np.linalg.inv(orient.inv_mat)



    if full_camera_suite:
        camera_set = OrderedDict([
                ['face',([0.,0.,1.],[0.,-1.,0],True)], #up is north=+y
                ['edge',([0.,1.,0.],[0.,0.,-1.],True)],#up is along z
                ['backface',([0.,0.,-1.],[0.,-1.,0],True)], #up is north=+y
                ['backedge',([0.,-1.,0.],[0.,0.,-1.],True)],#up is along z
                ['45',([0.,0.7071,0.7071],[0., 0., -1.],True)],
                ['Z-axis',([0.,0.,-1.],[0.,-1.,0],False)], #up is north=+y
                ['Y-axis',([0.,1.,0.],[0.,0.,-1.],False)],#up is along z
                ['X-axis',([1.,0.,0.],[0.,0.,-1.],False)],#up is along z
                ])  


        np.random.seed()
        ts_random = np.random.random(segments_random)*np.pi*2
        ps_random = array([np.math.acos(2*np.random.random()-1) for i in arange(segments_random)])

        np.random.seed(seed)
        ts_fixed = np.random.random(segments_fixed)*np.pi*2
        ps_fixed = array([np.math.acos(2*np.random.random()-1) for i in arange(segments_fixed)])

        ts = np.concatenate([ts_fixed, ts_random])
        ps = np.concatenate([ps_fixed, ps_random])

        for i,(theta, phi) in enumerate(zip(ts,ps)):
            print( theta, phi)
            pos = [np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),np.cos(phi)]
            vc = [np.cos(np.pi/2 - theta)*np.sin(np.pi/2-phi),np.sin(np.pi/2 - theta)*np.sin(np.pi/2-phi),np.cos(np.pi/2-phi)]
            
            if i < segments_fixed:
                camera_set['Fixed_%03i'%(i)]=(pos,vc,False)
            else:
                camera_set['Random_%03i'%(i-segments_fixed)]=(pos,vc,False)

    else:
        camera_set = OrderedDict([
                                ['face',([0.,0.,1.],[0.,-1.,0],True)], #up is north=+y
                                ['edge',([0.,1.,0.],[0.,0.,-1.],True)]
                                ])  

    i=0  
    cameras = OrderedDict()
    for name,(normal,north,do_rot)  in camera_set.items():
        print( name, normal, north, do_rot)
        
        orient = Orientation(normal_vector=normal, north_vector=north)
        if do_rot:
            drot = R.copy()
        else:
            drot = np.identity(3)
        sunrise_pos = np.dot(orient.normal_vector, drot)
        if name == 'Z-axis':
            sunrise_up = np.asarray([0.0,1.0,0.0])
        elif name == 'X-axis':
            sunrise_up = np.asarray([0.0,1.0,0.0])
        elif name =='Y-axis':
            sunrise_up = np.asarray([0.0,0.0,1.0])
        else:
            sunrise_up  = normal_vector.copy()

            
        if np.all(np.abs(sunrise_up-sunrise_pos)<1e-3):
            sunrise_up[0] *= 0.5 
        sunrise_direction = -1.0*sunrise_pos
        sunrise_afov = 2.0*np.arctan((fov/2.0)/distance)
        norm = lambda x: x/np.sqrt(np.sum(x*x))
        if np.all(np.abs(norm(sunrise_up)-norm(sunrise_pos))<1e-3):
            sunrise_up[0]*=0.5
            sunrise_up = norm(sunrise_up)
        line = (distance*sunrise_pos, distance*sunrise_direction, sunrise_up,
                sunrise_afov, fov, distance) 
        cameras[name] = line
        i+=1

    print( "Successfully generated cameras\n")
    return cameras

def write_cameras(prefix, cameras):
    print( "Writing cameras to ",  prefix+'.cameras')
    fn = prefix + '.cameras'
    campos = ()
    for name,row in cameras.items():
        campos += (tuple(row[1])+tuple(row[0])+tuple(row[2])+tuple([row[3]]),)
    campos = np.array(campos)
    np.savetxt(fn, campos)   
    fn =  prefix+'.camnames'
    fh = open(fn,'w')
    fh.write('\n'.join([c for c in cameras.keys()]))
    fh.close()


def export_fits(ds, center, export_radius, prefix, star_particles, max_level=None, no_gas_p = False, form = 'Enzo'):
    '''
    Convert the contents of a dataset to a FITS file format that Sunrise
    understands.
    '''
    import sunrise_octree_exporter

    print( "\nExporting data in %s to FITS for Sunrise"%ds.parameter_filename.split('/')[-1])
    
    filename = prefix+'.fits'
    center = center.in_units('kpc')
    width = export_radius.in_units('kpc')
    info = {}
    
    fle, fre, ile, ire, nrefined, nleafs, nstars, output, output_array = \
                                                                     sunrise_octree_exporter.export_to_sunrise(ds, filename, star_particles,  center, width, max_level=max_level, grid_structure_fn = prefix+'_grid_struct.npy', no_gas_p = no_gas_p, form=form)
    
    info['export_ile']=ile
    info['export_ire']=ire
    info['export_fle']=fle
    info['export_fre']=fre
    
    
    info['export_center']=center.value
    info['export_radius']=width.value
    info['export_max_level']=max_level
    info['export_nstars']=nstars
    info['export_nrefined']=nrefined
    info['export_nleafs']=nleafs
    info['input_filename']=filename
    
    print( "Successfully generated FITS for snapshot %s"%ds.parameter_filename.split('/')[-1])
    print( info,'\n')
    #return info,  output, output_array
    return info  #output arrays not actually used later



if __name__ == "__main__":
    args = parse_args()

    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    code_path = trackname.split('halo_tracks')[0]  
    track_dir = code_path + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    halo_c_v_name = track_dir + 'halo_c_v'
    snap_name = foggie_dir + run_loc + args.output + '/' + args.output
    ds, refine_box, refine_box_center, refine_width = load(snap_name, trackname, use_halo_c_v=args.use_halo_c_v, halo_c_v_name=halo_c_v_name)

    export_radius = ds.arr(1.2*args.cam_fov, 'kpc')
    gal_center = ds.halo_center_kpc

    prefix = args.run + '_' + args.halo + '_' + args.output
    L = np.array([1,0,0])
    if args.do_cameras:
        cameras = generate_cameras(normal_vector = L, seed = args.seed, distance = args.cam_dist, fov = args.cam_fov)
        write_cameras(prefix, cameras)

    if args.do_export:

        export_info = export_fits(ds, gal_center, export_radius, 
                                  prefix, star_particles = 'stars', form='ENZO')
        export_info['halo'] = args.halo
        export_info['run'] = args.run
        export_info['output'] = args.output
        export_info_file = prefix + '_export_info.npy'
        np.save(export_info_file, export_info)

















































