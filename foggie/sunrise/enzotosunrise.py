import os, sys, argparse
import yt
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.foggie_load import *
from collections import OrderedDict
from pathlib import Path
import time

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
    parser.set_defaults(system="pleiades_raymond")

    parser.add_argument('--full_camera_suite', dest='full_camera_suite', action='store_true',
                        help='use the full suite of cameras?')
    parser.set_defaults(full_camera_suite=False)


    parser.add_argument('--use_halo_c_v', dest='use_halo_c_v', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(use_halo_c_v=False)


    parser.add_argument('--do_export', dest='do_export', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(do_export=False)

    parser.add_argument('--do_setup', dest='do_setup', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(do_setup=False)

    parser.add_argument('--do_cameras', dest='do_cameras', action='store_true',
                        help='just use the pwd?, default is no')
    parser.set_defaults(do_cameras=False)


    parser.add_argument('--sunrise_directory', dest='sunrise_directory', type = str, action='store',
                        help='just use the pwd?, default is no')
    parser.set_defaults(sunrise_directory='/nobackupp2/rcsimons/sunrise/foggie')

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





    parser.add_argument('--nthreads', metavar='nthreads', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(nthreads="24")

    parser.add_argument('--queue', metavar='queue', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(queue="normal")


    parser.add_argument('--notify', metavar='notify', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(notify="rsimons@stsci.edu")


    parser.add_argument('--walltime_limit', metavar='walltime_limit', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(walltime_limit="02:00:00")


    parser.add_argument('--stub_dir', metavar='stub_dir', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(stub_dir="/u/gfsnyder/sunrise_data/stub_files")



    parser.add_argument('--model', metavar='model', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(model="has")


    parser.add_argument('--sunrise_data_dir', metavar='sunrise_data_dir', type=str, action='store',
                        help='which output? default is RD0020')
    parser.set_defaults(sunrise_data_dir="/u/gfsnyder/sunrise_data")



    args = parser.parse_args()
    return args

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


def generate_cameras(normal_vector, prefix, seed = 0, distance=100.0, fov=50.0, segments_random=7, segments_fixed=4, full_camera_suite = False):
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
                                ['Z-axis',([0.,0.,-1.],[0.,-1.,0],False)], #up is north=+y
                                ['Y-axis',([0.,1.,0.],[0.,0.,-1.],False)],#up is along z
                                ['X-axis',([1.,0.,0.],[0.,0.,-1.],False)],#up is along z

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

    write_cameras(prefix, cameras)
    print( "Successfully generated cameras\n")
    return cameras



def export_fits(ds, center, export_radius, prefix, args, star_particles, max_level=None, no_gas_p = False, form = 'Enzo'):
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
    

    info['halo'] = args.halo
    info['run'] = args.run
    info['output'] = args.output

    np.save(prefix + '_export_info.npy', info)


    print( "Successfully generated FITS for snapshot %s"%ds.parameter_filename.split('/')[-1])
    print( info,'\n')
    #return info,  output, output_array
    return info  #output arrays not actually used later


def check_paths(args):
    if not os.path.exists(args.sunrise_directory): 
        raise ValueError('The specified sunrise directory does not exist: %s. Pass an existing directory into --sunrise_directory'%args.sunrise_directory)
    output_directory = args.sunrise_directory + '/' + args.halo + '/' + args.run + '/' + args.output
    #checking for subdirectories, if they do not exist, create them
    Path(output_directory + '/inputs').mkdir(parents = True, exist_ok = True)
    prefix = output_directory + '/inputs/' + args.run + '_' + args.halo + '_' + args.output

    return output_directory, prefix







def setup_runs(ds, args, prefix, output_directory, list_of_types = ['images', 'grism', 'ifu']):

    import setupSunriseRun as ssR




    smf_images = open('%s/submit_sunrise_images.sh'%output_directory,'w')
    smf_ifu = open('%s/submit_sunrise_ifu.sh'%output_directory,'w')
    smf_grism = open('%s/submit_sunrise_grism.sh'%output_directory,'w')


    fits_file        = prefix + '.fits'
    export_info_file = prefix + '_export_info.npy'
    cam_file         = prefix + '.cameras'

    assert os.path.lexists(fits_file), 'Fits file %s not found'%fits_file
    assert os.path.lexists(export_info_file), 'Info file %s not found'%export_info_file
    assert os.path.lexists(cam_file),  'Cam file %s not found'%cam_file



    print ('\tFits file name: %s'%fits_file)
    print ('\tInfo file name: %s\n'%export_info_file)

    for run_type in list_of_types:
            run_dir = output_directory+'/%s'%run_type
            if not os.path.lexists(run_dir): os.mkdir(run_dir)
                    
            print ('\tGenerating sfrhist.config file for %s...'%run_type)
            sfrhist_fn   = 'sfrhist.config'
            sfrhist_stub = os.path.join(args.stub_dir,'sfrhist_base.stub')

            ssR.generate_sfrhist_config(run_dir = run_dir, 
                                        filename = sfrhist_fn, 
                                        stub_name = sfrhist_stub, 
                                        fits_file = fits_file, 
                                        center_kpc = ds.halo_center_kpc, 
                                        run_type = run_type, 
                                        sunrise_data_dir = args.sunrise_data_dir, 
                                        nthreads=args.nthreads)


            print ('\tGenerating mcrx.config file for %s...'%run_type)
            mcrx_fn   = 'mcrx.config'
            mcrx_stub = os.path.join(args.stub_dir,'mcrx_base.stub')

            ssR.generate_mcrx_config(run_dir = run_dir, filename = mcrx_fn, 
                                     stub_name = mcrx_stub, redshift = ds.current_redshift, 
                                     run_type = run_type, nthreads=args.nthreads, cam_file=cam_file)




            if run_type == 'images': 
                    print ('\tGenerating broadband.config file for %s...'%run_type)
                    broadband_fn   = 'broadband.config'
                    broadband_stub = os.path.join(args.stub_dir,'broadband_base.stub')

                    ssR.generate_broadband_config_images(run_dir = run_dir, 
                                                         filename = broadband_fn, 
                                                         stub_name = broadband_stub, 
                                                         redshift = ds.current_redshift,
                                                         sunrise_data_dir = args.sunrise_data_dir)
            if run_type == 'grism': 
                    print ('\tGenerating broadband.config file for %s...'%run_type)
                    broadband_fn   = 'broadband.config'
                    broadband_stub = os.path.join(args.stub_dir, 'broadband_base.stub')

                    ssR.generate_broadband_config_grism(run_dir = run_dir, 
                                                        filename = broadband_fn, 
                                                        stub_name = broadband_stub, 
                                                        redshift = ds.current_redshift,
                                                        sunrise_data_dir = args.sunrise_data_dir)


            print ('\tGenerating sunrise.qsub file for %s...'%run_type)
            qsub_fn   = 'sunrise.qsub'      
            final_fn = ssR.generate_qsub(run_dir = run_dir, filename = qsub_fn, run_type = run_type,
                                         ncpus=args.nthreads,model=args.model,queue=args.queue,
                                         email=args.notify,walltime=args.walltime_limit)
            
            submitline = 'qsub '+final_fn                
            if run_type=='images': smf_images.write(submitline+'\n')
            if run_type=='ifu'   : smf_ifu.write(submitline+'\n')
            if run_type=='grism' : smf_grism.write(submitline+'\n')


    smf_images.close()
    smf_ifu.close()
    smf_grism.close()



def load_sim(args):
    foggie_dir, output_dir, run_loc, trackname, haloname, spectra_dir, infofile = get_run_loc_etc(args)
    track_dir =  trackname.split('halo_tracks')[0]   + 'halo_infos/00' + args.halo + '/' + args.run + '/'
    snap_name = foggie_dir + run_loc + args.output + '/' + args.output
    ds, refine_box, refine_box_center, refine_width = load(snap = snap_name, 
                                                           trackfile = trackname, 
                                                           use_halo_c_v=args.use_halo_c_v, 
                                                           halo_c_v_name=track_dir + 'halo_c_v')
    return ds

if __name__ == "__main__":
    args = parse_args()
    output_directory, prefix = check_paths(args)
    ds = load_sim(args)

    if args.do_cameras: 
        cameras = generate_cameras(normal_vector = np.array([1,0,0]), 
                                   prefix = prefix,
                                   seed = args.seed, 
                                   distance = args.cam_dist, 
                                   fov = args.cam_fov)

    if args.do_export:
        export_info = export_fits(ds = ds, 
                                  center = ds.halo_center_kpc, 
                                  export_radius =  ds.arr(1.2*args.cam_fov, 'kpc'), 
                                  prefix = prefix, 
                                  args = args, 
                                  star_particles = 'stars', 
                                  form='ENZO')

    if args.do_setup:
        setup_runs(ds = ds, 
                   args = args, 
                   prefix = prefix, 
                   output_directory = output_directory)
































