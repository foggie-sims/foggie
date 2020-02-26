import os
import sys
import numpy as np
import glob
from numpy import *

'''
Generate the configuration files required to run Sunrise and
setup the Sunrise simulation directory with everything
necessary to submit.

'''


def generate_sfrhist_config(run_dir, filename, stub_name, fits_file, center_kpc, run_type, sunrise_data_dir, nthreads='1', idx = None):

    sf = open(run_dir+'/'+filename,'w+')
    sf.write('#Parameter File for Sunrise, sfrhist\n\n')
    sf.write('include_file              %s\n\n'%stub_name)
    sf.write('snapshot_file             %s\n'%fits_file)
    sf.write('output_file               %s\n\n'%(run_dir+'/sfrhist.fits'))
    sf.write('n_threads                 '+nthreads+'\n')

    sf.write('translate_origin          %.2f\t%.2f\t%.2f         / [kpc]\n'%(center_kpc[0], center_kpc[1], center_kpc[2]))

    if run_type == 'images':
        sf.write('min_wavelength            %s\n'%("0.02e-6"))
        sf.write('max_wavelength            %s\n\n'%("5.0e-6"))

        sf.write('mappings_sed_file         %s\n'%("%s/Smodel-lores128.fits"%sunrise_data_dir))
        sf.write('stellarmodelfile          %s\n'%("%s/Patrik-imfKroupa-Zmulti-ml.fits"%sunrise_data_dir))


    elif run_type == 'ifu':
        sf.write('min_wavelength            %s\n'%("0.6450e-6"))
        sf.write('max_wavelength            %s\n\n'%("0.6650e-6"))

        sf.write('mappings_sed_file         %s\n'%("%s/Smodel_full_hires.fits"))
        sf.write('stellarmodelfile          %s\n'%("%s/logspace-Patrik-imfKroupa-geneva-Zmulti-hires.fits"))

    elif run_type == 'grism':
        sf.write('min_wavelength            %s\n'%("0.02e-6"))
        sf.write('max_wavelength            %s\n\n'%("5.0e-6"))

        sf.write('mappings_sed_file         %s\n'%("%s/Mappings_Smodels_gfs.fits"))
        sf.write('stellarmodelfile          %s\n'%("%s/GFS_combined_nolines.fits"))   #or Patrik's hires.fits inputs



    sf.close()
    print('\t\tSuccessfully generated %s'%filename)

    return




def generate_mcrx_config(run_dir, filename, stub_name, redshift, run_type, nthreads='1',cam_file=''):
    mf = open(run_dir+'/'+filename,'w+')

    mf.write('#Parameter File for Sunrise, mcrx\n\n')
    mf.write('include_file         %s\n\n'%stub_name)
    mf.write('input_file           %s\n'%(run_dir+'/sfrhist.fits'))
    mf.write('output_file          %s\n'%(run_dir+'/mcrx.fits'))
    mf.write('n_threads                 '+nthreads+'\n')
    mf.write('camera_positions      %s\n'%(cam_file))

    if run_type != 'ifu':
        mf.write('use_kinematics       %s\n'%('false #True for IFU'))
    else:
        mf.write('use_kinematics       %s\n'%('true #False for images'))

    #move npixels to .config file

    if run_type == 'images':
        mf.write('npixels     800\n')
    elif run_type == 'ifu':
        mf.write('npixels     400\n')
    else:
        mf.write('npixels     200\n')


    mf.close()



    print('\t\tSuccessfully generated %s'%filename)

    return


def generate_broadband_config_images(run_dir, filename, stub_name, redshift, sunrise_data_dir):

    #copy sunrise filter folder to snap_dir+'/inputs/sunrise_filters/'

    bf = open(run_dir+'/'+filename,'w+')


    bf.write('#Parameter File for Sunrise, broadband\n\n')
    bf.write('include_file                      %s\n\n'%stub_name)
    bf.write('redshift                          %.3f\n\n'%redshift)
    bf.write('input_file                        %s\n'%(run_dir+'/mcrx.fits'))
    bf.write('output_file                       %s\n'%(run_dir+'/broadband.fits'))
    bf.write('filter_list                       %s\n'%('%s/sunrise_filters/filters_rest'%sunrise_data_dir))
    bf.write('filter_file_directory             %s\n'%('%s/sunrise_filters/'%sunrise_data_dir))
    bf.close()

    bfz = open(run_dir+'/'+filename.replace('broadband','broadbandz'),'w+')


    bfz.write('#Parameter File for Sunrise, broadband\n\n')
    bfz.write('include_file                      %s\n\n'%stub_name)
    bfz.write('redshift                          %.3f\n\n'%redshift)
    bfz.write('input_file                        %s\n'%(run_dir+'/mcrx.fits'))
    bfz.write('output_file                       %s\n'%(run_dir+'/broadbandz.fits'))
    bfz.write('filter_list                       %s\n'%('/u/gfsnyder/sunrise_data/sunrise_filters/filters_st'))
    bfz.write('filter_file_directory             %s\n'%('/u/gfsnyder/sunrise_data/sunrise_filters/'))
    bfz.close()




    print('\t\tSuccessfully generated %s'%filename)

    return


def generate_broadband_config_grism(run_dir, filename, stub_name, redshift, sunrise_data_dir):

    #copy sunrise filter folder to snap_dir+'/inputs/sunrise_filters/'
    #I uploaded these to '~gfsnyder/sunrise_data/' on Pleiades

    bfg = open(run_dir+'/'+filename.replace('broadband','broadbandgrism'),'w+')
    bfg.write('#Parameter File for Sunrise, broadband\n\n')
    bfg.write('include_file                      %s\n\n'%stub_name)
    bfg.write('redshift                          %.3f\n\n'%redshift)
    bfg.write('input_file                        %s\n'%(run_dir+'/mcrx.fits'))
    bfg.write('output_file                       %s\n'%(run_dir+'/grism.fits'))
    bfg.write('filter_list                       %s\n'%('%s/sunrise_filters/filters_grism'%sunrise_data_dir))
    bfg.write('filter_file_directory             %s\n'%('%s/sunrise_filters/'%sunrise_data_dir))
    bfg.close()




    print('\t\tSuccessfully generated %s'%filename)

    return




def generate_qsub(run_dir, filename, run_type, ncpus='12', model='wes', queue='normal',email='rsimons@jhu.edu',walltime='04:00:00',candelize=False):

    bsubf = open(run_dir+'/'+filename, 'w+')
    bsubf.write('#!/bin/bash\n')
    bsubf.write('#PBS -S /bin/bash\n')   #apparently this is a thing
    bsubf.write('#PBS -l select=1:ncpus='+ncpus+':model='+model+'\n')   #selects cpu model and number (sunrise uses 1 node)
    bsubf.write('#PBS -l walltime='+walltime+'\n')    #hh:mm:ss before job is killed
    bsubf.write('#PBS -q '+queue+'\n')       #selects queue to submit to 
    bsubf.write('#PBS -N sunrise_'+run_type+'\n')     #selects job name
    bsubf.write('#PBS -M '+email+'\n')  #notifies job info to this email address 
    bsubf.write('#PBS -m abe\n')  #set notification types (abe=abort, begin, end)
    bsubf.write('#PBS -o '+run_dir+'/sunrise_pbs.out\n')  #save standard output here
    bsubf.write('#PBS -e '+run_dir+'/sunrise_pbs.err\n')  #save standard error here
    bsubf.write('#PBS -V\n')    #export environment variables at start of job

    bsubf.write('cd '+run_dir+' \n')   #go to directory where job should run
    bsubf.write('/u/gfsnyder/bin/sfrhist sfrhist.config > sfrhist.out 2> sfrhist.err\n')
    bsubf.write('/u/gfsnyder/bin/mcrx mcrx.config > mcrx.out 2> mcrx.err\n')
    if run_type=='images':
        bsubf.write('/u/gfsnyder/bin/broadband broadbandz.config > broadbandz.out 2> broadbandz.err\n')
        bsubf.write('/u/gfsnyder/bin/broadband broadband.config > broadband.out 2> broadband.err\n')
        bsubf.write('rm -rf sfrhist.fits\n')   #enable this after testing
        bsubf.write('rm -rf mcrx.fits\n')   #enable this after testing
        if candelize==True:
                bsubf.write(os.path.expandvars('python $SYNIMAGE_CODE/candelize.py\n'))
        bsubf.write('pigz -9 -p '+str(ncpus)+' broadband.fits\n')
    elif run_type=='ifu':
        #bsubf.write('rm -rf sfrhist.fits\n')   #enable this after testing
                bsubf.write('gzip -9 mcrx.fits\n')
    elif run_type=='grism':
        bsubf.write('/u/gfsnyder/bin/broadband broadbandgrism.config > broadbandgrism.out 2> broadbandgrism.err\n')
        #bsubf.write('rm -rf sfrhist.fits\n')   #enable this after testing
        #bsubf.write('rm -rf mcrx.fits\n')   #enable this after testing

    bsubf.close()

    print('\t\tSuccessfully generated %s'%filename)



    return os.path.abspath(run_dir+'/'+filename)


def generate_candelize_qsub(run_dir, filename, run_type, ncpus='12', model='wes', queue='normal',email='rsimons@jhu.edu',walltime='04:00:00'):

    bsubf = open(run_dir+'/'+filename, 'w+')
    bsubf.write('#!/bin/bash\n')
    bsubf.write('#PBS -S /bin/bash\n')   #apparently this is a thing
    bsubf.write('#PBS -l select=1:ncpus=4:model=ivy\n')   #selects cpu model and number (sunrise uses 1 node)
    bsubf.write('#PBS -l walltime=05:00:00\n')    #hh:mm:ss before job is killed
    bsubf.write('#PBS -q '+queue+'\n')       #selects queue to submit to 
    bsubf.write('#PBS -N candelize_'+run_type+'\n')     #selects job name
    bsubf.write('#PBS -M '+email+'\n')  #notifies job info to this email address 
    bsubf.write('#PBS -m abe\n')  #set notification types (abe=abort, begin, end)
    bsubf.write('#PBS -o '+run_dir+'/candelize_pbs.out\n')  #save standard output here
    bsubf.write('#PBS -e '+run_dir+'/candelize_pbs.err\n')  #save standard error here
    bsubf.write('#PBS -V\n')    #export environment variables at start of job

    bsubf.write('cd '+run_dir+' \n')   #go to directory where job should run

    if run_type=='images':
                bsubf.write(os.path.expandvars('python $SYNIMAGE_CODE/candelize.py\n'))

    bsubf.close()

    print('\t\tSuccessfully generated %s'%filename)



    return os.path.abspath(run_dir+'/'+filename)


if __name__ == "__main__":

    if len(sys.argv)==2:
        snaps = np.asarray([sys.argv[1]])
    else:
        snaps = np.asarray(glob.glob("*.d"))


    #I'd suggest moving nthreads to the config files and passing this to the sfrhist and mcrx config creators
    nthreads = '24'  #comet
    queue='normal'   #options devel, debug, low, normal, long
    notify='gsnyder@stsci.edu'
    walltime_limit='02:00:00'

    stub_dir = '/u/gfsnyder/sunrise_data/stub_files'

    #list_of_types = ['images','ifu','grism']
    list_of_types = ['images']
    
    print("Generating Sunrise Runs for: ", snaps)

    abssnap = os.path.abspath(snaps[0])
    assert os.path.lexists(abssnap)

    dirname = os.path.dirname(abssnap)
    simname = os.path.basename(dirname) #assumes directory name for simulation name
    print("Simulation name:  ", simname)

    smf_images = open('submit_sunrise_images_gfs.sh','w')
    smf_ifu = open('submit_sunrise_ifu_gfs.sh','w')
    smf_grism = open('submit_sunrise_grism_gfs.sh','w')
    smf_candelize = open('submit_sunrise_candelize_gfs.sh','w')
    
    new_snapfiles = []

    for sn in snaps:
        aname = sn.split('_')[-1].rstrip('.d')

        snap_dir = os.path.join(simname+'_'+aname+'_sunrise')

        print("Sunrise directory: ", snap_dir)
        assert os.path.lexists(snap_dir)

        newf = os.path.join(snap_dir,sn)
        new_snapfiles.append(newf)
        assert os.path.lexists(newf)


    new_snapfiles = np.asarray(new_snapfiles)

    for isnap, snapfile in enumerate(new_snapfiles):
        snap_dir = os.path.abspath(os.path.dirname(snapfile))
        sunrise_dir = os.path.basename(snap_dir)
        snap_name = sunrise_dir.rstrip('_sunrise')

        fits_file = snap_dir+'/input/%s.fits'%(snap_name)
        info_file = fits_file.replace('.fits', '_export_info.npy')
        cam_file = fits_file.replace('.fits','.cameras')

        prop_file = os.path.abspath(simname+'_galprops.npy')

        #Clean exit for galaxies with no prop file
        if os.path.lexists(fits_file) and os.path.lexists(cam_file):
            print(prop_file)
            print(os.path.lexists(prop_file))
            assert os.path.lexists(prop_file), 'Prop file %s not found'%prop_file
            assert os.path.lexists(fits_file), 'Fits file %s not found'%fits_file
            assert os.path.lexists(info_file), 'Info file %s not found'%info_file
            assert os.path.lexists(cam_file), 'Cam file %s not found'%cam_file

            
            print ('\tFits file name: %s'%fits_file)
            print ('\tInfo file name: %s\n'%info_file)

            galprops_data = np.load(prop_file)[()]
            idx = np.argwhere(galprops_data['snap_files']==os.path.abspath(snapfile))[0][0]


            for run_type in list_of_types:
                run_dir = snap_dir+'/%s'%run_type
                if not os.path.lexists(run_dir):
                        os.mkdir(run_dir)
                        
                print ('\tGenerating sfrhist.config file for %s...'%run_type)
                sfrhist_fn   = 'sfrhist.config'
                sfrhist_stub = os.path.join(stub_dir,'sfrhist_base.stub')

                generate_sfrhist_config(run_dir = run_dir, filename = sfrhist_fn, 
                                        stub_name = sfrhist_stub,  fits_file = fits_file, 
                                        galprops_data = galprops_data, run_type = run_type, nthreads=nthreads, idx = idx)


                print ('\tGenerating mcrx.config file for %s...'%run_type)
                mcrx_fn   = 'mcrx.config'
                mcrx_stub = os.path.join(stub_dir,'mcrx_base.stub')

                generate_mcrx_config(run_dir = run_dir, snap_dir = snap_dir, filename = mcrx_fn, 
                                     stub_name = mcrx_stub,
                                     galprops_data = galprops_data, run_type = run_type, nthreads=nthreads, cam_file=cam_file, idx = idx)



                if run_type == 'images': 
                        print ('\tGenerating broadband.config file for %s...'%run_type)
                        broadband_fn   = 'broadband.config'
                        broadband_stub = os.path.join(stub_dir,'broadband_base.stub')

                        generate_broadband_config_images(run_dir = run_dir, snap_dir = snap_dir, filename = broadband_fn, 
                                                         stub_name = broadband_stub, 
                                                         galprops_data = galprops_data, idx = idx)
                if run_type == 'grism': 
                        print ('\tGenerating broadband.config file for %s...'%run_type)
                        broadband_fn   = 'broadband.config'
                        broadband_stub = os.path.join(stub_dir, 'broadband_base.stub')

                        generate_broadband_config_grism(run_dir = run_dir, snap_dir = snap_dir, filename = broadband_fn, 
                                                        stub_name = broadband_stub, 
                                                        galprops_data = galprops_data, idx = idx)





                print ('\tGenerating sunrise.qsub file for %s...'%run_type)
                qsub_fn   = 'sunrise.qsub'      
                final_fn = generate_qsub(run_dir = run_dir, snap_dir = snap_dir, filename = qsub_fn, 
                                         galprops_data = galprops_data, run_type = run_type,ncpus=nthreads,model=model,queue=queue,email=notify,walltime=walltime_limit, isnap=isnap)
                can_qsub_fn   = 'candelize.qsub'        
                can_final_fn = generate_candelize_qsub(run_dir = run_dir, snap_dir = snap_dir, filename = can_qsub_fn, 
                                         galprops_data = galprops_data, run_type = run_type,ncpus=nthreads,model=model,queue=queue,email=notify,walltime=walltime_limit, isnap=isnap)
                
                submitline = 'qsub '+final_fn
                can_submitline = 'qsub '+can_final_fn
                
                if run_type=='images':
                        smf_images.write(submitline+'\n')
                        smf_candelize.write(can_submitline+'\n')
                if run_type=='ifu':
                        smf_ifu.write(submitline+'\n')
                if run_type=='grism':
                        smf_grism.write(submitline+'\n')


    smf_images.close()
    smf_ifu.close()
    smf_grism.close()
    smf_candelize.close()
    










































