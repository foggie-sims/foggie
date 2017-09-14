import yt 
from astropy.table import Table 
import numpy as np 
import os 
import trident 

def plot_script(halo, run, axis, firstsnap, lastsnap, trackname): 

    wide = 250. 

    track_name = '/astro/simulations/FOGGIE/halo_00'+halo+'/'+trackname 
    print("opening track: "+track_name) 
    track = Table.read(track_name, format='ascii') 
    track.sort('col1') 

    outs = [x+firstsnap for x in range(lastsnap+1-firstsnap)]

    os.chdir('/astro/simulations/FOGGIE/halo_00'+halo+'/'+run) 
    prefix = '/Users/tumlinson/Dropbox/foggie/plots/halo_00'+halo+'/'+run+'/'

    for n in outs: 

        # load the snapshot 
        strset = 'DD00'+str(n) 
        if (n > 99): strset = 'DD0'+str(n) 
        snap_to_open = strset+'/'+strset  
        print('opening snapshot '+snap_to_open) 
        ds = yt.load(snap_to_open)
        trident.add_ion_fields(ds, ions=['C IV', 'O VI','H I','Si III'])
        zsnap = ds.get_parameter('CosmologyCurrentRedshift')

        def _msun_density(field, data):
            return data["density"]*1.0

        ds.add_field(("gas","Msun_density"),function=_msun_density, units="Msun/pc**3")
    
        # interpolate the center from the track
        centerx = np.interp(zsnap, track['col1'], track['col2']) 
        centery = np.interp(zsnap, track['col1'], track['col3']) 
        centerz = 0.5 * ( np.interp(zsnap, track['col1'], track['col4']) + np.interp(zsnap, track['col1'], track['col7'])) 
    
        center = [centerx, centery+20. / 143886., centerz] 
    
        x_left = np.interp(zsnap, track['col1'], track['col2']) 
        y_left = np.interp(zsnap, track['col1'], track['col3']) 
        z_left = np.interp(zsnap, track['col1'], track['col4']) 
        x_right = np.interp(zsnap, track['col1'], track['col5']) 
        y_right = np.interp(zsnap, track['col1'], track['col6']) 
        z_right = np.interp(zsnap, track['col1'], track['col7']) 
    
        refine_box_center = [0.5*(x_left+x_right), 0.5*(y_left+y_right), 0.5*(z_left+z_right)] 
        refine_box = ds.r[x_left:x_right, y_left:y_right, z_left:z_right] 
    
        box = ds.r[ center[0]-250./143886:center[0]+250./143886, center[1]-250./143886.:center[1]+250./143886., center[2]-250./143886.:center[2]+250./143886.]


        p = yt.ProjectionPlot(ds, axis, 'density', center=center, data_source=box, width=(wide, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="density", cmap='RdBu')
        p.set_zlim("density",1e-6,0.1)
        p.save(prefix+'plots/density_projection_map/'+axis+'/'+strset+'_box')

        p = yt.ProjectionPlot(ds, axis, 'density', center=[refine_box_center[0],refine_box_center[1],refine_box_center[2]], data_source=refine_box, width=(wide, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_unit('density', 'Msun / pc**2')
        p.set_zlim("density",0.01,1000)
        p.save(prefix+'plots/density_refine_map/'+axis+'/'+strset+'_refine')

        p = yt.SlicePlot(ds, axis, 'metallicity', center=[refine_box_center[0],center[1],center[2]], data_source=box, width=(wide, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="metallicity", cmap='rainbow')
        p.set_zlim("metallicity",1e-4,2.)
        p.save(prefix+'plots/metallicity_slice_map/'+axis+'/'+strset+'_box')
    
    
        p = yt.ProjectionPlot(ds, axis, 'H_p0_number_density', center=center, data_source=box, width=(wide,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_zlim("H_p0_number_density",1e11,1e17)
        p.save(prefix+'plots/HI_projection_map/'+axis+'/'+strset+'_HI')

        p = yt.ProjectionPlot(ds, axis, 'H_p0_number_density', center=center, data_source=refine_box, width=(wide,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_zlim("H_p0_number_density",1e11,1e17)
        p.save(prefix+'plots/HI_refine_map/'+axis+'/'+strset+'_HI_refine')
    
        p = yt.ProjectionPlot(ds, axis, "O_p5_number_density", center=center, data_source=box, width=(wide,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_zlim("O_p5_number_density",1e11,1e15)
        p.save(prefix+'plots/OVI_projection_map/'+axis+'/'+strset+'_OVI')
        p = yt.ProjectionPlot(ds, axis, "O_p5_number_density", center=center, data_source=refine_box, width=(wide,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_zlim("O_p5_number_density",1e11,1e15)
        p.save(prefix+'plots/OVI_refine_map/'+axis+'/'+strset+'_OVI_refine')
    
        p = yt.ProjectionPlot(ds, axis, "C_p3_number_density", center=center, data_source=box, width=(wide,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_zlim("C_p3_number_density",1e11,1e15)
        p.save(prefix+'plots/CIV_projection_map/'+axis+'/'+strset+'_CIV')
    
        p = yt.ProjectionPlot(ds, axis, "Si_p2_number_density", center=center, data_source=box, width=(wide,'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_zlim("Si_p2_number_density",1e11,1e15)
        p.save(prefix+'plots/SiIII_projection_map/'+axis+'/'+strset+'_SiIII')
    
        p = yt.ProjectionPlot(ds, axis, 'temperature', center=[refine_box_center[0],refine_box_center[1],refine_box_center[2]], data_source=refine_box, width=(wide, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="temperature", cmap='rainbow')
        p.set_zlim("temperature",1e29,2e30)
        p.save(prefix+'plots/temperature_refine_map/'+axis+'/'+strset+'_refine')
    
        p = yt.SlicePlot(ds, axis, 'entropy', center=[refine_box_center[0],refine_box_center[1],refine_box_center[2]], data_source=refine_box, width=(wide, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="entropy", cmap='rainbow')
        p.set_zlim("entropy",1e-4,1e3)
        p.save(prefix+'plots/entropy_slice_map/'+axis+'/'+strset+'_refine')
    
        p = yt.SlicePlot(ds, axis, 'temperature', center=[refine_box_center[0],refine_box_center[1],refine_box_center[2]], data_source=refine_box, width=(wide, 'kpc'))
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_cmap(field="temperature", cmap='rainbow')
        p.set_zlim("temperature",1e4,2e6) 
        p.save(prefix+'plots/temperature_slice_map/'+axis+'/'+strset+'_refine')
    
        p = yt.SlicePlot(ds, axis, 'metallicity', center=[refine_box_center[0],refine_box_center[1],refine_box_center[2]], data_source=refine_box, width=(wide, 'kpc'))
        p.set_cmap(field="metallicity", cmap='rainbow')
        p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True)
        p.set_zlim("metallicity",1e-4,2.0)
        p.save(prefix+'plots/metallicity_slice_map/'+axis+'/'+strset+'_refine')

    os.chdir('/astro/simulations/FOGGIE/'+halo+'/'+run) 
