import yt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from astropy.table import Table 
import numpy as np 
import trident 

def multiplot(): 

    field = 'density'  
    view = 'projection' 
    field = 'O_p5_number_density'

    track = Table.read('complete_track', format='ascii') 
    track.sort('col1') 

    outs = [x+55 for x in range(400)]

    for n in outs: 

        fig = plt.figure() 
        grid = AxesGrid(fig, (0.5,0.5,1.5,1.5),
                nrows_ncols = (1, 5),
                axes_pad = 0.1,
                label_mode = "1",
                share_all = True,
                cbar_location="right",
                cbar_mode="edge",
                cbar_size="5%",
                cbar_pad="0%")

        strset = 'DD00'+str(n) 
        if (n > 99): strset = 'DD0'+str(n) 
        fields = [field, field, field, field, field] 
        snaps = ['nref10_track_2/'+strset+'/'+strset, 'nref10_track_lowfdbk_1/'+strset+'/'+strset, 
             'nref10_track_lowfdbk_2/'+strset+'/'+strset, 'nref10_track_lowfdbk_3/'+strset+'/'+strset,
             'nref10_track_lowfdbk_4/'+strset+'/'+strset]

        for i, (field, snap) in enumerate(zip(fields, snaps)):
    
            ds = yt.load(snap) 
            zsnap = ds.get_parameter('CosmologyCurrentRedshift')
            trident.add_ion_fields(ds, ions=['C IV', 'O VI','H I','Si III'])

            centerx = np.interp(zsnap, track['col1'], 0.5*(track['col2']+track['col5']) ) 
            centery = np.interp(zsnap, track['col1'], track['col3']+30./143886.) 
            centerz = np.interp(zsnap, track['col1'], 0.5*(track['col4']+track['col7']) ) 
            center = [centerx, centery, centerz] 
    
            box = ds.r[ center[0]-200./143886:center[0]+200./143886, center[1]-250./143886.:center[1]+250./143886., center[2]-40./143886.:center[2]+40./143886.]
    
            # projection 
            p = yt.ProjectionPlot(ds, 'z', field, center=center, width=((120,'kpc'),(240,'kpc')), data_source=box)
            if (field == 'density'): 
                p.set_unit('density', 'Msun / pc**2')
                p.set_zlim('density', 0.01, 1000) 
            if ('O_p5' in field): 
                p.set_zlim("O_p5_number_density",1e11,1e15)
            if (i < 1): p.annotate_timestamp(corner='upper_left', redshift=True, draw_inset_box=True, text_args={'color':'white', 'size':'small'} )
    
            # This forces the ProjectionPlot to redraw itself on the AxesGrid axes.
            plot = p.plots[field]
            plot.figure = fig
            plot.axes = grid[i].axes
            p._setup_plots()   # Finally, redraw the plot.
    
        plt.savefig(strset+'_multiplot_'+field+'_projection.png', bbox_inches='tight')
