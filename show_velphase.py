import yt
import datashader as dshader
import numpy as np
import pandas as pd
import datashader.transfer_functions as tf
from astropy.table import Table 
import trident 
from datashader import reductions
import matplotlib.pyplot as plt

def show_velphase(ds, ray_df, ray_start, ray_end, triray, filename):
    # take in the yt dataset (ds) and a ray as a dataframe 
    
    # preliminaries 
    rs = ray_start.ndarray_view() 
    re = ray_end.ndarray_view() 

    imsize = 500 
    core_width = 10. 

    proper_box_size = ds.get_parameter('CosmologyComovingBoxSize') / ds.get_parameter('CosmologyHubbleConstantNow') * 1000. # in kpc 
    redshift = ds.get_parameter('CosmologyCurrentRedshift') 

    # take out a "core sample" that extends along the ray with a width given by core_width
    ad = ds.r[rs[0]:re[0],rs[1]-0.5*core_width/proper_box_size:rs[1]+0.5*core_width/proper_box_size,
                          rs[2]-0.5*core_width/proper_box_size:rs[2]+0.5*core_width/proper_box_size]

    cell_vol = ad["cell_volume"]
    cell_mass = ad["cell_mass"]
    cell_size = np.array(cell_vol)**(1./3.)*proper_box_size

    x_cells = ad['x'].ndarray_view() * proper_box_size# + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. )
    y_cells = ad['y'].ndarray_view() * proper_box_size# + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. )
    z_cells = ad['z'].ndarray_view() * proper_box_size# + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. )
    dens = np.log10(ad['density'].ndarray_view())
    temp = np.log10(ad['temperature'].ndarray_view())

    phase = np.chararray(np.size(temp), 4)
    phase[temp < 19.] = 'hot'
    phase[temp < 6.] = 'warm'
    phase[temp < 5.] = 'cool'
    phase[temp < 4.] = 'cold'
    phase_color_key = {b'cold':'salmon', b'hot':'yellow', b'warm':'#4daf4a', b'cool':'#984ea3'}     

    df = pd.DataFrame({'x':x_cells, 'y':y_cells, 'z':z_cells, 
                       'vx':ad["x-velocity"], 'vy':ad["y-velocity"], 'vz':ad["z-velocity"],
                       'temp':temp, 'dens':dens, 'phase':phase})
    df.phase = df.phase.astype('category')

    cvs = dshader.Canvas(plot_width=imsize, plot_height=imsize,
        x_range=(np.min(df['x']), np.max(df['x'])), y_range=(np.mean(df['y'])-100./0.695,np.mean(df['y'])+100./0.695))
    agg = cvs.points(df, 'x', 'y',  dshader.count_cat('phase'))
    img = tf.shade(agg, color_key=phase_color_key)  
    x_y = tf.spread(img, px=2, shape='square')
    x_y.to_pil().save(filename+'_x_vs_y.png')

    cvs = dshader.Canvas(plot_width=imsize, plot_height=imsize,
        x_range=(np.min(df['x']), np.max(df['x'])), y_range=(-0.008, 0.008))
    agg = cvs.points(df, 'x', 'vx',  dshader.count_cat('phase'))
    img = tf.shade(agg, color_key=phase_color_key)  
    x_vx = tf.spread(img, px=1, shape='square')
    x_vx.to_pil().save(filename+'_x_vs_vx.png')

    species_dict =  {'CIII':'C_p2_number_density', 'CIV':'C_p3_number_density', 'HI':'H_p0_number_density',
                     'MgII':'Mg_p1_number_density', 'OVI':'O_p5_number_density', 'SiIII':"Si_p2_number_density"}

    for species in species_dict.keys(): 
        cvs = dshader.Canvas(plot_width=imsize, plot_height=imsize, x_range=(rs[0],re[0]), y_range=(-0.008,0.008))
        vx = tf.shade(cvs.points(ray_df, 'x', 'x-velocity', agg=reductions.mean(species_dict[species])), how='eq_hist')
        tf.set_background(vx,"white")
        ray_vx = tf.spread(vx, px=4, shape='square')
        pil = ray_vx.to_pil()
        pil.save(filename+'_'+species+'_ray_vx.png', format='png')



