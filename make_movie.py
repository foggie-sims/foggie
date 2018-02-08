import datashader as dshader
from functools import partial
from datashader.utils import export_image
import datashader.transfer_functions as tf
import pandas as pd
import yt
import matplotlib.pyplot as plt 
import numpy as np 
from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno
from IPython.core.display import HTML, display

def movie_script(fname): 

    background = 'black'  
    export = partial(export_image, background = background, export_path="export")
    cm = partial(colormap_select, reverse=(background!="black"))

    ds=yt.load(fname) 
    region = ds.r[0.40:0.55, 0.40:0.55, 0.40:0.56]  

    ad = region 
    
    cell_vol = ad["cell_volume"]
    cell_size = np.array(cell_vol)**(1./3.)

    x_particles = ad['x'].ndarray_view() + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. ) 
    x_particles = ( x_particles - 0.4899 ) / 0.001 
    y_particles = ad['y'].ndarray_view() + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. ) 
    y_particles = ( y_particles - 0.47102 ) / 0.001  
    z_particles = ad['z'].ndarray_view() + cell_size * (np.random.rand(np.size(cell_vol)) * 2. - 1. ) 
    dens = np.log10(ad['density']) 
    temp = np.log10(ad['temperature']) 

    if (False): 
        #categorize by metallicity' 
        metallicity = ad['PartType0', 'Metallicity']
        metal_label = np.chararray(np.size(dens), 5)
        metal_label[metallicity < 10.] = 'high'
        metal_label[metallicity < 0.005] = 'solar'
        metal_label[metallicity < 0.000001] = 'low'
        metal_label[metallicity < 0.0000001] = 'poor'
        metal_color_key = {'high':'yellow', 'solar':'green', 'low':'purple', 'poor':'salmon'}

    #categorize by phase 
    phase = np.chararray(np.size(dens), 4)
    phase[temp < 9.] = 'hot'
    phase[temp < 6.] = 'warm'
    phase[temp < 5.] = 'cool'
    phase[temp < 4.] = 'cold'
    phase_color_key = {'hot':'yellow', 'warm':'green', 'cool':'purple', 'cold':'salmon'}
    dens = (dens + 24.) / 6. 
    temp = (temp - 6.5) / 3. 

    df = pd.DataFrame({'x':x_particles, 'y':y_particles, 'z':z_particles, \
         'temp':temp, 'dens':dens, 'phase':phase}) 
    df.phase = df.phase.astype('category')
    #df.metal_label = df.metal_label.astype('category')
    
    galaxy =    ((-1.,1.0), (-1.0,1.0)) 
    x_center = 0.49
    y_center = 0.471 
    d_center = 0.0   
    t_center = 0.0 

    background = "white"
    plot_width  = int(1080)
    plot_height = int(1080)

    # define the functions that create the images, then create them 
    def create_image(x_range, y_range, w=plot_width, h=plot_height):
        cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(df, 'x', 'y', dshader.count_cat('phase'))
        #img = tf.colorize(agg, phase_color_key, how='eq_hist')
        img = tf.shade(agg, color_key=phase_color_key, how='eq_hist')
        return img
    
    def create_image2(x_range, y_range, w=plot_width, h=plot_height):
        cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(df, 'dens', 'y', dshader.count_cat('phase'))
        img = tf.colorize(agg, phase_color_key, how='eq_hist')
        export(tf.shade(agg, cmap = cm(Greys9), how='linear'),"census_gray_linear")
        return img
    
    def create_image3(x_range, y_range, w=plot_width, h=plot_height):
        cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(df, 'dens', 'temp', dshader.count_cat('phase')) 
        img = tf.colorize(agg, phase_color_key, how='eq_hist')
        export(tf.shade(agg, cmap = cm(Greys9), how='linear'),"census_gray_linear")
        return img
    
    def cart2pol(x, y):
        return np.sqrt(x**2 + y**2), np.arctan2(y, x) 
    
    def pol2cart(rho, phi):
        return rho * np.cos(phi), rho * np.sin(phi) 
    
    # this function rotates from x/y plane to density / y 
    for ii in np.arange(100): 
        
        rr, phi = cart2pol(df['x'] - x_center, df['dens'] - d_center) 
        xxxx, yyyy = pol2cart(rr, phi - np.pi / 2. / 100.) 
        df.x = xxxx+x_center 
        df.dens = yyyy+d_center 
    
        export(create_image(*galaxy),"Phase_"+str(1000+ii))
        print(ii) 
    
    # now start with dens / y and gradually turn y into temp 
    for ii in np.arange(100): 
        
        rr, phi = cart2pol(df['y'] - y_center, df['temp'] - t_center) 
        xxxx, yyyy = pol2cart(rr, phi - np.pi / 2. / 100.) 
        df.y = xxxx+y_center 
        df.temp = yyyy+t_center 
    
        export(create_image(*galaxy),"Phase_"+str(2000+ii))
        print(ii) 
