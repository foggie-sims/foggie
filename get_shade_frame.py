import datashader as dshader
import datashader.transfer_functions as tf
import pandas as pd
import numpy as np 
from functools import partial
from datashader.utils import export_image
from astropy.table import Table 
from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno
from IPython.core.display import HTML, display

def shade_phase_diagram(refine_box, strset): 

    dens = np.log10(refine_box['density'].ndarray_view())
    temp = np.log10(refine_box['temperature'].ndarray_view())

    phase = np.chararray(np.size(dens), 4) #categorize by phase 
    phase[temp < 9.] = 'hot'
    phase[temp < 6.] = 'warm'
    phase[temp < 5.] = 'cool'
    phase[temp < 4.] = 'cold'
    phase_color_key = {'hot':'orange', 'warm':'green', 'cool':'purple', 'cold':'salmon'}
    temp = (temp - 2.5) / 5 # remaps the range logT = 2-8 to 0-1 (2 is min, 6 is interval) 
    dens = (dens + 3) / 10. # dens had been between -4 and +8, -4 is min 12 is interval 
    
    df = pd.DataFrame({'temp':temp, 'dens':dens, 'phase':phase}) 
    df.phase = df.phase.astype('category')
    
    plot_window =    ((-2.8,-2), (0.2,1.0))
   
    def create_image(x_range, y_range, w=1080, h=1080):
        cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(df, 'dens', 'temp', dshader.count_cat('phase')) 
        img = tf.colorize(agg, phase_color_key, how='eq_hist')
        return img
    
    export = partial(export_image, background = 'white')
    cvs = dshader.Canvas(1080, 1080, *plot_window).points(df,'dens','temp') 
    export(create_image(*plot_window),strset+"_phase") 
 

def shade_mass_diagram(refine_box, strset): 

    temp = np.log10(refine_box['temperature'].ndarray_view())
    mass = np.log10((refine_box['cell_mass'].in_units('Msun')).ndarray_view())

    phase = np.chararray(np.size(temp), 4) #categorize by phase 
    phase[temp < 9.] = 'hot'
    phase[temp < 6.] = 'warm'
    phase[temp < 5.] = 'cool'
    phase[temp < 4.] = 'cold'
    phase_color_key = {'hot':'orange', 'warm':'green', 'cool':'purple', 'cold':'salmon'}
    
    df = pd.DataFrame({'temp':temp, 'mass':mass, 'phase':phase}) 
    df.phase = df.phase.astype('category')
    
    plot_window = ((2,8), (-2,8))
   
    def create_image(x_range, y_range, w=1080, h=1080):
        cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(df, 'temp', 'mass', dshader.count_cat('phase')) 
        img = tf.colorize(agg, phase_color_key, how='eq_hist')
        return img
    
    export = partial(export_image, background = 'white')
    cvs = dshader.Canvas(1080, 1080, *plot_window).points(df,'temp','mass') 
    export(create_image(*plot_window),strset+"_mass") 






