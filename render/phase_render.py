import datashader as dshader
import datashader.transfer_functions as tf
import pandas as pd
import yt
import matplotlib.pyplot as plt 
import numpy as np 
from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno
from IPython.core.display import HTML, display

ds = yt.load('DD0127/DD0127')
halo_center = np.array([0.48984, 0.47133, 0.50956]) 
ad = ds.sphere(halo_center, (300., 'kpc'))

x_particles = ad['x'].ndarray_view() * 143886. 
y_particles = ad['y'].ndarray_view() * 143886. 
z_particles = ad['z'].ndarray_view() * 143886. 
dens = np.log10(ad['density'].ndarray_view()) 
temp = np.log10(ad['temperature'].ndarray_view()) 

#categorize by phase 
phase = np.chararray(np.size(dens), 4)
phase[temp < 9.] = 'hot'
phase[temp < 6.] = 'warm'
phase[temp < 5.] = 'cool'
phase[temp < 4.] = 'cold'
#z_particles[phase == 'hot'] = z_particles['hot'] + 1000. 
phase_color_key = {'hot':'white', 'warm':'white', 'cool':'white', 'cold':'salmon'}
phase_color_key = {'hot':'yellow', 'warm':'green', 'cool':'purple', 'cold':'salmon'}
temp = (temp - 2.5) / 5 # remaps the range logT = 2-8 to 0-1 (2 is min, 6 is interval) 
dens = (dens + 3) / 10. # dens had been between -4 and +8, -4 is min 12 is interval 

df = pd.DataFrame({'x':x_particles, 'y':y_particles, 'z':z_particles, \
     'temp':temp, 'dens':dens, 'phase':phase}) 
df.phase = df.phase.astype('category')

print df[phase == 'hot']

galaxy =    ((10.0,11.4), (7.8,9.2)) 
galaxy =    ((halo_center[0]-1000.,halo_center[0]+1000.), (halo_center[1]-1000.,halo_center[1]+1000.)) 
galaxy =    ((70200.0,70800.0), (67500,68100.0)) # xy 
galaxy =    ((70200.0,70800.0), (73000,73600.0)) # xz 

background = "white"
plot_width  = int(1080)
plot_height = int(1080)

# define the functions that create the images, then create them 
def create_image(x_range, y_range, w=plot_width, h=plot_height):
    cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(df, 'x', 'z', dshader.count_cat('phase'))
    img = tf.colorize(agg, phase_color_key, how='eq_hist')
    return tf.dynspread(img, threshold=0.3, max_px=4)

export = partial(export_image, background = background)
cm = partial(colormap_select, reverse=(background!="black"))
export(create_image(*galaxy),"Phase") 

