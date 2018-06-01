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

### obtain the EAGLE output
fname='snap_047_z000p000.0.hdf5'
ds=yt.load(fname,n_ref=8,over_refine_factor=1)
ad = ds.all_data()
coords = ad['PartType0', 'Coordinates']
x_particles = (coords[:,0].ndarray_view()-4.0)/4. + 10.
y_particles = (coords[:,1].ndarray_view()-5.0)/4. + 8.
z_particles = coords[:,2].ndarray_view()
dens = np.log10(ad['PartType0', 'Density'].ndarray_view())
temp = np.log10(ad['PartType0', 'Temperature'].ndarray_view())
mass = ad['PartType0', 'Mass'].ndarray_view()

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
temp = (temp - 2.5) / 5 # remaps the range logT = 2-8 to 0-1 (2 is min, 6 is interval)
dens = (dens + 3) / 10. # dens had been between -4 and +8, -4 is min 12 is interval

df = pd.DataFrame({'x':x_particles, 'y':y_particles, 'z':z_particles, \
     'mass':mass, 'temp':temp, 'dens':dens, 'metallicity':metallicity, 'phase':phase, 'metal_label':metal_label})
df.phase = df.phase.astype('category')
df.metal_label = df.metal_label.astype('category')

galaxy =    ((10.0,11.4), (7.8,9.2))
x_center = 10.7
y_center = 8.5
d_center = 0.5
t_center = 0.5

background = "white"
plot_width  = int(1080)
plot_height = int(1080)


# define the functions that create the images, then create them
def create_image(x_range, y_range, w=plot_width, h=plot_height):
    cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(df, 'x', 'y', dshader.count_cat('phase'))
    img = tf.colorize(agg, phase_color_key, how='eq_hist')
    return img

def create_image2(x_range, y_range, w=plot_width, h=plot_height):
    cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(df, 'dens', 'y', dshader.count_cat('phase'))
    img = tf.colorize(agg, phase_color_key, how='eq_hist')
    return img

def create_image3(x_range, y_range, w=plot_width, h=plot_height):
    cvs = dshader.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
    agg = cvs.points(df, 'dens', 'temp', dshader.count_cat('phase'))
    img = tf.colorize(agg, phase_color_key, how='eq_hist')
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

    export = partial(export_image, background = background)
    cm = partial(colormap_select, reverse=(background!="black"))

    cvs = dshader.Canvas(1080, 1080, *galaxy).points(df,'x','y')

    export(create_image(*galaxy),"Phase_"+str(1000+ii))
    print ii

# now start with dens / y and gradually turn y into temp
for ii in np.arange(100):

    rr, phi = cart2pol(df['y'] - y_center, df['temp'] - t_center)
    xxxx, yyyy = pol2cart(rr, phi - np.pi / 2. / 100.)
    df.y = xxxx+y_center
    df.temp = yyyy+t_center

    export = partial(export_image, background = background)
    cm = partial(colormap_select, reverse=(background!="black"))

    cvs2 = dshader.Canvas(1080, 1080, *galaxy).points(df, 'x', 'y')

    export(create_image(*galaxy),"Phase_"+str(2000+ii))
    print ii
