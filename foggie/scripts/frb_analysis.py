import glob
import yt
from foggie.utils.consistency import * 
from datashader.utils import export_image
import matplotlib.cm as cmaps
import pandas as pd
import datashader as dshader
import datashader.transfer_functions as tf

dir = '/home/tumlinson/Dropbox/FOGGIE/collab/data_dump/frbs/'

def frb_compile(dir, wildcard): 

    filelist = []
    filelist = glob.glob(dir+'cgm/DD*h5')
    filelist.sort()

    df = pd.DataFrame({'z':[1.], 'no6':[14.], 'nh1':[14.], 'radius':[100.]})

    for file in filelist: 
        ds = yt.load(file)
        ad = ds.all_data() 
        no6 = ad['O_p5_number_density']
        nh1 = ad['H_p0_number_density']
        z = no6 * 0.0 + ds.current_redshift
    
        radius_filename = dir+'radius/'+'DD'+str.split(file, 'DD')[1][0:4]+ '_radius.h5'
        print('I need to get radius file:  ', radius_filename)
        r = yt.load(radius_filename)
        r_ad = r.all_data()
        radius = r_ad['radius_corrected']
    
        dd = pd.DataFrame({'z':z, 'no6':np.log10(no6), 'nh1':np.log10(nh1), 'radius':radius })    
        df = df.append(dd)

    return df 

def plots(df): 
    
    cvs = dshader.Canvas(plot_width=800, plot_height=800, x_range=[0.0,2.5], y_range=[8,16])
    agg = cvs.points(df, 'z', 'no6', dshader.sum('no6'))
    o6img = tf.spread(tf.shade(agg, cmap = cmaps.get_cmap("magma"), how='eq_hist',min_alpha=0), px=2)
    export_image(o6img, 'o6_vs_z')

    cvs = dshader.Canvas(plot_width=800, plot_height=800, x_range=[0.0,2.5], y_range=[10,20])
    agg = cvs.points(df, 'z', 'nh1', dshader.sum('nh1'))
    h1img = tf.spread(tf.shade(agg, cmap = cmaps.get_cmap("plasma"), how='eq_hist',min_alpha=0), px=3)
    export_image(h1img, 'h1_vs_z')

    df['no6'][df['no6'] < 0.] = 0. 
    cvs = dshader.Canvas(plot_width=1600, plot_height=800, x_range=[0,200], y_range=[0, 2.5])
    agg = cvs.points(df, 'radius', 'z', dshader.max('no6'))
    o6img = tf.spread(tf.shade(agg, cmap = cmaps.get_cmap("magma"),how='eq_hist', min_alpha=0), px=2)
    export_image(o6img, 'o6_z_vs_r')

    df['nh1'][df['nh1'] < 0.] = 0. 
    cvs = dshader.Canvas(plot_width=1600, plot_height=800, x_range=[0,200], y_range=[0, 2.5])
    agg = cvs.points(df, 'radius', 'z', dshader.max('nh1'))
    h1img = tf.spread(tf.shade(agg, cmap = cmaps.get_cmap("plasma"),how='eq_hist', min_alpha=0), px=2)
    export_image(h1img, 'h1_z_vs_r')