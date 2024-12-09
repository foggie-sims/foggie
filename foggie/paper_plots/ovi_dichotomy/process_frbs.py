import glob
import yt
from foggie.utils.consistency import * 
import matplotlib.cm as cmaps
import pandas as pd
import datashader as dshader
import argparse
import datashader.transfer_functions as tf
from astropy.cosmology import WMAP9 as cosmo
from datashader.utils import export_image

directory = '/Users/tumlinson/Dropbox/FOGGIE/collab/data_dump/frbs'

def make_the_plots(phase):

    filelist = glob.glob(directory+'/'+phase+'/DD???0_'+phase+'_frb.h5')
    filelist.sort() 

    df = pd.DataFrame({'z':[], 'time':[], 'no6':[], 'nh1':[], 'radius':[]})

    for file in filelist: 
        ds = yt.load(file)
        ad = ds.all_data() 
        no6 = ad['O_p5_number_density']
        nh1 = ad['H_p0_number_density']
        z = no6 * 0.0 + ds.current_redshift
        time = cosmo.lookback_time(99.) - cosmo.lookback_time(z.ndarray_view())
    
        radius_filename = directory+'/radius/'+str.split(file, '/')[9][0:6]+ '_radius.h5'
        print('I need to get radius file:  ', radius_filename)
        r = yt.load(radius_filename)
        r_ad = r.all_data()
        radius = r_ad['radius_corrected']
    
        no6[no6 < 1e4] = 1e4
        nh1[nh1 < 1e4] = 1e4
        dd = pd.DataFrame({'z':z, 'time':time ,'no6':np.log10(no6), 'nh1':np.log10(nh1), 'radius':radius })
        df = df.append(dd)
    
        print(dd.head()) 

    cvs = dshader.Canvas(plot_width=400, plot_height=400, x_range=[0.0, 2.0], y_range=[8,16])
    agg = cvs.points(df, 'z', 'no6', dshader.sum('no6'))
    o6img = tf.spread(tf.shade(agg, cmap = cmaps.get_cmap("magma"), how='eq_hist',min_alpha=0), px=1)
    o6img
    export_image(o6img, 'frb_vs_z_2_to_0_o6_'+phase)  

    cvs = dshader.Canvas(plot_width=400, plot_height=400, x_range=[0.0, 2.0], y_range=[10, 20])
    agg = cvs.points(df, 'z', 'nh1', dshader.sum('nh1'))
    h1img = tf.spread(tf.shade(agg, cmap = cmaps.get_cmap("plasma"), how='eq_hist',min_alpha=0), px=1)
    h1img
    export_image(h1img, 'frb_vs_z_2_to_0_h1_'+phase)

    df.to_pickle(directory+'/pickles/'+phase+'.pkl')



def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    
    parser = argparse.ArgumentParser(description="processes FRBs into pickles and makes plots")

    parser.add_argument('--phase', dest='phase', action='store', help='name of phase in FRBs')
    parser.set_defaults(phase='cgm')

    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = parse_args()
    print(args.phase) 
    make_the_plots(args.phase)