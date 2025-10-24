'''
    Filename: shade_maps.py
    Author: JT
    Created: 10=12-25
    Last modified: 10=12-25 by JT
    This file works with fogghorn_analysis.py to incorprate shade maps into the analysis suite.
    If you add a new function to this script, then please also add the function name to the dictionary in fogghorn_analysis.py.

    NOTE: unlike the other FOGGHORN plotting scripts, this one generates multople plots per function call, so the function
    here is called 'phase_shade' but it makes many different shade maps. This prevents the FOGGHORN script from checking for 
    existing output files and skipping them, so be careful when using this script to avoid overwriting existing plots.
    This is a temporary solution until we can refactor the shade map code to be more modular, but was the most expedient way to
    reuse the existing rendering code from foggie/render/shade_maps.py
'''

from foggie.fogghorn.header import *
from foggie.fogghorn.util import *
import foggie.render.shade_maps as sm

def phase_shade(ds, region, args, output_filename):
    """ makes a suite of shade maps for the given dataset and region.
        This is a wrapper around foggie/render/shade_maps.py:simple_plot
    """
    region = 'cgm_z'  # force to trackbox for CGM plots

    for colorcode in ['phase', 'metal', 'cell_mass']:
        sm.simple_plot(ds.filename,args.trackfile,('gas','x'),('gas','y'), colorcode, ( (-100,100), (-100,100) ), output_filename+'_x_y_'+colorcode+'_'+region, region=region) 
        sm.simple_plot(ds.filename,args.trackfile,('gas', 'density'),('gas','temperature'),  colorcode, ((-32,-22), (1,8)), output_filename+'_density_temperature_'+colorcode+'_'+region, region=region ) 
        sm.simple_plot(ds.filename,args.trackfile,('gas', 'radius_corrected'), ('gas','temperature'),  colorcode, ((0,250), (1,8)), output_filename+'_radius_temperature_'+colorcode+'_'+region, region=region) 
        sm.simple_plot(ds.filename,args.trackfile,('gas', 'radius_corrected'), ('gas','metallicity'),  colorcode, ((0,250), (-6,2)), output_filename+'_radius_metallicity_'+colorcode+'_'+region, region=region) 
        sm.simple_plot(ds.filename,args.trackfile,('gas', 'radius_corrected'), ('gas','cooling_time'), colorcode, ((0,250), (4,12)), output_filename+'_radius_cooling_time_'+colorcode+'_'+region, region=region ) 
        sm.simple_plot(ds.filename,args.trackfile,('gas', 'radius_corrected'), ('gas','density'),      colorcode, ((0,250), (-32,-22)), output_filename+'_radius_density_'+colorcode+'_'+region, region=region ) 
        sm.simple_plot(ds.filename,args.trackfile,('gas', 'radius_corrected'), ('gas','O_p5_column_density'), colorcode, ((0,250), (8,14)), output_filename+'_radius_NOVI_'+colorcode+'_'+region, region=region ) 
        sm.simple_plot(ds.filename,args.trackfile,('gas', 'radius_corrected'), ('gas', 'H_p0_column_density'), colorcode, ((0,250), (8,14)), output_filename+'_radius_NHI_'+colorcode+'_'+region, region=region ) 
        sm.simple_plot(ds.filename,args.trackfile,('gas', 'radius_corrected'), ('gas', 'cell_mass'), colorcode, ((0,250), (-2,6)), output_filename+'_radius_cell_mass_'+colorcode+'_'+region, region=region ) 
        sm.simple_plot(ds.filename,args.trackfile,('gas', 'radial_velocity_corrected'), ('gas', 'cell_mass'), colorcode, ((-500,500), (-2,6)), output_filename+'_rv_cell_mass_'+colorcode+'_'+region, region=region ) 
        sm.simple_plot(ds.filename,args.trackfile,('gas', 'radial_velocity_corrected'), ('gas', 'tangential_velocity_corrected'), colorcode, ((-500,500),(-50,500)), output_filename+'_rv_tv_'+colorcode+'_'+region, region=region ) 
        sm.simple_plot(ds.filename,args.trackfile,('gas', 'radius_corrected'), ('gas', 'tangential_velocity_corrected'), colorcode, ((0,200),(-50,500)), output_filename+'_r_tv_'+colorcode+'_'+region, region=region ) 
        sm.simple_plot(ds.filename,args.trackfile,('gas', 'radius_corrected'), ('gas', 'radial_velocity_corrected'), colorcode, ((0,200),(-500,500)), output_filename+'_r_rv_'+colorcode+'_'+region, region=region ) 