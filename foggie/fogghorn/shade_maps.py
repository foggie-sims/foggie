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
    
    TRACKFILE = args.trackfile

    # These are the same plots that were in the original shade_maps.py file in foggie/render
    sm.simple_plot(ds.filename,TRACKFILE,('gas','x'),('gas','y'), 'phase', ( (-100,100), (-100,100) ), output_filename+'_x_y_phase_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'density'),('gas','temperature'),  'phase', ((-32,-22), (1,8)), output_filename+'_density_temperature_phase_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radius_corrected'), ('gas','temperature'),  'phase', ((0,250), (1,8)), output_filename+'_radius_temperature_phase_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radius_corrected'), ('gas','metallicity'),  'phase', ((0,250), (-6,2)), output_filename+'_radius_metallicity_phase_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radius_corrected'), ('gas','cooling_time'), 'phase', ((0,250), (4,12)), output_filename+'_radius_cooling_time_phase_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radius_corrected'), ('gas','density'),      'phase', ((0,250), (-32,-22)), output_filename+'_radius_density_phase_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radius_corrected'), ('gas','O_p5_column_density'), 'phase', ((0,250), (8,14)), output_filename+'_radius_NOVI_phase_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radius_corrected'), ('gas', 'H_p0_column_density'), 'phase', ((0,250), (8,14)), output_filename+'_radius_NHI_phase_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radius_corrected'), ('gas', 'cell_mass'), 'phase', ((0,250), (-2,6)), output_filename+'_radius_cell_mass_phase_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radial_velocity_corrected'), ('gas', 'cell_mass'), 'phase', ((-500,500), (-2,6)), output_filename+'_rv_cell_mass_phase_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radial_velocity_corrected'), ('gas', 'tangential_velocity_corrected'), 'phase', ((-500,500),(-50,500)), output_filename+'_rv_tv', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radius_corrected'), ('gas', 'tangential_velocity_corrected'), 'phase', ((0,200),(-50,500)), output_filename+'_r_tv', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radius_corrected'), ('gas', 'radial_velocity_corrected'), 'phase', ((0,200),(-500,500)), output_filename+'_r_rv', region='cgm') 

    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'density'),('gas','temperature'),  'metal', ((-32,-22), (1,8)), output_filename+'_density_temperature_metal_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radial_velocity_corrected'), ('gas', 'cell_mass'), 'metal', ((-500,500), (-2,6)), output_filename+'_rv_cell_mass_metal_cgm', region='cgm') 

    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'density'),('gas','temperature'),  'cell_mass', ((-32,-22), (1,8)), output_filename+'_density_temperature_cell_mass_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radial_velocity_corrected'), ('gas', 'tangential_velocity_corrected'), 'cell_mass', ((-500,500),(-50,500)), output_filename+'_rv_tv_cell_mass_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radius_corrected'), ('gas', 'radial_velocity_corrected'), 'cell_mass', ((0,200),(-500,500)), output_filename+'_r_rv_cell_mass_cgm', region='cgm') 
    sm.simple_plot(ds.filename,TRACKFILE,('gas', 'radial_velocity_corrected'), ('gas', 'cell_mass'), 'cell_mass', ((-500,500), (-2,6)), output_filename+'_rv_cell_mass_cell_mass_cgm', region='cgm') 
