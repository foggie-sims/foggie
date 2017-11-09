import matplotlib
matplotlib.use('Agg')

from comparative_analysis import *
import numpy as np
import builtins

## All of the RDs
#RDs = np.arange(27,43)
## Two Outputs for Testing
RDs = np.arange(41,43)
prefix = 'RD'
PLOTS_DIR = 'plots/'

sim_dirs = ['/astro/simulations2/FOGGIE/halo_008508/symmetric_box_tracking/nref10f_sym50kpc',
            '/astro/simulations/FOGGIE/halo_008508/natural/nref10']
builtins.track_name = '/astro/simulations/FOGGIE/halo_008508/complete_track_symmetric_50kpc'
file_base = PLOTS_DIR+'nref10_fn_'

## Radial profile evolution fields
fields = ['H_nuclei_density','Temperature','total_energy','metallicity','entropy']


for RD in RDs:
  if ((prefix == 'DD') & (RD > 100)):
      filenames = [x+('/'+prefix+'0'+str(RD))*2 for x in sim_dirs]
      file_out = file_base +prefix+'0'+str(RD)
  else:
      filenames = [x+('/'+prefix+'00'+str(RD))*2 for x in sim_dirs]
      file_out = file_base + prefix+'00'+str(RD)


  #ONCE PER RD FOR TWO FILENAMES
  #-----------------------------
  plot_phase_diagrams(filenames,file_out+'_phase.pdf')
  plot_cooling_time_histogram(filenames,file_out+'_cooltime_hist.pdf')

  if RD == RDs[-1]:
    #ONCE ON LATEST
    #--------------
    plot_SFHs(filenames,file_base+'SFH.pdf',redshift_limits=[1,0])


#DOES THE RD CYCLE ON ITS OWN
#----------------------------
for basename in sim_dirs:
  base_out = basename.split('/')[-1]
  base_out = PLOTS_DIR+base_out

  plot_mass_in_phase_evolution(basename,RDs,prefix,
                               base_out+'_phase_evol.pdf')
  for field in fields:
      if isinstance(field,tuple):
          fout = field[1] 
      else:
          fout = field
    
      if field == 'metallicity':
          plot_field_profile_evolution(basename,RDs,prefix,field,
                                       base_out+'_'+fout+'.pdf',
                                       plt_log=False)
      else:
          plot_field_profile_evolution(basename,RDs,prefix,field,
                                       base_out+'_'+fout+'.pdf')


## Holoviews Separate Loop since it currently uses bokeh and not matplotlib
hv.extension('bokeh')

for RD in RDs:
    if ((prefix == 'DD') & (RD > 100)):
        filenames = [x+('/'+prefix+'0'+str(RD))*2 for x in sim_dirs]
        file_out = file_base + prefix + '0'+str(RD)
    else:
        filenames = [x+('/'+prefix+'00'+str(RD))*2 for x in sim_dirs]
        file_out = file_base + prefix + '00' + str(RD)

    plot_holoviews_phase_diagrams(filenames,file_out+'_hvPhase')
    plot_holoviews_radial_profiles(filenames,file_out+'_hvProfiles')





