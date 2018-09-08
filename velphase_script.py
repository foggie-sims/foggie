
from foggie import show_velphase as sv
import os

#natural refinement
#for refine in ['natural','nref11n_nref10f_refine200kpc']:
for refine in ['nref11n_nref10f_refine200kpc']:
    for output in ['RD0018','RD0020']:
        dsname = '/Users/tumlinson/Dropbox/FOGGIE/outputs/halo_008508/nref11n/'+refine+'/'+output+'/'+output
        for subdir in ['lls','random']:
            os.chdir('/Users/tumlinson/Dropbox/FOGGIE/collab/plots_halo_008508/nref11n/'+refine+'/spectra/'+subdir)
            sv.drive_velphase(dsname, '*'+output.lower()+'*v6_los*fits.gz')
