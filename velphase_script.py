
from foggie import show_velphase as sv
import os

# Natural refinement, RD0020
dsname = '/Users/tumlinson/Dropbox/FOGGIE/outputs/halo_008508/nref11n/natural/RD0020/RD0020'

# LLS sightlines
os.chdir('/Users/tumlinson/Dropbox/FOGGIE/collab/plots_halo_008508/nref11n/natural/spectra/lls')
sv.drive_velphase(dsname, '*rd0020*v6_los*fits.gz')

# random sightlines
os.chdir('/Users/tumlinson/Dropbox/FOGGIE/collab/plots_halo_008508/nref11n/natural/spectra/random')
sv.drive_velphase(dsname, '*rd0020*v6_los*fits.gz')



# Natural refinement, RD0018
dsname = '/Users/tumlinson/Dropbox/FOGGIE/outputs/halo_008508/nref11n/natural/RD0018/RD0018'

# LLS sightlines
os.chdir('/Users/tumlinson/Dropbox/FOGGIE/collab/plots_halo_008508/nref11n/natural/spectra/lls')
sv.drive_velphase(dsname, '*rd0018*v6_los*fits.gz')

# random sightlines
os.chdir('/Users/tumlinson/Dropbox/FOGGIE/collab/plots_halo_008508/nref11n/natural/spectra/random')
sv.drive_velphase(dsname, '*rd0018*v6_los*fits.gz')





# nref10f, RD0020
dsname = '/Users/tumlinson/Dropbox/FOGGIE/outputs/halo_008508/nref11n/nref11n_nref10f_refine200kpc/RD0020/RD0020'

# LLS sightlines
os.chdir('/Users/tumlinson/Dropbox/FOGGIE/collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/lls')
sv.drive_velphase(dsname, '*rd0020*v6_los*fits.gz')

# random sightlines
os.chdir('/Users/tumlinson/Dropbox/FOGGIE/collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/random')
sv.drive_velphase(dsname, '*rd0020*v6_los*fits.gz')



# nref10f, RD0018
dsname = '/Users/tumlinson/Dropbox/FOGGIE/outputs/halo_008508/nref11n/nref11n_nref10f_refine200kpc/RD0018/RD0018'

# LLS sightlines
os.chdir('/Users/tumlinson/Dropbox/FOGGIE/collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/lls')
sv.drive_velphase(dsname, '*rd0018*v6_los*fits.gz')

# random sightlines
os.chdir('/Users/tumlinson/Dropbox/FOGGIE/collab/plots_halo_008508/nref11n/nref11n_nref10f_refine200kpc/spectra/random')
sv.drive_velphase(dsname, '*rd0018*v6_los*fits.gz')
