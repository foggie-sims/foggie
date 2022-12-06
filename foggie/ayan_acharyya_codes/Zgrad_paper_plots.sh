#!/bin/sh

# batch file to make all plots of metallicity gradient paper, including those in the appendix
# some parts of this file are to be run on the local machine, and some on pleiades, comment in/out accordingly

# For time series:

#python plot_Zevolution.py --system ayan_local --halo 8508 --upto_kpc 10 --weight mass --forpaper

# For time fraction plot pair:

#python plot_MZgrad.py --system ayan_local --halo 8508 --Zgrad_den kpc --upto_kpc 10 --weight mass --ycol Zgrad --xcol time --colorcol log_mass --zhighlight --plot_timefraction --Zgrad_allowance 0.05 --upto_z 2

# For *all halos* metallicity gradient vs redshift plot (with GLASS data):

#python plot_MZgrad.py --system ayan_local --halo 8508,5036,5016,4123,2878,2392 --Zgrad_den kpc --upto_kpc 10 --weight mass --glasspaper

# For single snapshot Z projection, Z profile gradient and Z histogram:

#python projection_plot.py --system ayan_local --halo 8508 --galrad 10 --output RD0030 --do metal
#python compute_MZgrad.py --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --xcol rad --keep --weight mass
#python compute_Zscatter.py --system ayan_local --halo 8508 --output RD0030 --upto_kpc 10 --res 0.1 --nbins 100 --weight mass --docomoving --fit_multiple --annotate_profile

# For 3x3 panel plots (projections, radial profiles, histograms) for multiple halos at redshifts 0, 1, 2 (run on pleiades):
# For Tempest
python projection_plot.py --system ayan_pleiades --halo 8508 --galrad 10 --output RD0020,RD0027,RD0042 --do metal
python compute_MZgrad.py --system ayan_pleiades --halo 8508 --output RD0020,RD0027,RD0042 --upto_kpc 10 --xcol rad --keep --weight mass
python compute_Zscatter.py --system ayan_pleiades --halo 8508 --output RD0020,RD0027,RD0042 --upto_kpc 10 --res 0.1 --nbins 100 --weight mass --docomoving --fit_multiple

# For Maelstrom
python projection_plot.py --system ayan_pleiades --halo 5036 --galrad 10 --output RD0020,RD0027,RD0042 --do metal
python compute_MZgrad.py --system ayan_pleiades --halo 5036 --output RD0020,RD0027,RD0042 --upto_kpc 10 --xcol rad --keep --weight mass
python compute_Zscatter.py --system ayan_pleiades --halo 5036 --output RD0020,RD0027,RD0042 --upto_kpc 10 --res 0.1 --nbins 100 --weight mass --docomoving --fit_multiple

# For Squall
python projection_plot.py --system ayan_pleiades --halo 5016 --galrad 10 --output RD0020,RD0027,RD0042 --do metal
python compute_MZgrad.py --system ayan_pleiades --halo 5016 --output RD0020,RD0027,RD0042 --upto_kpc 10 --xcol rad --keep --weight mass
python compute_Zscatter.py --system ayan_pleiades --halo 5016 --output RD0020,RD0027,RD0042 --upto_kpc 10 --res 0.1 --nbins 100 --weight mass --docomoving --fit_multiple
