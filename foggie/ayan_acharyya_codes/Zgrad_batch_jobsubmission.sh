#!/bin/sh
###########################################################################################
######### run this file in bash console as ./Zgrad_batch_jobsubmission.sh #################
###########################################################################################

#########################  original runs  ########################################

#########################  Z distribution  ########################################
# original python: plot distribution 10 pkpc
python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_Zscatter --prefix czs_10p --halo 8508 --queue ldan --opt_args "--clobber --do_all_sims --upto_kpc 10 --res 0.1 --nbins 100 --weight mass --fit_multiple --write_file --use_gasre"

# original python: plot distribution 10 ckpc: done already
#python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_Zscatter --prefix czs_10c --halo 8508 --queue ldan --opt_args "--clobber --do_all_sims --upto_kpc 10 --docomoving --res 0.1 --nbins 100 --weight mass --fit_multiple --write_file --use_gasre"

# original python: plot distribution 2.5 pkpc
python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_Zscatter --prefix czs_2.5p --halo 8508 --queue ldan --opt_args "--clobber --do_all_sims --upto_kpc 2.5 --res 0.1 --nbins 100 --weight mass --fit_multiple --write_file --use_gasre"

# original python: plot distribution 2.5 ckpc: done already
#python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_Zscatter --prefix czs_2.5c --halo 8508 --queue ldan --opt_args "--clobber --do_all_sims --upto_kpc 2.5 --docomoving --res 0.1 --nbins 100 --weight mass --fit_multiple --write_file --use_gasre"


#########################  Z profile  ########################################
# original python: plot profile 10 pkpc: done already
#python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_MZgrad --prefix cmzg_10p --halo 8508 --queue ldan --opt_args "--clobber --do_all_sims --upto_kpc 10 --xcol rad --weight mass --write_file"

# original python: plot profile 10 ckpc: done without plots, hence resubmitting
python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_MZgrad --prefix cmzg_10c --halo 8508 --queue ldan --opt_args "--clobber --do_all_sims --upto_kpc 10 --docomoving --xcol rad --weight mass --write_file"

# original python: plot profile 2.5 pkpc: done already
#python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_MZgrad --prefix cmzg_2.5p --halo 8508 --queue ldan --opt_args "--clobber --do_all_sims --upto_kpc 2.5 --xcol rad --weight mass --write_file"

# original python: plot profile 2.5 ckpc
python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_MZgrad --prefix cmzg_2.5c --halo 8508 --queue ldan --opt_args "--clobber --do_all_sims --upto_kpc 2.5 --docomoving --xcol rad --weight mass --write_file"


#########################  Z all  ########################################
# original python: plot all Z movie 10 pkpc
python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call plot_allZ_movie --prefix pazm_10p --halo 8508 --queue ldan --opt_args "--Zgrad_den kpc --upto_kpc 10 --res 0.1 --weight mass --zhighlight --overplot_smoothed 1000 --nocolorcoding --do_all_sims"

# original python: plot all Z movie 10 ckpc
python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call plot_allZ_movie --prefix pazm_10c --halo 8508 --queue ldan --opt_args "--Zgrad_den kpc --upto_kpc 10 --docomoving --res 0.1 --weight mass --zhighlight --overplot_smoothed 1000 --nocolorcoding --do_all_sims"

# original python: plot all Z movie 2.5 pkpc: not needed yet
#python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call plot_allZ_movie --prefix pazm_2.5p --halo 8508 --queue ldan --opt_args "--Zgrad_den kpc --upto_kpc 2.5 --res 0.1 --weight mass --zhighlight --overplot_smoothed 1000 --nocolorcoding --do_all_sims"

# original python: plot all Z movie 2.5 ckpc: not needed yet
#python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call plot_allZ_movie --prefix pazm_2.5c --halo 8508 --queue ldan --opt_args "--Zgrad_den kpc --upto_kpc 2.5 --docomoving --res 0.1 --weight mass --zhighlight --overplot_smoothed 1000 --nocolorcoding --do_all_sims"


######################  fdbk12  ########################################

#########################  Z distribution  ########################################
# fdbk12 python: plot distribution 10 pkpc
python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_Zscatter --prefix czs_10p_fb --halo 8508 --queue ldan --run feedback.12 --opt_args "--clobber --do_all_sims --upto_kpc 10 --res 0.1 --nbins 100 --weight mass --fit_multiple --write_file --use_gasre --foogie_dir /nobackup/jtumlins/ --forcepath"

# fdbk12 python: plot distribution 10 ckpc
python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_Zscatter --prefix czs_10c_fb --halo 8508 --queue ldan --run feedback.12 --opt_args "--clobber --do_all_sims --upto_kpc 10 --docomoving --res 0.1 --nbins 100 --weight mass --fit_multiple --write_file --use_gasre --foogie_dir /nobackup/jtumlins/ --forcepath"


#########################  Z profile  ########################################
# fdbk12 python: plot profile 10 pkpc
python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_MZgrad --prefix cmzg_10p_fb --halo 8508 --queue ldan --run feedback.12 --opt_args "--clobber --do_all_sims --upto_kpc 10 --xcol rad --weight mass --write_file --foogie_dir /nobackup/jtumlins/ --forcepath"

# fdbk12 python: plot profile 10 ckpc
python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call compute_MZgrad --prefix cmzg_10c_fb --halo 8508 --queue ldan --run feedback.12 --opt_args "--clobber --do_all_sims --upto_kpc 10 --docomoving --xcol rad --weight mass --write_file --foogie_dir /nobackup/jtumlins/ --forcepath"


#########################  Z all  ########################################
# original python: plot all Z movie 10 pkpc
python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call plot_allZ_movie --prefix pazm_10p --halo 8508 --queue ldan --run feedback.12 --opt_args "--Zgrad_den kpc --upto_kpc 10 --res 0.1 --weight mass --zhighlight --overplot_smoothed 1000 --nocolorcoding --do_all_sims --foogie_dir /nobackup/jtumlins/ --forcepath"

# original python: plot all Z movie 10 ckpc
python /nobackup/aachary2/ayan_codes/foggie/foggie/ayan_acharyya_codes/submit_jobs.py --call plot_allZ_movie --prefix pazm_10c --halo 8508 --queue ldan --run feedback.12 --opt_args "--Zgrad_den kpc --upto_kpc 10 --docomoving --res 0.1 --weight mass --zhighlight --overplot_smoothed 1000 --nocolorcoding --do_all_sims --foogie_dir /nobackup/jtumlins/ --forcepath"
