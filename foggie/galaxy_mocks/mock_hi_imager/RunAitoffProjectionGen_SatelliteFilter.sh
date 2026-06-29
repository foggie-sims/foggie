#!/bin/bash

#load ideal, noisy, filtered from h5py
#add a noise profile to ideal
#save all 3 as .fits


halo_id=$1 #008508
bminstr=$2 #'' or '_bmin5' or '_bmin10' etc
parfile=$3

case "$halo_id" in
  "008508")
    halo_name="Tempest"
    ;;
  "005036")
    halo_name="Maelstrom"
    ;;
  "005016")
    halo_name="Squall"
    ;;
  "004123")
    halo_name="Blizzard"
    ;;
  "002392")
    halo_name="Hurricane"
    ;;
  "002878")
    halo_name="Cyclone"
    ;;
  *)
    echo "Unknown halo_id: $halo_id"
    exit 1
    ;;
esac

base_path="/Users/ctrapp/Documents/foggie_analysis/analysis_tools/synthetic_HI_imager/aitoff_runs/tmp/"


for inclination in 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90; do
  for pa in 0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340; do
    FILE="./aitoff_runs/${halo_name}MhongooseLR_10Mpc_NHI1e18_RD0042_i${inclination}_pa${pa}.h5"
    if [ -f "$FILE" ]; then
      echo "$FILE exists, skipping..."
      continue
    fi
    echo "Filtering Satellites for halo $halo_id with inclination $inclination, position angle $pa"
    python make_hi_datacube.py --output ./aitoff_runs/tmp/${halo_name}_MhongooseLR_10Mpc --position_angle $pa --z .00233333333 --halo $halo_id --mock_suffix _NHI1e18_RD0042_GaussianHPF --memory_chunks 100 --survey MHONGOOSE_LR  --min_column_density 1e18   --high_pass_filter_type gaussian  --nthreads 15 --clean_gain 0.1 --set_res_auto 1 --fov_kpc 350 --inclination $inclination  --make_clump_cut_mask 1 --clump_cut_file  /Users/ctrapp/Documents/foggie_analysis/disk_project/disk_clumps/HI_Disk_Blizzard_RD0042__Satellite --clump_cut_suffix _Sat --n_clumps_to_cut 2  

    python remove_satellites_from_observable_cgm_fraction.py $inclination $pa $halo_name


  done
done