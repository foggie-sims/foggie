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
    echo "Running Aitoff projection for halo $halo_id with inclination $inclination, position angle $pa"
    python make_hi_datacube_v2.py --force_ideal_ifu 1 --output ./aitoff_runs/tmp/${halo_name}_MhongooseLR_10Mpc --position_angle $pa --z .00233333333 --halo $halo_id --mock_suffix _NHI1e18_RD0042_GaussianHPF --memory_chunks 100 --survey MHONGOOSE_LR  --min_column_density 1e18   --high_pass_filter_type gaussian  --nthreads 15 --clean_gain 0.1 --set_res_auto 1 --fov_kpc 350 --inclination $inclination  
    python make_hi_datacube_v2.py --make_disk_cut_mask 1 --output ./aitoff_runs/tmp/${halo_name}_MhongooseLR_10Mpc --position_angle $pa --z .00233333333 --halo $halo_id --mock_suffix _NHI1e18_RD0042_GaussianHPF --memory_chunks 100 --survey MHONGOOSE_LR  --min_column_density 1e18   --high_pass_filter_type gaussian  --nthreads 15 --clean_gain 0.1 --set_res_auto 1 --fov_kpc 350 --inclination $inclination  

    python convert_to_fits.py --halo $halo_id --input_survey _MhongooseLR_10Mpc_NHI1e18_RD0042 --image_type noisy --input_dir $base_path --bminstr -1
    python convert_to_fits.py --halo $halo_id --input_survey _MhongooseLR_10Mpc_NHI1e18_RD0042 --image_type smoothed --input_dir $base_path --bminstr -1
    python convert_to_fits.py --halo $halo_id --input_survey _MhongooseLR_10Mpc_NHI1e18_RD0042 --image_type filtered --input_dir $base_path --bminstr -1 --renormalize_noise 1


    #run sofia mask on all 3
    cd /Users/ctrapp/Documents/foggie_analysis/SoFiA-2-master
    if [ $bminstr = "-1" ]; then
      bminstr=""
    fi

    filename0="${halo_name}_MhongooseLR_10Mpc_NHI1e18_RD0042_noisy.fits"
    filename1="${halo_name}_MhongooseLR_10Mpc_NHI1e18_RD0042_smoothed.fits"
    filename2="${halo_name}_MhongooseLR_10Mpc_NHI1e18_RD0042_filtered.fits"

    echo "Running Sofia on $filename0"

    ./sofia "${parfile}" input.data="${base_path}${filename0}"
    echo "Running Sofia on $filename1"
    ./sofia "${parfile}" input.data="${base_path}${filename1}"
    echo "Running Sofia on $filename2"
    ./sofia "${parfile}" input.data="${base_path}${filename2}"

    cd /Users/ctrapp/Documents/foggie_analysis/analysis_tools/synthetic_HI_imager

    python save_observable_cgm_fraction.py $inclination $pa $halo_name


  done
done