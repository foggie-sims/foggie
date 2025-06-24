#!/bin/bash

#load ideal, noisy, filtered from h5py
#add a noise profile to ideal
#save all 3 as .fits


halo_id=$1 #008508
survey=$2 #Grid_MhongooseHR_NHI5e18_bmin29_RD0042
bminstr=$3 #'' or '_bmin5' or '_bmin10' etc

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

#base_path="/Volumes/FoggieCam/synthetic_HI_imager/outputs"
base_path="/Users/ctrapp/Documents/foggie_analysis/analysis_tools/synthetic_HI_imager/outputs/"

echo "Converting hdf5 to fits for halo $halo_id with survey $survey"
python convert_to_fits.py --halo $halo_id --input_survey $survey --image_type noisy --input_dir $base_path --bminstr $bminstr
python convert_to_fits.py --halo $halo_id --input_survey $survey --image_type smoothed --input_dir $base_path --bminstr $bminstr
python convert_to_fits.py --halo $halo_id --input_survey $survey --image_type filtered --input_dir $base_path --bminstr $bminstr # --renormalize_noise 1

#run sofia mask on all 3
cd /Users/ctrapp/Documents/foggie_analysis/SoFiA-2-master
#base_path="/Users/ctrapp/Documents/foggie_analysis/analysis_tools/synthetic_HI_imager/outputs"
if [ $bminstr = "-1" ]; then
  bminstr=""
fi

filename0="${halo_name}${survey}${bminstr}_noisy.fits"
filename1="${halo_name}${survey}${bminstr}_smoothed.fits"
filename2="${halo_name}${survey}${bminstr}_filtered.fits"

echo "Running Sofia on $filename0"
./sofia test_par_file.par input.data="${base_path}${filename0}"
echo "Running Sofia on $filename1"
./sofia test_par_file.par input.data="${base_path}${filename1}"
echo "Running Sofia on $filename2"
./sofia test_par_file.par input.data="${base_path}${filename2}"