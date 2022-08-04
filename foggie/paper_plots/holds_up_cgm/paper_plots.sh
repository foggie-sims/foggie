#!/bin/sh


# Make thin-slice projections of temperature, metallicity, and radial velocity
#python ../../../../projectionplots.py --output DD2427 --thin_proj --field temperature,metallicity,radial_velocity

# Make slices of velocity, smoothed velocity, and difference between velocity and smoothed velocity
# Note: This needs to be cropped to just the x-velocities to produce the plot in the paper
#python ../../pressure_support/pressure_support.py --output DD2427 --plot velocity_slice

# Make projections of density with and without the disk
#python ../../../../projectionplots.py --output DD2427 --simple_proj

# Make slices of pressures
#python ../../pressure_support/pressure_support.py --output DD2427 --plot pressure_slice --pressure_type all

# Make slices of forces
#python ../../pressure_support/pressure_support.py --output DD2427 --plot force_slice --force_type total

# Make PDFs of metallicity
#python ../../pressure_support/pressure_support.py --output DD0497,DD0970,DD1478,DD2427 --plot metallicity_PDF

# Make slices of smoothed forces with different smoothing scales
#python ../../pressure_support/pressure_support.py --output DD2427 --plot force_slice --force_type all --smoothed 20 --save_suffix smoothing-20kpc --force_type total
#python ../../pressure_support/pressure_support.py --output DD2427 --plot force_slice --force_type all --smoothed 50 --save_suffix smoothing-50kpc --force_type total
#python ../../pressure_support/pressure_support.py --output DD2427 --plot force_slice --force_type all --smoothed 100 --save_suffix smoothing-100kpc --force_type total

# Make plots of forces along rays
# Note: These need to be combined in powerpoint to make the final figure that's in the paper
#python ../../pressure_support/pressure_support.py --output DD2427 --plot force_rays

# Make plot of forces vs. radius for Tempest at z=0 as an example
#python ../../pressure_support/pressure_support.py --output DD2427 --plot force_vs_radius --load_stats --filename _T-split_cgm-only --normalized --save_suffix cgm-only_Tempest

# Make plots of support vs. radius for each halo at z=0
#python ../../pressure_support/pressure_support.py --halo 8508 --output DD2427 --plot support_vs_radius --filename _T-split_cgm-only --save_suffix cgm-only_Tempest
#python ../../pressure_support/pressure_support.py --halo 5016 --output DD2520 --plot support_vs_radius --filename _T-split_cgm-only --save_suffix cgm-only_Squall
#python ../../pressure_support/pressure_support.py --halo 5036 --output DD2520 --plot support_vs_radius --filename _T-split_cgm-only --save_suffix cgm-only_Maelstrom
#python ../../pressure_support/pressure_support.py --halo 4123 --output DD2520 --plot support_vs_radius --filename _T-split_cgm-only --save_suffix cgm-only_Blizzard

# Make plots of support vs. time for inner and outer CGM for all halos
#python ../../pressure_support/pressure_support.py --halo 8508 --run nref11c_nref9f --output DD0497-DD2427 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.1,0.5]" --save_suffix 0p1-0p5Rvir_cgm-only_Tempest
#python ../../pressure_support/pressure_support.py --halo 8508 --run nref11c_nref9f --output DD0497-DD2427 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.5,1]" --save_suffix 0p5-1Rvir_cgm-only_Tempest
#python ../../pressure_support/pressure_support.py --halo 5016 --run nref11c_nref9f --output DD0590-DD2520 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.1,0.5]" --save_suffix 0p1-0p5Rvir_cgm-only_Squall
#python ../../pressure_support/pressure_support.py --halo 5016 --run nref11c_nref9f --output DD0590-DD2520 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.5,1]" --save_suffix 0p5-1Rvir_cgm-only_Squall
#python ../../pressure_support/pressure_support.py --halo 5036 --run nref11c_nref9f --output DD0590-DD2520 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.1,0.5]" --save_suffix 0p1-0p5Rvir_cgm-only_Maelstrom
#python ../../pressure_support/pressure_support.py --halo 5036 --run nref11c_nref9f --output DD0590-DD2520 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.5,1]" --save_suffix 0p5-1Rvir_cgm-only_Maelstrom
#python ../../pressure_support/pressure_support.py --halo 4123 --run nref11c_nref9f --output DD0590-DD2520 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.1,0.5]" --save_suffix 0p1-0p5Rvir_cgm-only_Blizzard
#python ../../pressure_support/pressure_support.py --halo 4123 --run nref11c_nref9f --output DD0590-DD2520 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.5,1]" --save_suffix 0p5-1Rvir_cgm-only_Blizzard

# Make support vs. radius plots averaged over time
#python ../../pressure_support/pressure_support.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot support_vs_radius_time_averaged --filename Z-split_cgm-only --region_filter metallicity --save_suffix cgm-only_Tempest
#python ../../pressure_support/pressure_support.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot support_vs_radius_time_averaged --filename Z-split_cgm-only --region_filter metallicity --save_suffix cgm-only_Squall
#python ../../pressure_support/pressure_support.py --halo 5036 --output DD0590-DD2520 --output_step 10 --plot support_vs_radius_time_averaged --filename Z-split_cgm-only --region_filter metallicity --save_suffix cgm-only_Maelstrom
#python ../../pressure_support/pressure_support.py --halo 4123 --output DD0590-DD2520 --output_step 10 --plot support_vs_radius_time_averaged --filename Z-split_cgm-only --region_filter metallicity --save_suffix cgm-only_Blizzard

# Make plots of support vs. time for inner and outer CGM for z=0.4 to 0 for the different feedback runs
#python ../../pressure_support/pressure_support.py --halo 8508 --run nref11c_nref9f --output DD1627-DD2427 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.1,0.5]" --save_suffix 0p1-0p5Rvir_z0p4-0_cgm-only_fiducial --feedback_diff
#python ../../pressure_support/pressure_support.py --halo 8508 --run nref11c_nref9f --output DD1627-DD2427 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.5,1]" --save_suffix 0p5-1Rvir_z0p4-0_cgm-only_fiducial --feedback_diff
#python ../../pressure_support/pressure_support.py --halo 8508 --run low_feedback_06 --output DD1627-DD2427 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.1,0.5]" --save_suffix 0p1-0p5Rvir_z0p4-0_cgm-only_weak --feedback_diff
#python ../../pressure_support/pressure_support.py --halo 8508 --run low_feedback_06 --output DD1627-DD2427 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.5,1]" --save_suffix 0p5-1Rvir_z0p4-0_cgm-only_weak --feedback_diff
#python ../../pressure_support/pressure_support.py --halo 8508 --run feedback_return --output DD1627-DD2237 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.1,0.5]" --save_suffix 0p1-0p5Rvir_z0p4-0_cgm-only_strong --feedback_diff
#python ../../pressure_support/pressure_support.py --halo 8508 --run feedback_return --output DD1627-DD2237 --output_step 10 --plot support_vs_time --filename _T-split_cgm-only --radius_range "[0.5,1]" --save_suffix 0p5-1Rvir_z0p4-0_cgm-only_strong --feedback_diff

# Make 2D plots of support vs. energy output and radius
# NOTE: These did not make it into the paper
#python ../../pressure_support/pressure_support.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot support_vs_energy_output --filename _T-split_cgm-only --save_suffix cgm-only_Tempest
#python ../../pressure_support/pressure_support.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot support_vs_energy_output --filename _T-split_cgm-only --save_suffix cgm-only_Squall
#python ../../pressure_support/pressure_support.py --halo 5036 --output DD0590-DD2520 --output_step 10 --plot support_vs_energy_output --filename _T-split_cgm-only --save_suffix cgm-only_Maelstrom
#python ../../pressure_support/pressure_support.py --halo 4123 --output DD0590-DD2520 --output_step 10 --plot support_vs_energy_output --filename _T-split_cgm-only --save_suffix cgm-only_Blizzard

# Make 2D plots of support vs. energy output and radius for the different feedback runs
# NOTE: These did not make it into the paper
#python ../../pressure_support/pressure_support.py --halo 8508 --run nref11c_nref9f --output DD1477-DD1727 --output_step 10 --plot support_vs_energy_output --filename _T-split_cgm-only --save_suffix cgm-only_strong-before
#python ../../pressure_support/pressure_support.py --halo 8508 --run feedback_return --output DD1747-DD2427 --output_step 10 --plot support_vs_energy_output --filename _T-split_cgm-only --save_suffix cgm-only_strong-after
#python ../../pressure_support/pressure_support.py --halo 8508 --run low_feedback_06 --output DD1627-DD2427 --output_step 10 --plot support_vs_energy_output --filename _T-split_cgm-only --save_suffix cgm-only_weak

# Make 2D plots of support vs. mass flux and radius
# NOTE: These did not make it into the paper
#python ../../pressure_support/pressure_support.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot support_vs_mass_flux --filename _T-split_cgm-only --save_suffix cgm-only_Tempest
#python ../../pressure_support/pressure_support.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot support_vs_mass_flux --filename _T-split_cgm-only --save_suffix cgm-only_Squall
#python ../../pressure_support/pressure_support.py --halo 5036 --output DD0590-DD2520 --output_step 10 --plot support_vs_mass_flux --filename _T-split_cgm-only --save_suffix cgm-only_Maelstrom
#python ../../pressure_support/pressure_support.py --halo 4123 --output DD0590-DD2520 --output_step 10 --plot support_vs_mass_flux --filename _T-split_cgm-only --save_suffix cgm-only_Blizzard

# Make 2D plots of support vs. radius and time
#python ../../pressure_support/pressure_support.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot support_vs_time_radius --filename T-split_cgm-only --save_suffix cgm-only_Tempest
#python ../../pressure_support/pressure_support.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot support_vs_time_radius --filename T-split_cgm-only --save_suffix cgm-only_Squall
#python ../../pressure_support/pressure_support.py --halo 5036 --output DD0590-DD2520 --output_step 10 --plot support_vs_time_radius --filename T-split_cgm-only --save_suffix cgm-only_Maelstrom
#python ../../pressure_support/pressure_support.py --halo 4123 --output DD0590-DD2520 --output_step 10 --plot support_vs_time_radius --filename T-split_cgm-only --save_suffix cgm-only_Blizzard

# Make 2D plots of support vs. radius and time for the different feedback runs
#python ../../pressure_support/pressure_support.py --halo 8508 --run nref11c_nref9f --output DD1627-DD2427 --output_step 10 --plot support_vs_time_radius --filename T-split_cgm-only --save_suffix cgm-only_fiducial --feedback_diff
#python ../../pressure_support/pressure_support.py --halo 8508 --run low_feedback_06 --output DD1627-DD2427 --output_step 10 --plot support_vs_time_radius --filename T-split_cgm-only --save_suffix cgm-only_weak --feedback_diff
#python ../../pressure_support/pressure_support.py --halo 8508 --run feedback_return --output DD1627-DD2427 --output_step 10 --plot support_vs_time_radius --filename T-split_cgm-only --save_suffix cgm-only_strong --feedback_diff

# Make plots of velocity dispersion vs. mass and spatial resolution
python ../../turbulence/turbulence.py --halo 8508 --output DD2427 --plot vdisp_vs_mass_res
python ../../turbulence/turbulence.py --halo 8508 --output DD2427 --plot vdisp_vs_spatial_res
