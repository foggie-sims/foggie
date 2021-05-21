#!/bin/sh

# Make mass vs time plots
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot mass_vs_time --save_suffix Tempest
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot mass_vs_time --save_suffix Squall
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD0590-DD2520 --output_step 10 --plot mass_vs_time --save_suffix Maelstrom

# Make energy vs time plots
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot energy_vs_time --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Tempest
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot energy_vs_time --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Squall
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD0590-DD2520 --output_step 10 --plot energy_vs_time --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Maelstrom

# Make energy vs radius plots
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD2427 --plot energy_vs_radius --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Tempest
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD2360 --plot energy_vs_radius --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Squall
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD2510 --plot energy_vs_radius --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Maelstrom

# Make temperature vs time plots
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot temperature_vs_time --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered --save_suffix Tempest
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot temperature_vs_time --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered --save_suffix Squall
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD0590-DD2520 --output_step 10 --plot temperature_vs_time --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered --save_suffix Maelstrom

# Make temperature vs radius plots
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD2427 --plot temperature_vs_radius --hist_from_file --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered --save_suffix Tempest
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD2360 --plot temperature_vs_radius --hist_from_file --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered --save_suffix Squall
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD2510 --plot temperature_vs_radius --hist_from_file --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered --save_suffix Maelstrom

# Make energy-SFR cross-correlation plots
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --plot energy_SFR_xcorr --filename totals_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Tempest
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0580-DD2520 --plot energy_SFR_xcorr --filename totals_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Squall
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD0590-DD2520 --plot energy_SFR_xcorr --filename totals_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Maelstrom

# Make temperature-SFR cross-correlation plots
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --plot temp_SFR_xcorr --filename totals_mass_sphere_vcut-p5vff_cgm-filtered_Tvirbins --save_suffix Tempest
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0590-DD2520 --plot temp_SFR_xcorr --filename totals_mass_sphere_vcut-p5vff_cgm-filtered_Tvirbins --save_suffix Squall
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD0590-DD2520 --plot temp_SFR_xcorr --filename totals_mass_sphere_vcut-p5vff_cgm-filtered_Tvirbins --save_suffix Maelstrom

# Make a temperature vs. radius and an energy vs. radius plot for all outputs
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot temperature_vs_radius --hist_from_file --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered --save_suffix Tempest
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot energy_vs_radius --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Tempest
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot temperature_vs_radius --hist_from_file --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered --save_suffix Squall
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot energy_vs_radius --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Squall
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD2400-DD2520 --output_step 10 --plot temperature_vs_radius --hist_from_file --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered --save_suffix Maelstrom
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD2400-DD2520 --output_step 10 --plot energy_vs_radius --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Maelstrom

# Make appendix A plots
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD2077 --plot energy_vs_radius --Rvir_compare --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Tempest_Rvir-compare
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot energy_vs_time --Rvir_compare --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Tempest_Rvir-compare

# Make visualization plots
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD2427 --plot visualization --save_suffix Tempest_z0_test

# Make velocity component distribution plots
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD2427 --plot velocity_PDF --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered --save_suffix Tempest_z0
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD1477 --plot velocity_PDF --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered --save_suffix Tempest_z0p5

# Make plots using a larger bin in R
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot energy_vs_time --large_Rbin --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Tempest_largeR-bin
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot energy_vs_time --large_Rbin --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Squall_largeR-bin
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD0590-DD2290 --output_step 10 --plot energy_vs_time --large_Rbin --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Maelstrom_largeR-bin
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD2427 --plot temperature_vs_radius --hist_from_file --filename stats_temperature_velocity_pdf_sphere_mass-weighted_vcut-vp5vff_cgm-filtered_bigRbins --save_suffix Tempest_largeR-bin
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD2427 --plot velocity_PDF --filename stats_temperature_velocity_pdf_sphere_mass-weighted_vcut-vp5vff_cgm-filtered_bigRbins --save_suffix Tempest_z0_largeR-bin

# Test if there's a difference when using only cells in the refine box for the shell at Rvir
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD2427 --plot energy_vs_radius --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered_refined-only --save_suffix Tempest_refined-only
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD2427 --plot temperature_vs_radius --hist_from_file --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered_refined-only --save_suffix Tempest_refined-only
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD2360 --plot energy_vs_radius --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered_refined-only --save_suffix Squall_refined-only
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD2360 --plot temperature_vs_radius --hist_from_file --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered_refined-only --save_suffix Squall_refined-only
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD2290 --plot energy_vs_radius --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered_refined-only --save_suffix Maelstrom_refined-only
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD2290 --plot temperature_vs_radius --hist_from_file --filename stats_temperature_energy_velocity_pdf_sphere_mass-weighted_vcut-p5vff_dbl-gauss-fit_cgm-filtered_refined-only --save_suffix Maelstrom_refined-only

# Test the difference between singular isothermal sphere assumption (shells) or explicitly measuring PE and pressure boundary (cumulative) for Appendix B
#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot energy_vs_radius_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Tempest_inner-r0p3Rvir --inner_r 0.3
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot energy_vs_radius_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Squall_inner-r0p3Rvir --inner_r 0.3
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD0590-DD2520 --output_step 10 --plot energy_vs_radius_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Maelstrom_inner-r0p3Rvir --inner_r 0.3

#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot energy_vs_radius_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Tempest_inner-r0p5Rvir --inner_r 0.5
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot energy_vs_radius_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Squall_inner-r0p5Rvir --inner_r 0.5
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD0590-DD2520 --output_step 10 --plot energy_vs_radius_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Maelstrom_inner-r0p5Rvir --inner_r 0.5

#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot energy_vs_radius_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Tempest_inner-r0p7Rvir --inner_r 0.7
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot energy_vs_radius_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Squall_inner-r0p7Rvir --inner_r 0.7
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD0590-DD2520 --output_step 10 --plot energy_vs_radius_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Maelstrom_inner-r0p7Rvir --inner_r 0.7

#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot energy_vs_time_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Tempest_inner-r0p3Rvir --inner_r 0.3
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot energy_vs_time_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Squall_inner-r0p3Rvir --inner_r 0.3
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD0590-DD2290 --output_step 10 --plot energy_vs_time_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Maelstrom_inner-r0p3Rvir --inner_r 0.3

#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot energy_vs_time_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Tempest_inner-r0p5Rvir --inner_r 0.5
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot energy_vs_time_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Squall_inner-r0p5Rvir --inner_r 0.5
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD0590-DD2290 --output_step 10 --plot energy_vs_time_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Maelstrom_inner-r0p5Rvir --inner_r 0.5

#python mod_vir_temp_paper_plots.py --halo 8508 --output DD0497-DD2427 --output_step 10 --plot energy_vs_time_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Tempest_inner-r0p7Rvir --inner_r 0.7
#python mod_vir_temp_paper_plots.py --halo 5016 --output DD0590-DD2520 --output_step 10 --plot energy_vs_time_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Squall_inner-r0p7Rvir --inner_r 0.7
#python mod_vir_temp_paper_plots.py --halo 5036 --output DD0590-DD2290 --output_step 10 --plot energy_vs_time_comp --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered --save_suffix Maelstrom_inner-r0p7Rvir --inner_r 0.7

# Plot cooling time vs radius for 2nd referee report
python mod_vir_temp_paper_plots.py --halo 8508 --output DD2427 --plot tcool_vs_radius_hist --save_suffix Tempest
python mod_vir_temp_paper_plots.py --halo 8508 --output DD2427 --plot tcool_vs_radius_tot --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered_cooling --save_suffix Tempest
python mod_vir_temp_paper_plots.py --halo 5016 --output DD2360 --plot tcool_vs_radius_hist --save_suffix Squall
python mod_vir_temp_paper_plots.py --halo 5016 --output DD2360 --plot tcool_vs_radius_tot --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered_cooling --save_suffix Squall
python mod_vir_temp_paper_plots.py --halo 5036 --output DD2510 --plot tcool_vs_radius_hist --save_suffix Maelstrom
python mod_vir_temp_paper_plots.py --halo 5036 --output DD2510 --plot tcool_vs_radius_tot --filename totals_mass_volume_energy_sphere_vcut-p5vff_cgm-filtered_cooling --save_suffix Maelstrom
