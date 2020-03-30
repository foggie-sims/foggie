mocky_way code structure:
prepare the data set to load in necessary function
    * from foggie.mocky_way.core_funcs import prepdata
    * ds, ds_paras = prepdata(dd_name, sim_name=sim_name)
    * prepdata does these following things:
        * load data (i.e., ds)
        * generate a dict called ds_paras which includes all needed information for the dataset,
            * dd_name: e.g, ‘DD2175, same as input
            * sim_name: e.g., ‘nref11n_nref10f’, same as input
            * data_path: path to the needed output
            * zsnap: redshift of the snapshot
            * halo_center: center of halo, see core_funcs.find_halo_center_yz
            * rvir: pre calculated viral radius of the galaxy
            * L_vec, sun_vec, phi_vec: the xyz vector in the disk coordinate system
            * disk_bulkvel: disk bulk velocity within the sphere that used to calculate the gas angular momentum, 
              see core_funcs.get_sphere_ang_mom_vecs
            * disk_rs, disk_zs: pre calculated disk scale length and scale height, see core_funcs.dict_disk_rs_zs, 
              and disk_scale_length_rs.py and disk_scale_height_zs.py
            * offcenter_location, offcenter_bulkvel: call core_funcs.offcenter_observer to decide where the off 
              center observer location and velocity are.
        * call trident.add_ion_fields, and ion fields, including SiII, SiIII, SiIV, CII, CIV, OVI, NV, OVII, OVIII, NeVIII, NeVIII
        * call foggie.mocky_way.mocky_way_fields to add 3 self defined fields: (‘gas’, ‘los_velocity_mw’), (‘gas’, ‘l’), (‘gas’, ‘b')
        * calculate the angular momentum and the xyz vectors for the disk, called L_vec, sun_vec, and phi_vec
        * if shift_obs_location is set True when calling prepdata, then the xyz coordinate vectors are  
          shifted by shift_n45*45 degrees in the disk plane, where shift_n45 = 1, 2, …. 7
        * call locate_offcenter_observer  to shift the observer from disk center to some redefined off center location

Figure 1 - face on and edge on projection plots
    * fig1_find_flat_disk_allskyproj.py: to be run on pleiades, inside out view to make sure disk is flat
    * fig1_find_flat_disk_offaxproj.py: external view to make sure disk is flat

Figure 2: decide the disk scale height and scale length
    * disk_scale_height_zs.py
    * disk_scale_length_rs.py

Figure 3: cgm mass distribution
    * fig3_cgm_mass_phases.ipynb

Figure 4:
    * vrot_vcirc_cs.py

Figure 5ab and 6ab:
    * phase_diagram.py

Figure 5c and 6c:
    * offaxproj_dshader.py

Figure 5de and 6de:
    * offaxproj_ytfunc_logT_Z.py

Figure 7 - comparison of NHI between sim and HI4PI
    * fig7_allsky_HI4PI_NHI.ipynb
    * need to run alllsky_diff_ions.py on pleiades to get the allsky result for
      different ions, and write it into fits files, then use the python notebook fig7_xx to make the plots
    * qsub job_allsky_diffions.sh

Figure 8, bias cartoon,
    * fig8_hvc_bias_cartoon.ipynb

Figure 9 - dMdv plot, Figure 10 - dMdt plots
    * $ python dMdv_run.py nref11n_nref10f DD2175
        * the code will calculate the mass flux per velocity bin, split into different temperature ranges, cold, cool, warm, hot;
          and split into low and high galactic latitudes, decide to be at b = 20
        * for off center location, need to run the code through all 8 locations
        * data results are recorded as something like:
          figs/dM_dv/fits/nref11n_nref10f_DD2175_dMdv_cgm-15kpc_offcenter_location_b30.fits
    * $ python dMdt_constant.py nref11n_nref10f DD2175
        * the code will calculate constant flux dM/dt, where dt=r/v
        * save to something as: figs/dM_dt/fits/nref11n_nref10f_DD2175_dMdt_cgm-15kpc_offcenter_location_b20.fits
    * go to dMdv_dMdt_fig9_fig10.ipynb to make the plots!

Figure 11 - QuaStar ish pair sightline comparison
    * use fig11_column_density_star_qso_plt.py to make the plot, figure saved in figs/Nr_star_qso/...
    * run column_density_inview_ray_pleiades.py on pleiades to make a whole bunch of random rays through
      the halo to record N, l, b, r within certain r range.
    *
Figure 12 - SiIII and OVI ion profiles
    * fig12_ion_nr_profile.py

Figure 13 - Column density inside vs outside views
    * column_density_in_vs_ex_plt.py
