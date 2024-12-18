#metal_distribution_orig_comp.py
#Summary: Plot metal mass, total mass, and metallicity over time for stars, ISM, CGM as a whole, and compare with original simulations at similar redshifts.
#Author: Kathleen Hamilton-Campos, SASP intern at STScI, summer 2019 - kahamil@umd.edu

#Import library
import matplotlib.pyplot as plt

#Close all open plots to avoid accidentally overwriting them
plt.close("all")

#Matching colors to the Tumlinson, Peeples, Werk ARAA plots
star_color = '#d73027'
ism_color = '#4575b4'
cool_cgm_color = '#984ea3'
metal_color = "tab:gray"

#Allowing the metallicity scatter to be translucent
scatter_alpha = 0.25

#halo_008508_arrays_orig.py
if True:
	#Simulations
	sims_comp = [400, 600, 1700]
	sims_orig = [250, 455, 1550]
	
	#Analysis
	virial_radius_orig = [59., 87., 183.]
	redshift_orig = [2.3571555801473, 1.677118408032, 0.35392820449609]
	lookbacks_orig = [10.908, 9.809, 3.934]
	
	#Metal Mass
	log_star_metal_mass_orig = [8.27967203, 8.62687501, 9.01736104]
	log_ism_metal_mass_orig = [7.79593561, 7.57382815, 7.48639303]
	log_cgm_metal_mass_orig = [6.85174739, 7.09608892, 7.02361404]
	log_pink_metal_mass_orig = [6.35545952, 6.58290669, 5.74780548]
	log_purple_metal_mass_orig = [6.43258401, 6.51643174, 6.42203665]
	log_green_metal_mass_orig = [6.20255677, 6.51954992, 6.77913773]
	log_yellow_metal_mass_orig = [5.73167703, 6.31320618, 6.1280774]
	
	#Metals Returned
	log_metals_returned_orig = [8.41404047, 8.6952249, 9.05867348]
	
	#Total Mass
	log_star_total_mass_orig = [9.92165933, 10.18739563, 10.54359872]
	log_ism_total_mass_orig = [9.44682044, 9.31918962, 9.74443232]
	log_cgm_total_mass_orig = [9.80992384, 9.98673964, 10.1792452]
	log_pink_total_mass_orig = [8.57561187, 8.75885292, 7.87184315]
	log_purple_total_mass_orig = [9.64407931, 9.64153167, 9.4613384]
	log_green_total_mass_orig = [9.19129074, 9.63784841, 10.07082654]
	log_yellow_total_mass_orig = [8.07650558, 8.60335519, 8.56889723]
	
	#Metallicity
	log_star_metallicity_orig = [0.24574293, 0.32720961, 0.36149255]
	log_ism_metallicity_orig = [0.2368454, 0.14236875, -0.37030906]
	log_cgm_metallicity_orig = [-1.07044622, -1.00292048, -1.26790093]
	log_pink_metallicity_orig = [-0.33242211, -0.288216, -0.23630743]
	log_purple_metallicity_orig = [-1.32376507, -1.23736969, -1.15157152]
	log_green_metallicity_orig = [-1.10100374, -1.23056825, -1.40395858]
	log_yellow_metallicity_orig = [-0.45709832, -0.40241877, -0.5530896]
	
	#Metallicity Scatter
	star_above_avg_orig = [0.51404015, 0.54087866, 0.55735691]
	star_below_avg_orig = [-0.17715665, 0.02982983, 0.09712507]
	ism_above_avg_orig = [0.37462286, 0.23943871, -0.24393135]
	ism_below_avg_orig = [0.04552997, 0.01536253, -1.13715439]
	cgm_above_avg_orig = [-0.7996867, -0.75316947, -1.01861675]
	cgm_below_avg_orig = [-3.96733466, -3.17887115, -2.49623137]
	pink_above_avg_orig = [0.07383393, -0.10947414, -0.08739769]
	pink_below_avg_orig = [-1.13138661, -0.58413697, -0.45436215]
	purple_above_avg_orig = [-1.1179716, -1.13061229, -0.88582501]
	purple_below_avg_orig = [-4.32977224, -3.75629327, -2.98142037]
	green_above_avg_orig = [-0.80161794, -0.96113137, -1.1323016]
	green_below_avg_orig = [-3.21344981, -2.69067105, -2.36118116]
	yellow_above_avg_orig = [-0.23975276, -0.12661945, -0.34809554]
	yellow_below_avg_orig = [-0.87754965, -1.04611762, -1.69174646]


#halo_008508_arrays.py
if True:
	#Simulations
	simulations = [400, 600, 1700]
	
	#Analysis
	virial_radius = [60., 88., 186.]
	redshift = [2.327015122116, 1.67203864, 0.34961832640547]
	lookbacks = [10.870, 9.798, 3.896]
	density_cut = [1.e-25, 7.5e-25, 5.e-27]
	central_radius_cut = [4., 6., 30.]
	satellite_distance_limit = [1., 2., 12.]
	
	#Metal Mass
	log_star_metal_mass = [8.31624128, 8.75613115, 9.13019357]
	log_ism_metal_mass = [7.55292066, 7.41610945, 8.13887159]
	log_cgm_metal_mass = [6.83262931, 6.96129056, 6.61793625]
	log_pink_metal_mass = [5.99509999, 6.42264011, 5.13140503]
	log_purple_metal_mass = [6.34281394, 6.30830197, 6.02109504]
	log_green_metal_mass = [6.32039906, 6.42147444, 6.45361668]
	log_yellow_metal_mass = [6.18146326, 6.23788467, 5.07618744]
	
	#Metals Returned
	log_metals_returned = [8.40945534, 8.80915847, 9.1913927]
	
	#Total Mass
	log_star_total_mass = [9.90770785, 10.29722092, 10.68165432]
	log_ism_total_mass = [9.34775906, 9.11001965, 10.24605476]
	log_cgm_total_mass = [9.71643101, 9.9224015, 9.92151206]
	log_pink_total_mass = [8.35530319, 9.05796593, 7.15325512]
	log_purple_total_mass = [9.62239667, 9.76319086, 9.65198902]
	log_green_total_mass = [8.80655537, 9.08459454, 9.57993238]
	log_yellow_total_mass = [8.16428112, 8.30490154, 7.55635207]
	
	#Metallicity
	log_star_metallicity = [0.29626366, 0.34664047, 0.33626948]
	log_ism_metallicity = [0.09289183, 0.19382003, -0.21945294]
	log_cgm_metallicity = [-0.99607146, -1.07338071, -1.41584558]
	log_pink_metallicity = [-0.47247297, -0.74759559, -0.13411985]
	log_purple_metallicity = [-1.3918525, -1.56715866, -1.74316375]
	log_green_metallicity = [-0.59842608, -0.77538986, -1.23858547]
	log_yellow_metallicity = [-0.09508762, -0.17928663, -0.5924344]
	
	#Metallicity Scatter
	star_above_avg = [0.52653738, 0.55433248, 0.52993782]
	star_below_avg = [-0.08768771, 0.02717052, 0.04241433]
	ism_above_avg = [0.28270069, 0.38546582, 0.03681575]
	ism_below_avg = [-0.20938902, -0.08314182, -1.11220268]
	cgm_above_avg = [-0.91738959, -0.9251934, -1.13350629]
	cgm_below_avg = [-2.68514364, -2.47887251, -2.49443093]
	pink_above_avg = [-0.14732756, -0.52669418, 0.25069198]
	pink_below_avg = [-1.69096653, -1.76299288, -0.81249839]
	purple_above_avg = [-1.29859655, -1.36388222, -1.54725029]
	purple_below_avg = [-2.79927485, -2.65307067, -2.83945443]
	green_above_avg = [-0.32013754, -0.51010363, -1.00103731]
	green_below_avg = [-1.38052043, -1.47768456, -1.76304542]
	yellow_above_avg = [0.13166812, 0.03782155, -0.38456845]
	yellow_below_avg = [-0.59851372, -0.79755886, -1.11730288]


#Plot metallicity, metal mass, and total mass for stars, ISM, and CGM
if True:
	fig, axes = plt.subplots(1,3, figsize = (25,8))

	#fig.suptitle('Metal Mass, Total Mass, and Metallicity over Time')

	axes[0].plot(redshift, log_star_metal_mass, '-', color = star_color, label = 'Star Metal Mass')
	axes[0].plot(redshift, log_ism_metal_mass, '-', color = ism_color, label = 'ISM Metal Mass')
	axes[0].plot(redshift, log_cgm_metal_mass, '-', color = cool_cgm_color, label = 'CGM Metal Mass')
	axes[0].plot(redshift, log_metals_returned, '-', color = metal_color, label = 'Available Metals')

	axes[0].plot(redshift_orig, log_star_metal_mass_orig, ':', color = star_color, label = 'Star Metal Mass Orig')
	axes[0].plot(redshift_orig, log_ism_metal_mass_orig, ':', color = ism_color, label = 'ISM Metal Mass Orig')
	axes[0].plot(redshift_orig, log_cgm_metal_mass_orig, ':', color = cool_cgm_color, label = 'CGM Metal Mass Orig')
	axes[0].plot(redshift_orig, log_metals_returned_orig, ':', color = metal_color, label = 'Available Metals Orig')

	axes[1].plot(redshift, log_star_total_mass, '-', color = star_color, label = 'Star Total Mass')
	axes[1].plot(redshift, log_ism_total_mass, '-', color = ism_color, label = 'ISM Total Mass')
	axes[1].plot(redshift, log_cgm_total_mass, '-', color = cool_cgm_color, label = 'CGM Total Mass')

	axes[1].plot(redshift_orig, log_star_total_mass_orig, ':', color = star_color, label = 'Star Total Mass Orig')
	axes[1].plot(redshift_orig, log_ism_total_mass_orig, ':', color = ism_color, label = 'ISM Total Mass Orig')
	axes[1].plot(redshift_orig, log_cgm_total_mass_orig, ':', color = cool_cgm_color, label = 'CGM Total Mass Orig')

	axes[2].plot(redshift, log_star_metallicity, '-', color = star_color, label = 'Star Metallicity')
	axes[2].fill_between(redshift, star_above_avg, star_below_avg, alpha = scatter_alpha, color = star_color, label = 'Star Scatter')

	axes[2].plot(redshift, log_ism_metallicity, '-', color = ism_color, label = 'ISM Metallicity')
	axes[2].fill_between(redshift, ism_above_avg, ism_below_avg, alpha = scatter_alpha, color = ism_color, label = 'ISM Scatter')

	axes[2].plot(redshift, log_cgm_metallicity, '-', color = cool_cgm_color, label = 'CGM Metallicity')
	axes[2].fill_between(redshift, cgm_above_avg, cgm_below_avg, alpha = scatter_alpha, color = cool_cgm_color, label = 'CGM Scatter')

	axes[2].plot(redshift_orig, log_star_metallicity_orig, ':', color = star_color, label = 'Star Metallicity Orig')
	axes[2].fill_between(redshift_orig, star_above_avg_orig, star_below_avg_orig, alpha = scatter_alpha, color = star_color, label = 'Star Scatter Orig')

	axes[2].plot(redshift_orig, log_ism_metallicity_orig, ':', color = ism_color, label = 'ISM Metallicity Orig')
	axes[2].fill_between(redshift_orig, ism_above_avg_orig, ism_below_avg_orig, alpha = scatter_alpha, color = ism_color, label = 'ISM Scatter Orig')

	axes[2].plot(redshift_orig, log_cgm_metallicity_orig, ':', color = cool_cgm_color, label = 'CGM Metallicity Orig')
	axes[2].fill_between(redshift_orig, cgm_above_avg_orig, cgm_below_avg_orig, alpha = scatter_alpha, color = cool_cgm_color, label = 'CGM Scatter Orig')


	axes[0].set_ylabel('Log Mass (Msun)')
	axes[1].set_xlabel('Redshift (z)')
	axes[2].set_ylabel('Log Metallicity (Zsun)')
	axes[2].yaxis.tick_right()
	axes[2].yaxis.set_label_position("right")

	axes[0].legend(loc='lower left', prop={'size': 8}, bbox_to_anchor= (0.0, 1.01), ncol=2, borderaxespad=0, frameon=False)
	axes[1].legend(loc='lower left', prop={'size': 8}, bbox_to_anchor= (0.0, 1.01), ncol=2, borderaxespad=0, frameon=False)
	axes[2].legend(loc='lower left', prop={'size': 8}, bbox_to_anchor= (0.0, 1.01), ncol=4, borderaxespad=0, frameon=False)

fig.savefig('MetalDistributionOrigComp.png')