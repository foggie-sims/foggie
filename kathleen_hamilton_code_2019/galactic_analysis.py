#galactic_analysis.py
#Summary: Analyze FOGGIE halo_008508 to create temperature-density phase plots, galactic/ISM/CGM face-on/edge-on density/metal mass projection plots, 
#and calculate star/ISM/CGM metal mass/total mass/metallicity. Used to gather information in halo_008508_arrays.py.
#Author: Kathleen Hamilton-Campos, SASP intern at STScI, summer 2019 - kahamil@umd.edu


#Prevent errors when running
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Import necessary libraries
import yt
import numpy as np
from utils.get_refine_box import get_refine_box
from astropy.table import Table
import matplotlib.pyplot as plt
import math
import seaborn as sns

#Defining new fields, filters, and functions
def _metal_mass(field, data): return data[star_metallicity_fraction]*data[star_particle_mass]
def _stars(pfilter, data): return data[(pfilter.filtered_type, "particle_type")] == 2
def weighted_avg_and_stdev(values, weights):
	average = np.average(values, weights=weights)
	variance = np.average((values-average)**2, weights=weights)
	return (average, math.sqrt(variance))

#Prevent new plots from overwriting open existing plots
plt.close("all")

#DDs currently working with
simulations = [250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1300, 1400, 1700]

#ISM/CGM density and temperature cuts
density_cut = [7.5e-24, 10.e-25, 1.e-25, 7.5e-25, 7.5e-25, 7.5e-25, 7.5e-25, 1.e-25, 1.e-25, 1.e-25, 2.5e-26, 2.5e-26, 5.e-27]
temp = 1.5e4

#Radii for looking at the center of the galaxy, the stars in the galaxy, and cutting out satellites
sphere_rad = 100.
star_rad = 20.
central_radius_cut = [4., 4., 4., 4., 6., 4., 4., 8., 10., 10., 20., 20., 30.]

#Creating uniform graphs
proj_width = 75.
temp_phase_min = 1.
temp_phase_max = 10.**9
dens_phase_min = 10.**-31
dens_phase_max = 10.**-21
dens_proj_min = 10.**-5
dens_proj_max = 0.5
metal_proj_min = 10.**54
metal_proj_max = 10.**61

#Removing satellites that are too close to the central galaxy
satellite_distance_limit = [2., 2., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 12.]
sat_sph_rad = 3.

#virial radii of the galaxy
virial_radius = [35., 44., 60., 73., 88., 102., 114., 124., 135., 151., 159., 167., 186.]

#CGM temperature cuts
low_temp = 1.e4
med_temp = 1.e5
high_temp = 1.e6

#FOGGIE specs
StarMetalYield = 0.025
StarMassEjectionFraction = 0.25

#Defining where to pull information from for metal mass and metallicity calculations
star_metallicity_fraction = ("stars", "metallicity_fraction")
star_particle_mass = ("stars", "particle_mass")
metallicity_field = ("gas", "metallicity")
metallicity_weight = ("gas", "cell_mass")

#Color maps from consistency.py
density_color_map = sns.blend_palette(("black", "#4575b4", "#4daf4a", "#ffe34d", "darkorange"), as_cmap=True)
metal_color_map = sns.blend_palette(("black", "#4575b4", "#984ea3", "#984ea3", "#d73027", "darkorange", "#ffe34d"), as_cmap=True)

#Setting arrays to zero for filling
redshift = np.zeros(len(simulations))
log_star_metallicity = np.zeros(len(simulations))
log_star_metal_mass = np.zeros(len(simulations))
log_star_total_mass = np.zeros(len(simulations))
log_ism_metallicity = np.zeros(len(simulations))
log_ism_metal_mass =  np.zeros(len(simulations))
log_ism_total_mass =  np.zeros(len(simulations))
log_cgm_metallicity = np.zeros(len(simulations))
log_cgm_metal_mass =  np.zeros(len(simulations))
log_cgm_total_mass =  np.zeros(len(simulations))
log_pink_metallicity = np.zeros(len(simulations))
log_pink_metal_mass =  np.zeros(len(simulations))
log_pink_total_mass =  np.zeros(len(simulations))
log_purple_metallicity = np.zeros(len(simulations))
log_purple_metal_mass =  np.zeros(len(simulations))
log_purple_total_mass =  np.zeros(len(simulations))
log_green_metallicity = np.zeros(len(simulations))
log_green_metal_mass =  np.zeros(len(simulations))
log_green_total_mass =  np.zeros(len(simulations))
log_yellow_metallicity = np.zeros(len(simulations))
log_yellow_metal_mass =  np.zeros(len(simulations))
log_yellow_total_mass =  np.zeros(len(simulations))
star_above_avg = np.zeros(len(simulations))
star_below_avg = np.zeros(len(simulations))
ism_above_avg = np.zeros(len(simulations))
ism_below_avg = np.zeros(len(simulations))
cgm_above_avg = np.zeros(len(simulations))
cgm_below_avg = np.zeros(len(simulations))
pink_above_avg = np.zeros(len(simulations))
pink_below_avg = np.zeros(len(simulations))
purple_above_avg = np.zeros(len(simulations))
purple_below_avg = np.zeros(len(simulations))
green_above_avg = np.zeros(len(simulations))
green_below_avg = np.zeros(len(simulations))
yellow_above_avg = np.zeros(len(simulations))
yellow_below_avg = np.zeros(len(simulations))
log_metals_returned = np.zeros(len(simulations))

#Select timestamp for analysis
for DD_ind, DD in enumerate(simulations):

	#Center galaxy
	if True:
		#Insert own path to centering file
		cen_fits = np.load("/Users/khamilton/Desktop/Scripts/nref11n_nref10f_interpolations_DD0150_new.npy", allow_pickle = True)[()]

		cen_x = cen_fits['CENTRAL']['fxe'](DD)
		cen_y = cen_fits['CENTRAL']['fye'](DD)
		cen_z = cen_fits['CENTRAL']['fze'](DD)

		cen_cen = yt.YTArray([cen_x, cen_y, cen_z], 'kpc')

	#Load dataset, define new filter and field, and define initial sphere with directions
	if True:
		#Insert own path to simulation data
		if DD < 1000:
			ds = yt.load('/Users/khamilton/Desktop/halo_008508/nref11n_nref10f/orig/DD0{}/DD0{}'.format(DD,DD))
		else:
			ds = yt.load('/Users/khamilton/Desktop/halo_008508/nref11n_nref10f/orig/DD{}/DD{}'.format(DD,DD))

		yt.add_particle_filter("stars",function=_stars, filtered_type='all',requires=["particle_type"])
		ds.add_particle_filter('stars')

		ds.add_field(("all", "metal_mass"), function=_metal_mass, units="Msun", particle_type = True, force_override = True)

		ob_sphere = ds.sphere(cen_cen, (sphere_rad, "kpc"))

		#Setting face-on and edge-on projection directions
		gas_ang_mom_x = ob_sphere.quantities.total_quantity([("gas", "angular_momentum_x")])
		gas_ang_mom_y = ob_sphere.quantities.total_quantity([("gas", "angular_momentum_y")])
		gas_ang_mom_z = ob_sphere.quantities.total_quantity([("gas", "angular_momentum_z")])
		gas_ang_mom = yt.YTArray([gas_ang_mom_x, gas_ang_mom_y, gas_ang_mom_z])
		gas_ang_mom_tot = np.sqrt(sum(gas_ang_mom**2))
		gas_ang_mom_norm = gas_ang_mom / gas_ang_mom_tot

		edge_on_dir = np.random.randn(3)
		edge_on_dir -= edge_on_dir.dot(gas_ang_mom_norm) * gas_ang_mom_norm / np.linalg.norm(gas_ang_mom_norm)**2

		W = yt.YTArray([15, 15, 15], 'kpc')
		north_vector = [-1, 1, 0]

	#Create Temperature-Density diagram color-colored by cell mass
	if True:
		phaseplot = yt.PhasePlot(ob_sphere, "density", "temperature", "cell_mass")
		phaseplot.set_xlim(dens_phase_min, dens_phase_max)
		phaseplot.set_ylim(temp_phase_min, temp_phase_max)
		phaseplot.save('PhaseTDCM{}.png'.format(DD))

	#Face-on and edge-on projection plots
	if True:
		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('FaceOn{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('EdgeOn{}.png'.format(DD))

	#Create the box and spheres
	if True:
		zsnap = ds.get_parameter('CosmologyCurrentRedshift')
		redshift[DD_ind] = zsnap

		#Insert own path to halo track
		trackname = "/Users/khamilton/Desktop/Scripts/halo_track_full"
		track = Table.read(trackname, format='ascii')

		refine_box, refine_box_center, refine_width = get_refine_box(ds, zsnap, track)

		cen_sphere = ds.sphere(cen_cen, (central_radius_cut[DD_ind], "kpc"))
		star_sphere = ds.sphere(cen_cen, (star_rad, 'kpc'))
		cgm_sphere = ds.sphere(cen_cen, (virial_radius[DD_ind], 'kpc'))

	#Projection box color-coded by density
	if True:
		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = refine_box)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('FaceOnBox{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = refine_box)
		#prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('FaceOnBoxMetals{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = refine_box)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('EdgeOnBox{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = refine_box)
		prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('EdgeOnBoxMetals{}.png'.format(DD))

	#Remove satellites
	if True:
		#Find satellites and calculate their distance
		for satn in np.arange(6):
			sat_x = cen_fits['SAT_0{}'.format(satn)]['fxe'](DD)
			sat_y = cen_fits['SAT_0{}'.format(satn)]['fye'](DD)
			sat_z = cen_fits['SAT_0{}'.format(satn)]['fze'](DD)

			sat_cen = yt.YTArray([sat_x, sat_y, sat_z], 'kpc')

			sat_dist = np.sqrt((cen_x - sat_x)**2. + (cen_y - sat_y)**2. + (cen_z - sat_z)**2.)

			#Remove satellites that are too far
			if sat_dist > satellite_distance_limit[DD_ind]:
				sub_sphere = ds.sphere(sat_cen, (sat_sph_rad, "kpc"))
				refine_box -= sub_sphere
				star_sphere -= sub_sphere
				cgm_sphere -= sub_sphere
				cen_sphere -= sub_sphere

		#Remove ISM area from CGM sphere
		cgm_sphere -= cen_sphere

	#Create cuts
	if True:
		temp_cut = refine_box.cut_region(["(obj['temperature'] < {} )".format(temp)])
		dens_cut = refine_box.cut_region(["(obj['density'] > {})".format(density_cut[DD_ind])])
		ism_cut = cen_sphere.cut_region(["(obj['temperature'] < {} ) & (obj['density'] > {})".format(temp,density_cut[DD_ind])])
		cgm_cut = cgm_sphere.cut_region(["(obj['temperature'] > {} ) | (obj['density'] < {})".format(temp,density_cut[DD_ind])])
		pink_cut = cgm_sphere.cut_region(["(obj['temperature'] < {} ) & (obj['density'] < {})".format(low_temp, density_cut[DD_ind])])
		purple_cut = cgm_sphere.cut_region(["(obj['temperature'] > {}) & (obj['temperature'] < {}) & (obj['density'] < {})".format(low_temp, med_temp, density_cut[DD_ind])])
		green_cut = cgm_sphere.cut_region(["(obj['temperature'] > {}) & (obj['temperature'] < {}) & (obj['density'] < {})".format(med_temp, high_temp, density_cut[DD_ind])])
		yellow_cut = cgm_sphere.cut_region(["(obj['temperature'] > {} ) & (obj['density'] < {})".format(high_temp, density_cut[DD_ind])])

	#Cut on Temperature
	if False:	
		prj = yt.ProjectionPlot(ds, 'z', "density", center = cen_cen, data_source = temp_cut, width = (proj_width, 'kpc'))
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.save('TempCut{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = temp_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.save('FaceOnTemp{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = temp_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.save('EdgeOnTemp{}.png'.format(DD))

	#Cut on Density
	if False:
		prj = yt.ProjectionPlot(ds, 'z', "density", center = cen_cen, data_source = dens_cut, width = (proj_width, 'kpc'))
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.save('DensCut{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = dens_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.save('FaceOnDens{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = dens_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.save('EdgeOnDens{}.png'.format(DD))

	#Cut on Temperature and Density for ISM
	if True:
		#prj = yt.ProjectionPlot(ds, 'z', "density", center = cen_cen, data_source = ism_cut, width = (proj_width, 'kpc'))
		#prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		#prj.set_cmap(field="density", cmap=density_color_map)
		#prj.save('TDCut{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = ism_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('FaceOnISM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = ism_cut)
		#prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('FaceOnMetalISM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = ism_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('EdgeOnISM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = ism_cut)
		#prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('EdgeOnMetalISM{}.png'.format(DD))

	#Cut on Temperature and Density for main CGM
	if True:
		#prj = yt.ProjectionPlot(ds, 'z', "density", center = cen_cen, data_source = cgm_cut, width = (proj_width, 'kpc'))
		#prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		#prj.set_cmap(field="density", cmap=density_color_map)
		#prj.save('TDNeg{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = cgm_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('FaceOnCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = cgm_cut)
		#prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('FaceOnMetalCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = cgm_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('EdgeOnCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = cgm_cut)
		#prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('EdgeOnMetalCGM{}.png'.format(DD))

	#Cut on Temperature and Density for pink CGM
	if True:
		#prj = yt.ProjectionPlot(ds, 'z', "density", center = cen_cen, data_source = pink_cut, width = (proj_width, 'kpc'))
		#prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		#prj.set_cmap(field="density", cmap=density_color_map)
		#prj.save('TDNegPink{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = pink_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('FaceOnPinkCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = pink_cut)
		#prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('FaceOnMetalPinkCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = pink_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('EdgeOnPinkCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = pink_cut)
		#prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('EdgeOnMetalPinkCGM{}.png'.format(DD))

	#Cut on Temperature and Density for purple CGM
	if True:
		#prj = yt.ProjectionPlot(ds, 'z', "density", center = cen_cen, data_source = purple_cut, width = (proj_width, 'kpc'))
		#prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		#prj.set_cmap(field="density", cmap=density_color_map)
		#prj.save('TDNegPurple{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = purple_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('FaceOnPurpleCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = purple_cut)
		#prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('FaceOnMetalPurpleCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = purple_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('EdgeOnPurpleCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = purple_cut)
		#prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('EdgeOnMetalPurpleCGM{}.png'.format(DD))

	#Cut on Temperature and Density for green CGM
	if True:
		#prj = yt.ProjectionPlot(ds, 'z', "density", center = cen_cen, data_source = green_cut, width = (proj_width, 'kpc'))
		#prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		#prj.set_cmap(field="density", cmap=density_color_map)
		#prj.save('TDNegGreen{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = green_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('FaceOnGreenCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = green_cut)
		#prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('FaceOnMetalGreenCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = green_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('EdgeOnGreenCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = green_cut)
		#prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('EdgeOnMetalGreenCGM{}.png'.format(DD))

	#Cut on Temperature and Density for yellow CGM
	if True:
		#prj = yt.ProjectionPlot(ds, 'z', "density", center = cen_cen, data_source = yellow_cut, width = (proj_width, 'kpc'))
		#prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		#prj.set_cmap(field="density", cmap=density_color_map)
		#prj.save('TDNegYellow{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = yellow_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('FaceOnYellowCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = gas_ang_mom_norm, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = yellow_cut)
		#prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('FaceOnMetalYellowCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'density', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = yellow_cut)
		prj.set_zlim(('gas','density'), zmin = dens_proj_min, zmax = dens_proj_max)
		prj.set_cmap(field="density", cmap=density_color_map)
		prj.save('EdgeOnYellowCGM{}.png'.format(DD))

		prj = yt.OffAxisProjectionPlot(ds, normal = edge_on_dir, fields = 'metal_mass', center = cen_cen, width=(proj_width, 'kpc'), north_vector=north_vector, data_source = yellow_cut)
		#prj.set_zlim(('gas','metal_mass'), zmin = metal_proj_min, zmax = metal_proj_max)
		prj.set_cmap(field="metal_mass", cmap=metal_color_map)
		prj.save('EdgeOnMetalYellowCGM{}.png'.format(DD))

	#Calculate metallicity for stars, ISM, and CGM
	if True:
		star_metallicity = star_sphere.quantities.weighted_average_quantity(star_metallicity_fraction, star_particle_mass)
		ism_metallicity = ism_cut.quantities.weighted_average_quantity(metallicity_field, metallicity_weight)
		cgm_metallicity = cgm_cut.quantities.weighted_average_quantity(metallicity_field, metallicity_weight)
		pink_metallicity = pink_cut.quantities.weighted_average_quantity(metallicity_field, metallicity_weight)
		purple_metallicity = purple_cut.quantities.weighted_average_quantity(metallicity_field, metallicity_weight)
		green_metallicity = green_cut.quantities.weighted_average_quantity(metallicity_field, metallicity_weight)
		yellow_metallicity = yellow_cut.quantities.weighted_average_quantity(metallicity_field, metallicity_weight)

		log_star_metallicity[DD_ind] = np.log10(star_metallicity.in_units('Zsun').value)
		log_ism_metallicity[DD_ind] = np.log10(ism_metallicity.in_units('Zsun').value)
		log_cgm_metallicity[DD_ind] = np.log10(cgm_metallicity.in_units('Zsun').value)
		log_pink_metallicity[DD_ind] = np.log10(pink_metallicity.in_units('Zsun').value)
		log_purple_metallicity[DD_ind] = np.log10(purple_metallicity.in_units('Zsun').value)
		log_green_metallicity[DD_ind] = np.log10(green_metallicity.in_units('Zsun').value)
		log_yellow_metallicity[DD_ind] = np.log10(yellow_metallicity.in_units('Zsun').value)

	#Calculate metal mass for stars, ISM, and CGM
	if True:
		star_metal_mass = star_sphere.quantities.total_quantity([("all", "metal_mass")])
		ism_metal_mass = ism_cut.quantities.total_quantity([("gas", "metal_mass")])
		cgm_metal_mass = cgm_cut.quantities.total_quantity([("gas", "metal_mass")])
		pink_metal_mass = pink_cut.quantities.total_quantity([("gas", "metal_mass")])
		purple_metal_mass = purple_cut.quantities.total_quantity([("gas", "metal_mass")])
		green_metal_mass = green_cut.quantities.total_quantity([("gas", "metal_mass")])
		yellow_metal_mass = yellow_cut.quantities.total_quantity([("gas", "metal_mass")])

		log_star_metal_mass[DD_ind] = np.log10(star_metal_mass.in_units('Msun').value)
		log_ism_metal_mass[DD_ind] = np.log10(ism_metal_mass.in_units('Msun').value)
		log_cgm_metal_mass[DD_ind] = np.log10(cgm_metal_mass.in_units('Msun').value)
		log_pink_metal_mass[DD_ind] = np.log10(pink_metal_mass.in_units('Msun').value)
		log_purple_metal_mass[DD_ind] = np.log10(purple_metal_mass.in_units('Msun').value)
		log_green_metal_mass[DD_ind] = np.log10(green_metal_mass.in_units('Msun').value)
		log_yellow_metal_mass[DD_ind] = np.log10(yellow_metal_mass.in_units('Msun').value)

	#Calculate total mass for stars, ISM, and CGM
	if True:
		star_total_mass = star_sphere.quantities.total_quantity(star_particle_mass)
		ism_total_mass = ism_cut.quantities.total_quantity([("gas", "cell_mass")])
		cgm_total_mass = cgm_cut.quantities.total_quantity([("gas", "cell_mass")])
		pink_total_mass = pink_cut.quantities.total_quantity([("gas", "cell_mass")])
		purple_total_mass = purple_cut.quantities.total_quantity([("gas", "cell_mass")])
		green_total_mass = green_cut.quantities.total_quantity([("gas", "cell_mass")])
		yellow_total_mass = yellow_cut.quantities.total_quantity([("gas", "cell_mass")])

		log_star_total_mass[DD_ind] = np.log10(star_total_mass.in_units('Msun').value)
		log_ism_total_mass[DD_ind] = np.log10(ism_total_mass.in_units('Msun').value)
		log_cgm_total_mass[DD_ind] = np.log10(cgm_total_mass.in_units('Msun').value)
		log_pink_total_mass[DD_ind] = np.log10(pink_total_mass.in_units('Msun').value)
		log_purple_total_mass[DD_ind] = np.log10(purple_total_mass.in_units('Msun').value)
		log_green_total_mass[DD_ind] = np.log10(green_total_mass.in_units('Msun').value)
		log_yellow_total_mass[DD_ind] = np.log10(yellow_total_mass.in_units('Msun').value)

	#Calculate metallicity scatter for stars, ISM, and CGM
	if True:
		star_values = star_sphere[star_metallicity_fraction].in_units('Zsun').value
		star_weight_values = star_sphere[star_particle_mass].value
		log_star_values = np.log10(star_values)
		star_average, star_stdev = weighted_avg_and_stdev(log_star_values, star_weight_values)
		star_above_avg[DD_ind] = star_average + star_stdev
		star_below_avg[DD_ind] = star_average - star_stdev

		ism_values = ism_cut["metallicity"].in_units('Zsun').value
		ism_weight_values = ism_cut["cell_mass"].value
		log_ism_values = np.log10(ism_values)
		ism_average, ism_stdev = weighted_avg_and_stdev(log_ism_values, ism_weight_values)
		ism_above_avg[DD_ind] = ism_average + ism_stdev
		ism_below_avg[DD_ind] = ism_average - ism_stdev

		cgm_values = cgm_cut["metallicity"].in_units('Zsun').value
		cgm_weight_values = cgm_cut["cell_mass"].value
		log_cgm_values = np.log10(cgm_values)
		cgm_average, cgm_stdev = weighted_avg_and_stdev(log_cgm_values, cgm_weight_values)
		cgm_above_avg[DD_ind] = cgm_average + cgm_stdev
		cgm_below_avg[DD_ind] = cgm_average - cgm_stdev

		pink_values = pink_cut["metallicity"].in_units('Zsun').value
		pink_weight_values = pink_cut["cell_mass"].value
		log_pink_values = np.log10(pink_values)
		pink_average, pink_stdev = weighted_avg_and_stdev(log_pink_values, pink_weight_values)
		pink_above_avg[DD_ind] = pink_average + pink_stdev
		pink_below_avg[DD_ind] = pink_average - pink_stdev

		purple_values = purple_cut["metallicity"].in_units('Zsun').value
		purple_weight_values = purple_cut["cell_mass"].value
		log_purple_values = np.log10(purple_values)
		purple_average, purple_stdev = weighted_avg_and_stdev(log_purple_values, purple_weight_values)
		purple_above_avg[DD_ind] = purple_average + purple_stdev
		purple_below_avg[DD_ind] = purple_average - purple_stdev

		green_values = green_cut["metallicity"].in_units('Zsun').value
		green_weight_values = green_cut["cell_mass"].value
		log_green_values = np.log10(green_values)
		green_average, green_stdev = weighted_avg_and_stdev(log_green_values, green_weight_values)
		green_above_avg[DD_ind] = green_average + green_stdev
		green_below_avg[DD_ind] = green_average - green_stdev

		yellow_values = yellow_cut["metallicity"].in_units('Zsun').value
		yellow_weight_values = yellow_cut["cell_mass"].value
		log_yellow_values = np.log10(yellow_values)
		yellow_average, yellow_stdev = weighted_avg_and_stdev(log_yellow_values, yellow_weight_values)
		yellow_above_avg[DD_ind] = yellow_average + yellow_stdev
		yellow_below_avg[DD_ind] = yellow_average - yellow_stdev

	#Calculate metals returned
	if True:
		current_stellar_mass = star_sphere[star_particle_mass]
		Zstar = star_sphere[star_metallicity_fraction]
		equation_returned = ((1 - Zstar) * StarMetalYield + StarMassEjectionFraction * Zstar)
		M_return = ((current_stellar_mass/(1 - equation_returned))*equation_returned).in_units("Msun")
		M_return_total = np.sum(M_return)
		log_metals_returned[DD_ind] = np.log10(M_return_total)