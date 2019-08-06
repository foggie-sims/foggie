#virial_radii.py
#Summary: Determine virial radii for halo_008508 at each redshift and plot results
#Author: Kathleen Hamilton-Campos, SASP intern at STScI, summer 2019 - kahamil@umd.edu

#Import libraries
from astropy.cosmology import Planck15 as cosmo
import numpy as np
import yt
import math
import matplotlib.pyplot as plt

#Close all open plots to prevent over-writing
plt.close("all")

#Input guesses for minimum and maximum virial radii
min_rad = [88.3, 101.7, 113.6, 123.6, 135.2, 143.3, 151.2, 158.9, 167.2]
max_rad = [88.4, 101.8, 113.7, 123.7, 135.3, 143.4, 151.3, 159.0, 167.3]

#Choose how precisely the program looks between min_rad and max_rad
rad_del = .1

#List of DDs
simulations = [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]

#Find the virial radii
for ind, DD in enumerate(simulations):

	#Center galaxy
	if True:
		cen_fits = np.load("/Users/khamilton/Desktop/Scripts/nref11n_nref10f_interpolations_DD0150_new.npy", allow_pickle = True)[()]

		cen_x = cen_fits['CENTRAL']['fxe'](DD)
		cen_y = cen_fits['CENTRAL']['fye'](DD)
		cen_z = cen_fits['CENTRAL']['fze'](DD)

		cen_cen = yt.YTArray([cen_x, cen_y, cen_z], 'kpc')

	#Load dataset and add particle filters
	if True:
		if DD < 1000:
			ds = yt.load('/Users/khamilton/Desktop/halo_008508/nref11n_nref10f/DD0{}/DD0{}'.format(DD,DD))
		else:
			ds = yt.load('/Users/khamilton/Desktop/halo_008508/nref11n_nref10f/DD{}/DD{}'.format(DD,DD))

		def _stars(pfilter, data): return data[(pfilter.filtered_type, "particle_type")] == 2
		def _darkmatter(pfilter, data): return data[(pfilter.filtered_type, "particle_type")] == 4
		yt.add_particle_filter("stars",function=_stars, filtered_type='all',requires=["particle_type"])
		yt.add_particle_filter("darkmatter",function=_darkmatter, filtered_type='all',requires=["particle_type"])
		ds.add_particle_filter('stars')
		ds.add_particle_filter('darkmatter')

	#Get cosmological information
	if True:
		zsnap = ds.get_parameter('CosmologyCurrentRedshift')

		p_crit = (cosmo.critical_density(zsnap))

		virial = ((200*p_crit)).value

	#Compute density
	if False:
		count = 0

		dens_rad = yt.YTArray(np.arange(min_rad[ind], max_rad[ind], rad_del), 'kpc')

		ob_sphere = ds.sphere(cen_cen, (dens_rad[count]))

		star_mass = ob_sphere.quantities.total_quantity([("stars","particle_mass")])
		darkmatter_mass = ob_sphere.quantities.total_quantity([("darkmatter", "particle_mass")])
		gas_mass = ob_sphere.quantities.total_quantity([("gas", "matter_mass")])

		total_mass = star_mass + darkmatter_mass + gas_mass

		total_vol = ((4/3)*math.pi*(dens_rad[count]**3)).to("cm**3")

		total_dens = total_mass / total_vol

		#Narrow in on virial radius
		while total_dens > virial:
			count += 1

			if count > len(dens_rad)-1:
				print('Array length exceeded')
				break

			ob_sphere = ds.sphere(cen_cen, (dens_rad[count]))
			print("Radius: ", dens_rad[count])

			star_mass = ob_sphere.quantities.total_quantity([("stars","particle_mass")])
			darkmatter_mass = ob_sphere.quantities.total_quantity([("darkmatter", "particle_mass")])
			gas_mass = ob_sphere.quantities.total_quantity([("gas", "matter_mass")])

			total_mass = star_mass + darkmatter_mass + gas_mass

			total_vol = ((4/3)*math.pi*(dens_rad[count]**3)).to("cm**3")

			total_dens = total_mass / total_vol
			print("Total Density: ", total_dens)
			print("Virial: ", virial)

	#Plotting results
	if True:
		vir_rad = (min_rad[ind] + max_rad[ind])/2
		plt.plot(zsnap,vir_rad,'*', label = '{}'.format(DD))
		plt.title('Virial Radius across Time')
		plt.xlabel('Redshift (z)')
		plt.ylabel('Virial Radius (kpc)')
		plt.legend(loc = "upper right")
		plt.savefig("VirialRedshifts.png")

#Converting to a plot with lookback times and thousands of light years
if True:
	virial = np.array([88. , 102., 114., 124., 135., 151., 159., 167.])
	kpc_kly = 3.26
	virial_kly = virial*kpc_kly
	lookbacks = [9.798, 9.262, 8.725, 8.189, 7.653, 6.580, 6.043, 5.506]
	plt.plot(lookbacks,virial_kly,'*')
	plt.title('Virial Radius across Time')
	plt.xlabel('Light Travel Time (billions of years)')
	plt.ylabel('Virial Radius (thousands of lightyears)')
	plt.savefig("VirialLightTravel.png")