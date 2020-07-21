from astropy.table import Table
import matplotlib.pyplot as plt
import foggie.render.shade_maps as sm
from foggie.utils.consistency import *
import numpy as np 

absorber_axis_limits = {'radius_corrected':(0,200), 'radial_velocity_corrected':(-500,500), 
                        'b_effective':(0,50), 'vel_dispersion':(-10,100), 'vx_corrected':(-500,500), 
                        'vy_corrected':(-500,500), 'vz_corrected':(-500,500), 'redshift':(2,0)}
absorber_axis_labels = {'radius_corrected':'Radius [kpc]', 'b_effective':'effective b [km s$^{-1}$]', 
                        'radial_velocity_corrected':'Radial Velocity [km s$^{-1}$]', 'redshift':'z'}


def read_absorber_catalog(filename):
    #read the table
    ab = Table.read(filename, format='ascii.basic')
    ab['phase'] = categorize_by_temp(np.log10(ab['temperature']))

    #assign colors to phases using FOGGIE utilities 
    colors = []
    for p in zip(ab['phase']):
        colors.append( new_phase_color_key[str.encode(p[0])] ) 

    ab['colors'] = colors   

    # add labels for each component 
    sightline_label = []
    absorber_label = []
    for i in ab['absorber_index']: 
        sightline_label.append(i[0:4])
        absorber_label.append(i[-1])
    
    ab['sightline_label'] = sightline_label
    ab['absorber_label'] = absorber_label

    ab['b_thermal'] = (1.38e-16*ab['temperature'] / 16. / 1.67e-24)**0.5 / 1e5 # thermal b-value in km/s
    ab['b_nonthermal'] = ab['vel_dispersion'] / 2.
    ab['b_effective'] = (ab['b_thermal']**2 + ab['b_nonthermal']**2)**0.5

    ab['b_thermal'] = (1.38e-16*ab['temperature'] / 16. / 1.67e-24)**0.5 / 1e5 # thermal b-value in km/s

    ab['cooling_time'] = ab['cooling_time'] / 3.156e7 

    for field in ['temperature', 'density', 'cooling_time', 'pressure', 'entropy']: 
        ab[field] = np.log10(ab[field]) 

    return ab


def select_absorber_temperature(ab,T): 
    ab_t = ab[(ab['temperature'] > T[0]) & (ab['temperature'] < T[1])]
    return ab_t

def select_absorber_redshift(ab,z): 
    ab_z = ab[(ab['col_dens'] > 11) & (ab['redshift'] > z-0.03) & (ab['redshift'] < z+0.03)]
    return ab_z

#make the plots 
def make_absorber_plot(ab_z, var1, var2): 
 
    zlabel = str(np.round(ab_z['redshift'][0]*10.) / 10.)
    
    plt.figure(figsize=(8,8)) 
    for rr, rrv, rcolor in zip(ab_z[var1], ab_z[var2], ab_z['colors']): 
        plt.plot(rr, rrv, 'o', markersize=8, color='#000000')
        plt.plot(rr, rrv, 'o', markersize=5, color=rcolor)
    plt.xlim(absorber_axis_limits[var1])   
    plt.ylim(absorber_axis_limits[var2])
    plt.xlabel(absorber_axis_labels[var1])
    plt.ylabel(absorber_axis_labels[var2])
    plt.savefig('abs_'+var1+'_'+var2+'_z_'+zlabel+'.png',transparent=True)