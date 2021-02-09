from astropy.table import Table, unique
import matplotlib.pyplot as plt
import foggie.render.shade_maps as sm
from foggie.utils.consistency import *
import numpy as np 

absorber_axis_limits = {'radius_corrected':(0,200), 'radial_velocity_corrected':(-500,500), 
                        'b_effective':(0,50), 'vel_dispersion':(-10,100), 'vx_corrected':(-500,500), 
                        'vy_corrected':(-500,500), 'vz_corrected':(-500,500), 'redshift':(2,0)}
absorber_axis_labels = {'radius_corrected':'Radius [kpc]', 'b_effective':'effective b [km s$^{-1}$]', 
                        'radial_velocity_corrected':'Radial Velocity [km s$^{-1}$]', 'redshift':'z'}

ion_colors={'H I': '#888888', 'C II': '#440000', 'C III':'#990000', 'C IV':'#FF0000', 'Mg II':'#999900', 'Si II':'#000044', 'Si III':'#000099', 'Si IV':'#0000FF', 'O VI':'#00FF00'}

def read_absorber_catalog(filename, format='astropy'):
    #read the table
    ab = Table.read(filename, format='ascii.basic')
    
    #assign colors to phases using FOGGIE utilities 
    ab['phase'] = categorize_by_temp(np.log10(ab['temperature']))
    colors = []
    for p in ab['phase']:   
        colors.append( new_phase_color_key[str.encode(p)] ) 

    ab['temp_colors'] = colors   

    ab['metal'] = categorize_by_metals(ab['metallicity'])
    metals = []
    for p in ab['metal']:   
        metals.append( new_metals_color_key[str.encode(p)] ) 

    ab['metal_colors'] = metals   

    # add labels for each component 
    sightline_label = []
    absorber_label = []
    for i in ab['absorber_index']: 
        sightline_label.append(i[0:4])
        absorber_label.append(i[-1])
    
    ab['sightline_label'] = sightline_label
    ab['absorber_label'] = absorber_label

    ab['b_thermal'] = (1.38e-16*ab['temperature'] / 16. / 1.67e-24)**0.5 / 1e5 # thermal b-value in km/s O VI ONLY!!!!!
    ab['b_nonthermal'] = ab['vel_dispersion'] / 2.
    ab['b_effective'] = (ab['b_thermal']**2 + ab['b_nonthermal']**2)**0.5

    ab['cooling_time'] = ab['cooling_time'] / 3.156e7 
    ab['radius_corrected'] = ab['radius_corrected'] / 3.086e21
    for v in ['vx_corrected', 'vy_corrected','vz_corrected', 'radial_velocity_corrected', 'tangential_velocity_corrected']: 
        ab[v] = ab[v] / 1e5 
    ab['cell_mass'] = ab['cell_mass'] / 1.989e33
    ab['cell_size'] = ab['cell_volume']**0.333333333 / 3.086e21

    for field in ['temperature', 'density', 'cooling_time', 'pressure', 'entropy']: 
        ab[field] = np.log10(ab[field]) 

    if ('pandas' in format): 
        return ab.to_pandas()
    else: 
        return ab

def select_absorber_temperature(ab,T): 
    ab_t = ab[(ab['temperature'] > T[0]) & (ab['temperature'] < T[1])]
    return ab_t

def select_absorber_redshift(ab,z): 
    ab_z = ab[(ab['col_dens'] > 11) & (ab['redshift'] > z-0.03) & (ab['redshift'] < z+0.03)]
    return ab_z

#make the plots 
def abs_z(ab_z, var1, var2): 
 
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


def plot_absorbers(abcat, var1, var2, limit1, limit2, ion, code, prefix): 
    """ plots the absorbers color-coded by temperature in arbitrary FOGGIE axes"""
    plt.figure(figsize=(8,8)) 
    for rr, rrv, rcolor in zip(abcat[var1][abcat['name'] == ion], abcat[var2][abcat['name'] == ion], abcat[code][abcat['name'] == ion]): 
        plt.plot(rr, rrv, 'o', markersize=8, color='#FFFFFF')
        plt.plot(rr, rrv, 'o', markersize=5, color=rcolor)
    plt.xlim(limit1)   
    plt.ylim(limit2)
    plt.xlabel(axes_label_dict[var1])
    plt.ylabel(axes_label_dict[var2])
    plt.savefig(prefix+'_'+ion.replace(' ','')+'_'+var1+'_'+var2+'_'+code+'.png',transparent=True)
    plt.close()

    
def plot_ions(abcat, var1, var2, limit1, limit2, prefix): 
    """ plots the absorbers color coded by ion in arbitrary FOGGIE axes"""
    plt.figure(figsize=(8,8)) 
    
    for ion in unique(abcat, 'name')['name']: 
        ab_ion = abcat[abcat['name'] == ion]
        for rr, rrv in zip(ab_ion[var1], ab_ion[var2]): 
            plt.plot(rr, rrv, '.', markersize=7, color=ion_colors[ion], alpha=0.2, fillstyle='full', markeredgewidth=0.0)

    plt.xlim(limit1)   
    plt.ylim(limit2)
    plt.xlabel(axes_label_dict[var1])
    plt.ylabel(axes_label_dict[var2])
    plt.savefig(prefix+'_abs_'+var1+'_'+var2+'_ion.png',transparent=True, dpi=300)
    plt.close()
    