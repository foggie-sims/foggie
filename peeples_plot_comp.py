#! /usr/bin/env python

#peeples_plot_comp.py
#Summary: Uses Molly's code from creating fig 9a in Tumlinson, Peeples, Werk 2017 ARAA. Adds a subplot showing distribution of metals in halo_008508.
#Author: modifications by Kathleen Hamilton-Campos, SASP intern at STScI, summer 2019 - kahamil@umd.edu

#Import libraries
import sys
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
mpl.rcParams['font.family'] = 'stixgeneral'
from astropy import units as u

#Close all open plots to prevent over-writing
plt.close("all")

#Matching colors to the Tumlinson, Peeples, Werk ARAA plots
star_color = '#d73027'
ism_color = '#4575b4'
cool_cgm_color = '#984ea3'
metal_color = "tab:gray"
pink_color = "lightcoral"
purple_color = "blueviolet"
green_color = "forestgreen"
yellow_color = "gold"

#halo_008508_arrays.py
if True:
    #Simulations
    simulations = [600, 700, 800, 900, 1000, 1200, 1300, 1400]
    
    #Analysis
    virial_radius = [88., 102., 114., 124., 135., 151., 159., 167.]
    redshift = [1.67203864, 1.44355343, 1.25523385, 1.09649559, 0.96024456, 0.73680431, 0.64341981, 0.55938858]
    lookbacks = [9.798, 9.262, 8.725, 8.189, 7.653, 6.580, 6.043, 5.506]
    density_cut = [7.5e-25, 7.5e-25, 7.5e-25, 1.e-25, 1.e-25, 1.e-25, 2.5e-26, 2.5e-26]
    central_radius_cut = [6., 4., 4., 8., 10., 10., 20., 20.]
    
    #Metal Mass
    log_star_metal_mass = [8.75613115, 8.90493106, 8.97880156, 9.01590204, 9.03431259, 9.07707769, 9.09095877, 9.0989926]
    log_ism_metal_mass = [7.41610945, 7.01884864, 7.22399948, 7.34495633, 7.62640389, 7.90247196, 7.95563369, 7.97245207]
    log_cgm_metal_mass = [6.96129056, 7.21389993, 7.16571149, 6.960262, 6.93344747, 6.88127286, 6.70369428, 6.72991688]
    log_pink_metal_mass = [6.42264011, 6.79510715, 6.58298529, 6.01051374, 5.92205321, 5.76084686, 5.10499359, 5.37331619]
    log_purple_metal_mass = [6.30830197, 6.57219856, 6.6217368, 6.56151396, 6.38860369, 6.24905933, 6.26831868, 6.23935507]
    log_green_metal_mass = [6.42147444, 6.53451834, 6.69423788, 6.64148014, 6.57539623, 6.65215261, 6.46032455, 6.45545263]
    log_yellow_metal_mass = [6.23788467, 6.44667773, 6.22615504, 4.88659956, 6.18598284, 5.88536981, 5.27000911, 5.73545304]
    
    #Metals Returned
    log_metals_returned = [8.80915847, 8.94573277, 9.01935911, 9.05387849, 9.07721023, 9.12616953, 9.14387134, 9.15165251]
    
    #Total Mass
    log_star_total_mass = [10.29722092, 10.43045056, 10.50401991, 10.53781784, 10.56252562, 10.61318673, 10.63192447, 10.63963946]
    log_ism_total_mass = [9.11001965, 8.98071814, 9.23518941, 9.6813505, 9.87878766, 9.91896161, 10.02418574, 10.08030778]
    log_cgm_total_mass = [9.9224015, 10.00454427, 10.02347962, 9.94589795, 9.94444685, 9.96049841, 9.92833352, 9.96914984]
    log_pink_total_mass = [9.05796593, 9.0728697, 8.96851864, 8.40568938, 8.22681277, 8.39579674, 7.50616937, 7.74316116]
    log_purple_total_mass = [9.76319086, 9.84856084, 9.87171705, 9.8203237, 9.79942426, 9.73414711, 9.76129191, 9.78848361]
    log_green_total_mass = [9.08459454, 9.17492063, 9.29628956, 9.28774726, 9.31396042, 9.52426397, 9.42077554, 9.47766643]
    log_yellow_total_mass = [8.30490154, 8.56170041, 8.31070213, 7.34861815, 8.4293867, 8.06439883, 7.59941348, 8.04383945]
    
    #Metallicity
    log_star_metallicity = [0.34664047, 0.36221073, 0.36251189, 0.36581443, 0.3595172, 0.3516212, 0.34676453, 0.34708337]
    log_ism_metallicity = [0.19382003, -0.07413927, -0.12345969, -0.44866394, -0.36465354, -0.12875942, -0.18082181, -0.22012547]
    log_cgm_metallicity = [-1.07338071, -0.90291412, -0.9700379, -1.09790571, -1.12326916, -1.19149532, -1.33690901, -1.35150273]
    log_pink_metallicity = [-0.74759559, -0.39003232, -0.49780312, -0.50744541, -0.41702933, -0.74721965, -0.51344555, -0.48211474]
    log_purple_metallicity = [-1.56715866, -1.38863205, -1.36225003, -1.37107951, -1.52309034, -1.59735754, -1.605243, -1.66139831]
    log_green_metallicity = [-0.77538986, -0.75267205, -0.71432145, -0.7585369, -0.85083396, -0.98438113, -1.07272076, -1.13448357]
    log_yellow_metallicity = [-0.17928663, -0.22729246, -0.19681686, -0.57428835, -0.35567363, -0.29129879, -0.44167413, -0.42065618]
    
    #Metallicity Scatter
    star_above_avg = [0.55433248, 0.56024463, 0.55303596, 0.55282943, 0.55087418, 0.5472402 , 0.54247595, 0.5409719 ]
    star_below_avg = [0.02717052, 0.06617214, 0.08230201, 0.09301548, 0.07755932, 0.05813121, 0.05176889, 0.05600171]
    ism_above_avg = [ 0.38546582,  0.10261412,  0.04352738, -0.25387694, -0.18070932, 0.08769022,  0.11404643,  0.07332398]
    ism_below_avg = [-0.08314182, -0.37300153, -0.43350158, -0.87829904, -0.76536607, -0.54628381, -0.89091468, -1.07389127]
    cgm_above_avg = [-0.9251934 , -0.77780706, -0.80729691, -0.92058695, -0.96053067, -0.94194043, -1.10535983, -1.0861565 ]
    cgm_below_avg = [-2.47887251, -2.70707491, -2.68641952, -2.63861884, -2.67774697, -2.49657654, -2.52120572, -2.63578073]
    pink_above_avg = [-0.52669418,  0.0109657 , -0.13167021, -0.25709479, -0.11785395, -0.49672121, -0.19709807, -0.1645786 ]
    pink_below_avg = [-1.76299288, -1.4438627 , -1.49683293, -1.39289017, -1.25354003, -1.4588543 , -1.19449295, -1.71464562]
    purple_above_avg = [-1.36388222, -1.33220844, -1.2839254 , -1.27928819, -1.42027794, -1.3794305 , -1.44164538, -1.41830124]
    purple_below_avg = [-2.65307067, -2.94154968, -2.91530981, -2.82387858, -2.87247583, -2.79522841, -2.72492237, -2.88990756]
    green_above_avg = [-0.51010363, -0.53585434, -0.42278747, -0.46685428, -0.53784839, -0.71658536, -0.80754616, -0.87721567]
    green_below_avg = [-1.47768456, -1.51671746, -1.58872672, -1.51199499, -1.61955669, -1.72127415, -1.69460371, -1.77089699]
    yellow_above_avg = [ 0.03782155,  0.01271684,  0.01202342, -0.3258607 , -0.13297435, -0.05795236, -0.26776021, -0.18813011]
    yellow_below_avg = [-0.79755886, -0.83139605, -0.65441034, -1.21310098, -0.93744487, -0.99958113, -1.28577526, -1.05028427]

#Setting arrays to zero for filling
star_frac = np.zeros(len(simulations))
ism_frac = np.zeros(len(simulations))
cgm_frac = np.zeros(len(simulations))
pink_frac = np.zeros(len(simulations))
purple_frac = np.zeros(len(simulations))
green_frac = np.zeros(len(simulations))
yellow_frac = np.zeros(len(simulations))

#Filling arrays - the fraction of available metals is each section's metal mass divided by the metals returned
for DD_ind, DD in enumerate(simulations):
    star_frac[DD_ind] = (10**log_star_metal_mass[DD_ind])/(10**log_metals_returned[DD_ind])
    ism_frac[DD_ind] = (10**log_ism_metal_mass[DD_ind])/(10**log_metals_returned[DD_ind])
    cgm_frac[DD_ind] = (10**log_cgm_metal_mass[DD_ind])/(10**log_metals_returned[DD_ind])
    pink_frac[DD_ind] = (10**log_pink_metal_mass[DD_ind])/(10**log_metals_returned[DD_ind])
    purple_frac[DD_ind] = (10**log_purple_metal_mass[DD_ind])/(10**log_metals_returned[DD_ind])
    green_frac[DD_ind] = (10**log_green_metal_mass[DD_ind])/(10**log_metals_returned[DD_ind])
    yellow_frac[DD_ind] = (10**log_yellow_metal_mass[DD_ind])/(10**log_metals_returned[DD_ind])

#For checking what the raw numbers look like
if False:
    plt.plot(log_star_total_mass, star_frac)
    plt.plot(log_star_total_mass, ism_frac)
    plt.plot(log_star_total_mass, cgm_frac)
    plt.plot(log_star_total_mass, pink_frac)
    plt.plot(log_star_total_mass, purple_frac)
    plt.plot(log_star_total_mass, green_frac)
    plt.plot(log_star_total_mass, yellow_frac)

#Creating side-by-side plots, one star/ISM/CGM, the other star/ISM/CGM temperature divisions
#Pink, purple, green, and yellow refer to the colors in 8c-d of Tumlinson, Peeples, Werk 2017 ARAA
#Pink: T < 10^4K
#Purple: 10^4K < T < 10^5K
#Green: 10^5K < T < 10^6K
#Yellow: T > 10^6K
if True:
    fig, axes = plt.subplots(1,2, figsize = (12,6))

    #Creating a starting point for the lowest fraction
    zeros = np.zeros(len(log_star_total_mass))

    #Calculating the average so the plots will have smooth lines
    star_frac_avg = np.average(star_frac)
    ism_frac_avg = np.average(ism_frac)
    cgm_frac_avg = np.average(cgm_frac)
    pink_frac_avg = np.average(pink_frac)
    purple_frac_avg = np.average(purple_frac)
    green_frac_avg = np.average(green_frac)
    yellow_frac_avg = np.average(yellow_frac)

    #Creating the stars/ISM/CGM subplot
    axes[0].plot([8.,11.5],[1,1], linestyle='dashed', color='black', linewidth=2)
    axes[0].annotate('Peeples et al. (2014)', xy=(11.45, 1.02), xytext=(11.45, 1.02), size=14, ha='right')
    starbar1 = axes[0].fill_between(log_star_total_mass, ism_frac_avg, star_frac_avg, facecolor = star_color, edgecolor = star_color, label = "stars")
    ismbar1 = axes[0].fill_between(log_star_total_mass, cgm_frac_avg, ism_frac_avg, facecolor = ism_color, edgecolor = ism_color, label = "ism")
    cgmbar = axes[0].fill_between(log_star_total_mass, zeros, cgm_frac_avg, facecolor = cool_cgm_color, edgecolor = cool_cgm_color, label = "cgm")

    lg1 = axes[0].legend(handles=[starbar1, ismbar1,cgmbar], loc="upper left", fontsize=12, ncol=3, columnspacing=0.5, 
        labelspacing=0.4, borderpad=0.1, handletextpad=0.1, handlelength=3.2 )
    lg1.draw_frame(False)

    axes[0].set_xlim([8.5,11.5])
    axes[0].set_ylim([0,1.18])
    axes[0].set_xlabel(r'log M$_{\star}$/M$_{\odot}$',fontsize=15)
    axes[0].set_ylabel(r'fraction of available metals, < 150kpc',fontsize=15)

    #Creating the stars/ISM/CGM temperature divisions subplot
    axes[1].plot([8.,11.5],[1,1], linestyle='dashed', color='black', linewidth=2)
    axes[1].annotate('Peeples et al. (2014)', xy=(11.45, 1.02), xytext=(11.45, 1.02), size=14, ha='right')
    starbar2 = axes[1].fill_between(log_star_total_mass, ism_frac_avg, star_frac_avg, facecolor = star_color, edgecolor = star_color, label = "stars")
    ismbar2 = axes[1].fill_between(log_star_total_mass, green_frac_avg, ism_frac_avg, facecolor = ism_color, edgecolor = ism_color, label = "ism")
    pinkbar = axes[1].fill_between(log_star_total_mass, yellow_frac_avg, pink_frac_avg, facecolor = pink_color, edgecolor = pink_color, label = "pink")
    purplebar = axes[1].fill_between(log_star_total_mass, pink_frac_avg, purple_frac_avg, facecolor = purple_color, edgecolor = purple_color, label = "purple")
    greenbar = axes[1].fill_between(log_star_total_mass, purple_frac_avg, green_frac_avg, facecolor = green_color, edgecolor = green_color, label = "green")
    yellowbar = axes[1].fill_between(log_star_total_mass, zeros, yellow_frac_avg, facecolor = yellow_color, edgecolor = yellow_color, label = "yellow")

    lg2 = axes[1].legend(handles=[starbar2, ismbar2, pinkbar, purplebar, greenbar, yellowbar], loc="upper left", fontsize=12, ncol=3, columnspacing=0.5, 
        labelspacing=0.4, borderpad=0.1, handletextpad=0.1, handlelength=3.2 )
    lg2.draw_frame(False)

    axes[1].set_xlim([8.5,11.5])
    axes[1].set_ylim([0,1.18])
    axes[1].set_xlabel(r'log M$_{\star}$/M$_{\odot}$',fontsize=15)
    axes[1].set_ylabel(r'fraction of available metals, < 150kpc',fontsize=15)

    plt.tight_layout()

    plt.savefig('metal-sim-bar.png')

#constants.py
if True:
    HELIUM_CORR = 1.366  # factor by which to correct H masses for existence of He, as used in Peeples et al. (2014)
    
    ## atomic masses
    O_AMU = 15.999
    FE_AMU = 55.845
    
    ####### Solar metallicities, Caffau et al. (2011), used in Peeples et al. (2014)
    Z_SUN = 0.0153
    ## number abundances, 12+log(X/H):
    T_LOG_OH_SUN = 8.76
    T_LOG_FEH_SUN = 7.52
    ## mass abundances
    Z_SUN_O = (O_AMU / HELIUM_CORR) * np.power(10.0,T_LOG_OH_SUN - 12)
    Z_SUN_FE = (FE_AMU / HELIUM_CORR) * np.power(10.0,T_LOG_FEH_SUN - 12)
    
    ## yields, as used in Peeples et al. (2014)
    Y_Z_II = 0.03
    Y_O_II = 0.015
    
    DUST_OXY_FRAC = 0.27  ## as adopted in Peeples et al. (2014)
    
    Zejmax = Y_O_II / 0.2 ## approx. metallicity of Type II SNe ejecta, assumes 20% of Mstar in high mass stars
    
    
    ## CHABRIER (2003) IMF cumulative fraction of mass lass by stellar particle
    ## Leitner & Kravtsov (2011) and Jungweirt et al. (2001), using buggy SPS
    FML_C0_OLD = 0.046
    FML_LAMBDA_OLD = 2.76e5*u.yr
    ## New values from Behroozi, Wechsler, and Conroy (2013):
    FML_C0 = 0.05
    FML_LAMBDA = 1.4e6*u.yr
    CHI_RECY = 0.40  ## approx assymptotic limit for fml(t~5Gyr)

#peeples2014.py
if True:
    ###### this matches what is in metals.c from the Peeples et al. (2014) metal census paper #####
    ###### in each case, the input variable "x" is log Mstar in solar masses at redshift = 0
    ###### the returned value is in units of log Msun
    ###### assumes Chabrier (2003) IMF everywhere
    
    
    ## kefits() from Kewley & Ellison (2008) on a Chabrier IMF
    ## average of all of the coefficients except two lowest ones
    ## as calculated for Peeples et al. (2014)
    KE_a = 27.8612
    KE_b = -7.05649
    KE_c = 0.818368
    KE_d = -0.0302926
    DUST_CORR = 0.1 ### assumed assumed dust depletion correction in O/H calibrations
    
    def Fg(x):
        """F_g = Mgas / Mstar"""
        return 0.5*(np.power(10.0,(-0.43*x + 3.75 + np.log10(HELIUM_CORR))) + np.power(10.0,(-0.4814*x + 4.3676)))
    
    def metalslost(x):
        """metals lost = metals made - metals in stars - metals in ISM """
        value = np.power(10.0,metalsmade(x)) - np.power(10.0,starz(x)) - np.power(10.0,ismz(x)) - np.power(10.0,dustz(x))
        return np.log10(value)
    
    def oxylost(x):
        """oxygen lost = oxygen made - oxygen in stars - oxygen in ISM """
        value = np.power(10.0,oxymade(x)) - np.power(10.0,staroxy(x)) - np.power(10.0,ismoxy(x)) - np.power(10.0,dustoxy(x))
        return np.log10(value)
        
    def metalsmade(x):
        return np.log10(np.power(10.0,zii(x)) + np.power(10.0,zia(x)) + np.power(10.0,zagb(x)))
    
    def zii(x):
        return (1.0146*x + np.log10(Y_Z_II) + 0.109109)
    
    def zia(x):
        return (1.043*x - 2.678) ## t^-1 DTD
    
    def zagb(x):
        m = x - 10.5
        return (7.5+ 0.611 + 0.914*m - 0.1797*np.sin(m + np.cos(0.7188*m + 0.336*np.sin(0.611 + 1.797*m) + 0.218*m*np.sin(0.611 + 1.797*m) - 0.218*m*m)))
    
    def starz(x):
        if (x >= 9.05):
            lzstar = (x-10.72)/(4.005+4.275*(x-10.72)+2.051*(x-10.72)*(x-10.72)) + 0.04
        else:
            lzstar = -0.11 + 0.40*(x-6) + np.log10(0.019)
        lmz = 1.08*lzstar - 0.16
    
        return (x + lmz + np.log10(Z_SUN))
    
    def starz_test(x):
        lzstar = np.ones_like(x)
        for i in np.arange(0,np.size(x)):
            if x[i] >= 9.05:
                lzstar[i] = (x[i]-10.72)/(4.005+4.275*(x[i]-10.72)+2.051*(x[i]-10.72)*(x[i]-10.72)) + 0.04
            else:
                lzstar[i] = -0.11 + 0.40*(x[i]-6) + np.log10(0.019)
        lmz = 1.08*lzstar - 0.16
    
        return (x + lmz + np.log10(Z_SUN))
    
    def ismz(x):
        return (ismoxy(x) + np.log10(Z_SUN/Z_SUN_O))
    
    def oxymade(x):
        return np.log10(np.power(10.0,oxyii(x)) + np.power(10.0,oxyia(x)) + np.power(10.0,oxyagb(x)))
    
    def oxyii(x):
        return (1.0146*x + np.log10(Y_O_II) + 0.109109)
    
    def oxyia(x):
        return (1.0216*x - 3.37046)
    
    def oxyagb(x):
        m = x - 10.5
        if(x >= 9.79): 
            lagb = 6.52532793646306 + m + np.sin(0.558326101347158*m/np.cos(m*m - m) + np.cos(0.707421425956075*m*m/np.cos(0.101415124817729 + -0.558326101347158*m/np.cos(m*m - m)) - m));
            return (-1.0*np.power(10.0,lagb))  ## OXYGEN DESTRUCTION!
        else:
            return (np.pow(10.0, 7.198 + 1.255*m + 0.01446/np.cos(2.212*m) + (0.9399 + m)/(0.5036 + m)))
        return -10
    
    def staroxy(x):
        alphafe =  -0.495 + 0.28*(x - 0.63)/4.52
        return (alphafe + starz(x) + np.log10(Z_SUN_O/Z_SUN))
    
    def ismoxy(x):
        print("--> not checking lMstar range validity for ismoxy !!! <---")
        lMg = np.log10(Fg(x)) + x
        oh = KE_a + KE_b*x + KE_c*x*x + KE_d*x*x*x
        oh = oh - DUST_CORR
        zg = (O_AMU / HELIUM_CORR) * np.power(10.0, oh-12)
        return (np.log10(zg) + lMg)
    
    def dustz(x):
        return (0.864*x - 1.3065)
    
    def dustoxy(x):
        return (dustz(x) + np.log10(DUST_OXY_FRAC))
    
    def ovioxy(x):
        if ((x >= 9.33) and (x < 10.83)):
            return (np.log10(2e7))
        else:
            return -20
    
    def oviz(x):
        return (ovioxy(x) + np.log10(Z_SUN/Z_SUN_O))
    
    def cgmz(x):
        if((x >= 9.33) and (x <= 10.83)):
            return 7.365
        else:
            return -20
    
    def cgmoxy(x):
        return cgmz(x) + np.log10(Z_SUN_O/Z_SUN)
    
    def hotz(x):
        if((x >= 10.37) and (x <= 11.3)):
            return (0.98*x - 2.89)
        else:
            return -20
    
    def hotoxy(x):
        return(hotz(x) + np.log10(Z_SUN_O/Z_SUN))
    
    def igdustz(x):
        if((x >= 9.929) and (x <= 10.429)):
            return (np.log10(5.e7))
        else:
            return -20
    
    def igdustoxy(x):
        return (igdustz(x) + np.log10(DUST_OXY_FRAC))

#metal-census-bar-newcolors.py
if True:
    #-----------------------------------------------------------------------------------------------------
    
    def fmr(lmstar, lsfr):
        ## copied from code for Peeples et al. (2014), probably from Mannucci et al. (2010)?
        ## takes log stellar mass and log star formation rate and returns 12+log(O/H)
        m = lmstar - 10
        return (8.90 + 0.37*m - 0.14*lsfr - 0.19*m*m + 0.12*m*lsfr - 0.054*lsfr*lsfr)
    
    #-----------------------------------------------------------------------------------------------------
    
    
    def peek_igdustz(x):
        if((x >= 9.33) and (x <= 10.83)):
            return (np.log10(6.e7))
        else:
            return -20
    #-----------------------------------------------------------------------------------------------------
    

    fig, axes = plt.subplots(1,2, figsize = (12,6))
    
    lmstar, lmzmade, lmzstar, lmzism, lmzdust, lmzovi, lmzcgm, lmzhot, lmzigdust = np.loadtxt('metalmasses.dat',usecols=(0,1,3,5,7,11,13,15,17), unpack=True)
    A = np.vstack([lmstar, np.ones(len(lmstar))]).T
    m, b = np.linalg.lstsq(A, lmzmade)[0]

    lmstar = np.arange(7,12,0.01)
    mstar = np.power(10.0,lmstar)
    mzmade = np.power(10.0,metalsmade(lmstar))
    mzstar = np.power(10.0,[starz(lmstar[i]) for i in np.arange(0,np.size(lmstar))])
    mzism = np.power(10.0,ismz(lmstar))
    mzdust = np.power(10.0,dustz(lmstar))
    mzovi = np.power(10.0,[oviz(lmstar[i]) for i in np.arange(0,np.size(lmstar))])
    mzcgm = np.power(10.0,[cgmz(lmstar[i]) for i in np.arange(0,np.size(lmstar))])
    mzhot = np.power(10.0,[hotz(lmstar[i]) for i in np.arange(0,np.size(lmstar))])
    mzigdust = np.power(10.0,[peek_igdustz(lmstar[i]) for i in np.arange(0,np.size(lmstar))])

    lstars = lmstar
    starsz = mzstar/mzmade
    ismz = mzism/mzmade
    coolz = mzcgm/mzmade
    warmz = mzovi/mzmade
    hotz = mzhot/mzmade
    dustz = mzdust/mzmade
    igdustz = mzigdust/mzmade

    z = np.zeros( lmstar.size ) ###0.0*lstars

    starbar = axes[0].fill_between(lstars,z,starsz,facecolor='#d73027',edgecolor='#d73027', zorder=10, label="stars")   # stars, red
    
    ismbar = axes[0].fill_between(lstars,starsz,starsz+ismz,facecolor='#4575b4',edgecolor='#4575b4', zorder=10, label="ISM gas")  ## ISM, blue
    
    dustbar = axes[0].fill_between(lstars,starsz+ismz,starsz+ismz+dustz,facecolor='darkorange',edgecolor='darkorange', zorder=10, label="ISM dust")

    fidstar, delta = 10.1, 0.0005 ## this is the center of the bar, so will grow things from there
    index = np.where((lmstar > (fidstar-delta)) & (lmstar < (fidstar+delta)))
    galfrac = starsz+ismz+dustz
    galfid = galfrac[index]
    coolfid = coolz[index]
    warmfid = warmz[index]
    dustfid = igdustz[index]

    coolfidmax = coolfid * (10**(7.36+0.15+0.45))/(10**7.36)
    coolfidmin = coolfid * (10**(7.36-0.15-0.38))/(10**7.36)
    warmfidmax = warmfid * 5./2
    dustfidmax = dustfid * (5*(10**7.5) / (5*(10**7.)))

    coolbar = axes[0].bar(fidstar - 0.25, galfid+coolfid, 0.5, color="#984ea3", edgecolor="#984ea3", zorder = 6, label="low ionization")  # cool, purple
    warmbar = axes[0].bar(fidstar - 0.25, warmfid, 0.5, bottom=galfid+coolfid, color="#4daf4a", edgecolor="#4daf4a", zorder=5, label="O VI traced") # OVI, green
    cgmdustbar = axes[0].bar(fidstar - 0.25, dustfid, 0.5, bottom=galfid+coolfid+warmfid, color="#9C6635", edgecolor="#9C6635", zorder=4, label="CGM dust") # dust, brown

    axes[0].bar(fidstar - 0.25, coolfidmax-coolfid, 0.5,  bottom=galfid+coolfid+warmfid+dustfid, color="#984ea3", edgecolor="#984ea3", fill=False, hatch="//")
    axes[0].bar(fidstar - 0.25, warmfidmax - warmfid, 0.5, bottom=galfid+coolfidmax+warmfid+dustfid, color="#4daf4a", edgecolor="#4daf4a", fill=False, hatch="\\\\",  zorder=7)
    axes[0].bar(fidstar - 0.25, dustfidmax - dustfid, 0.5, bottom=galfid+coolfidmax+warmfidmax+dustfid, color="#9C6635", edgecolor="#9C6635", fill=False, hatch="X",  zorder=7)

    hotbar = axes[0].fill_between(lstars,starsz+ismz+dustz,starsz+ismz+dustz+hotz,facecolor='#ffe34d',edgecolor='#ffe34d',zorder=9, label="X-ray traced")
    axes[0].fill_between(lstars,starsz+ismz+dustz+hotz,starsz+ismz+dustz+hotz+(8/6.)*hotz,facecolor='none',edgecolor='#ffe34d',hatch="XX")


    ### COS-Dwarfs
    fidstar, delta = 9.5, 0.0005
    index = np.where((lmstar > (fidstar-delta)) & (lmstar < (fidstar+delta)))
    galfrac = starsz+ismz+dustz
    galfid = galfrac[index]
    coolfid = coolz[index]
    warmfid = warmz[index]
    dustfid = igdustz[index]
    denom = mzmade[index]

    #### RONGMON CII/CIV TRACED CGM ####
    # log M*  8- 9.5   (9.3)                Carbon mass >   1.1 *1e6 M_solar
    # log M* 9.5-10    (9.81)               Carbon mass >    2.5 *1e6 M_solar
    ####################################
    ### assume C/Z = 0.18 #####

    coolmass = (1/0.18) * 2e6
    coolfid = coolmass / denom
    coolfidmax = coolfid*5
    print(coolfid, denom)
    axes[0].bar(fidstar - 0.25, galfid+coolfid, 0.5, color="#984ea3", edgecolor="#984ea3", zorder = 6)  # cool, purple
    axes[0].bar(fidstar - 0.25, coolfidmax-coolfid, 0.5,  bottom=galfid+coolfid, color="#984ea3", edgecolor="#984ea3", fill=False, hatch="//")

    fidstar, delta = 9.81, 0.0005
    index = np.where((lmstar > (fidstar-delta)) & (lmstar < (fidstar+delta)))
    galfrac = starsz+ismz+dustz
    galfid = galfrac[index]
    coolfid = coolz[index]
    warmfid = warmz[index]
    dustfid = igdustz[index]
    denom = mzmade[index]
    coolmass = (1/0.18) * 2.5e6
    coolfid = coolmass / denom
 
    axes[0].plot([8.,11.5],[1,1], linestyle='dashed', color='black', linewidth=2)
    axes[0].annotate('Peeples et al. (2014)', xy=(11.45, 1.02), xytext=(11.45, 1.02), size=14, ha='right')

    axes[0].set_xlim([8.5,11.5])
    axes[0].set_ylim([0,1.18])
    axes[0].set_xlabel(r'log M$_{\star}$/M$_{\odot}$',fontsize=15)
    axes[0].set_ylabel(r'fraction of available metals, < 150kpc',fontsize=15)

    lg1 = axes[0].legend(handles=[starbar, ismbar, dustbar, coolbar, warmbar, cgmdustbar, hotbar], # star_frac, ism_frac, cgm_frac],
        loc="upper left", fontsize=12, ncol=3, columnspacing=0.5,
        labelspacing=0.4, borderpad=0.1,
        handletextpad=0.1, handlelength=3.2 )
    lg1.draw_frame(False)

    #Adding the subplot for halo_008508 data
    axes[1].plot([8.,11.5],[1,1], linestyle='dashed', color='black', linewidth=2)
    axes[1].annotate('Peeples et al. (2014)', xy=(11.45, 1.02), xytext=(11.45, 1.02), size=14, ha='right')
    starbar1 = axes[1].fill_between(log_star_total_mass, ism_frac_avg, star_frac_avg, facecolor = star_color, edgecolor = star_color, label = "stars")
    ismbar1 = axes[1].fill_between(log_star_total_mass, cgm_frac_avg, ism_frac_avg, facecolor = ism_color, edgecolor = ism_color, label = "ism")
    cgmbar = axes[1].fill_between(log_star_total_mass, zeros, cgm_frac_avg, facecolor = cool_cgm_color, edgecolor = cool_cgm_color, label = "cgm")

    axes[1].set_xlim([8.5,11.5])
    axes[1].set_ylim([0,1.18])
    axes[1].set_xlabel(r'log M$_{\star}$/M$_{\odot}$',fontsize=15)
    axes[1].set_ylabel(r'fraction of available metals, within virial radii',fontsize=15)

    lg2 = axes[1].legend(handles=[starbar1, ismbar1, cgmbar], loc="upper left", fontsize=12, ncol=3, columnspacing=0.5, 
        labelspacing=0.4, borderpad=0.1, handletextpad=0.1, handlelength=3.2 )
    lg2.draw_frame(False)

    plt.tight_layout()

    plt.savefig('metal-census-and-sim-comp.png')