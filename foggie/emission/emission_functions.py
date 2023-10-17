import numpy as np
from scipy import interpolate
import yt
import yt.units as u

##########
## patt is the path to the Cloudy files being used for the emission calculations
##########

#patt = "/Users/dalek/data/cloudy_data/emissivity_tables/bertone_z1/bertone_run%i.dat"
#patt = "/Users/dalek/data/Ryan/cloudy_out/bertone/bertone_run%i.dat"
patt = "/Users/dalek/repos/Aspera/self-shielding-tests/updated_ciaoloop/z0/sh_z0_HM12_run%i.dat"

## Old function for finding the refine box. Your method may vary!
def get_refine_box(ds, zsnap, track):
    ## find closest output, modulo not updating before printout
    diff = track['col1'] - zsnap
    this_loc = track[np.where(diff == np.min(diff[np.where(diff > 1.e-6)]))]
    #print "using this loc:", this_loc
    x_left = this_loc['col2'][0]
    y_left = this_loc['col3'][0]
    z_left = this_loc['col4'][0]
    x_right = this_loc['col5'][0]
    y_right = this_loc['col6'][0]
    z_right = this_loc['col7'][0]
    refine_box_center = [0.5*(x_left+x_right), 0.5*(y_left+y_right), 0.5*(z_left+z_right)]
    refine_box = ds.r[x_left:x_right, y_left:y_right, z_left:z_right]
    refine_width = np.abs(x_right - x_left)
    return refine_box, refine_box_center, refine_width


# The Cloudy calculations assumed a single metallicity (typically solar).
# This function scales the emission by the metalliciit of the gas itself to
# account for this discrepancy.
def scale_by_metallicity(values,assumed_Z,wanted_Z):
        wanted_ratio = (10.**(wanted_Z))/(10.**(assumed_Z))
        return values*wanted_ratio

# Cloudy loves making ascii files. CIAOLoop already combines many of these files
# into an output file per run. We need to read each of these files.
def parse_cloudy_lines_file(filein,fileout):
    f = open(filein,'r')
    fh = open(filein.split('.lines')[0]+'.dat','r')
    fo = open(fileout,'w')

    foL = fh.readlines()
    hden = foL[5]

    linenum = 0
    header = True
    fo.write('# \n# Cooling Map File \n# Loop values: \n# \n')
    fo.write(hden)
    fo.write('\n# Data Columns: \n # log10 Te [K] \n')
    fo.write('# log10 Emissivities / n_H^2 [erg s^-1 cm^3] \n# \n')
    header_words = '#Te'

    for line in f:
        if linenum == 0:
            words = line.split()
            temp = str(np.log10(float(words[-1][:-1])))
            linenum = linenum + 1
            continue
        if linenum == 1:
            if header == True:
                words = line.split('depth')[1]
                header_words = header_words + words+' \n'
                fo.write(header_words)
                header = False
            linenum = linenum + 1
            continue
        if linenum == 2:
            nums = line.split()[1:]
            line_out = temp +'\t'+ '\t'.join(nums) + '\n'
            fo.write(line_out)
            linenum = 0
            continue
    fo.write('\n')
    fo.close()
    return

# This function takes all of the Cloudy files and compiles them into one table
# for use in the emission functions
# table_index is the column in the Cloudy output files that is being read.
# each table_index value corresponds to a different emission line
def make_Cloudy_table(table_index):
        # this is the the range and number of bins for which Cloudy was run
        # i.e. the temperature and hydrogen number densities gridded in the
        # Cloudy run. They must match or the table will be incorrect.
        hden_n_bins, hden_min, hden_max = 15, -5, 2 #17, -6, 2 #23, -9, 2
        T_n_bins, T_min, T_max = 51, 3, 8 #71, 2, 8

        hden=np.linspace(hden_min,hden_max,hden_n_bins)
        T=np.linspace(T_min,T_max, T_n_bins)
        table = np.zeros((hden_n_bins,T_n_bins))
        for i in range(hden_n_bins):
                table[i,:]=[float(l.split()[table_index]) for l in open(patt%(i+1)) if l[0] != "#"]
        return hden,T,table


# These are the typical units that Lauren uses
# NOTE: This is a volumetric unit since it's for the emissivity of each cell
# Emission / surface brightness comes from the projections
emission_units = 's**-1 * cm**-3 * steradian**-1'
ytEmU = u.s**-1 * u.cm**-3 * u.steradian**-1

# These are a second set of units that a lot of observers prefer
# NOTE: This is a volumetric unit since it's for the emissivity of each cell
# Emission / surface brightness comes from the projections
emission_units_ALT = 'erg * s**-1 * cm**-3 * arcsec**-2'
ytEmUALT = u.erg * u.s**-1 * u.cm**-3 * u.arcsec**-2

####################################
## BEGIN CREATING EMISSION FIELDS ##
####################################

# To make the emissivity fields, you need to follow a number of steps
# 1. Read in the Cloudy values for a given emission line
# 2. Create the n_H and T grids that represent the desired range of values
# 3. Set up interpolation function for the emissivity values across the grids
#    so the code can use the n_H and T values of a simulation grid cell to
#    interpolate the correct emissivity value
# 4. Define the emission field for the line
# 5. Add the line as a value in yt


###########################
## H Alpha Emission Line ##
###########################
# 1. Read cloudy file
hden_pts,T_pts,table_HA = make_Cloudy_table(2)
# 2. Create grids
hden_pts,T_pts = np.meshgrid(hden_pts,T_pts)
pts = np.array((hden_pts.ravel(),T_pts.ravel())).T

#set up interpolation fundtion
sr_HA = table_HA.T.ravel()
bl_HA = interpolate.LinearNDInterpolator(pts,sr_HA)

# 4. write emission function
def _Emission_HAlpha(field,data):
    H_N = np.log10(np.array(data['H_nuclei_density']))
    Temperature = np.log10(np.array(data['Temperature']))
    dia1 = bl_HA(H_N,Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line=(10.**dia1)*((10.**H_N)**2.0)
    emission_line = emission_line/(4.*np.pi*3.03e-12)
    return emission_line*ytEmU

# 5. Add field
yt.add_field(('gas','Emission_HAlpha'),units=emission_units,function=_Emission_HAlpha,take_log=True,force_override=True,sampling_type='cell')

###############################
## Lyman Alpha Emission Line ##
###############################
hden_pts,T_pts,table_LA = make_Cloudy_table(1)
sr_LA = table_LA.T.ravel()
bl_LA = interpolate.LinearNDInterpolator(pts,sr_LA)

def _Emission_LyAlpha(field,data):
    H_N=np.log10(np.array(data["H_nuclei_density"]))
    Temperature=np.log10(np.array(data["Temperature"]))
    dia1 = bl_LA(H_N,Temperature)
    idx = np.isnan(dia1)
    dia1[idx] = -200.
    emission_line = (10**dia1)*((10.0**H_N)**2.0)
    emission_line = emission_line/(4.*np.pi*1.63e-11)
    return emission_line*ytEmU

yt.add_field(('gas','Emission_LyAlpha'),units=emission_units,function=_Emission_LyAlpha,take_log=True,force_override=True,sampling_type='cell')

## Commented out because I didn't run them in the self-shielding Cloudy runs
##
#hden1, T1, table_SiIV_1 = make_Cloudy_table(13)
#hden1, T1, table_SiIV_2 = make_Cloudy_table(14)
#sr_SiIV_1 = table_SiIV_1.T.ravel()
#sr_SiIV_2 = table_SiIV_2.T.ravel()
#bl_SiIV_1 = interpolate.LinearNDInterpolator(pts,sr_SiIV_1)
#bl_SiIV_2 = interpolate.LinearNDInterpolator(pts,sr_SiIV_2)

#def _Emission_SiIV(field,data):
#        H_N=np.log10(np.array(data["H_nuclei_density"]))
#        Temperature=np.log10(np.array(data["Temperature"]))
#        dia1 = bl_SiIV_1(H_N,Temperature)
#        dia2 = bl_SiIV_2(H_N,Temperature)
#        idx = np.isnan(dia1)
#        dia1[idx] = -200.
#        dia2[idx] = -200.
#        emission_line=((10.0**dia1)+(10**dia2))*((10.0**H_N)**2.0)
#        emission_line = emission_line/(4.*np.pi*1.42e-11)
#        emission_line = scale_by_metallicity(emission_line,0.0,np.log10(np.array(data['metallicity'])))
#        return emission_line*ytEmU

#yt.add_field(("gas","Emission_SiIV"),units=emission_units,function=_Emission_SiIV,take_log=True,force_override=True,sampling_type='cell')

##############
## CIII 977 ##
##############
hden1,T1,table_CIII_977 = make_Cloudy_table(7)
sr_CIII_977 = table_CIII_977.T.ravel()
bl_CIII_977 = interpolate.LinearNDInterpolator(pts,sr_CIII_977)

def _Emission_CIII_977(field,data):
        H_N=np.log10(np.array(data["H_nuclei_density"]))
        Temperature=np.log10(np.array(data["Temperature"]))
        dia1 = bl_CIII_977(H_N,Temperature)
        idx = np.isnan(dia1)
        dia1[idx] = -200.
        emission_line=(10.0**dia1)*((10.0**H_N)**2.0)
        emission_line = emission_line/(4.*np.pi*2.03e-11)
        emission_line = scale_by_metallicity(emission_line,0.0,np.log10(np.array(data['metallicity'])))
        return emission_line*ytEmU

yt.add_field(("gas","Emission_CIII_977"),units=emission_units,function=_Emission_CIII_977,take_log=True,force_override=True,sampling_type='cell')

############################
## CIV (2 lines combined) ##
############################
hden1, T1, table_CIV_1 = make_Cloudy_table(3)
hden1, T1, table_CIV_2 = make_Cloudy_table(4)
sr_CIV_1 = table_CIV_1.T.ravel()
sr_CIV_2 = table_CIV_2.T.ravel()
bl_CIV_1 = interpolate.LinearNDInterpolator(pts,sr_CIV_1)
bl_CIV_2 = interpolate.LinearNDInterpolator(pts,sr_CIV_2)

def _Emission_CIV(field,data):
        H_N=np.log10(np.array(data["H_nuclei_density"]))
        Temperature=np.log10(np.array(data["Temperature"]))
        dia1 = bl_CIV_1(H_N,Temperature)
        dia2 = bl_CIV_2(H_N,Temperature)
        idx = np.isnan(dia1)
        dia1[idx] = -200.
        dia2[idx] = -200.
        emission_line=((10.0**dia1)+(10**dia2))*((10.0**H_N)**2.0)
        emission_line = emission_line/(4.*np.pi*1.28e-11)
        emission_line = scale_by_metallicity(emission_line,0.0,np.log10(np.array(data['metallicity'])))
        return emission_line*ytEmU

yt.add_field(("gas","Emission_CIV"),units=emission_units,function=_Emission_CIV,take_log=True,force_override=True,sampling_type='cell')

############################
## OVI (2 lines combined) ##
############################
hden1, T1, table_OVI_1 = make_Cloudy_table(5)
hden1, T1, table_OVI_2 = make_Cloudy_table(6)
sr_OVI_1 = table_OVI_1.T.ravel()
sr_OVI_2 = table_OVI_2.T.ravel()
bl_OVI_1 = interpolate.LinearNDInterpolator(pts,sr_OVI_1)
bl_OVI_2 = interpolate.LinearNDInterpolator(pts,sr_OVI_2)

def _Emission_OVI(field,data):
        H_N=np.log10(np.array(data["H_nuclei_density"]))
        Temperature=np.log10(np.array(data["Temperature"]))
        dia1 = bl_OVI_1(H_N,Temperature)
        dia2 = bl_OVI_2(H_N,Temperature)
        idx = np.isnan(dia1)
        dia1[idx] = -200.
        dia2[idx] = -200.
        emission_line=((10.0**dia1)+(10**dia2))*((10.0**H_N)**2.0)
        emission_line = emission_line/(4.*np.pi*1.92e-11)
        emission_line = scale_by_metallicity(emission_line,0.0,np.log10(np.array(data['metallicity'])))
        return emission_line*ytEmU

yt.add_field(("gas","Emission_OVI"),units=emission_units,function=_Emission_OVI,take_log=True,force_override=True,sampling_type='cell')

######################################
## OVI ALT UNITS (2 lines combined) ##
######################################
def _Emission_OVI_ALTunits(field,data):
        H_N=np.log10(np.array(data["H_nuclei_density"]))
        Temperature=np.log10(np.array(data["Temperature"]))
        dia1 = bl_OVI_1(H_N,Temperature)
        dia2 = bl_OVI_2(H_N,Temperature)
        idx = np.isnan(dia1)
        dia1[idx] = -200.
        dia2[idx] = -200.
        emission_line=((10.0**dia1))*((10.0**H_N)**2.0)
        emission_line= emission_line + ((10**dia2))*((10.0**H_N)**2.0)
        emission_line = emission_line/(4.*np.pi)
        emission_line = emission_line/4.25e10 # convert steradian to arcsec**2
        emission_line = scale_by_metallicity(emission_line,0.0,np.log10(np.array(data['metallicity'])))
        return emission_line*ytEmUALT

yt.add_field(("gas","Emission_OVI_ALT"),units=emission_units_ALT,function=_Emission_OVI_ALTunits,take_log=True,force_override=True,sampling_type='cell')

####################
## OVI 1032A only ##
####################
def _Emission_OVI_1032(field,data):
        H_N=np.log10(np.array(data["H_nuclei_density"]))
        Temperature=np.log10(np.array(data["Temperature"]))
        dia1 = bl_OVI_1(H_N,Temperature)
        idx = np.isnan(dia1)
        dia1[idx] = -200.
        emission_line=((10.0**dia1))*((10.0**H_N)**2.0)
        emission_line = emission_line/(4.*np.pi*1.92e-11) #1.9249e-11)
        emission_line = scale_by_metallicity(emission_line,0.0,np.log10(np.array(data['metallicity'])))
        return emission_line*ytEmU

yt.add_field(("gas","Emission_OVI_1032"),units=emission_units,function=_Emission_OVI_1032,take_log=True,force_override=True,sampling_type='cell')

####################
## OVI 1038A only ##
####################
def _Emission_OVI_1038(field,data):
        H_N=np.log10(np.array(data["H_nuclei_density"]))
        Temperature=np.log10(np.array(data["Temperature"]))
        dia2 = bl_OVI_2(H_N,Temperature)
        idx = np.isnan(dia2)
        dia2[idx] = -200.
        emission_line=((10**dia2))*((10.0**H_N)**2.0)
        emission_line = emission_line/(4.*np.pi*1.92e-11) #1.9137e-11)
        emission_line = scale_by_metallicity(emission_line,0.0,np.log10(np.array(data['metallicity'])))
        return emission_line*ytEmU

yt.add_field(("gas","Emission_OVI_1038"),units=emission_units,function=_Emission_OVI_1038,take_log=True,force_override=True,sampling_type='cell')
