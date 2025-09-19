import yt, os 
import numpy as np
from astropy.table import Table
import argparse
from astropy.io import ascii

run = True 
# select halo of interest
ID = XXXX 
halos = ascii.read('/nobackupnfs1/jtumlins/25Mpc_new_cosmology/halo_catalogs_256/256/z0/out_0.list', header_start=0, data_start=2)
thishalo = halos[halos['ID'] == ID]
print(thishalo) 
x0 = thishalo['X'].value[0]/25.
y0 = thishalo['Y'].value[0]/25.
z0 = thishalo['Z'].value[0]/25.
rvir = np.max([thishalo['Rvir'].value[0], 200.])
print("Analyzing halo "+str(ID)+" at:")
print('The specified halo center is: ', x0, y0, z0)
print('With Rvir = ', rvir)

def set_0to1_conf():

    command = "awk '/halo_center /{$3="+str(x0)+";$4=\",\";$5="+str(y0)+";$6=\",\";$7="+str(z0)+"}1' ../halo_template_256/halo_DM_NtoN.conf > halo"+str(ID)+"_DM_0to1.temp"
    print(command)
    if (run): os.system(command)

    command = "awk '/halo_radius /{$3="+str(rvir)+"}1' ./halo"+str(ID)+"_DM_0to1.temp > halo"+str(ID)+"_DM_0to1.conf"
    print(command)
    if (run): os.system(command)

def run_0to1_music():
    command = "python ../../enzo-mrp-music/enzo-mrp-music.py halo"+str(ID)+"_DM_0to1.conf 1 "
    print(command)
    if (run): os.system(command)

def copy_template_files(level):
    command = "awk '{sub(/XXXX/,"+str(ID)+"); print}' /nobackupnfs1/jtumlins/25Mpc_new_cosmology/halo_template_256/RunScript.sh > ./RunScript.temp"
    print(command)
    if (run): os.system(command)

    command = "awk '{sub(/LX/,\"L"+level+"\"); print}' RunScript.temp > ./RunScript.sh"
    print(command)
    if (run): os.system(command)

    command = "cp -rp /nobackupnfs1/jtumlins/25Mpc_new_cosmology/halo_template_256/25Mpc_DM_256-L"+level+".enzo ."
    print(command)
    if (run): os.system(command)

    command = "cp -rp /nobackupnfs1/jtumlins/25Mpc_new_cosmology/halo_template_256/simrun.pl ." 
    print(command)
    if (run): os.system(command)

def get_0to1_shifts():

    #create the l0_to_l1_shifts file whicn otherwise does not exist until L2 is created 

    os.system("grep shift_x ../25Mpc_DM_256-L1.conf_log.txt | awk '{print $7}' > shift_x")
    os.system("grep shift_y ../25Mpc_DM_256-L1.conf_log.txt | awk '{print $7}' > shift_y")
    os.system("grep shift_z ../25Mpc_DM_256-L1.conf_log.txt | awk '{print $7}' > shift_z" )
    os.system("paste shift_x shift_y shift_z > l0_to_l1_shifts")

def copy_gas_template_files(level):
    command = "awk '{sub(/XXXX/,"+str(ID)+"); print}' /nobackupnfs1/jtumlins/25Mpc_new_cosmology/halo_template_256/RunScriptGas.sh > ./RunScript.temp"
    print(command)
    if (run): os.system(command)

    command = "awk '{sub(/LX/,\"L"+level+"\"); print}' RunScript.temp > ./RunScript.sh"
    print(command)
    if (run): os.system(command)

    command = "cp -rp /nobackupnfs1/jtumlins/25Mpc_new_cosmology/halo_template_256/25Mpc_DM_256-L"+level+"-gas.enzo ."
    print(command)
    if (run): os.system(command)

def mod_param_file(level):
    command = "grep CosmologySimulationGrid parameter_file.txt > grid_parameters.txt"
    print(command)
    if (run): os.system(command)

    command = "cat 25Mpc_DM_256-L"+level+".enzo grid_parameters.txt > pars.temp"
    print(command)
    if (run): os.system(command)

    command = "mv pars.temp 25Mpc_DM_256-L"+level+".enzo"
    print(command)
    if (run): os.system(command)

def mod_gas_param_file(level):
    command = "grep CosmologySimulationGrid parameter_file.txt > grid_parameters.txt"
    print(command)
    if (run): os.system(command)

    command = "cat 25Mpc_DM_256-L"+level+"-gas.enzo grid_parameters.txt > pars.temp"
    print(command)
    if (run): os.system(command)

    command = "mv pars.temp 25Mpc_DM_256-L"+level+"-gas.enzo"
    print(command)
    if (run): os.system(command)

def set_1to2_conf():

    os.system("cp -rp halo"+str(ID)+"_DM_0to1.conf halo"+str(ID)+"_DM_1to2.conf")

    os.system("grep shift_x 25Mpc_DM_256-L1.conf_log.txt | awk '{print $7}' > shift_x")
    os.system("grep shift_y 25Mpc_DM_256-L1.conf_log.txt | awk '{print $7}' > shift_y")
    os.system("grep shift_z 25Mpc_DM_256-L1.conf_log.txt | awk '{print $7}' > shift_z" )
    os.system("paste shift_x shift_y shift_z > l0_to_l1_shifts")

    file = open('l0_to_l1_shifts')
    line = file.read()
    print(line)

    xshift = int(str.split(line)[0])
    yshift = int(str.split(line)[1])
    zshift = int(str.split(line)[2])

    print("shifts to be aplied are ", xshift, yshift, zshift)

    command = "awk '/halo_center /{$3 = "+str(x0+xshift/255.)+";$5 = "+str(y0+yshift/255.)+";$7= "+str(z0+zshift/255.)+"}1' halo"+str(ID)+"_DM_0to1.conf > asdf"
    print(command)
    os.system(command)

    command = "awk '/simulation_run_directory/{$3 = \""+os.getcwd()+"\"}1' asdf > halo"+str(ID)+"_DM_1to2.conf"
    print(command)
    os.system(command)

def set_2to3_conf():

    os.system("cp -rp halo"+str(ID)+"_DM_1to2.conf halo"+str(ID)+"_DM_2to3.conf")

    os.system("grep shift_x 25Mpc_DM_256-L2.conf_log.txt | awk '{print $7}' > shift_x")
    os.system("grep shift_y 25Mpc_DM_256-L2.conf_log.txt | awk '{print $7}' > shift_y")
    os.system("grep shift_z 25Mpc_DM_256-L2.conf_log.txt | awk '{print $7}' > shift_z" )
    os.system("paste shift_x shift_y shift_z > l1_to_l2_shifts")

    file = open('l1_to_l2_shifts')
    line = file.read()
    print(line)

    xshift = int(str.split(line)[0])
    yshift = int(str.split(line)[1])
    zshift = int(str.split(line)[2])

    print("shifts to be aplied are ", xshift, yshift, zshift)

    command = "awk '/halo_center /{$3 = "+str(x0+xshift/255.)+";$5 = "+str(y0+yshift/255.)+";$7= "+str(z0+zshift/255.)+"}1' halo"+str(ID)+"_DM_1to2.conf > asdf"
    print(command)
    os.system(command)

    command = "awk '/simulation_run_directory/{$3 = \""+os.getcwd()+"\"}1' asdf > halo"+str(ID)+"_DM_2to3.conf"
    print(command)
    os.system(command)

def set_3to4_conf():

    os.system("cp -rp halo"+str(ID)+"_DM_2to3.conf halo"+str(ID)+"_DM_3to4.conf")

    os.system("grep shift_x 25Mpc_DM_256-L3.conf_log.txt | awk '{print $7}' > shift_x")
    os.system("grep shift_y 25Mpc_DM_256-L3.conf_log.txt | awk '{print $7}' > shift_y")
    os.system("grep shift_z 25Mpc_DM_256-L3.conf_log.txt | awk '{print $7}' > shift_z" )
    os.system("paste shift_x shift_y shift_z > l2_to_l3_shifts")

    file = open('l2_to_l3_shifts')
    line = file.read()
    print(line)

    xshift = int(str.split(line)[0])
    yshift = int(str.split(line)[1])
    zshift = int(str.split(line)[2])

    print("shifts to be aplied are ", xshift, yshift, zshift)

    command = "awk '/halo_center /{$3 = "+str(x0+xshift/255.)+";$5 = "+str(y0+yshift/255.)+";$7= "+str(z0+zshift/255.)+"}1' halo"+str(ID)+"_DM_2to3.conf > asdf"
    print(command)
    os.system(command)

    command = "awk '/simulation_run_directory/{$3 = \""+os.getcwd()+"\"}1' asdf > halo"+str(ID)+"_DM_3to4.conf"
    print(command)
    os.system(command)

def run_music(level):
    command = "python ../../enzo-mrp-music/enzo-mrp-music.py halo"+str(ID)+"_DM_"+str(int(level)-1)+"to"+level+".conf "+level
    print(command)
    if (run): os.system(command)

def convert_to_gas(level):

    command = "awk '/baryons/{$3 = \"yes\"}1' 25Mpc_DM_256-L"+str(level)+".conf | awk '/Omega_b/{$3 = 0.0461}1' | awk '/filename/{$3 = \"25Mpc_DM_256-L"+level+"-gas\"}1' > 25Mpc_DM_256-L"+level+"-gas.conf"
    print(command)
    if (run): os.system(command)




#main
parser = argparse.ArgumentParser()
parser.add_argument('--level', type=str, required=True)
parser.add_argument('--gas', type=str, required=True)
args = parser.parse_args()
print('Hello your level is:', args.level)
print('Are you including gas?', args.gas)


if (args.gas == 'no'):
    if ('1' in args.level):
        set_0to1_conf()

    if ('2' in args.level):
        set_1to2_conf()

    if ('3' in args.level):
        set_2to3_conf()

    if ('4' in args.level):
        set_3to4_conf()

    run_music(args.level)
    os.chdir('25Mpc_DM_256-L'+args.level)
    copy_template_files(args.level)
    mod_param_file(args.level)
    get_0to1_shifts() 
    os.system('rm *temp')
    os.system('qsub -koed RunScript.sh')
else:
    print("calculating with gas")
    convert_to_gas(args.level)
    os.system("/u/jtumlins/installs/music/MUSIC 25Mpc_DM_256-L"+args.level+"-gas.conf")
    os.chdir('25Mpc_DM_256-L'+args.level+'-gas')
    copy_gas_template_files(args.level)
    mod_gas_param_file(args.level)
    os.system('rm *temp')
    os.system('qsub -koed RunScript.sh')


