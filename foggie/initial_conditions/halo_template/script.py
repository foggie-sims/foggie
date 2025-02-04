import yt
from astropy.table import Table
import numpy as np
import os
from astropy.table import Table
import argparse

run = True
# select halo of interest
ID = 12922
halos = Table.read('../25Mpc_256_shielded-L0/BigBox_z2_rockstar/out_0.list', format='ascii',header_start=0)
index = [halos['ID'] == ID]
thishalo = halos[index]
x0 = thishalo['X'][0]/25.
y0 = thishalo['Y'][0]/25.
z0 = thishalo['Z'][0]/25.
rvir = np.max([thishalo['Rvir'][0], 200.])
print("Analyzing halo "+str(ID)+" at:")
print(x0)
print(y0)
print(z0)
print(rvir)

def set_0to1_conf():

    command = "awk '/halo_center /{$3="+str(x0)+";$4=\",\";$5="+str(y0)+";$6=\",\";$7="+str(z0)+"}1' ../halo_template/halo_DM_NtoN.conf > halo"+str(ID)+"_DM_0to1.temp"
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
    command = "awk '{sub(/XXXX/,"+str(ID)+"); print}' /nobackup/jtumlins/CGM_bigbox/halo_template/RunScript.sh > ./RunScript.temp"
    print(command)
    if (run): os.system(command)

    command = "awk '{sub(/LX/,\"L"+level+"\"); print}' RunScript.temp > ./RunScript.sh"
    print(command)
    if (run): os.system(command)

    command = "cp -rp /nobackup/jtumlins/CGM_bigbox/halo_template/25Mpc_DM_256-L"+level+".enzo ."
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

    print("shifts to be applied are ", xshift, yshift, zshift)

    command = "awk '/halo_center /{$3 = "+str(x0+xshift/255.)+";$5 = "+str(y0+yshift/255.)+";$7= "+str(z0+zshift/255.)+"}1' halo"+str(ID)+"_DM_0to1.conf > asdf"
    print(command)
    os.system(command)

    command = "awk '/simulation_run_directory/{$3 = \""+os.getcwd()+"\"}1' asdf > halo"+str(ID)+"_DM_1to2.conf"
    print(command)
    os.system(command)
    os.system('rm shift*')

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

    print("shifts to be applied are ", xshift, yshift, zshift)

    command = "awk '/halo_center /{$3 = "+str(x0+xshift/255.)+";$5 = "+str(y0+yshift/255.)+";$7= "+str(z0+zshift/255.)+"}1' halo"+str(ID)+"_DM_1to2.conf > asdf"
    print(command)
    os.system(command)

    command = "awk '/simulation_run_directory/{$3 = \""+os.getcwd()+"\"}1' asdf > halo"+str(ID)+"_DM_2to3.conf"
    print(command)
    os.system(command)


def run_music(level):
    command = "python ../../enzo-mrp-music/enzo-mrp-music.py halo"+str(ID)+"_DM_"+str(int(level)-1)+"to"+level+".conf "+level
    print(command)
    if (run): os.system(command)


parser = argparse.ArgumentParser()
parser.add_argument('--level', type=str, required=True)
args = parser.parse_args()
print('Hello your level is:', args.level)

if ('1' in args.level):
    set_0to1_conf()

if ('2' in args.level):
    set_1to2_conf()

if ('3' in args.level):
    set_2to3_conf()

run_music(args.level)
os.system('rm *temp')
os.chdir('25Mpc_DM_256-L'+args.level)
copy_template_files(args.level)
mod_param_file(args.level)
os.system('rm *temp')
os.system('qsub -koed RunScript.sh')

