from astropy.table import Table, vstack, hstack 
import os
import numpy as np 
import matplotlib.pyplot as plt 
import glob 

def get_memory_by_node():    
    
    list_of_memory_trace_files = glob.glob('memory*gov') 
    list_of_memory_tables = [] 
    #read each memory table separately 
    for file in list_of_memory_trace_files: 
        print('opening memory trace file: ', file) 
        os.system('grep time ' + file + ' | grep -v out | cut -b 8-1000000 | awk "NF > 1{print}" > trace_'+file) # quickly preprocess the memory trace files 
        this_memory_table = Table.read('trace_'+file, format='ascii')
        list_of_memory_tables.append(this_memory_table) 
        print(this_memory_table) 

    all_memory_table = vstack(list_of_memory_tables) 
    all_memory_table.rename_column('col1', 'timestamp')
    for k in all_memory_table.keys()[1:]: 
        all_memory_table.rename_column(k, 'node'+str(int(k[3:])-1)) 
    print(all_memory_table) 
    return all_memory_table 

def get_times_and_redshifts(): 

    os.system("grep 'comoving_expansion redshift ' pbs_output.txt | awk '{print $2, $6}' > times_and_zs")
    os.system("grep 'simulation num-particles total' pbs_output.txt | awk '{print $7}' > nparticles") 
    os.system("grep 'Simulation cycle' pbs_output.txt | awk '{print $5}' > cycles ")  

    times = Table.read('times_and_zs', format='ascii') 
    npart = Table.read('nparticles', format='ascii') 
    cycle = Table.read('cycles', format='ascii')  
    table = hstack([cycle, times, npart])

    names = table.keys() 
    table.rename_column(names[0], 'cycle')
    table.rename_column(names[1], 'timestamp')
    table.rename_column(names[2], 'redshift')
    table.rename_column(names[3], 'nparticles')

    #os.system('rm times_and_zs nparticles cycles') 

    return table 



mm = get_memory_by_node() # this returns a table with the memory traces in it, by node
mm['timestamp'] = 1.0 * (mm['timestamp'] - mm['timestamp'][0])  
mm.sort('timestamp') 
print(mm) 

tz = get_times_and_redshifts()
tz.sort('timestamp') 
tz = tz[:-2] 
print(tz) 

number_of_nodes = len(mm.keys()) - 1 
print('there are :', number_of_nodes, ' nodes')
mm['redshift'] = np.interp(mm['timestamp'], tz['timestamp'], tz['redshift'])
print(mm) 

plt.figure(figsize=(12,4))
for i in range(number_of_nodes):
    plt.plot(mm['redshift'], mm['node'+str(i+1)], alpha=0.5)

plt.plot(tz['redshift'], (tz['nparticles'])/1e6, color='red', alpha=0.8) 
number_of_particles = (tz['nparticles'][-1])/1e6 

plt.xlim(25,0)
plt.title(os.getcwd().split('/')[-1]) 
plt.ylim(0,125)
plt.xlabel('Redshift') 
plt.text(mm['redshift'][-1]-0.05, number_of_particles, str(number_of_particles)[0:4]+' million', color='red', horizontalalignment='right') 
plt.ylabel('Free Memory Per Node (GB)') 
plt.savefig('ee_memory_trace.png')

print('Total Number Of Particles So Far: ', (tz['nparticles'][-1]-tz['nparticles'][0])/1e6, ' million') 
