

from astropy.table import Table, vstack 
import os, time, datetime, glob
import numpy as np 
import matplotlib.pyplot as plt 
start_time = time.time()

def get_memory_by_node():    
    
    list_of_memory_trace_files = glob.glob('memory*gov') 
    list_of_memory_tables = [] 
    #read each memory table separately 
    for file in list_of_memory_trace_files: 
        print('opening memory trace file: ', file)
        try:
            os.system('grep time ' + file + ' | grep -v out | cut -b 8-1000000 | awk "NF > 1{print}" > trace_'+file) # quickly preprocess the memory trace files
            this_memory_table = Table.read('trace_'+file, format='ascii')
            list_of_memory_tables.append(this_memory_table)
            #print(this_memory_table)
        except Exception as e:
            #print(file, 'cannot be processed because following error:', e)
            print(file, 'cannot be processed')
            pass

    print('Successfully read in ' + str(len(list_of_memory_tables)) + ' memory files out of ' + str(len(list_of_memory_trace_files)))
    all_memory_table = vstack(list_of_memory_tables)
    all_memory_table.rename_column('col1', 'timestamp')
    for k in all_memory_table.keys()[1:]: 
        all_memory_table.rename_column(k, 'node'+str(int(k[3:])-1)) 
    #print(all_memory_table)
    return all_memory_table 

def get_times_and_redshifts(): 
    os.system("grep ntRed ?D????/?D???? | awk '{print $3}' > zs")
    os.system("grep TimeId ?D????/?D???? | awk '{print $3}' > times")
    os.system("grep NumberOfParticles ?D????/?D???? | awk '{print $3}' > nparticles") 
    os.system("paste times zs nparticles > times_zs_nparts")
    times_and_redshifts = Table.read('times_zs_nparts', format='ascii')
    times_and_redshifts.rename_column('col1', 'timestamp')
    times_and_redshifts.rename_column('col2', 'redshift')
    times_and_redshifts.rename_column('col3', 'nparticles')
    return times_and_redshifts

mm = get_memory_by_node() # this returns a table with the memory traces in it, by node
print(mm) 
mm.sort('timestamp') 
tz = get_times_and_redshifts()
tz.sort('timestamp') 
print(tz) 

number_of_nodes = len(mm.keys()) - 1 
print('there are :', number_of_nodes, ' nodes')
mm['redshift'] = np.interp(mm['timestamp'], tz['timestamp'], tz['redshift'])

print(mm) 

plt.figure(figsize=(12,4))
for i in range(number_of_nodes):
    plt.plot(mm['redshift'], mm['node'+str(i+1)], alpha=0.5)

plt.plot(tz['redshift'], (tz['nparticles']-tz['nparticles'][0])/1e6, color='red', alpha=0.8) 

number_of_particles = (tz['nparticles'][-1]-tz['nparticles'][0])/1e6 

#plt.xlim(6,0)
plt.xlim(13, 0)
plt.title(os.getcwd().split('/')[-2] + ' ' + os.getcwd().split('/')[-1]) 
plt.ylim(0,125)
plt.xlabel('Redshift') 
plt.text(mm['redshift'][-1]-0.05, number_of_particles, str(number_of_particles)[0:4]+' million', color='red', horizontalalignment='right') 
plt.ylabel('Free Memory Per Node (GB)') 
plt.savefig('node_memory_trace.png')
plt.show(block=False)

print('Total Number Of Particles So Far: ', (tz['nparticles'][-1]-tz['nparticles'][0])/1e6, ' million')
print('Completed in %s minutes' % datetime.timedelta(seconds=(time.time() - start_time)))
