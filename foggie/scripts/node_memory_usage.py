
from astropy.table import Table
import os
import numpy as np 
import matplotlib.pyplot as plt 

def get_memory_by_node():    
    os.system('grep time mem* | cut -b 43-1000000 > mems') # quickly preprocess the memory trace files 
    memory_table = Table.read('mems', format='ascii')
    memory_table.rename_column('col1', 'timestamp')
    for k in memory_table.keys()[1:]: 
        memory_table.rename_column(k, 'node'+str(int(k[-1])-1))
    return memory_table

def get_times_and_redshifts(): 
    os.system("grep ntRed RD????/RD00?? | awk '{print $3}' > zs")
    os.system("grep TimeId RD????/RD00?? | awk '{print $3}' > times")
    os.system("paste times zs > times_and_redshifts ")
    times_and_redshifts = Table.read('times_and_redshifts', format='ascii')
    times_and_redshifts.rename_column('col1', 'timestamp')
    times_and_redshifts.rename_column('col2', 'redshift')
    return times_and_redshifts

mm = get_memory_by_node() # this returns a table with the memory traces in it, by node
tz = get_times_and_redshifts()

mm['redshift'] = np.interp(mm['timestamp'], tz['timestamp'], tz['redshift'])

plt.figure(figsize=(12,4))
plt.plot(mm['redshift'], mm['node1'])
plt.plot(mm['redshift'], mm['node2'])
plt.plot(mm['redshift'], mm['node3'])
plt.xlim(4,1.5)
plt.ylim(0,125)
plt.savefig('node_memory_trace.png')

