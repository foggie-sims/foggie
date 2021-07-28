import glob
from glob import glob
import numpy as np

halos = ['2392', '2878', '4123', '5016', '5036', '8508']


props = {}
for halo in halos:
    print (halo)
    props[halo] = {}
    fls = np.sort(glob('/nobackupp2/rcsimons/foggie/angular_momentum/profiles/%s/Lprof*npy'%halo))
    for fl in fls:
        DD_name = fl.split('_')[-1].strip('.npy').lstrip('DD')
        print (DD_name)
        a = np.load(fl, allow_pickle = True)[()]
        props[halo][DD_name] = {}
        for key in a['props'].keys():
            props[halo][DD_name][key] = a['props'][key]     
np.save('/nobackupp2/rcsimons/foggie/angular_momentum/halo_props.npy', props)
