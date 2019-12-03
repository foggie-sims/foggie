import PIL
import numpy as np
from PIL import Image
import glob
from glob import glob
fls = glob('/Users/rsimons/Desktop/foggie/figures/simulated_plunges/*nref11c_nref9f.png')

imgs = [PIL.Image.open(fl) for fl in fls]

min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

imgs_comb = PIL.Image.fromarray( imgs_comb)
imgs_comb.save('/Users/rsimons/Desktop/foggie/figures/simulated_plunges/all_plunges.png')    

