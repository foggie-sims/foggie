import PIL
import numpy as np
from PIL import Image
import glob
from glob import glob
from PIL import ImageDraw
import os
from PIL import ImageFont
import matplotlib.pyplot as plt
plt.close('all')

translate = {}




translate['halo_002392'] = 'Hurricane'
translate['halo_002878'] = 'Cyclone'
translate['halo_004123'] = 'Blizzard'
translate['halo_005016'] = 'Squall'
translate['halo_005036'] = 'Maelstrom'
translate['halo_008508'] = 'Tempest'




fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')




if True: extra = '_nolabels'
else: extra = ''


for orient in ['vertical', 'horizontal'][0:1]:

    for typ in ['gas', 'stars']:
        fls = glob('/Users/rsimons/Dropbox/foggie/figures/for_paper/central_projections/*nref11c_nref9f*%s*%s.png'%(typ, extra))
        fls = np.sort(fls)
        imgs = [PIL.Image.open(fl) for fl in fls]
        if (typ == 'gas') & (extra == ''):
            #add halo names
            for i, (img, fl) in enumerate(zip(imgs, fls)):
                name = translate[fl.strip('_nref11c_nref9f_x_gas_projection.png').split('/')[-1]]
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype('/Library/Fonts/arial.ttf', 350)
                draw.text((100, 200),name,(255,255,255), font = font)

                font2 = ImageFont.truetype('/Library/Fonts/arial.ttf', 250)



        min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
        
        if orient == 'horizontal':
            imgs_comb_1 = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs[:3] ) )
            imgs_comb_2 = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs[3:] ) )
        elif orient == 'vertical':
            imgs_comb_1 = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs[:3] ) )
            imgs_comb_2 = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs[3:] ) )



        for i, imgs_comb in enumerate([imgs_comb_1, imgs_comb_2]):
            imgs_comb = PIL.Image.fromarray( imgs_comb)
            imgs_comb.save('/Users/rsimons/Dropbox/foggie/figures/for_paper/central_projections/%s_%s_%s%s.png'%(typ, i, orient, extra))

    for i in np.arange(2):
        fls = glob('/Users/rsimons/Dropbox/foggie/figures/for_paper/central_projections/*_%s_%s%s.png'%(i, orient, extra))
        if i == 1: fls = fls[::-1]
        imgs = [PIL.Image.open(fl) for fl in fls]
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
        if orient == 'horizontal':
            imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
        if orient == 'vertical':
            imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

        imgs_comb = PIL.Image.fromarray( imgs_comb)
        imgs_comb.save('/Users/rsimons/Dropbox/foggie/figures/for_paper/central_projections/combined_%s_%s%s.png'%(i, orient, extra))

    '''
    fls = glob('/Users/rsimons/Dropbox/foggie/figures/for_paper/central_projections/combined_*_%s%s.png'%(orient, extra))
    imgs = [PIL.Image.open(fl) for fl in fls]
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    if orient == 'horizontal':
        imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    if orient == 'vertical':
        imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save('/Users/rsimons/Dropbox/foggie/figures/for_paper/central_projections/combined_%s%s.png'%(orient, extra))
    '''







    if False:

        for typ in ['gas', 'stars']:
            fls = glob('/Users/rsimons/Dropbox/foggie/figures/for_paper/central_projections/all_%s_*.png'%typ)
            imgs = [PIL.Image.open(fl) for fl in fls]
            min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
            imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

            imgs_comb = PIL.Image.fromarray( imgs_comb)
            imgs_comb.save('/Users/rsimons/Dropbox/foggie/figures/for_paper/central_projections/all_%s.png'%typ)    

        else:

            for i in np.arange(2):
                fls = glob('/Users/rsimons/Dropbox/foggie/figures/for_paper/central_projections/all_*_%s.png'%i)
                imgs = [PIL.Image.open(fl) for fl in fls]
                min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
                imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

                imgs_comb = PIL.Image.fromarray( imgs_comb)
                imgs_comb.save('/Users/rsimons/Dropbox/foggie/figures/for_paper/central_projections/all_%s.png'%i)    


















