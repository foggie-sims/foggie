import matplotlib
matplotlib.use('Agg')

import yt
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl

def get_refine_box(ds, zsnap, track):
    ## find closest output, modulo not updating before printout
    diff = track['col1'] - zsnap
    this_loc = track[np.where(diff == np.min(diff[np.where(diff > 1.e-6)]))]
    print "using this loc:", this_loc
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

fontrc ={'fontname':'Osaka','fontsize':30}
mpl.rc('text', usetex=True)

baseREF = "/lou/s2m/mpeeples/halo_008508/nref11n/nref11f_refine200kpc_z4to2"
baseNAT = "/lou/s2m/mpeeples/halo_008508/nref11n/natural"
track_name = baseREF+"/halo_track"
track = Table.read(track_name, format='ascii')
track.sort('col1')

RDs = np.arange(19,140,1)

for RD in RDs:
    if RD < 100:
        RD = 'RD00'+str(RD)
    else:
        RD = 'RD0'+str(RD)

    print '###############################'
    print '########## '+RD+' #############'
    print '###############################'

    fnREF = baseREF+"/"+RD+"/"+RD
    fnNAT = baseNAT+"/"+RD+"/"+RD
    fileout = 'AAS_'+RD+'.pdf'

    dsREF = yt.load(fnREF)
    dsNAT = yt.load(fnNAT)
    rbR, rb_centerR, rb_widthR = get_refine_box(dsREF, dsREF.current_redshift, track)
    rbN, rb_centerN, rb_widthN = get_refine_box(dsNAT, dsNAT.current_redshift, track)

    fig,ax = plt.subplots(2,3)#,sharex=True,sharey=True)
    fig.set_size_inches(20,12)

    ### Dens Ref ###
    densR = yt.ProjectionPlot(dsREF,'x','H_nuclei_density',data_source=rbR,center=rb_centerR,
                              width=(rb_widthR,'code_length'))
    densR.set_zlim('H_nuclei_density',10**18,10**24)
    frbDR = np.log10(densR.frb['H_nuclei_density'])

    im = ax[0,0].imshow(frbDR,vmin=18.,vmax=24.,interpolation='none',cmap='Blues_r',origin='lower')
    im.axes.set_xticks([])
    im.axes.set_yticks([])
    ax[0,0].text(0.95, 0.95, 'Density',
                verticalalignment='top', horizontalalignment='right',
                transform=ax[0,0].transAxes,color='white', fontsize=30)

    ### Temp Ref ###
    tempR = yt.ProjectionPlot(dsREF,'x','temperature',data_source=rbR,center=rb_centerR,
                              width=(rb_widthR,'code_length'),weight_field='Density')
                              tempR.set_zlim('temperature',10**3,10**7)
    frbTR = np.log10(tempR.frb['temperature'])

    im2 = ax[0,1].imshow(frbTR,vmin=3.,vmax=7.,interpolation='none',cmap='Reds_r',origin='lower')
    im2.axes.set_xticks([])
    im2.axes.set_yticks([])
    ax[0,1].text(0.95, 0.95, 'Temperature',
                verticalalignment='top', horizontalalignment='right',
                transform=ax[0,1].transAxes,color='black', fontsize=30)

    ### Metal Ref ###
    metalR = yt.ProjectionPlot(dsREF,'x','metallicity',data_source=rbR,center=rb_centerR,
                               width=(rb_widthR,'code_length'),weight_field='Density')
    metalR.set_zlim('metallicity',10**-3,1.5)
    frbZR = np.log10(metalR.frb['metallicity'])

    im3 = ax[0,2].imshow(frbZR,vmin=-3.,vmax=0.18,interpolation='none',cmap='viridis',origin='lower')
    im3.axes.set_xticks([])
    im3.axes.set_yticks([])
    ax[0,2].text(0.95, 0.95, 'Metallicity',
                verticalalignment='top', horizontalalignment='right',
                transform=ax[0,2].transAxes,color='white', fontsize=30)

    ## Dens Nat ###
    densN = yt.ProjectionPlot(dsNAT,'x','H_nuclei_density',data_source=rbN,center=rb_centerN,
                              width=(rb_widthN,'code_length'))
    densN.set_zlim('H_nuclei_density',10**18,10**24)
    frbDN = np.log10(densN.frb['H_nuclei_density'])

    im = ax[1,0].imshow(frbDN,vmin=18.,vmax=24.,interpolation='none',cmap='Blues_r',origin='lower')
    im.axes.set_xticks([])
    im.axes.set_yticks([])


    ## Temp Nat ###
    tempN = yt.ProjectionPlot(dsNAT,'x','temperature',data_source=rbN,center=rb_centerN,
                              width=(rb_widthN,'code_length'),weight_field='Density')
    tempN.set_zlim('temperature',10**3,10**7)
    frbTN = np.log10(tempN.frb['temperature'])

    im2 = ax[1,1].imshow(frbTN,vmin=3.,vmax=7.,interpolation='none',cmap='Reds_r',origin='lower')
    im2.axes.set_xticks([])
    im2.axes.set_yticks([])

    ### Metal Nat ###
    metalN = yt.ProjectionPlot(dsNAT,'x','metallicity',data_source=rbN,center=rb_centerN,
                               width=(rb_widthN,'code_length'),weight_field='Density')
    metalN.set_zlim('metallicity',10**-3,1.5)
    frbZN = np.log10(metalN.frb['metallicity'])
    im3 = ax[1,2].imshow(frbZN,vmin=-3.,vmax=0.18,interpolation='none',cmap='viridis',origin='lower')
    im3.axes.set_xticks([])
    im3.axes.set_yticks([])

    ## Reformat ##
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.065)

    ## Colorbars ##
    cbaxes = fig.add_axes([0.0328, 0.03, 0.273, 0.02])
    cb = plt.colorbar(im, cax = cbaxes,orientation='horizontal')
    axcb = cb.ax
    text = axcb.yaxis.label
    font = mpl.font_manager.FontProperties(family='Osaka', size=20)
    text.set_font_properties(font)

    cbaxes2 = fig.add_axes([0.363, 0.03, 0.273, 0.02])
    cb2 = plt.colorbar(im2, cax = cbaxes2,orientation='horizontal')
    axcb = cb2.ax
    text = axcb.yaxis.label
    font = mpl.font_manager.FontProperties(family='Osaka', size=20)
    text.set_font_properties(font)

    cbaxes3 = fig.add_axes([0.689, 0.03, 0.273, 0.02])
    cb3 = plt.colorbar(im3, cax = cbaxes3,orientation='horizontal')
    axcb = cb3.ax
    text = axcb.yaxis.label
    font = mpl.font_manager.FontProperties(family='Osaka', size=20)
    text.set_font_properties(font)

    plt.savefig(fileout)
    plt.close()
