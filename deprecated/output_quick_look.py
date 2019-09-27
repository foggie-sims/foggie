#! /usr/bin/env python

'''
AUTHOR: Molly Peeples
DATE: 10/06/16
NAME: output_quick_look.py
DESCRIPTION: makes plots and a table for Enzo blob outputs
'''

import yt

from astropy.table import Table, vstack

import matplotlib.pyplot as plt
import numpy as np

import os
import sys
import glob

import argparse

#-----------------------------------------------------------------------------------------------------

def parse_args():
    '''
    Parse command line arguments.  Returns args object.
    '''
    parser = argparse.ArgumentParser(description = \
                                     "makes some plots")

    parser.add_argument('--clobber', dest='clobber', action='store_true')
    parser.add_argument('--no-clobber', dest='clobber', action='store_false', help="default is no clobbering")

    parser.set_defaults(clobber=False)

    args = parser.parse_args()
    return args



#-----------------------------------------------------------------------------------------------------
def make_yt_plots(ds):

    ## density projection
    p = yt.ProjectionPlot(ds,'z','Density')
    p.set_unit('Density','gm/cm**2')
    p.save('Density_z_'+ds.basename+'.png')


    ## temperature
    p = yt.ProjectionPlot(ds,'z','Temperature',weight_field="Density")
    p.save('Temperature_z_'+ds.basename+'.png')

    ## metallicity?
    ## p = yt.ProjectionPlot(ds,'z','Metallicity')
    ## p.save('Metallicity_z_'+ds.basename+'.png')

    ## phase diagrams
    ### why isn't this working ?!
#    p = yt.PhasePlot(ds,"Density","Temperature",["cell_mass"])
#    p.save("Density_Temperature_PhasePlot"+ds.basename+".png")

    ## what else ?!

#-----------------------------------------------------------------------------------------------------

def make_quicklook_page(ds):
    # put some stuff on an html page
    outfilename = ds.basename + "_quicklook.html"
    outfile = open(outfilename, "w")
    info = """<html>
    <head><title>"""+ds.basename+"""</title>
    <h1 style="text-align:center;font-size:350%">"""+ds.basename+"""</h1></head>
    <body>
    <p style="text-align:center;font-size=250%">t = """+str(ds.current_time)+"""<br>
    <hr />
    """
    outfile.write(info)

    ## now add the figures
    print "figure names should be dynamic or passed, but not currently"
    density_string = 'Density_z_'+ds.basename+'.png'
    temperature_string = 'Temperature_z_'+ds.basename+'.png'

    addfig = r"""<img src='"""+density_string+r"""' style="width:35%">"""
    outfile.write(addfig)
    addfig = r"""<img src='"""+temperature_string+r"""' style="width:35%">"""
    outfile.write(addfig)

    outfile.write("</body></html>")
    outfile.close()




#-----------------------------------------------------------------------------------------------------

def look_at_output(clobber):
    ## is there an output defined? if not, do all!
    print " ---> WARNING OUTPUT NAMES ARE HARDCODED, ASSUMES DD#### FORMAT <---- "
    DATA_DIR = "."
    PLOTS_DIR = "./plots"
    dataset_list = glob.glob(os.path.join(DATA_DIR, 'DD????/DD????'))

    print " ---> I'm not checking the plots have been made I'm just gonna make them <---- "
    run_catalog = Table(names=("rootname","time","Density","Temperature"),dtype=("S200","f8","S200","S200"))
    for output in dataset_list:
        ds = yt.load(output)
        density_string = 'Density_z_'+ds.basename+'.png'
        temperature_string = 'Temperature_z_'+ds.basename+'.png'
        if clobber or not (os.path.exists(density_string) and os.path.exists(density_string)):
            make_yt_plots(ds)
        if clobber:
            make_quicklook_page(ds)
        ds_link = '<a href=\"'+ ds.basename + '_quicklook.html\">' + ds.basename + '</a>'
        density_html_string = '<a href=\"'+density_string+'\"><img height=\"200px\" src=\"'+density_string+'\"></a>'
        temperature_html_string = '<a href=\"'+temperature_string+'\"><img height=\"200px\" src=\"'+temperature_string+'\"></a>'
        run_catalog.add_row([ds_link, ds.current_time, density_html_string,  temperature_html_string])

    ## make a table with the thumbnails and basic information
    run_catalog.write("run_catalog.temp",format="jsviewer")
    os.system('sed "s/&lt;/</g" run_catalog.temp | sed "s/&gt;/>/g" > run_catalog.html')
    os.system('rm run_catalog.temp')

    ## now generate QL pages


    return "yay simulations!"

#-----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    ## which outputs?
    args = parse_args()
    clobber = args.clobber

    message = look_at_output(clobber)
    sys.exit(message)
