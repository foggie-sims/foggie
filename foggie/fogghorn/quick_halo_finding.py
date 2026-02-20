#!/usr/bin/env python
# coding: utf-8

import yt
import argparse, os
from yt_astro_analysis.halo_analysis import HaloCatalog, add_quantity
from foggie.utils.foggie_load import foggie_load
from foggie.utils.halo_quantity_callbacks import * 
from datetime import datetime
from astropy.table import QTable
from astropy.table import Table
from foggie.utils.analysis_utils import *

def prep_dataset_for_halo_finding(simulation_dir, snapname, trackfile=None, boxwidth=0.04): 
    """Prepare a small dataset subregion around a halo center for finding.

    This helper loads the dataset for the given `snapname` (located under
    `simulation_dir`), reads the halo center from the trackfile-provided
    dataset fields, and returns the full dataset along with a cubic
    subregion (`box`) centered on the halo with side length `boxwidth`
    (in code units).

    Parameters
    ----------
    simulation_dir : str
        Base directory containing the snapshot subdirectory named
        `snapname`.
    snapname : str
        Snapshot name (used to construct the dataset path).
    trackfile : str
        Path to the track file used by `foggie_load` to determine the
        halo center (passed through as ``trackfile_name``).
    boxwidth : float, optional
        Side length of the returned cubic subregion in code units
        (default: 0.04).

    Returns
    -------
    ds : yt.Dataset
        The loaded dataset object for the requested snapshot.
    box : yt.Region
        A cubic `yt.Region` subvolume centered on the halo center with
        side length `boxwidth` (in code units).
    """

    dataset_name = simulation_dir+'/'+snapname+'/'+snapname

    if (trackfile == None): # if no trackfile, use min and max positions of the smallest DM particles to define a subregion
        ds, region = foggie_load(dataset_name, central_halo = False) 
    else: 
        ds, region = foggie_load(dataset_name, central_halo = False, trackfile_name = trackfile)

    return ds, region 

def halo_finding_step(ds, box, simulation_dir='./', threshold=400.): 
    """Run the HOP halo finder on a dataset subvolume and create a catalog.

    Parameters
    ----------
    ds : yt.Dataset
        The dataset to operate on.
    box : yt.Region
        Subvolume region to provide to the halo finder (used as ``subvolume``).
    simulation_dir : str, optional
        Directory where the `halo_catalogs` output will be written (default: './').
    threshold : float, optional
        Overdensity threshold for the HOP finder (default: 400.0).

    Returns
    -------
    HaloCatalog
        The created `HaloCatalog` instance (after ``create()`` has been called).
    """

    hc = HaloCatalog(data_ds=ds, finder_method='hop', 
        finder_kwargs={"subvolume": box, "threshold":threshold, "ptype":"nbody", "save_particles":True}, 
        output_dir=simulation_dir+'/halo_catalogs') 
                      
    hc.create()

    return hc

def repair_halo_catalog(ds, simulation_dir, snapname, min_rvir=10., min_halo_mass=1e10): 
    """Load, repair, and enrich a halo catalog for a snapshot.

    This function loads an existing halo catalog produced by the halo
    finder for the given `snapname`, wraps it with a `HaloCatalog`,
    applies callbacks and filters to remove spurious halos, registers
    a set of derived halo quantities (via `add_quantity`) and then
    recreates/saves the repaired catalog to the `halo_catalogs` output
    directory.

    Parameters
    ----------
    ds : yt.Dataset
        The dataset used as the data context for the halo catalog.
    simulation_dir : str
        Base directory containing the `halo_catalogs` output folder.
    snapname : str
        Snapshot name (used to locate the catalog file).
    min_rvir : float, optional
        Minimum virial radius (in kpc) to keep halos (default: 10.0).
    min_halo_mass : float, optional
        Minimum halo mass (in Msun) to keep halos (default: 1e10).

    Returns
    -------
    HaloCatalog
        The repaired and augmented `HaloCatalog` instance.
    """
    hds = yt.load(simulation_dir+'/halo_catalogs/'+snapname+'/'+snapname+'.0.h5')
    hc = HaloCatalog(data_ds=ds, halos_ds=hds, output_dir=simulation_dir+'/halo_catalogs')
    hc.add_callback("sphere")

    hc.add_filter("quantity_value", "virial_radius", ">", min_rvir, "kpc")
    hc.add_filter("quantity_value", "particle_mass", ">", min_halo_mass, "Msun") #<--- this supresses a lot of bogus halos

    add_quantity("corrected_rvir", halo_corrected_rvir)
    hc.add_quantity("corrected_rvir")

    quantities = {"overdensity":halo_overdensity, "average_temperature":halo_average_temperature, 
              "average_metallicity":halo_average_metallicity, "total_mass":halo_total_mass, 
              "total_gas_mass":halo_total_gas_mass, "total_ism_gas_mass":halo_ism_gas_mass, 
              "total_ism_HI_mass":halo_ism_HI_mass, "total_ism_HII_mass":halo_ism_HII_mass, 
              "total_cgm_gas_mass":halo_cgm_gas_mass, "total_cold_cgm_gas_mass": halo_cold_cgm_gas_mass,
              "total_cool_cgm_gas_mass":halo_cool_cgm_gas_mass, "total_warm_cgm_gas_mass":halo_warm_cgm_gas_mass,
              "total_hot_cgm_gas_mass":halo_hot_cgm_gas_mass, "total_star_mass":halo_total_star_mass, 
              "total_metal_mass":halo_total_metal_mass, "total_young_stars7_mass":halo_total_young_stars7_mass, 
              "actual_baryon_fraction": halo_actual_baryon_fraction, "sfr7":halo_sfr7, "sfr8":halo_sfr8,
              "total_young_stars8_mass": halo_total_young_stars8_mass, "max_metallicity": halo_max_metallicity, 
              "max_gas_density": halo_max_gas_density, "max_dm_density": halo_max_dm_density, 
              "outflow_mass_300":halo_outflow_300, "outflow_mass_500":halo_outflow_500} 
    
    #These quantities will also be calculated with 2Rvir to capture, e.g. outflows that have passed beyond Rvir but are still associated with the halo.
    quantities_2rvir = {"total_gas_mass_2rvir":halo_total_gas_mass, "total_cgm_gas_mass_2rvir":halo_cgm_gas_mass, 
                        "total_cold_cgm_gas_mass_2rvir": halo_cold_cgm_gas_mass, "total_cool_cgm_gas_mass_2rvir":halo_cool_cgm_gas_mass, 
                        "total_warm_cgm_gas_mass_2rvir":halo_warm_cgm_gas_mass, "total_hot_cgm_gas_mass_2rvir":halo_hot_cgm_gas_mass}

    for q in quantities.keys(): 
        add_quantity(q, quantities[q])
        hc.add_quantity(q, correct=True) 

    for q in quantities_2rvir.keys():
        add_quantity(q, quantities[q])
        hc.add_quantity(q, correct=True, rvir_factor=2.) 

    if (ds.parameters['MultiSpecies'] == 2): # this is necessary because older runs do not have this field 
        add_quantity("average_fH2", halo_average_fH2) 
        hc.add_quantity("average_fH2", correct=True)

        add_quantity("total_ism_H2_mass", halo_ism_H2_mass)
        hc.add_quantity("total_ism_H2_mass", correct=True)

    hc.create()

    return hc 

def export_to_astropy(simulation_dir, snapname): 
    new_ds = yt.load(simulation_dir+'/halo_catalogs/'+snapname+'/'+snapname+'.0.h5')
    all_data = new_ds.all_data()
    halo_table = QTable() 

    for field in new_ds.field_list: 
        if (field[0] == 'halos'):
            halo_table[field[1]] = all_data[field].to_astropy() 

    halo_table.write(simulation_dir+'/halo_catalogs/'+snapname+'/'+snapname+'.0.fits', format='fits', overwrite=True) 
    halo_table.write(simulation_dir+'/halo_catalogs/'+snapname+'/'+snapname+'.0.txt', format='ascii', overwrite=True) 

def find_root_particles(simulation_dir, ds, hc):
    '''Find and save to file the indices of DM particles within 1Rvir of the
    halo in the catalog hc that is closest to the center of the old track box.
    Recommend to use the z = 4 snapshot for identifying root particles.'''

    halos = hc.all_data()
    halo_mass = halos[('halos','total_mass')].in_units('Msun')
    halo_radius = halos[('halos','corrected_rvir')].in_units('kpc')
    halo_position = halos[('halos','particle_position')].in_units('kpc')

    dist_from_old_track = np.sqrt((halo_position[:,0]-ds.halo_center_kpc[0])**2. + (halo_position[:,1]-ds.halo_center_kpc[1])**2. + (halo_position[:,2]-ds.halo_center_kpc[2])**2.)

    center = halo_position[dist_from_old_track==np.min(dist_from_old_track)]
    radius = halo_radius[dist_from_old_track==np.min(dist_from_old_track)].in_units('kpc').v

    sph = ds.sphere(center=center[0], radius=(radius[0], 'kpc'))
    DM_in_sph = sph[('dm','particle_index')].v

    a = Table() 
    a['root_index'] = DM_in_sph
    a.write(simulation_dir + '/halo_catalogs/root_index.txt', format='ascii', overwrite=True)

def make_halo_plots(ds, simulation_dir, snapname): 

    print("Looking for halo catalog at, to plot them: ", simulation_dir+'/halo_catalogs/'+snapname+'/'+snapname+'.0.h5') 
    new_ds = yt.load(simulation_dir+'/halo_catalogs/'+snapname+'/'+snapname+'.0.h5')
    all_data = new_ds.all_data()

    x = all_data["halos", "particle_position_x"]
    y = all_data["halos", "particle_position_y"]
    z = all_data["halos", "particle_position_z"]
    corrected_rvir = all_data["halos", "corrected_rvir"].in_units('kpc') 
    total_halo_mass = all_data["halos", "total_mass"].in_units('Msun')
    sfr7 = all_data["halos", "sfr7"].in_units('Msun/yr') 
    actual_baryon_fraction = all_data["halos", "actual_baryon_fraction"]
    overdensity = all_data["halos", "overdensity"]

    for index in np.arange(len(x)):
        center0 = [float(x.in_units('code_length')[index]), float(y.in_units('code_length')[index]), float(z.in_units('code_length')[index])] 
        halo0 = ds.sphere(center0, radius = corrected_rvir[index] ) 

        p = yt.ProjectionPlot(ds, 'z', 'density', weight_field='density', data_source=halo0, center=center0, width=2.5 * corrected_rvir[index])
        p.set_cmap('density', density_color_map)
        current_datetime = datetime.now()
        p.annotate_title(os.getcwd().split('/')[-1] + '  ' + simulation_dir + '/' + ds._input_filename[-6:] + '    ' + str(current_datetime.date()) )
        p.annotate_timestamp(redshift=True)
        p.set_zlim('density', 1e-28, 1e-21)
        p.annotate_sphere(halo0.center, halo0.radius, circle_args={"color": "green", "linewidth": 2, "linestyle": "dashed"}) 
        p.annotate_text([0.05, 0.95], 'Virial Radius = ' + str(corrected_rvir[index])[0:5], coord_system='axis', text_args={"color": "black"}) 
        p.annotate_text([0.40, 0.95], 'Virial Mass = ' + str(total_halo_mass[index]/1e10)[0:5]+'e10', coord_system='axis', text_args={"color": "black"}) 
        p.annotate_text([0.05, 0.92], 'Baryon Fraction = ' + str(actual_baryon_fraction[index])[0:5], coord_system='axis', text_args={"color": "black"}) 
        p.annotate_text([0.40, 0.92], 'Overdensity = ' + str(overdensity[index])[0:7], coord_system='axis', text_args={"color": "black"}) 
        p.annotate_text([0.75, 0.92], 'SFR7 = ' + str(sfr7[index])[0:7], coord_system='axis', text_args={"color": "black"}) 
        p.set_origin('native') 
        p.save(simulation_dir+'/halo_catalogs/'+snapname+'/'+snapname+'_index'+str(index))

        p = yt.ProjectionPlot(ds, 'z', ('enzo', 'Dark_Matter_Density'), data_source=halo0, center=center0, width=2.5 * corrected_rvir[index])
        p.set_cmap(('enzo', 'Dark_Matter_Density'), density_color_map)
        p.annotate_title(os.getcwd().split('/')[-1] + '  ' + simulation_dir + '/' + ds._input_filename[-6:] + '    ' + str(current_datetime.date()) )
        p.annotate_timestamp(redshift=True)
        p.set_unit(('enzo', 'Dark_Matter_Density'), 'Msun/pc**2') 
        p.set_zlim(('enzo', 'Dark_Matter_Density'), 1e-27, 1e-23) 
        p.annotate_sphere(halo0.center, halo0.radius, circle_args={"color": "green", "linewidth": 2, "linestyle": "dashed"}) 
        p.annotate_text([0.05, 0.95], 'Virial Radius = ' + str(corrected_rvir[index])[0:5], coord_system='axis', text_args={"color": "black"}) 
        p.annotate_text([0.40, 0.95], 'Virial Mass = ' + str(total_halo_mass[index]/1e10)[0:5]+'e10', coord_system='axis', text_args={"color": "black"}) 
        p.annotate_text([0.05, 0.92], 'Baryon Fraction = ' + str(actual_baryon_fraction[index])[0:5], coord_system='axis', text_args={"color": "black"}) 
        p.annotate_text([0.40, 0.92], 'Overdensity = ' + str(overdensity[index])[0:7], coord_system='axis', text_args={"color": "black"}) 
        p.set_origin('native') 
        p.save(simulation_dir+'/halo_catalogs/'+snapname+'/'+snapname+'_index'+str(index))

    return True 

def parallel_loop_over_halos(snap, args):
    """Call halo finding steps on the snapshot 'snap'. This function is used for
    running in parallel, with one output per process.
    
    Parameters
    ----------
    args: parser.parse_args list
        command-line arguments for this script
    snap: str
        the name of the snapshot
    
    Returns
    -------
    nothing
    """

    ds, box = prep_dataset_for_halo_finding(args.directory, snap, args.trackfile, boxwidth=args.boxwidth) 
    hc = halo_finding_step(ds, box, simulation_dir=args.directory, threshold=args.threshold) 
    hc = repair_halo_catalog(ds, args.directory, snap, min_rvir = args.min_rvir, min_halo_mass=args.min_mass) 
    export_to_astropy(args.directory, snap)
    if (args.make_plots): make_halo_plots(ds, args.directory, args.output)


if __name__ == "__main__":

    """Command-line entrypoint for quick halo finding.

    Usage:
      python quick_halo_finding.py --output SNAPNAME --trackfile TRACKFILE [--boxwidth 0.04] [--threshold 400] [--min_rvir 10] [--min_mass 1e10]

    This script prepares a small dataset subregion around a halo center,
    runs a HOP-based halo finder to produce a halo catalog, then repairs
    and filters that catalog by adding quantities and applying callbacks.

    This code is a custom extension of yt's built-in HOP algorithm: 
        "halo_finding_step" wraps the HOP halo finder with a high overdensity threshold (NOT the same as the canonical overdensity defining virialization)
        "repair_halo_catalog" uses the halo callbacks to generate corrected Rvir for overdensity = 200 
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--output', metavar='output', type=str, action='store', default=None, required=True, help='Output to run a halo catalog for. Single output or comma-separated list or range like DD0400-DD0500.') 
    parser.add_argument('--directory', metavar='directory', type=str, action='store', default='./', required=False, help='Pathname to simulation directory') 
    parser.add_argument('--trackfile', metavar='trackfile', type=str, action='store', default=None, required=False, help='Track file for this halo (center of box subregion)')
    parser.add_argument('--boxwidth', metavar='boxwidth', type=float, action='store', default=0.04, required=False, help='Width of subregion box in code units')
    parser.add_argument('--threshold', metavar='threshold', type=float, action='store', default=400., required=False, help='Overdensity thresold for HOP algorithm (default = 400)')
    parser.add_argument('--min_rvir', metavar='min_rvir', type=float, action='store', default=10, required=False, help='Filter halo catalogs to this min Rvir [kpc]')
    parser.add_argument('--min_mass', metavar='min_mass', type=float, action='store', default=1e10, required=False, help='Filter halo catalogs to this min mass [Msun]')
    parser.add_argument('--nproc', metavar='nproc', type=int, action='store', default=1, required=False, help='Use this many processors to run in parallel (one snapshot per process, default 1')
    parser.add_argument('--save_root', dest='save_root', action='store_true', default=False, required=False, help='After finding halos, save to file the particle indices in most massive halo? Default False')
    parser.add_argument('--make_plots', dest='make_plots', action='store_true', default=True, required=False, help='After finding halos, make plots of each one for verification and reference')

    args = parser.parse_args()

    print('ARGS args = ', args)
    if (args.nproc==1):
        ds, box = prep_dataset_for_halo_finding(args.directory, args.output, trackfile=args.trackfile, boxwidth=args.boxwidth) 
        hc = halo_finding_step(ds, box, simulation_dir=args.directory, threshold=args.threshold) 
        hc = repair_halo_catalog(ds, args.directory, args.output, min_rvir = args.min_rvir, min_halo_mass=args.min_mass) 
        export_to_astropy(args.directory, args.output)
        if (args.make_plots): make_halo_plots(ds, args.directory, args.output) 
        if (args.save_root):
            hc = yt.load(args.directory+'/halo_catalogs/'+args.output+'/'+args.output+'.0.h5')
            find_root_particles(args.directory, ds, hc)

    else:
        import multiprocessing as multi
        outs = make_output_list(args.output)
        for i in range(len(outs)//args.nproc):
            threads = []
            for j in range(args.nproc):
                snap = outs[args.nproc*i+j]
                threads.append(multi.Process(target=parallel_loop_over_halos, args=[snap, args]))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        # For any leftover snapshots, run one per processor
        threads = []
        for j in range(len(outs)%args.nproc):
            snap = outs[-(j+1)]
            threads.append(multi.Process(target=parallel_loop_over_halos, args=[snap, args]))
        for t in threads:
            t.start()
        for t in threads:
            t.join()