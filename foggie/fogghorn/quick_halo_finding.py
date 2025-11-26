#!/usr/bin/env python
# coding: utf-8

import yt, argparse 
from yt_astro_analysis.halo_analysis import HaloCatalog, add_quantity
from foggie.utils.foggie_load import foggie_load
from foggie.utils.halo_quantity_callbacks import * 
from astropy.table import QTable

def prep_dataset_for_halo_finding(simulation_dir, snapname, trackfile, boxwidth=0.04): 
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

    ds, _ = foggie_load(dataset_name, trackfile_name = trackfile)

    box = ds.r[ds.halo_center_code.value[0]-boxwidth/2:ds.halo_center_code.value[0]+boxwidth/2, 
               ds.halo_center_code.value[1]-boxwidth/2:ds.halo_center_code.value[1]+boxwidth/2, 
               ds.halo_center_code.value[2]-boxwidth/2:ds.halo_center_code.value[2]+boxwidth/2] 

    return ds, box 


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
              "max_gas_density": halo_max_gas_density, "max_dm_density": halo_max_dm_density} 


    print('made it this far in repair_halo_catalog 1')
    for q in quantities.keys(): 
        add_quantity(q, quantities[q])
        hc.add_quantity(q, correct=True) 

    if (ds.parameters['MultiSpecies'] == 2): # this is necessary because older runs do not have this field 
        add_quantity("average_fH2", halo_average_fH2) 
        hc.add_quantity("average_fH2", correct=True)

        add_quantity("total_ism_H2_mass", halo_ism_H2_mass)
        hc.add_quantity("total_ism_H2_mass", correct=True)

    print('made it this far in repair_halo_catalog 2')

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

    parser.add_argument('--output', metavar='output', type=str, action='store', default=None, required=True, help='Output to run a halo catalog for') 
    parser.add_argument('--directory', metavar='directory', type=str, action='store', default='./', required=False, help='Pathname to simulation directory') 
    parser.add_argument('--trackfile', metavar='trackfile', type=str, action='store', default=None, required=True, help='Track file for this halo (center of box subregion)')
    parser.add_argument('--boxwidth', metavar='boxwidth', type=float, action='store', default=0.04, required=False, help='Width of subregion box in code units')
    parser.add_argument('--threshold', metavar='threshold', type=float, action='store', default=400., required=False, help='Overdensity thresold for HOP algorithm (default = 400)')
    parser.add_argument('--min_rvir', metavar='min_rvir', type=float, action='store', default=10, required=False, help='Filter halo catalogs to this min Rvir [kpc]')
    parser.add_argument('--min_mass', metavar='min_mass', type=float, action='store', default=1e10, required=False, help='Filter halo catalogs to this min mass [Msun]')

    args = parser.parse_args()

    ds, box = prep_dataset_for_halo_finding('./', args.output, args.trackfile, boxwidth=args.boxwidth) 
    hc = halo_finding_step(ds, box, simulation_dir='./', threshold=args.threshold) 
    hc = repair_halo_catalog(ds, './', args.output, min_rvir = args.min_rvir, min_halo_mass=args.min_mass) 
    export_to_astropy('./', args.output)