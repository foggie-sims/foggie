#!/usr/bin/env python
# coding: utf-8

import yt
from yt.units import dimensions
from yt.fields.api import ValidateParameter # req yt 4 for correctness
import numpy as np

# Field Functions

def _overdensity(field, data):
    odthresh = data.get_field_parameter("overdensity_threshold")
    return (data[("gas","density")] > odthresh).astype(float)

def _velocity_divergence(field, data):
    return (data[("gas", "velocity_divergence")] < 0).astype(float)

def _freefall_time(field, data):
    total_density = data[("gas","density")] + data[("gas", "dark_matter_density")]
    return np.sqrt(3*np.pi / (32 * yt.units.gravitational_constant * total_density))

def _cold_or_cooling(field, data):
    tempthresh = data.get_field_parameter("temperature_threshold")
    cold = data[("gas", "temperature")] < tempthresh
    cooling = data[("gas", "cooling_time")] < data[("gas", "freefall_time")]
    return np.logical_or(cold, cooling).astype(float)

def _jeans_mass(field, data):
    """
    WARNING: Enzo uses a constant assumption for mean molecular weight,
    whereas this field does not.
    """
    return (data[("gas", "cell_mass")] > data[("gas","jeans_mass")]).astype(float)

def _mass_threshold(field, data):
    data._debug
    if data.has_field_parameter("stellar_mass_efficiency"):
        masseff = data.get_field_parameter("stellar_mass_efficiency")
    else:
        masseff = 1
    massthresh = data.get_field_parameter("stellar_mass_threshold")
    return ((masseff * data[("gas", "cell_mass")]) > massthresh).astype(float)

# Add fields to yt

yt.add_field(
     ("index", "overdensity_criterion"),
     function = _overdensity,
     sampling_type = "cell",
     dimensions = dimensions.dimensionless,
     take_log = False,
     validators = [ValidateParameter(["overdensity_threshold"])]
)

yt.add_field(
    ("index", "velocity_divergence_criterion"),
    function = _velocity_divergence,
    sampling_type = "cell",
    dimensions = dimensions.dimensionless,
    take_log = False
)

yt.add_field(
    ("gas", "freefall_time"),
    function = _freefall_time,
    sampling_type = "cell",
    dimensions = dimensions.time,
    take_log = True,
)

yt.add_field(
    ("index", "temperature_criterion"),
    function = _cold_or_cooling,
    sampling_type = "cell",
    dimensions = dimensions.dimensionless,
    take_log = False,
    validators = [ValidateParameter(["temperature_threshold"])]
)

yt.add_field(
    ("index", "jeans_mass_criterion"),
    function = _jeans_mass,
    sampling_type = "cell",
    dimensions = dimensions.dimensionless,
    take_log = False
)

yt.add_field(
    ("index", "minimum_mass_criterion"),
    function = _mass_threshold,
    sampling_type = "cell",
    dimensions = dimensions.dimensionless,
    take_log = False,
    validators = [ValidateParameter(["stellar_mass_threshold"])]
)

# Sample usage

if __name__ == "__main__":

    # run:
    #    from fields_sf_criteria import *
    # to add all fields to yt
    
    ds = yt.load("galaxy0030/galaxy0030")

    dsk = ds.disk('c', [0,0,1], (30,'kpc'), (3,'kpc'))

    dsk.set_field_parameter("overdensity_threshold",
                            ds.quan(1e-24, 'g/cm**3'))
    dsk.set_field_parameter("temperature_threshold", ds.quan(3e3, 'K'))
    dsk.set_field_parameter("stellar_mass_threshold", ds.quan(1e4, 'Msun'))

    print("Overdensity threshold:", dsk.field_parameters["overdensity_threshold"].to('g/cm**3'))
    print("Temperature threshold:", dsk.field_parameters["temperature_threshold"])
    print("Stellar mass threshold:", dsk.field_parameters["stellar_mass_threshold"])

    slc = yt.SlicePlot(ds, 'z', center='c', width=(30,'kpc'),
                       fields=[("index","overdensity_criterion"),
                               ("index","velocity_divergence_criterion"),
                               ("index","temperature_criterion"),
                               ("index","jeans_mass_criterion"),
                               ("index","minimum_mass_criterion"),
                               ("gas","density")],
                       data_source=dsk)

    slc.save()

    def _combine_criteria(field, data):
        return data[("index","overdensity_criterion")] * \
               data[("index","velocity_divergence_criterion")] * \
               data[("index","temperature_criterion")] * \
               data[("index","jeans_mass_criterion")] * \
               data[("index","minimum_mass_criterion")]

    ds.add_field(name=("index","star_formation_criteria"),
                 function=_combine_criteria,
                 sampling_type="cell",
                 units='1')

    # Using method "sum" behaves incorrectly for off-axis plots
    prj = yt.ProjectionPlot(ds, 'z', center='c', width=(30,'kpc'),
                            fields=("index","star_formation_criteria"),
                            method="sum",
                            data_source=dsk)

    prj.save()
