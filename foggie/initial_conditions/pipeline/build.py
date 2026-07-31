"""
Rendering of the per-level Enzo parameter files and run scripts.

The three DM templates 25Mpc_DM_512-L{1,2,3}.enzo differed only in
CosmologySimulationNumberOfInitialGrids, MustRefineParticlesRefineToLevel and
MinimumOverDensityForRefinement, so they collapse to a single DM-LX.enzo plus
the keyword table below.  `ic_pipeline validate-templates` proves the collapse
is faithful by re-rendering each level and diffing against the originals.
"""

import os
import re


# ---------------------------------------------------------------------------
# Level-dependent values
# ---------------------------------------------------------------------------

# The tail of the MinimumOverDensityForRefinement vector, after the two
# level-dependent leading entries.  The DM and gas templates were written with
# different lengths; preserved verbatim rather than normalised, so the rendered
# files stay byte-comparable with the originals.
_OVERDENSITY_TAIL_DM = "1. 1. 1. 1. 1. 1. 1. 1."
_OVERDENSITY_TAIL_GAS = "1. 1. 1. 1. 1"


def _fmt_overdensity(value):
    """Format a power of 1/8 the way the hand-written templates did."""
    if value == 1.0:
        return "1."
    return repr(value)


def min_overdensity(level, phase="DM"):
    """MinimumOverDensityForRefinement for a given zoom level.

    Divide by 8 for each additional level, per Britton Smith's notes: level 1
    gives 1., level 2 gives 0.125, level 3 gives 0.015625.
    """
    value = 8.0 ** -(level - 1)
    lead = _fmt_overdensity(value)
    tail = _OVERDENSITY_TAIL_GAS if phase == "gas" else _OVERDENSITY_TAIL_DM
    return "%s %s %s" % (lead, lead, tail)


def enzo_keywords(box, level, phase="DM", grid_parameters=""):
    """Keyword table for the .enzo templates."""
    kw = {
        "__NUM_INITIAL_GRIDS__": str(level + 1),
        "__MRP_REFINE_TO_LEVEL__": str(level),
        "__GRID_PARAMETERS__": grid_parameters,
    }
    if phase == "gas":
        kw["__MIN_OVERDENSITY_GAS__"] = min_overdensity(level, "gas")
        kw["__MAX_REFINE_LEVEL__"] = str(box.gas_max_refine_level)
    else:
        kw["__MIN_OVERDENSITY__"] = min_overdensity(level, "DM")
    return kw


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

def replace_keywords(text, mapping):
    """Substitute __KEYWORD__ placeholders, erroring on anything left behind."""
    for key, value in mapping.items():
        text = text.replace(key, str(value))
    leftover = sorted(set(re.findall(r"__[A-Z_]+__", text)))
    if leftover:
        raise RuntimeError("Unsubstituted keyword(s): %s" % ", ".join(leftover))
    return text


def render_enzo_param(box, level, phase="DM", grid_parameters=""):
    """Render the Enzo parameter file for one stage."""
    name = "gas-LX.enzo" if phase == "gas" else "DM-LX.enzo"
    template = os.path.join(box.template_dir_path(), name)
    with open(template) as fp:
        text = fp.read()
    return replace_keywords(text, enzo_keywords(box, level, phase, grid_parameters))


def read_grid_parameters(parameter_file_txt):
    """Pull the CosmologySimulationGrid* block out of MUSIC's parameter_file.txt.

    Replaces the old `grep ... > grid_parameters.txt; cat template
    grid_parameters.txt > pars.temp; mv pars.temp template` shell sequence.
    """
    with open(parameter_file_txt) as fp:
        return "".join(l for l in fp if l.startswith("CosmologySimulationGrid"))


def render_runscript(box, halo_id, level, phase="DM", pipeline_hook=""):
    """Render the PBS run script for one stage."""
    template = os.path.join(box.template_dir_path(), "RunScript.sh")
    with open(template) as fp:
        text = fp.read()

    is_gas = phase == "gas"
    walltime = box.gas_walltime if is_gas else box.dm_walltime
    hours, minutes, seconds = (int(x) for x in walltime.split(":"))
    mapping = {
        "__JOBNAME__": "halo%s-L%d%s" % (halo_id, level, "-gas" if is_gas else ""),
        "__GROUP__": box.group_list,
        "__SELECT__": box.gas_select if is_gas else box.dm_select,
        "__WALLTIME__": walltime,
        "__QUEUE__": "normal",
        "__NRANKS__": box.gas_nranks if is_gas else box.dm_nranks,
        # simrun.pl needs the walltime in seconds so it knows when to resubmit.
        "__SIMRUN_WALL__": hours * 3600 + minutes * 60 + seconds,
        "__EMAIL__": box.email,
        "__ENZO_EXE__": box.enzo_exe,
        "__PARAM_FILE__": box.param_filename(level, phase),
        "__PIPELINE_HOOK__": pipeline_hook,
    }
    return replace_keywords(text, mapping)
