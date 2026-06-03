# Generating Emission Line Maps with CLOUDY for FOGGIE

**Originally documented by:** Lauren Corlies  
**Updated by:** Vida Saeedzadeh 
**Last updated:** May 2026  

> This README describes how to install CLOUDY and CIAOLoop, set up a parameter file, and generate CLOUDY tables for use with the FOGGIE simulation suite. It is based on a guide originally written by Lauren Corlies, with updates added by Vida Saeedzadeh.

---

## Table of Contents

1. [Overview](#overview)
2. [Installing CLOUDY](#installing-cloudy)
3. [Installing CIAOLoop](#installing-ciaoloop)
4. [Running the Code](#running-the-code)
5. [The Input Parameter File](#the-input-parameter-file)
6. [Sanity Checks](#sanity-checks)
7. [Notes and Tips](#notes-and-tips)

---

## Overview

Generating emission lines for FOGGIE cells requires two pieces of software:

- **CLOUDY** — a photoionization and spectral synthesis code that computes emissivities across a grid of physical conditions.
- **CIAOLoop** — a wrapper that automates running CLOUDY over a parameter grid (density, temperature, redshift, etc.) using a single input parameter file.

Because emissivities in many FOGGIE simulation cells are extremely low, special care is needed during compilation and setup to avoid numerical precision issues.

---

## Installing CLOUDY

CLOUDY is under active development and releases new versions regularly. Always refer to the official installation guide for the most up-to-date steps:

**Official installation guide:**  
https://gitlab.nublado.org/cloudy/cloudy/-/wikis/StepByStep

The Step-by-Step instructions on that page are generally straightforward to follow. The one critical difference for FOGGIE emission work is the compilation flags used.

### Compilation (FOGGIE-specific)

When compiling CLOUDY, use the following commands **instead of** the default `make`:

```bash
make distclean
make -j 4 EXTRA=-DFLT_IS_DBL &> /dev/null
```

The `EXTRA=-DFLT_IS_DBL` flag compiles CLOUDY in double-precision floating point mode. This is essential for FOGGIE because emissivities in many simulation cells are extremely low, and without this flag, CLOUDY will produce `NaN` values across much of the parameter space.

### Smoke Test

After compiling, run the built-in smoke test as described in the official guide to verify the installation is working correctly before proceeding.

### Set Environment Variable

For convenience, add the following line to your `.bashrc` (or `.zshrc`) so that CLOUDY can always find its data files:

```bash
export CLOUDY_DATA_PATH="/path/to/cloudy/cxx.xx/data"
```

Replace `/path/to/cloudy/cxx.xx/` with your actual install path (e.g., `/Users/yourname/repos/c22.01/`).

---

## Installing CIAOLoop

CIAOLoop was originally written by our own **Britton Smith**, who is a good resource for questions about the code.

**Original repository:**  
https://github.com/brittonsmith/cloudy_cooling_tools

> **Note For FOGGIE Emission Maps:** Use `CIAOLoop` version with Lauren's updates. For now it can be found here in repo under directory:/cgm_emission/How-to-Run-CLOUDY/CIAOLoop_lauren

The file that we are most interested in is  `CIAOLoop`. This is the code that generates the correct Cloudy runs using a `.par` parameter file where you define the parameters that you want to set fo takes a set number of inputs. Britton has some README instructions with the repository. 


---

## Running the Code

CIAOLoop is driven entirely by an input parameter file. To run it:

```bash
./CIAOLoop NAME.par
```

where `NAME.par` is your parameter file (see the section below for a template and explanation of all fields).

CIAOLoop will:
- Set up the CLOUDY runs based on your parameter grid
- Execute CLOUDY for each combination of parameters
- Collect and store the output line emissivities

---

## The Input Parameter File

The parameter file used to generate the CLOUDY tables in `/cgm_emission/cloudy_extended_z0_selfshield/` is included in this directory as `parameters.par`. Use it as your starting point.

Below is a complete, annotated example of what a parameter file looks like, with explanations for each field:

```
###########################################################
################## Line Emissivity Maps ###################
###########################################################

###########################################################
##################### Run parameters ######################

# Full path to the CLOUDY executable
cloudyExe = /path/to/cloudy/cxx.xx/source/cloudy.exe

# Set to 1 to suppress saving raw CLOUDY output files (saves disk space)
saveCloudyOutputFiles = 1

# Set to 1 to exit the loop if CLOUDY crashes on any run
exitOnCrash = 1

# A descriptive name for this run (used as a prefix for output files)
outputFilePrefix = TEST_z1_HM12_sh

# Directory where output files will be written
outputDir = /path/to/output/directory

# Index of the first run (usually 1)
runStartIndex = 1

# Set to 0 for a real run; set to 1 to test the setup without executing CLOUDY
test = 0

# Run mode: 4 = line emissivity map mode
cloudyRunMode = 4

###########################################################
################## Line Map Parameters ####################

# Temperature grid (log space)
coolingMapTmin    = 1e3   # Minimum temperature [K]
coolingMapTmax    = 1e8   # Maximum temperature [K]
coolingMapTpoints = 51    # Number of temperature steps (log-spaced)

# Set to 0 to disable Jeans length criterion
coolingMapUseJeansLength = 0

# Emission lines to map (format: element  ionization_state  wavelength_angstrom)
# Add or remove lines as needed for your science case
lineMapLine = H  1 1216    # Lyman-alpha
lineMapLine = H  1 6563    # H-alpha
lineMapLine = C  4 1548    # CIV
lineMapLine = C  4 1551    # CIV 
lineMapLine = O  6 1032    # OVI
lineMapLine = O  6 1038    # OVI 
lineMapLine = C  3 977     # CIII
lineMapLine = C  3 1907    # CIII
lineMapLine = C  3 1910    # CIII

############################################################
############ Commands to be executed every time ############

# Allow up to 200 CLOUDY failures before aborting
command failures 200 times map

# Iterate to convergence (max 2 iterations, 20% error tolerance)
command iterate to convergence max=2 error=0.20

# Set the CMB radiation field at the target redshift
command CMB redshift 0.0

# Set the UV background radiation field (Haardt & Madau 2012 here; adjust as needed)
command Table HM12 redshift 0.0

# Numerical stability and physical approximations 
command set WeakHeatCool -20
command no H2 molecule
command no charge transfer

#######################################################
############# Commands to be looped over ##############

# Density grid: loop over hydrogen number density [log cm^-3]
# Format: (min; max; step) in log space
loop [hden] (-5;2;0.5)

# Metallicity: set metals to zero (log relative to Solar)
command metals 0 log
```

### Key Parameters to Modify for a New Run

| Parameter | What to change |
|---|---|
| `cloudyExe` | Path to your compiled CLOUDY executable |
| `outputFilePrefix` | Descriptive label for this run (e.g., redshift, UVB model) |
| `outputDir` | Where output files will be written |
| `CMB redshift` | Target redshift for the CMB radiation field |
| `Table HM12 redshift` | UV background model and redshift |
| `loop [hden]` | Density grid range and step size |
| `coolingMapTmin/Tmax/Tpoints` | Temperature grid |
| `lineMapLine` | Which emission lines to compute |

---

## Sanity Checks

This paper has a published a fairly comprehensive set of cooling curves that can serve as a sanity check for new runs, even though it’s dated at this point. 

> Bertone et al. (2013), *MNRAS*, 430, 3292  
> https://ui.adsabs.harvard.edu/abs/2013MNRAS.430.3292B/abstract

**Lauren Note:** Also, Serena Bertone was helpful when I was getting started so I like to reference her in all of my emission papers. 

---

## Notes

- In response to Cassi Lochhaas request, Vida made CLOUDY tables including lines ionization rates. The parameter file used for making those tables called `parameter_file_oxygen_rates.par` and can be found in this directory. The relevant tables are saved in  `/cgm_emission/cloudy_extended_z0_selfshield/rates/`.

