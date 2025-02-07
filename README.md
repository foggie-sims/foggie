# foggie

This repository contains code for analyzing the FOGGIE simulations.
It may be installed as a package using e.g. `pip install --editable .` for 
easier access to imports such as `foggie.utils.consistency`
or you may directly use the scripts within.

### Using with Conda

If you use a Conda distribution such as Anaconda or Miniforge to manage your Python environments, you are encouraged to use the accompanying `foggie-env.yml` file to create the `foggie` environment:

```
conda env create -f foggie-env.yml
```

## Documentation

The `doc` directory contains documentation on both the FOGGIE simulations
and how to use scripts within this repository. The docs can be built as HTML using:
```
cd doc
make html
```


## Analysis Modules

The `foggie` directory contains several subdirectories
with scripts from various FOGGIE analysis projects. A description of their contents is included below.

| Folder/Module        | Description |
|----------------------|-------------|
| `angular_momentum`   | Characterizing galaxy angular momentum. Scripts used for [Simons et al. 2020](https://iopscience.iop.org/article/10.3847/1538-4357/abc5b8). |
| `cgm_absorption`     | CGM absorption spectra generation and analysis. Scripts used for [Peeples et al. 2019](https://iopscience.iop.org/article/10.3847/1538-4357/ab0654). |
| `cgm_emission`       | Scripts for post-processing CGM emission using [CLOUDY](https://gitlab.nublado.org/cloudy/cloudy). |
| `clumps`             | Finding & analyzing clumps. Scripts used for Augustin et al. in prep.|
| `deprecated`         | Deprecated scripts. |
| `edges`              | Basic code for edge analysis of FOGGIE regions. |
| `examples`           |  |
| `flux_tracking`      | Scripts and Notebooks for estimating fluxes such as accretion. |
| `fogghorn`           | Diagnostic plot pipeline for planning new simulations. |
| `galaxy_mocks`       | Scripts for making mock observations. |
| `gas_metallicity`    | Scripts for gas phase metallicity analysis (and some stellar metallicity). |
| `halo_infos`         | Catalog files; see accompanying README. |
| `halo_tracks`        | Halo tracks used for running the FOGGIE simulations. |
| `initial_conditions` | Old initial conditions for a 25 Mpc simulation box. |
| `interns`            | Previous student intern projects. |
| `movies`             | Scripts for making movies. |
| `notebooks`          | A folder of Jupyter Notebooks to contain `git` change tracking chaos. |
| `paper_plots`        | Scripts for making plots in the various FOGGIE publications. |
| `plots`              | Generate basic sanity-check plots. |
| `pressure_support`   | Thermal and non-thermal pressure estimations. Scripts used for [Lochhaas et al. 2023](https://iopscience.iop.org/article/10.3847/1538-4357/acbb06). |
| `radial_quantities`  | Radial profiles of common quantities. |
| `render`             | Scripts using the `datashader` package. |
| `satellites`         |  |
| `scripts`            | A random assortment of things. |
| `turbulence`         | Calculate velocity structure functions and velocity dispersions. |
| `utils`              | Utility scripts. |

## For FOGGIE Developers

As you add new directories and modules to `foggie`, it is requested that you add/update `README` files in the respective directory. Please also keep your module description in this `README` up to date.

The `foggie` repository contains a wide variety of artifacts, from scripts to data tables to command line tools. Developers are encouraged to follow the relevant guidelines for integrating their additions into the `foggie` package.

### Making Importable (Sub)modules

If you install `foggie` as a package with `pip install .` (or, more preferably, `pip install --editable .`) you can import any individual file anywhere on your system as `import.<submodule>.<filename>`. For example, `import foggie.utils.consistency` accesses `foggie/utils/consistency.py`.

But what if you want to load multiple files from a single module folder at once, such as `foggie/utils/consistency.py` and `foggie/utils/foggie_utils.py`? You can define (sub)modules with `__init__.py` files. These files contain `import` statements that will be run when the (sub)module is loaded.

For example, the file `foggie/utils/__init__.py` contains the following:
```python
from foggie.utils.consistency import *
from foggie.utils.foggie_utils import *
```
This means that when you run `import foggie.utils`, all of the variables and functions contained in `consitency.py` and `foggie_utils.py` will be accessible to you via the `foggie.utils` submodule; e.g., `foggie.utils.axes_label_dict`.

If you want functions or submodules to be loaded every time you load the main `foggie` module with `import foggie`, add their respective import statements to `foggie/__init__.py`. For example, this file contains the following:

```python
import foggie.utils
from foggie.utils.foggie_load import foggie_load
```

This makes it so that the `foggie.utils` submodule will always be available when `import foggie` is run. Additionally, the `foggie_load()` function in `foggie/utils/foggie_load.py` will be made available as `foggie.foggie_load()`. Not all submodules have to be loaded with the main `foggie` module but it does make them easier to access.