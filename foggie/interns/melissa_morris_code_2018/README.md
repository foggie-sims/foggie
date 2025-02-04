AUTHOR: Melissa Morris
DATE: 08/18/2017
Directory: simulation_analysis

This directory contains a set of tools used to analyze the circumgalactic medium in simulation outputs, accompanied by notebooks explaining the various ways in which the CGM is analyzed.

File Name: autosiman.py
Description: allows automatic analysis of simulations, such as velocity flux calculation, mass flow calculation, and movie making
Notes: Can be run completely alone without siman.py, may need to change the base directory in which the files are located

File Name: center_finding.ipynb
Description: compares different methods of deriving the center of the galaxy
Notes: Does not contain any ground breaking information, but may be useful to look back on

File Name: flowstrength.ipynb
Description: demonstrates how outflows and inflows are examined in these simulations and how this works
Notes: Does not need siman.py

File Name: siman.ipynb
Description: contains functions for performing various calculations
Notes: notebook version of siman.py

File Name: siman.py
Description: contains functions for performing various calculations gone over in flowstrength and velocity_flux; includes how these functions perform their calculations and how to use them
Notes: Other notebooks import this

File Name: velocity_flux.ipynb
Description: goes over various forms of calculating the volume- and mass-weighted velocity flux profiles for outflows and inflows above and below the galaxy.
Notes: Needs siman.py
