Frequently Asked Questions
==========================

Building Enzo
-------------


**Q: I’m getting a compilation error related to HDF5. What is HDF5 and how to I get it?**

A: HDF5 is a data format with accompanying library for writing very large
data sets. Enzo uses HFD5 for data output. If you do not have a version of HDF5
available on your machine, you can download binaries or source code for HDF5
from https://www.hdfgroup.org/downloads/hdf5/. Once you have a version of HDF5
installed on your machine, you need to notify Enzo where it is located for the
build process in the Makefile (eg. ``Make.mach.linux-gnu`` or
``Make.mach.my-machine``). For example, if HDF5 was installed in
``/home/enzo-user/local/hdf5/``, you would edit the line
::

  LOCAL_HDF5_INSTALL = /home/enzo-user/local/hdf5

then run
:: 

  $ make machine-linux-gnu
  $ make clean
  $ make

to rebuild enzo.exe with your HDF5 installation. When running enzo.exe, make
sure that the HDF5 library is in ``LD_LIBRARY_PATH``. In this example, if you
are running bash, run the command
::

  $ export LD_LIBRARY_PATH=/home/enzo-user/local/hdf5/lib/:$LD_LIBRARY_PATH 

to put the HDF5 library in the library path before running Enzo.


Running Simulations
-------------------

Common Crashes
--------------


Misc.
-----


**Q: What is the difference between enzo-dev (week-of-code) and the stable
branch? Should I only use the stable branch?**

A:

The "week-of-code" branch of enzo-dev is the primary development branch, which
is updated on a fairly regular basis (the name "week-of-code" is historical).
Changes are migrated into the stable branch on a roughly annual basis. In
general, if you want code that is somewhat more reliable but may be
significantly behind the cutting-edge Enzo version, you should use the 'stable'
branch. If you are comfortable with more recent (and thus possibly less
reliable) code, you should use the "week-of-code" branch.


