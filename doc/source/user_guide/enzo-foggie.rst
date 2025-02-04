Compiling enzo-foggie on Pleiades
=================================

To make a clean Enzo build using the enzo-foggie fork:

1.  From the command line:
    ::

        > git clone https://github.com/foggie-sims/enzo-foggie.git 
        > cd enzo-foggie
        > ./configure 
        > cd src/enzo 
        > make machine-pleiades-mpich
        > make grackle-yes 
        > make opt-high 
        > module load comp-intel/2020.4.304 hdf5/1.8.18_serial
        > make clean 
        > make -j16

    [wait while Enzo builds] 

    Note that ``Make.mach.pleiades-mpich`` links to JT's mpich libraries:

    ``u/jtumlins/installs/mpich-4.0.3/usr/lib``

    and grackle: 

    ``/u/jtumlins/grackle-mpich/build-mpich/``

    ``/u/jtumlins/grackle-mpich/src/clib/*.o``

    This means you don't need to install grackle yourself!

    If you get an error at the linking step of the compile when you're reasonably sure it should be working,
    try submitting the final make commands as a PBS script to the processors you are planning on running Enzo
    on (likely Broadwell). Don't forget to include the module loads in the PBS script, and you can submit to
    the devel queue to get it to run right away. Should only take a few minutes. There is an example PBS script
    for this in the ``enzo-foggie repo``, in ``enzo-foggie/src/qsub_compile_enzo.sh``.

2.  Add the following lines to your ``.bash_profile``:
    ::

        export LD_LIBRARY_PATH="/u/jtumlins/installs/mpich-4.0.3/usr/local/lib":"/u/jtumlins/installs/mpich-4.0.3/usr/lib":$LD_LIBRARY_PATH
        export PATH="/nobackup/jtumlins/anaconda3/bin:/u/scicon/tools/bin/:/u/jtumlins/installs/mpich-4.0.3/usr/local/bin:$PATH"


3.  Put a new file in your home directory called memory_gauge.sh, containing this:
    ::

        #!/bin/bash
        count=1
        echo "---------------------------Nodes------------------------"
        /u/scicon/tools/bin/qsh.pl $1 "cat /proc/meminfo"   | grep r  | grep i | grep n | grep -v SU | awk '{print $1}' | fmt -1000
        echo "-----------------------Total Memory---------------------"
        /u/scicon/tools/bin/qsh.pl $1 "cat /proc/meminfo" | grep MemTotal: | awk '{print $2/1024/1024}' | fmt -1000
        echo "----------------------Available Memory---------------------"
        while [ $count -le 999999 ]
        do
            sleep 2
                (echo "time = "; date +'%s'; /u/scicon/tools/bin/qsh.pl $1 "cat /proc/meminfo" | grep Available: | awk '{print $2/1024/1024}') | tr '\n' '\t' | fmt -1000
            ((count++))

Now you can follow the instructions in either `Restarting an Existing FOGGIE Run <restart.html>`_ or `Starting a Fresh FOGGIE Run from Initial Conditions <clean_from_ICs.html>`_!