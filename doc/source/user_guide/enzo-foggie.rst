Compiling enzo-foggie on Pleiades
=================================

To make a clean Enzo build using the enzo-foggie fork:

1. First load the compiler modules needed:
   ::

        > module load comp-intel/2020.4.304 hdf5/1.8.18_serial


2. Install grackle. In the below command line prompts, enter in your own Pleiades username. From the command line:
   ::

        > cd /nobackup/<USERNAME>
        > git clone https://github.com/grackle-project/grackle
        > cd grackle
        > git submodule update --init
        > cmake -DCMAKE_INSTALL_PREFIX=/nobackup/<USERNAME>/grackle/build -DBUILD_SHARED_LIBS=ON 
        -B /nobackup/<USERNAME>/grackle/build
        > cmake --build /nobackup/<USERNAME>/grackle/build
        > cmake --install /nobackup/<USERNAME>/grackle/build

   You can run one of the test problems to see if it built properly:

   ::

        > cd build/examples/
        > ./cxx_example

   If it outputs a bunch of physics stuff to the terminal (as opposed to any errors), it worked!

3.  Now configure enzo. From the command line:
    ::

        > git clone https://github.com/foggie-sims/enzo-foggie.git 
        > cd enzo-foggie
        > ./configure 
        > cd src/enzo

    In the ``src/enzo`` directory, there will be a file called ``Make.mach.pleiades-mpich``. There are a couple of lines in
    this file that need to be updated:

    Change the line that says ``LOCAL_GRACKLE_INSTALL`` to:

    ``LOCAL_GRACKLE_INSTALL = /nobackup/<USERNAME>/grackle/build``

    Change the line that says ``LOCAL_LIBS_GRACKLE`` to:

    ``LOCAL_LIBS_GRACKLE = -L$(LOCAL_GRACKLE_INSTALL)/lib64 -lgrackle``

    Note that this ``Make.mach.pleiades-mpich`` file links to JT's mpich libraries. There should be no need to change this.

4. Now we can compile enzo. This needs to be done as a job submitted to a node rather than straight from the command line 
   on the pfe's. There is an example PBS script for this in the ``enzo-foggie repo``, in ``enzo-foggie/src/qsub_compile_enzo.sh``.
   In ``qsub_compile_enzo.sh``, you will need to change the directories:

   In the line ``#PBS -o /home5/clochhaa/FOGGIE/output_compile_enzo``, change this to your own home or nobackup directory. 
   This is where the output from the command line will go. If the compile fails, examine this file to see what happened.

   In the line ``cd /home5/clochhaa/enzo-foggie/src/enzo``, change this to the directory where you cloned enzo-foggie. 
   Make sure the end of this path is ``src/enzo``.

   After making these changes, submit this script at the command line:
   ::

        > qsub qsub_compile_enzo.sh

   And wait for enzo to compile! It should take just a few minutes. Check the ``output_compile_enzo`` file to make sure it ends with "Success!"

5.  Add the following lines to your ``.profile`` in your home directory. This file will be executed on start of a PBS job:
    ::

        export LD_LIBRARY_PATH="/u/jtumlins/installs/mpich-4.0.3/usr/local/lib":
        "/u/jtumlins/installs/mpich-4.0.3/usr/lib":"/nobackup/<USERNAME>/grackle/build/lib64":$LD_LIBRARY_PATH
        export PATH="/nobackup/jtumlins/anaconda3/bin:/u/scicon/tools/bin/:/u/jtumlins/installs/mpich-4.0.3/usr/local/bin:$PATH"


6.  Put a new file in your home directory called memory_gauge.sh, containing this:
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
        done

Now you can follow the instructions in either `Restarting an Existing FOGGIE Run <restart.html>`_ or `Starting a Fresh FOGGIE Run from Initial Conditions <clean_from_ICs.html>`_!