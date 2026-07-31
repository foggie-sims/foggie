#!/bin/bash
#
# Pipeline poller, rendered from templates_512/PollScript.sh.
#
# The job-chained hook in each RunScript is the primary trigger and fires within
# seconds of a run finishing.  This is the safety net for the case that hook
# never runs at all -- a hard node failure, a job killed outside Enzo, a chain
# broken by something nobody anticipated.
#
# The sweep is cheap (reads OutputLog and RunFinished, calls qstat) and takes a
# couple of seconds, so this job runs briefly and then schedules its own
# successor with `qsub -a` rather than sleeping on an allocated node.
#
#PBS -N ic-pipeline-poll
#PBS -W group_list=__GROUP__
#PBS -l select=1:ncpus=1
#PBS -l walltime=00:10:00
#PBS -q devel
#PBS -j oe
#PBS -V
#PBS -o __LOG_DIR__/poll.log

export PATH="/nobackup/jtumlins/anaconda3/bin:$PATH"

cd __LOG_DIR__

echo "=== poll sweep $(date) ==="
python3 __SCRIPT__ advance
python3 __SCRIPT__ status --include-manual --write

# Schedule the next sweep.  `qsub -a` defers the start time, so nothing holds a
# node while waiting.
__RESCHEDULE__
