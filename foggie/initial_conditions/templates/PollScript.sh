#!/bin/bash
#
# Pipeline poller, rendered from templates_512/PollScript.sh.
#
# The job-chained hook in each RunScript is the primary trigger and fires within
# seconds of a run finishing.  This is the safety net for the case where that
# hook never runs at all: a hard node failure, a job killed outside Enzo, a
# chain broken by something nobody anticipated.
#
# The sweep is cheap -- it reads OutputLog and RunFinished and calls qstat -- so
# this job runs for well under a minute and then schedules its own successor
# with `qsub -a` rather than sleeping on an allocated node.
#
#PBS -N ic-pipeline-poll
#PBS -W group_list=__GROUP__
#PBS -l select=__POLL_SELECT__
#PBS -l walltime=__POLL_WALLTIME__
__POLL_QUEUE_LINE__
#PBS -j oe
#PBS -V
#PBS -o __LOG_DIR__/poll.log

export PATH="/nobackup/jtumlins/anaconda3/bin:$PATH"

cd __LOG_DIR__

# Reschedule FIRST, deliberately.  This is a watchdog: if the sweep below dies
# on a bad state or a transient filesystem error, the next poller must still
# run.  Rescheduling afterwards would let a single failed sweep silently end
# the whole chain of pollers -- the exact failure this job exists to catch.
__RESCHEDULE__

echo "=== poll sweep $(date) ==="
python3 __SCRIPT__ advance
python3 __SCRIPT__ status --include-manual --write __NOTIFY_ARGS__
echo "=== sweep done $(date) ==="
