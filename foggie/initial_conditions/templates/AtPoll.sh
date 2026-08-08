#!/bin/bash
#
# Pipeline poller driven by a self-rescheduling `at` chain, rendered from
# templates_512/AtPoll.sh.
#
# Why `at` and not cron: user crontabs are accepted on the NAS front ends but
# never executed -- crond is not running there (verified with a bare `date` on
# a */1 schedule producing nothing in four minutes).  atd IS running, so a
# chain of one-shot `at` jobs gives the same effect.
#
# Why `at` and not PBS: a sweep is ~1.4 s and ~90 MB.  Pleiades allocates whole
# nodes, so a PBS poller wakes an entire Aitken node every interval to read a
# few log files.  This costs no allocation at all.
#
# The chain is only as durable as the front end it runs on; if afe reboots, the
# chain stops.  That is acceptable for a watchdog whose job is to catch a
# missed RunScript hook, and `poll --install` remains available as the PBS
# fallback.

export FOGGIE_REPO="__FOGGIE_REPO__"
export FOGGIE_ICS_DIR="__FOGGIE_ICS_DIR__"
LOG="__LOG_DIR__/poll.log"

# Reschedule FIRST, deliberately.  This is a watchdog: if the sweep below dies
# on a bad state or a transient filesystem error, the next poller must still
# run.  Rescheduling afterwards would let one failed sweep silently end the
# chain -- exactly the failure this is here to catch.
__RESCHEDULE__

{
  echo "=== poll sweep $(date) ==="
  __PYTHON__ __SCRIPT__ advance
  __PYTHON__ __SCRIPT__ status --include-manual --write __NOTIFY_ARGS__
  echo "=== sweep done $(date) ==="
} >> "$LOG" 2>&1
