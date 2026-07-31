"""
Stage state detection.

Everything here is derived from files Enzo and simrun.pl already write, so the
pipeline holds no authoritative state of its own and existing hand-built runs
are readable without migration.  Nothing in this module imports yt or opens an
HDF5 file -- `status` must stay instant on a login node.

Two traps found in the real data, both guarded against below:

  * halo15097/25Mpc_DM_512-L2 has RunFinished ("Finished on cycle 272") but its
    OutputLog stops at DD0008.  RunFinished on its own does NOT mean the run
    reached z = 0, so DONE requires the final redshift dump as well.

  * halo15097/25Mpc_DM_512-L1 has a .message reading "finished!" while the run
    is stuck at RD0259 with no RunFinished.  simrun.pl leaves .message behind
    from earlier attempts, so it is never treated as evidence of completion.
"""

import glob
import os
import re
import subprocess


# Stage states, ordered from least to most progressed.
BLOCKED = "BLOCKED"
READY = "READY"
BUILDING = "BUILDING"
QUEUED = "QUEUED"
RUNNING = "RUNNING"
STALLED = "STALLED"
DONE = "DONE"


# ---------------------------------------------------------------------------
# Enzo parameter file
# ---------------------------------------------------------------------------

_REDSHIFT_RE = re.compile(r"^\s*CosmologyOutputRedshift\[(\d+)\]\s*=\s*([-\d.eE+]+)")


def find_param_file(stage_dir):
    """The .enzo parameter file inside a run directory, or None."""
    candidates = sorted(glob.glob(os.path.join(stage_dir, "*.enzo")))
    return candidates[0] if candidates else None


def output_redshifts(param_file):
    """Map of output index -> redshift, parsed from CosmologyOutputRedshift[i]."""
    redshifts = {}
    if not param_file or not os.path.exists(param_file):
        return redshifts
    with open(param_file, errors="replace") as fp:
        for line in fp:
            m = _REDSHIFT_RE.match(line)
            if m:
                redshifts[int(m.group(1))] = float(m.group(2))
    return redshifts


def final_dump(param_file):
    """Name of the last redshift dump this run is configured to produce.

    Derived from the output list rather than hardcoded to RD0265, so it stays
    correct if the list ever changes.
    """
    redshifts = output_redshifts(param_file)
    if not redshifts:
        return None
    return "RD%04d" % max(redshifts)


# ---------------------------------------------------------------------------
# OutputLog
# ---------------------------------------------------------------------------

class OutputLog:
    """Parsed OutputLog: `DATASET WRITTEN <path> <cycle> <time> <elapsed> <dt>`."""

    def __init__(self, dumps, last_name, last_cycle, mtime):
        self.dumps = dumps            # set of dump names seen
        self.last_name = last_name    # e.g. "RD0265"
        self.last_cycle = last_cycle
        self.mtime = mtime

    def __bool__(self):
        return self.last_name is not None


def read_output_log(stage_dir):
    path = os.path.join(stage_dir, "OutputLog")
    if not os.path.exists(path):
        return OutputLog(set(), None, None, None)

    dumps, last_name, last_cycle = set(), None, None
    with open(path, errors="replace") as fp:
        for line in fp:
            fields = line.split()
            if len(fields) < 4 or fields[0] != "DATASET":
                continue
            name = os.path.basename(fields[2].rstrip("/"))
            dumps.add(name)
            last_name = name
            try:
                last_cycle = int(fields[3])
            except ValueError:
                last_cycle = None
    return OutputLog(dumps, last_name, last_cycle, os.path.getmtime(path))


def read_run_finished(stage_dir):
    """Contents of RunFinished, or None.  Necessary but not sufficient for DONE."""
    path = os.path.join(stage_dir, "RunFinished")
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as fp:
        return fp.read().strip()


def dump_redshift(stage_dir, dump_name, redshifts):
    """Redshift of a dump: from the output list for RD, from the dump itself for DD."""
    if dump_name is None:
        return None
    if dump_name.startswith("RD"):
        try:
            return redshifts.get(int(dump_name[2:]))
        except ValueError:
            return None
    # DD dumps are not in the output list; read the small text parameter file
    # the dump writes alongside itself.  This is what status.sh greps for.
    path = os.path.join(stage_dir, dump_name, dump_name)
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as fp:
        for line in fp:
            if line.startswith("CosmologyCurrentRedshift"):
                try:
                    return float(line.split("=")[1])
                except (IndexError, ValueError):
                    return None
    return None


# ---------------------------------------------------------------------------
# Failure diagnosis
# ---------------------------------------------------------------------------

def stall_reason(stage_dir, log, run_finished, final):
    """Best-effort explanation for why a stage is not progressing.

    The two mirror-image anomalies come first because they are the most
    informative, and both are real: halo46205 L1 reached RD0265 with no
    RunFinished, halo15097 L2 has RunFinished but stopped at DD0008.  Neither
    is ever treated as DONE -- an ambiguous completion signal is exactly the
    case where a human should look rather than the pipeline advancing.
    """
    if final and final in log.dumps and not run_finished:
        return "reached %s but no RunFinished" % final
    if run_finished and final and final not in log.dumps:
        return "RunFinished but never reached %s" % final

    for pbs in sorted(glob.glob(os.path.join(stage_dir, "pbs_output*.txt")),
                      key=os.path.getmtime, reverse=True)[:2]:
        try:
            with open(pbs, errors="replace") as fp:
                for line in fp:
                    if "walltime" in line and "exceeded" in line:
                        return "walltime kill"
                    if "PBS: job killed" in line:
                        return line.strip().split("PBS:")[-1].strip()
        except OSError:
            pass

    # Note: .message saying "finished!" is NOT evidence of completion -- it is
    # left behind by earlier attempts.  Only the trouble signal is useful.
    message = os.path.join(stage_dir, ".message")
    if os.path.exists(message):
        with open(message, errors="replace") as fp:
            if "in trouble" in fp.read():
                return "simrun.pl: in trouble"

    if not log:
        return "no outputs written"
    return "no live job"


# ---------------------------------------------------------------------------
# PBS
# ---------------------------------------------------------------------------

_PBS_STATES = {"R": RUNNING, "Q": QUEUED, "H": QUEUED, "S": QUEUED,
               "T": QUEUED, "W": QUEUED, "B": RUNNING, "E": RUNNING}


def qstat_states(user=None):
    """Map of PBS job id prefix -> our state, for the current user's jobs."""
    user = user or os.getenv("USER")
    try:
        out = subprocess.check_output(["qstat", "-u", user],
                                      stderr=subprocess.DEVNULL, timeout=30).decode()
    except Exception:
        return {}

    states = {}
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 4 or "." not in fields[0]:
            continue
        jobid = fields[0].split(".")[0]
        for token in reversed(fields):
            if token in _PBS_STATES:
                states[jobid] = _PBS_STATES[token]
                break
    return states


# ---------------------------------------------------------------------------
# Stage state
# ---------------------------------------------------------------------------

class StageState:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def stage_state(stage_dir, job_state=None, prereq_done=True):
    """Classify one stage directory.

    job_state is the pipeline's view of any live PBS job for this stage (from
    the ledger crossed with qstat); None means no live job is known.
    """
    if not os.path.isdir(stage_dir):
        return StageState(state=READY if prereq_done else BLOCKED,
                          last=None, cycle=None, redshift=None,
                          final=None, note="" if prereq_done else "prerequisite not done",
                          updated=None)

    param_file = find_param_file(stage_dir)
    redshifts = output_redshifts(param_file)
    final = final_dump(param_file)
    log = read_output_log(stage_dir)
    run_finished = read_run_finished(stage_dir)

    # DONE needs BOTH the final redshift dump and RunFinished.  Either alone is
    # a false positive on real data -- see the module docstring.
    reached_final = bool(final) and final in log.dumps
    if reached_final and run_finished:
        state, note = DONE, ""
    elif job_state in (RUNNING, QUEUED, BUILDING):
        state, note = job_state, ""
    else:
        state, note = STALLED, stall_reason(stage_dir, log, run_finished, final)

    return StageState(
        state=state,
        last=log.last_name,
        cycle=log.last_cycle,
        redshift=dump_redshift(stage_dir, log.last_name, redshifts),
        final=final,
        note=note,
        updated=log.mtime,
    )


def discover_stage_dirs(halo_dir, sim_name):
    """Stage directories present on disk, in level order.

    Matches <sim_name>-L<N> and <sim_name>-L<N>-<suffix>, so physics variants
    such as -gas-therm and -gas-radius3 are visible to `status` even though the
    pipeline does not manage them.
    """
    pattern = re.compile(r"^%s-L(\d+)(.*)$" % re.escape(sim_name))
    found = []
    for entry in sorted(os.listdir(halo_dir)):
        path = os.path.join(halo_dir, entry)
        if not os.path.isdir(path):
            continue
        m = pattern.match(entry)
        if m:
            found.append((int(m.group(1)), m.group(2), path))
    return sorted(found, key=lambda t: (t[0], t[1]))
