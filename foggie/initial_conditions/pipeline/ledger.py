"""
Per-halo submission ledger and the guard against hand-built directories.

The ledger records what the pipeline submitted and when.  It is deliberately
NOT the source of truth for whether a stage finished -- that is derived from
OutputLog and RunFinished (see state.py).  The ledger answers only "is there a
job of mine already in the queue for this stage?", which is what keeps
`advance` from double-submitting when the job-chained hook and the poller fire
at the same moment.

Its presence also marks a halo directory as pipeline-managed, which is how the
frozen hand-built runs are protected.
"""

import errno
import fcntl
import json
import os
import socket
import time
from contextlib import contextmanager


PIPELINE_SUBDIR = ".pipeline"


def pipeline_dir(halo_dir):
    return os.path.join(halo_dir, PIPELINE_SUBDIR)


def ledger_path(halo_dir):
    return os.path.join(pipeline_dir(halo_dir), "ledger.json")


def is_managed(halo_dir):
    """True if the pipeline created this halo directory."""
    return os.path.isdir(pipeline_dir(halo_dir))


def ensure_managed(halo_dir):
    os.makedirs(pipeline_dir(halo_dir), exist_ok=True)


class UnmanagedHaloError(RuntimeError):
    pass


def guard_unmanaged(halo_dir, sim_name, adopt=False):
    """Refuse to write into a halo directory the pipeline did not create.

    A directory holding <sim_name>-L* run directories but no .pipeline/ is a
    hand-built run.  Writing into one would mix generated files into a
    simulation somebody produced by hand, so it takes an explicit --adopt.
    """
    if adopt or not os.path.isdir(halo_dir) or is_managed(halo_dir):
        return
    existing = [e for e in sorted(os.listdir(halo_dir))
                if e.startswith("%s-L" % sim_name) and os.path.isdir(os.path.join(halo_dir, e))]
    if existing:
        raise UnmanagedHaloError(
            "%s holds stage directories but no %s/ -- it looks hand-built.\n"
            "  found: %s\n"
            "Refusing to write into it.  Pass --adopt only if you are certain the\n"
            "pipeline should take this directory over." %
            (halo_dir, PIPELINE_SUBDIR, ", ".join(existing[:6]) + ("..." if len(existing) > 6 else "")))


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

def _lock_identity(state):
    return ("%s host=%s pid=%d job=%s at=%s\n"
            % (state, socket.gethostname(), os.getpid(),
               os.environ.get("PBS_JOBID", "-"),
               time.strftime("%Y-%m-%d %H:%M:%S")))


@contextmanager
def halo_lock(halo_dir, timeout=30):
    """Exclusive lock for one halo, so the hook and the poller cannot race.

    The lock file RECORDS ITS HOLDER, because a bare timeout message is
    actively misleading. `fcntl.flock` is released by the kernel when the
    holding process dies, so a leftover lock FILE is never a held lock and an
    orphaned lock cannot block anyone -- if acquisition times out, some live
    process really does hold it. But pipeline builds run on compute nodes under
    PBS, so that holder is invisible to `pgrep` on the login node, and the
    obvious conclusion ("stale lock, delete it") is both wrong and unsafe:
    removing the file lets the next process create a new inode and take a
    second lock, which is exactly the race this exists to prevent. Recording
    host, pid and PBS job id turns the question into one a person can answer.

    Opened with O_CREAT|O_RDWR and never truncated on entry. The previous
    `open(path, "w")` truncated BEFORE flock was acquired, so every process
    that merely waited erased the live holder's identity -- which is why the
    file was always empty when anyone looked at it.
    """
    ensure_managed(halo_dir)
    path = os.path.join(pipeline_dir(halo_dir), "lock")
    deadline = time.time() + timeout
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o664)
    fp = os.fdopen(fd, "r+")
    try:
        while True:
            try:
                fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.time() > deadline:
                    try:
                        fp.seek(0)
                        who = fp.read().strip() or "(no identity recorded)"
                    except Exception:
                        who = "(unreadable)"
                    raise RuntimeError(
                        "Timed out after %ds waiting for the lock on %s.\n"
                        "  Held by: %s\n"
                        "  A LIVE process holds this. flock is dropped by the "
                        "kernel when a process dies, so an abandoned lock file "
                        "cannot block -- do NOT delete it. The holder is most "
                        "likely a build running on a compute node, which "
                        "pgrep on the login node cannot see; check the job id "
                        "above with qstat, or wait for it to finish."
                        % (timeout, path, who))
                time.sleep(1)
        fp.seek(0)
        fp.truncate()
        fp.write(_lock_identity("HELD by"))
        fp.flush()
        yield
    finally:
        try:
            try:
                fp.seek(0)
                fp.truncate()
                fp.write(_lock_identity("released by"))
                fp.flush()
            except Exception:
                pass
            fcntl.flock(fp, fcntl.LOCK_UN)
        finally:
            fp.close()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def read_ledger(halo_dir):
    path = ledger_path(halo_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fp:
            return json.load(fp)
    except (ValueError, OSError):
        return []


def append_record(halo_dir, record):
    ensure_managed(halo_dir)
    records = read_ledger(halo_dir)
    record = dict(record)
    record.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    records.append(record)
    path = ledger_path(halo_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(records, fp, indent=2)
        fp.write("\n")
    os.replace(tmp, path)
    return record


def stage_key(level, phase):
    return "L%d-%s" % (level, phase)


def last_qc(halo_dir, kind):
    """The most recent qc submission of this kind ("density"/"contamination")."""
    for record in reversed(read_ledger(halo_dir)):
        if record.get("stage") == "qc" and record.get("action") == "qc-%s" % kind:
            return record
    return None


def qc_in_flight(halo_dir, kind, qstat_states):
    """True if a qc job of this kind is still queued or running.

    Without this the trigger would resubmit on every poll sweep between a level
    finishing and its qc job starting, which on a busy queue is hours.
    """
    if qstat_states is None:      # queue state unknown -- see state.qstat_states
        qstat_states = {}
    record = last_qc(halo_dir, kind)
    if not record or not record.get("jobid"):
        return False
    return str(record["jobid"]).split(".")[0] in qstat_states


def live_job(halo_dir, level, phase, qstat_states):
    """The (jobid, state, action) of a live job for this stage.

    Cross-references the ledger against the queue, so a job id that has since
    finished or been deleted does not read as live.  `action` is "build" for
    IC generation and "enzo" for the run itself, which lets the caller show a
    queued IC build as BUILDING rather than QUEUED.
    """
    if qstat_states is None:      # queue state unknown -- see state.qstat_states
        qstat_states = {}
    key = stage_key(level, phase)
    for record in reversed(read_ledger(halo_dir)):
        if record.get("stage") != key:
            continue
        jobid = record.get("jobid")
        if not jobid:
            continue
        short = str(jobid).split(".")[0]
        if short in qstat_states:
            return jobid, qstat_states[short], record.get("action")
    return None, None, None
