"""
Notification of stage state changes.

Reports transitions rather than status.  A poller running every 30 minutes that
mails the current table produces ~48 messages a day that almost all say nothing
changed, and a notification nobody reads is worse than none.

The previous snapshot is the status.ecsv from the last sweep, read just before
it is overwritten, so there is no extra state file to keep in sync.

Note: simrun.pl has a send_email() that only ever wrote a .message file and
never sent anything, which is why those "supercomputer says ... finished!"
notes accumulate unread in run directories.  This actually sends.
"""

import os
import subprocess


# Transitions worth waking someone for.  A stage going RUNNING or QUEUED is
# routine progress and is reported in the body, but does not by itself justify
# a message.
_NOTABLE = {"DONE", "STALLED"}


def _key(row):
    return (row["halo"], row["box"], row["stage"])


def read_snapshot(path):
    """Previous status as {(halo, box, stage): state}, or {} if there is none."""
    if not os.path.exists(path):
        return {}
    try:
        from astropy.table import Table
        table = Table.read(path)
    except Exception:
        return {}
    return {(str(r["halo"]), str(r["box"]), str(r["stage"])): str(r["state"])
            for r in table}


def diff_states(previous, rows):
    """Transitions since the previous snapshot.

    Returns (changes, notable) where changes is every (halo, stage, was, now)
    and notable is the subset that justifies sending a message.  A stage absent
    from the previous snapshot is reported as new rather than as a change, so
    the first ever sweep does not mail the entire fleet.
    """
    changes, notable = [], []
    first_run = not previous
    for row in rows:
        key = _key(row)
        was = previous.get(key)
        now = row["state"]
        if was is None:
            continue          # newly discovered stage, or the first sweep
        if was == now:
            continue
        change = (row["halo"], row["stage"], was, now, row)
        changes.append(change)
        if now in _NOTABLE:
            notable.append(change)
    return changes, notable, first_run


def format_message(changes, notable, halo_rows):
    """Subject and body for a set of transitions."""
    if notable:
        bits = ["%s %s %s" % (c[0], c[1], c[3]) for c in notable[:3]]
        more = "" if len(notable) <= 3 else " (+%d more)" % (len(notable) - 3)
        subject = "FOGGIE ICs: " + ", ".join(bits) + more
    else:
        subject = "FOGGIE ICs: %d stage change(s)" % len(changes)

    lines = ["Stage changes since the last sweep:", ""]
    for halo, stage, was, now, row in changes:
        detail = ""
        if row.get("last") and row["last"] != "-":
            detail = "  at %s" % row["last"]
            if row.get("z") and row["z"] != "-":
                detail += " (z=%s)" % row["z"]
        note = "  -- %s" % row["note"] if row.get("note") else ""
        lines.append("  halo %-8s %-10s %s -> %s%s%s" % (halo, stage, was, now, detail, note))

    lines += ["", "Current state by halo:", ""]
    for h in halo_rows:
        lines.append("  halo %-8s %-10s %-9s %s"
                     % (h["halo"], h["current_stage"], h["state"],
                        h["note"] or ""))
    lines += ["", "-- ic_pipeline"]
    return subject, "\n".join(lines)


def send_email(to, subject, body, dry_run=False):
    """Send via mailx, which is present on the NAS front ends."""
    if dry_run:
        print("[dry-run] would email %s" % to)
        print("  Subject: %s" % subject)
        print("".join("  | %s\n" % l for l in body.splitlines()))
        return True
    try:
        proc = subprocess.run(["mailx", "-s", subject, to],
                              input=body.encode(), timeout=60,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as exc:
        print("notify: could not send mail: %s" % exc)
        return False
    if proc.returncode != 0:
        print("notify: mailx exit %d: %s" % (proc.returncode, proc.stdout.decode().strip()))
        return False
    print("notify: emailed %s -- %s" % (to, subject))
    return True


def notify_changes(previous_path, rows, halo_rows, to, dry_run=False, quiet_ok=True):
    """Compare against the previous snapshot and mail anything that changed."""
    previous = read_snapshot(previous_path)
    changes, notable, first_run = diff_states(previous, rows)

    if first_run:
        print("notify: no previous snapshot, establishing a baseline (not sending)")
        return False
    if not changes:
        if not quiet_ok:
            print("notify: nothing changed")
        return False

    subject, body = format_message(changes, notable, halo_rows)
    return send_email(to, subject, body, dry_run=dry_run)
