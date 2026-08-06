"""Rendering of the pipeline progress table (text, ECSV, HTML)."""

import datetime
import html
import os


# `src` distinguishes rows the pipeline will act on from hand-built directories
# it only observes.  It is a column of its own rather than a note, so a stall
# reason can never mask the fact that a run is frozen.
COLUMNS = ["halo", "box", "stage", "src", "state", "last", "z", "cycle", "jobid", "note"]

# The written table carries more than the terminal view: the registry settings
# that produced each stage, plus where it lives on disk.  Status is written to
# its own file rather than back into the registry -- the registry is
# hand-curated input under version control, this is derived output that changes
# every sweep, and the two must not be entangled.
ECSV_COLUMNS = ["halo", "box", "stage", "level", "phase", "src", "state",
                "last", "final", "z", "cycle", "jobid", "note",
                "enabled", "final_level", "gas", "rvir_min",
                "updated", "stage_dir"]

_STATE_ORDER = ["DONE", "RUNNING", "QUEUED", "BUILDING", "BUILT", "READY", "BLOCKED", "STALLED"]


def _fmt(value, kind=None):
    if value is None:
        return "-"
    if kind == "z":
        return "%.3f" % value
    return str(value)


def _fmt_time(mtime):
    if not mtime:
        return "-"
    return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")


def to_rows(records):
    """Flatten StageState records into plain dicts for rendering."""
    rows = []
    for rec in records:
        st = rec["state"]
        rows.append({
            "halo": rec["halo"],
            "box": rec["box"],
            "stage": rec["stage"],
            "level": _fmt(rec.get("level")),
            "phase": _fmt(rec.get("phase")),
            "src": "frozen" if rec.get("frozen") else "registry",
            "state": st.state,
            "last": _fmt(st.last),
            "final": _fmt(st.final),
            "z": _fmt(st.redshift, "z"),
            "cycle": _fmt(st.cycle),
            "jobid": _fmt(rec.get("jobid")),
            "note": st.note or "",
            "enabled": _fmt(rec.get("enabled")),
            "final_level": _fmt(rec.get("final_level")),
            "gas": _fmt(rec.get("gas")),
            "rvir_min": _fmt(rec.get("rvir_min")),
            "updated": _fmt_time(st.updated),
            "stage_dir": _fmt(rec.get("stage_dir")),
        })
    return rows


def to_halo_rows(rows):
    """One row per halo: where it is in the ladder and what it is waiting on."""
    order = {s: i for i, s in enumerate(_STATE_ORDER)}
    halos = []
    for key in dict.fromkeys((r["halo"], r["box"], r["src"]) for r in rows):
        halo, box, src = key
        mine = [r for r in rows if (r["halo"], r["box"], r["src"]) == key]
        done = [r for r in mine if r["state"] == "DONE"]
        # The current stage is the first that is not DONE; a halo with none is
        # complete.  This mirrors how `advance` walks the ladder.
        pending = [r for r in mine if r["state"] != "DONE"]
        current = pending[0] if pending else None
        halos.append({
            "halo": halo,
            "box": box,
            "src": src,
            "stages": str(len(mine)),
            "done": str(len(done)),
            "furthest_done": done[-1]["stage"] if done else "-",
            "current_stage": current["stage"] if current else "-",
            "state": current["state"] if current else "COMPLETE",
            "last": current["last"] if current else "-",
            "z": current["z"] if current else "-",
            "jobid": current["jobid"] if current else "-",
            "note": current["note"] if current else "all stages done",
            "updated": max((r["updated"] for r in mine if r["updated"] != "-"), default="-"),
        })
    halos.sort(key=lambda h: (h["src"] != "registry", order.get(h["state"], 99), h["halo"]))
    return halos


HALO_COLUMNS = ["halo", "box", "src", "stages", "done", "furthest_done",
                "current_stage", "state", "last", "z", "jobid", "updated", "note"]


def render_text(rows, columns=None):
    columns = columns or COLUMNS
    if not rows:
        return "(no stages found)"
    widths = {c: max(len(c.upper()), max(len(r[c]) for r in rows)) for c in columns}
    out = ["  ".join(c.upper().ljust(widths[c]) for c in columns).rstrip()]
    out.append("  ".join("-" * widths[c] for c in columns))
    for r in rows:
        out.append("  ".join(r[c].ljust(widths[c]) for c in columns).rstrip())
    return "\n".join(out)


def summarize(rows):
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    parts = ["%s %d" % (s, counts[s]) for s in _STATE_ORDER if s in counts]
    for s in sorted(counts):
        if s not in _STATE_ORDER:
            parts.append("%s %d" % (s, counts[s]))
    return "  ".join(parts)


_ECSV_HEADER = [
    "FOGGIE IC pipeline status -- GENERATED, do not edit.",
    "Written by `ic_pipeline status --write`; regenerated on every sweep.",
    "State is derived from OutputLog and RunFinished on each sweep, so this file",
    "is a snapshot and never an authority.  The hand-curated input it reflects is",
    "$FOGGIE_ICS_DIR/halo_registry.ecsv, which is deliberately a separate",
    "file: that one is edited by hand, this one churns.",
]


def write_ecsv(rows, path, columns=None, extra_comments=()):
    from astropy.table import Table

    if not rows:
        return
    columns = columns or ECSV_COLUMNS
    table = Table({c: [r[c] for r in rows] for c in columns}, names=columns)
    table.meta["comments"] = _ECSV_HEADER + list(extra_comments) + [
        "Generated %s" % datetime.datetime.now().isoformat(timespec="seconds"),
        "Summary: %s" % summarize(rows),
    ]
    table.write(path, format="ascii.ecsv", overwrite=True)


_STATE_COLOR = {
    "DONE": "#1f7a52", "RUNNING": "#2f6fb5", "QUEUED": "#6247aa",
    "BUILDING": "#2f6fb5", "BUILT": "#6247aa", "READY": "#5c6370",
    "BLOCKED": "#5c6370", "STALLED": "#a8571b",
}


def write_html(rows, path):
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cells = []
    for r in rows:
        color = _STATE_COLOR.get(r["state"], "#5c6370")
        tds = []
        for c in COLUMNS:
            value = html.escape(r[c])
            if c == "state":
                value = '<b style="color:%s">%s</b>' % (color, value)
            tds.append("<td>%s</td>" % value)
        cells.append("<tr>%s</tr>" % "".join(tds))

    doc = """<!doctype html>
<html><head><meta charset="utf-8"><title>FOGGIE IC pipeline status</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
     margin:40px;color:#16181d;background:#fff}}
h1{{font-size:20px;margin:0 0 4px}}
p.meta{{color:#5c6370;font-size:13px;margin:0 0 20px;font-family:ui-monospace,Menlo,Consolas,monospace}}
table{{border-collapse:collapse;font-size:14px;font-family:ui-monospace,Menlo,Consolas,monospace}}
th,td{{padding:6px 12px;border-bottom:1px solid #d3d7de;text-align:left;white-space:nowrap}}
th{{font-size:11px;letter-spacing:.06em;color:#5c6370;text-transform:uppercase}}
@media (prefers-color-scheme:dark){{
  body{{background:#14161a;color:#e6e8ec}} th,td{{border-bottom-color:#333944}} p.meta{{color:#9aa1ad}}
}}
</style></head><body>
<h1>FOGGIE IC pipeline status</h1>
<p class="meta">generated {generated} &middot; {summary}</p>
<table><thead><tr>{head}</tr></thead><tbody>
{body}
</tbody></table>
</body></html>
""".format(generated=generated,
           summary=html.escape(summarize(rows)),
           head="".join("<th>%s</th>" % c.upper() for c in COLUMNS),
           body="\n".join(cells))

    with open(path, "w") as fp:
        fp.write(doc)
