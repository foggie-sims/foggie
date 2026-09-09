# Reports

Snapshots of ATP ultrafaint campaign state and write-ups, committed so they can
be read off Aitken. These are point-in-time records, not living documents --
regenerate rather than edit.

| file | what it is | how to regenerate |
|---|---|---|
| `ignition-gate.html` | The Ten Solar Mass Gate -- literature assessment and research plan, with the resolution and code comparison tables. Standalone; open it in a browser. | authored; source of record is `../SF_IGNITION_LITERATURE.md` |
| `fleet_status_20260909.txt` | Fleet status table, 2026-09-09 08:20 PDT | `python3 halocat/scripts/fleet_status.py` |

Also published as a private Claude artifact:
https://claude.ai/code/artifact/9f010b57-040e-4676-9a1a-0235aa39f30c

## The two plans this reports on

- `../SF_IGNITION_PLAN.md` -- the systematic plan to ignite star formation in
  the smallest halos (tiers of physics and numerical changes).
- `../SF_IGNITION_LITERATURE.md` -- where the study sits against LYRA, EDGE2,
  FIRE-2, ChaNGa and Kuhlen+ 2012, and what is genuinely novel.
