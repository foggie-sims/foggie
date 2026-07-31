Generating New Zoom Initial Conditions
======================================

**Authors: JT and Claude**

This describes how to generate the initial conditions for a cosmological
zoom-in simulation and run it through its refinement levels, using the
automated pipeline in ``foggie/initial_conditions/pipeline``.

The pipeline replaces the older hand-driven ``script512.py`` / ``script256.py``
workflow, in which each refinement level had to be submitted by hand once the
previous one finished. You now curate a single table of halos and the pipeline
does the rest: it generates the ICs for each level with enzo-mrp-music and
MUSIC, submits the Enzo run, waits for it to reach z = 0, and then starts the
next level on its own.

Once the ICs exist, `Starting a Fresh FOGGIE Run from Initial Conditions
<clean_from_ICs.html>`_ covers what to do with them.

.. contents::
   :local:
   :depth: 2


Overview
--------

A zoom is built as a ladder of **stages**. Each stage is one refinement level
of one halo, and each is two jobs:

* a short **build** job, which traces the halo's Lagrangian region back to
  z = 99 and generates the ICs, and
* a long **run** job, which evolves those ICs to z = 0 with Enzo.

Level N cannot be built until level N-1 has finished, because enzo-mrp-music
loads the previous level's Enzo outputs to work out which particles end up in
the halo. That dependency is why the old workflow needed a human at every
level, and it is the thing the pipeline automates.

The default ladder is ``L1-DM -> L2-DM -> L3-DM``, with a gas run at the final
level as an optional extra stage.


Prerequisites
-------------

Two environment variables must be set. Both are read on every invocation::

    export FOGGIE_REPO=/path/to/foggie/foggie          # the package dir
    export FOGGIE_ICS_DIR=/path/to/25Mpc_new_cosmology # where runs live

You also need MUSIC built in place at ``$FOGGIE_REPO/initial_conditions/music/MUSIC``.
The source is in the repo; the compiled binary is deliberately gitignored, so
build it once on the machine you are running on.

The Enzo executable is set per box in ``pipeline/config.py`` (``enzo_exe``).


Quick start
-----------

To add a halo to the fleet, add one row to the registry and let the pipeline
pick it up::

    $FOGGIE_REPO/initial_conditions/halo_registry.ecsv

Then check it resolves::

    python3 $FOGGIE_REPO/initial_conditions/pipeline/ic_pipeline.py validate-registry

That is the whole workflow. The poller submits the first build within its
sweep interval. To start immediately instead of waiting::

    python3 .../ic_pipeline.py advance --halo 51541

and to watch progress::

    python3 .../ic_pipeline.py status --by-halo

.. note::

   Invoke the script **by path**, as above, rather than as
   ``python -m foggie.initial_conditions.pipeline.ic_pipeline``. Importing the
   ``foggie`` package pulls in yt through ``foggie/__init__.py``, and nothing
   in the pipeline needs it. Running by path keeps ``status`` down to about a
   second.


The halo registry
-----------------

``foggie/initial_conditions/halo_registry.ecsv`` is the single hand-curated
input, and the only file you normally edit. **You edit it by hand, in a text
editor, and commit it like any other source file.** Nothing writes to it: the
pipeline only ever reads it, so your edits are never overwritten and two people
changing it resolve as an ordinary git conflict.

It is ECSV -- plain text, with a commented header declaring the column types,
followed by one whitespace-separated row per halo::

    halo_id box          enabled final_level gas   rvir_min allow_mixed_outputs queue  nodes model   notes
    51541   25Mpc_DM_512 True    3           False 0.0      False               normal 1     mil_ait "started fresh"
    79628   25Mpc_DM_512 True    3           False 0.0      False               normal 1     mil_ait "started fresh"

To add a halo, append a row in the order given by the header line. A few
things to get right, because a malformed row fails when the file is read
rather than when you save it:

* Fields are separated by whitespace, so **quote anything containing spaces**.
  In practice that means ``notes``, which should always be in double quotes.
* ``enabled``, ``gas`` and ``allow_mixed_outputs`` are booleans: write ``True``
  or ``False``, capitalised.
* ``rvir_min`` is a float, so write ``0.0`` rather than ``0``.
* Do not edit the commented header block above the column line. It declares
  the column types, and changing it will produce confusing parse errors.

Then check it before waiting on a sweep::

    python3 .../ic_pipeline.py validate-registry

which confirms the file parses, every ``halo_id`` resolves in that box's
Rockstar catalog, ``final_level`` is within range, and reports the effective
zoom radius for each halo.

Columns
~~~~~~~

One row per halo:

``halo_id``
    Rockstar ID, which must resolve in the box's catalog.

``box``
    Parent box, e.g. ``25Mpc_DM_512``. Defined in ``pipeline/config.py``.

``enabled``
    Only ``True`` rows are ever acted on. Set ``False`` to park a halo without
    deleting its row.

``final_level``
    Top DM level, normally 3.

``gas``
    Whether to run a gas stage at the final level.

``rvir_min``
    Floor on the zoom radius in kpc, overriding the box default. ``0`` uses the
    box default (80 kpc for ``25Mpc_DM_512``). The radius is
    ``max(catalog Rvir, floor)``: several of these dwarfs have Rvir well under
    80 kpc, and too small a Lagrangian region makes a poor zoom.

``allow_mixed_outputs``
    Accept a halo whose levels disagree about their redshift output list. Leave
    ``False`` unless you know why you need it -- see `Guards`_.

``queue``, ``nodes``, ``model``
    PBS hints.

``notes``
    Free text. Please say why a halo is unusual; it is the only place that
    context survives.

Status is **not** written back here. The registry is versioned input; status is
regenerated constantly and lives separately, under ``$FOGGIE_ICS_DIR``.


Monitoring
----------

``status`` derives everything from files Enzo and ``simrun.pl`` already write,
so it is read-only, opens no HDF5, and needs no yt::

    python3 .../ic_pipeline.py status              # one row per stage
    python3 .../ic_pipeline.py status --by-halo    # one row per halo
    python3 .../ic_pipeline.py status --include-manual   # also hand-built dirs
    python3 .../ic_pipeline.py status --write      # write the tables to disk

``--write`` produces three files in ``$FOGGIE_ICS_DIR``:

* ``status.ecsv`` -- one row per stage, with the registry settings that
  produced it
* ``status_by_halo.ecsv`` -- one row per halo
* ``status.html`` -- the same table, browsable

Stage states
~~~~~~~~~~~~

===============  ==============================================================
State            Meaning
===============  ==============================================================
``BLOCKED``      Waiting on the level below.
``READY``        Prerequisite done, no ICs yet. ``advance`` will build it.
``BUILDING``     IC generation job in the queue or running.
``BUILT``        ICs exist, Enzo never started. ``advance`` will submit it.
``QUEUED``       Enzo job queued.
``RUNNING``      Enzo running.
``DONE``         Reached its final redshift dump **and** wrote ``RunFinished``.
``STALLED``      Produced output, then stopped. Needs a human; never retried.
===============  ==============================================================

``DONE`` deliberately requires both conditions. Either alone is a false
positive on real data: one hand-built run has ``RunFinished`` while its
``OutputLog`` stops far short of z = 0, and another reached z = 0 with no
``RunFinished`` at all. Both are reported with an explanatory note and neither
is treated as complete, because an ambiguous completion signal is exactly where
the pipeline should stop rather than advance a level on a guess.

Note also that ``.message`` files saying "finished!" are **not** evidence of
anything. ``simrun.pl`` leaves them behind across attempts, so one may sit in a
directory whose run died hours earlier.


How stages advance
------------------

``advance`` walks a halo's ladder, acts on the first stage that is not
``DONE``, and stops. Only ``READY`` and ``BUILT`` are actionable; everything
else means the correct action is to do nothing.

It is called from two places, and both must be safe to run at any moment:

**The job-chained hook.** Every generated ``RunScript.sh`` ends with a call to
``advance``. It fires within seconds of an Enzo job exiting. There is
deliberately no "did it finish?" test in the shell script -- ``simrun.pl``
exits three different ways (finished, resubmitted itself for walltime, or
died), and only one should advance. ``advance`` re-derives the state and is a
no-op in the other two cases.

**The poller.** A periodic sweep that catches chains broken by a hard node
failure, where the hook never ran at all.

Concurrency is safe because state is re-derived from disk every time, each halo
is locked with ``flock``, and the submission ledger is cross-checked against
``qstat`` before anything is submitted.

Running the poller
~~~~~~~~~~~~~~~~~~

A sweep costs about a second, so it belongs on a front end rather than in a
PBS job::

    python3 .../ic_pipeline.py poll --install-at --notify

This starts a self-rescheduling ``at`` chain: each sweep schedules its
successor **before** doing any work, so a sweep that fails does not end the
chain. Inspect with ``atq``; stop with ``atrm <job>`` and delete
``$FOGGIE_ICS_DIR/AtPoll.sh``.

.. warning::

   ``cron`` does not work on the NAS front ends. ``crontab`` accepts an entry
   and it is never executed. Use ``--install-at``. ``poll --install`` submits a
   PBS poller instead, for sites where front-end scheduling is unavailable, but
   it wakes a whole node for a one-second sweep.

The chain lives on the front end it was started from, so a reboot stops it. If
``atq`` is unexpectedly empty, restart it with the command above.

Notifications
~~~~~~~~~~~~~

``--notify`` emails stage **transitions**, not status. The previous sweep's
``status.ecsv`` is the comparison snapshot, so there is no extra state to keep.
The first sweep establishes a baseline silently rather than mailing the whole
fleet, an unchanged sweep says nothing, and ``DONE`` or ``STALLED`` are named
in the subject line. The recipient is the box ``email`` in ``config.py``, or
``--notify-to``.


Guards
------

Two refusals protect against the mistakes that are otherwise silent.

**Hand-built directories.** ``build`` refuses to write into a halo directory
that holds ``<sim>-L*`` run directories but no ``.pipeline/``, which means it
was built by hand. This is what stops a halo added to the registry from
overwriting somebody's existing work. ``--adopt`` overrides it, and will write
generated files into that directory, so do not pass it casually.

**Mixed output cadence.** ``build`` refuses to generate level N when its
redshift output list differs from level N-1. Editing the list in a template
part-way through a ladder otherwise produces a halo whose levels are not
comparable -- the same ``RD`` number means a different redshift at each level,
and nothing downstream would notice. Set ``allow_mixed_outputs`` in the
registry to accept it for a specific halo.


Recovering a stalled run
------------------------

``STALLED`` is never retried automatically, because resubmitting into whatever
killed the run just burns allocation. The note in ``status`` says what
happened: ``walltime kill``, ``reached RD0014 but no RunFinished``,
``RunFinished but never reached RD0014``, and so on.

To restart a stalled Enzo run, resubmit it by hand from its stage directory::

    cd $FOGGIE_ICS_DIR/halo51541/25Mpc_DM_512-L1
    qsub -koed RunScript.sh

``simrun.pl`` restarts from the last output automatically. Once it reaches its
final dump the chain resumes on its own.


What gets generated, and where
------------------------------

Inside ``$FOGGIE_ICS_DIR/halo<ID>/``:

* ``halo<ID>_DM_<N-1>to<N>.conf`` -- enzo-mrp-music config for each level
* ``<sim>-L<N>.conf`` and ``.conf_log.txt`` -- the MUSIC config it generated,
  and MUSIC's log. The log is where the domain shift is read from.
* ``<sim>-L<N>/`` -- the stage directory: ICs, ``.enzo`` parameter file,
  ``RunScript.sh``, ``simrun.pl``, and Enzo's outputs
* ``.pipeline/ledger.json`` -- what the pipeline submitted and when. Marks the
  directory as pipeline-managed.
* ``BuildScript-L<N>-<phase>.sh`` -- the IC generation job
* ``pipeline.log`` -- what the job-chained hook did

Templates live in ``foggie/initial_conditions/templates_512/``. There is one
``DM-LX.enzo`` and one ``gas-LX.enzo`` rather than a file per level: the
per-level files differed only in ``CosmologySimulationNumberOfInitialGrids``,
``MustRefineParticlesRefineToLevel`` and ``MinimumOverDensityForRefinement``,
all of which are now substituted at build time.

If you change a template, re-approve it::

    python3 .../ic_pipeline.py validate-templates              # check
    python3 .../ic_pipeline.py validate-templates --rebaseline # approve

The check compares against an approved baseline under
``templates_512/baseline/``, so an intentional edit is approved once while an
accidental change to a refinement or cosmology parameter still fails.


Command reference
-----------------

All commands take ``--registry`` to use a registry other than the default, and
most take ``--dry-run``.

``status``
    Progress table. ``--by-halo``, ``--include-manual``, ``--include-gas``,
    ``--write``, ``--out-dir``, ``--notify``.

``advance``
    Submit the next actionable stage. ``--halo <id>`` for one halo, otherwise
    every enabled halo. This is what the hook and the poller call.

``build``
    Generate ICs for one stage and submit it. ``--halo``, ``--level``,
    ``--phase {DM,gas}``, ``--as-job`` (run IC generation on a compute node --
    it needs about 10 GB and must not run on a front end), ``--no-submit``,
    ``--adopt``, ``--allow-mixed-outputs``.

``poll``
    One sweep. ``--install-at`` to start the recurring chain, ``--interval``,
    ``--notify``.

``validate-registry``
    Check the registry parses, every halo resolves in the catalog, and report
    the effective zoom radius for each.

``validate-templates``
    Check the templates render as approved. ``--rebaseline`` to approve the
    current state, ``--original`` to compare against the original hand-written
    per-level files in git.


Adding a new parent box
-----------------------

Add an entry to ``BOXES`` in ``pipeline/config.py``. Everything that used to be
a literal in the old scripts is a field there: the parent grid size, the halo
catalog, the MUSIC and Enzo binaries, the baryon fraction, the PBS resources
and the Rvir floor.

The grid size matters more than it looks. MUSIC reports domain shifts in units
of the parent grid, so converting them to code units divides by
``parent_ngrid - 1`` -- 511 for a 512 box, 255 for a 256 box. That was a bare
literal in the old scripts and is the kind of thing that silently displaces a
zoom region if it is wrong.
