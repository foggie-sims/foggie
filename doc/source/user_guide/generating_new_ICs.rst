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


Getting started
---------------

Start here if you have a working Enzo build and nothing else. Nine steps, in
order. Later sections explain what any of it means.

**What you need first**

* A **working Enzo executable**. The pipeline does not build Enzo.
* The **parent box already run to z = 0**. A zoom is cut out of an existing
  unigrid simulation, so the L0 run must exist and be finished before any zoom
  can be made. For the 25 Mpc box that is ``25Mpc_DM_512-L0`` under
  ``$FOGGIE_ICS_DIR``, along with the MUSIC config that produced it
  (``25Mpc_DM_512_planck18.conf``) and its log (``.conf_log.txt``).
* A **Rockstar halo catalog** for that box, so halos can be looked up by ID.
  One is in the repo at ``initial_conditions/halo_catalogs_512/512/z0/out_0.list``.

If you do not have the parent box, you are not ready for this page: make that
first, the ordinary way, and come back.

**1. Get the code**

::

    git clone git@github.com:foggie-sims/foggie.git
    cd foggie

Nothing needs installing. The pipeline is run as a script.

**2. Set two environment variables**

Everything reads these, so put them in your shell profile::

    export FOGGIE_REPO=/path/to/foggie/foggie            # note: the inner dir
    export FOGGIE_ICS_DIR=/path/to/25Mpc_new_cosmology   # where runs live

``FOGGIE_REPO`` points at the *package* directory inside the clone, the one
containing ``initial_conditions/``.

**3. Build MUSIC**

The source is in the repo; the compiled binary is not, so build it once on the
machine you will run on::

    cd $FOGGIE_REPO/initial_conditions/music
    make

MUSIC needs a C++ compiler, **FFTW3**, **GSL** and **HDF5**. On a cluster,
load those modules first. The ``Makefile`` has include and library paths
hardcoded near the top -- edit ``CC``, ``CPATHS`` and ``LPATHS`` to match where
yours are installed. On NAS that looks like::

    module load comp-intel hdf5/1.8.18_serial

The result must be an executable at::

    $FOGGIE_REPO/initial_conditions/music/MUSIC

That exact path is where the pipeline looks. Check with ``./MUSIC --help``.

**4. Point the pipeline at your Enzo**

Edit ``enzo_exe`` in ``$FOGGIE_REPO/initial_conditions/pipeline/config.py``,
in the ``25Mpc_DM_512`` entry, to your Enzo executable. While you are there,
check ``group_list`` is an account you can charge to, and that the PBS
resource lines (``dm_select``, ``build_select``) name a node model your site
has.

**5. Check it is wired up correctly**

::

    cd $FOGGIE_REPO/initial_conditions/pipeline
    python3 ic_pipeline.py validate-templates
    python3 ic_pipeline.py validate-registry

.. note::

   Always run it as a **script**, as above, not as
   ``python -m foggie.initial_conditions.pipeline.ic_pipeline``. Importing the
   ``foggie`` package pulls in yt, which nothing here needs and which slows
   ``status`` from about a second to ten.

The first confirms the parameter-file templates render as approved. The second
confirms the registry parses and every halo in it resolves in the catalog. Both
should say ``OK``. Fix anything they report before going further -- they are
much cheaper than finding the same problem inside a job.

**6. Pick a halo and add it to the registry**

Open ``$FOGGIE_REPO/initial_conditions/halo_registry.ecsv`` in a text editor
and append one row, using the column order in the header line::

    82812 25Mpc_DM_512 True 3 False 0.0 normal 1 mil_ait "my first zoom"

That is: halo ID, box, enabled, top level, gas on/off, Rvir floor override,
queue, nodes, node model, a note. Re-run
``validate-registry`` -- it will print the halo's virial radius, which is a
good check you chose the ID you meant.

**7. Launch it**

::

    python3 ic_pipeline.py advance --halo 82812

This submits the level 1 IC generation job. That job generates the ICs and then
submits the Enzo run itself.

Use ``--dry-run`` first if you want to see exactly what it would do without
submitting anything.

**8. Watch**

::

    python3 ic_pipeline.py status --by-halo

Your halo moves ``READY -> BUILDING -> QUEUED -> RUNNING -> DONE``, then the
next level starts on its own. A full three-level ladder takes days, mostly
Enzo time.

**9. Turn on the poller**

::

    python3 ic_pipeline.py poll --install-at --notify

Each Enzo run already triggers the next level when it finishes. The poller is
the backstop for when that trigger does not fire -- a node dies, a job is
killed -- and it emails you when anything changes state. Recommended, not
required.

That is the whole thing. From here the pipeline runs unattended; you add halos
by adding rows.

.. note::

   Two habits worth forming early. Run ``status`` before assuming anything is
   wrong -- it will usually tell you. And if a stage says ``STALLED``, read the
   note in that row before restarting it: nothing restarts automatically, on
   purpose, and restarting without fixing the cause just stalls again.


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
level for halos whose registry row asks for one.

The gas stage is **not** the next rung of that ladder. It hangs off the DM
build at the same level and runs alongside the DM run -- see `The gas stage`_.


The halo registry
-----------------

``foggie/initial_conditions/halo_registry.ecsv`` is the single hand-curated
input, and the only file you normally edit. **You edit it by hand, in a text
editor, and commit it like any other source file.** Nothing writes to it: the
pipeline only ever reads it, so your edits are never overwritten and two people
changing it resolve as an ordinary git conflict.

It is ECSV -- plain text, with a commented header declaring the column types,
followed by one whitespace-separated row per halo::

    halo_id box          enabled final_level gas   rvir_min queue  nodes model   notes
    51541   25Mpc_DM_512 True    3           False 0.0      normal 1     mil_ait "started fresh"
    79628   25Mpc_DM_512 True    3           False 0.0      normal 1     mil_ait "started fresh"

To add a halo, append a row in the order given by the header line. A few
things to get right, because a malformed row fails when the file is read
rather than when you save it:

* Fields are separated by whitespace, so **quote anything containing spaces**.
  In practice that means ``notes``, which should always be in double quotes.
* ``enabled`` and ``gas`` are booleans: write ``True`` or ``False``,
  capitalised.
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
    Top DM level. Three is usual; four works (halo80181 ran a full L1-L4).

``gas``
    Run a gas stage at the final level. It runs in parallel with the DM run at
    that level rather than after it; see `The gas stage`_.

``rvir_min``
    Floor on the zoom radius in kpc, overriding the box default. ``0`` uses the
    box default (80 kpc for ``25Mpc_DM_512``). The radius is
    ``max(catalog Rvir, floor)``: several of these dwarfs have Rvir well under
    80 kpc, and too small a Lagrangian region makes a poor zoom.

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

The gas stage
~~~~~~~~~~~~~

Set ``gas`` to ``True`` in a halo's registry row and it gains a gas stage at
its final level. It is enabled by default; ``--no-gas`` on ``status`` or
``advance`` suppresses it.

Gas ICs are not made with enzo-mrp-music. They are made by running MUSIC
directly on the DM MUSIC config for the same level, with ``baryons = yes`` and
the box's ``omega_b``. So the gas stage depends on that config file existing::

    $FOGGIE_ICS_DIR/halo<ID>/<sim>-L<N>.conf

which is written by the level-N DM **build**, and which in turn requires level
N-1's Enzo run to have finished. In the usual three-level setup that means
**L2 must be done before L3-gas is possible.**

What it does *not* depend on is level N's own Enzo run. Nothing about the gas
ICs comes from it. So once the DM build at that level has written its config,
the gas stage can be generated and submitted **while the DM run at the same
level is still going**::

    halo 42189 L2-DM  is QUEUED -- nothing to do
    halo 42189 L2-gas is READY  -- submitting IC build

``advance`` therefore treats gas as an independent branch rather than the next
rung, and can act on the DM ladder and the gas stage in the same sweep. Until
the config exists the stage reports what it is waiting for::

    halo 42189 L3-gas is BLOCKED -- waiting on 25Mpc_DM_512-L3.conf

Gas runs get their own PBS resources (``gas_select``, ``gas_nranks``,
``gas_walltime`` in the box config), which are larger than the DM ones, and
their own ``gas-LX.enzo`` template. That template is not a collapsed copy of
the DM one: the gas physics genuinely differs by level, so it derives from the
L3 gas file rather than being merged across levels.

.. warning::

   **A gas run writes about 40 TB if nothing is deleted**, and that is the
   number to budget against, not the 12 TB a finished run appears to occupy.

   Two things are writing snapshots, and they are roughly the same size --
   47--56 GB each, growing as structure forms:

   * the 266-entry ``CosmologyOutputRedshift`` list, giving 266 ``RD`` dumps
     (~12 TB);
   * ``dtDataDump = 1``, giving a ``DD`` dump every code time unit -- 609 of
     them in the reference run (~28 TB).

   The completed runs *look* like 12 TB because their ``DD`` dumps were deleted
   afterwards; their ``OutputLog`` still records all 609. Plan either to delete
   ``DD`` dumps periodically, as was done for those runs, or to raise
   ``dtDataDump`` before starting. Enable one gas halo at a time regardless.

Do **not** bring the two output lists into step. They differ deliberately:
``DM-LX.enzo`` keeps a 15-entry list, and ``gas-LX.enzo`` carries the 266-entry
list every completed gas run used. Trimming it would leave a gas run with about
fifteen redshift snapshots, and would also remove the output at *z* = 15 that
the next section depends on.

The z = 15 cooling transition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grackle's cooling changes character around *z* = 15, where self-shielding
starts to matter. A gas run is therefore **two legs**, not one:

* **First leg**, *z* = 99 to 15, with unshielded cooling. The template ships
  ``H2FormationOnDust``, ``self_shielding_method`` and ``H2_self_shielding`` all
  at 0, and ``CosmologyFinalRedshift`` at 15.
* **Second leg**, *z* = 15 to 0, with those set to 1, 3 and 1 and
  ``CosmologyFinalRedshift`` at 0.

Enzo cannot change these mid-run, so the handoff is: stop, rewrite the restart
parameter file, restart. This used to be a hand edit between two submissions --
and nothing recorded that it was needed. The pipeline now does it
automatically, inside the same PBS job, and you should not have to touch it.

What happens in the run directory, in order:

#. Enzo reaches *z* = 15, writes its dump, and writes ``RunFinished``.
#. ``simrun.pl`` sees ``RunFinished`` and exits normally.
#. ``RunScript.sh`` notices the run stopped at the transition rather than at the
   end, writes a ``new_pars`` file with the four parameters, deletes
   ``RunFinished``, and records ``gas_transition.done``.
#. It calls ``simrun.pl`` again. ``simrun.pl`` applies ``new_pars`` to the
   restart parameter file, renames it ``new_pars.old`` so it cannot be applied
   twice, and restarts Enzo from the *z* = 15 dump.

So a healthy gas run leaves both ``gas_transition.done`` and ``new_pars.old``
behind, and ``run.log`` records the switch::

    Switching H2FormationOnDust to 1.
    Switching self_shielding_method to 3.
    Switching H2_self_shielding to 1.
    Switching CosmologyFinalRedshift to 0.

Two details worth knowing if you are debugging one:

* ``RunFinished`` at *z* = 15 is a **false positive** for completion --
  ``simrun.pl`` reads it as "done" and so does this pipeline's state machine.
  That is why the handoff deletes it. The block keys off
  ``gas_transition.done``, not off ``RunFinished``, because the second leg ends
  by writing ``RunFinished`` too; keying off the latter would loop forever.
* ``simrun.pl`` restarts its own walltime clock on each invocation, so the
  second leg is given only what the job has left. If that is under
  ``gas_transition_min_seconds`` (default 1800), the handoff still happens but
  the leg is left to a fresh submission rather than started with too little time
  to reach an output.

The stop redshift and the four parameters live in ``config.Box`` as
``gas_stop_redshift`` and ``gas_transition_pars``, so the ``.enzo`` template and
the shell block cannot disagree about where the run stops. The names are
applied verbatim to the parameter file: a misspelling is silently ignored by
Enzo rather than raising an error, so check any change against
``ReadParameterFile.C``.

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


Diagnostic plots
----------------

``qc`` makes one figure per halo, a row per refinement level, centred on the
target::

    python3 .../ic_pipeline.py qc --halo 79628
    python3 .../ic_pipeline.py qc --halo 79628 --as-job     # needs yt and ~10 GB

Each row shows the halo at three zooms and a species panel, and the run prints
a contamination table. Two questions are being answered:

**Is the target a single object?** A catalog entry can turn out to be two
clumps mid-merger, or a chance superposition. The wide panels show what is
around it -- halo79628 has a comparable companion about 150 kpc away, obvious
at 8 x Rvir and easy to miss otherwise.

**Is the high-resolution region clean?** A zoom is only usable if the coarse
particles from the parent box stayed out of it. The species panel colours fine
particles blue and coarse red, and the table gives, per species, how many
particles lie inside Rvir, how close the nearest one comes, and the fraction of
the mass inside Rvir that is coarse. Under 1 % reads ``CLEAN``.

Three things worth knowing about how it works, each of which produced a wrong
answer before it was handled:

* **The halo is not where the analytic centre says.** Catalog position plus the
  MUSIC domain shift is only a starting guess; the object's z = 0 position
  differs between the parent box and the zoom. For halo79628 it lands about
  270 kpc away -- ten virial radii -- and L1 and L2 agree on that, so it is the
  halo having moved rather than the shift arithmetic being wrong. ``qc``
  therefore locates the halo by shrinking spheres from the guess and reports
  the offset. A large offset is itself a result: check it is still the object
  you meant.
* **At L0 the catalog position is used unchanged.** Rockstar was run on that
  very output, so the position is definitional and re-centring can only walk
  onto a brighter neighbour.
* **Levels that have not reached z = 0 are marked ``IN PROGRESS``** rather than
  judged. Rvir and the centre are z = 0 quantities; applying them to a halo at
  z = 4 gives a confident and meaningless verdict.

Run it after a level finishes. It is deliberately not wired into ``advance``:
it needs yt and real memory, and the useful moment to look is once a level is
done, not while it is filling in.


Guards
------

One refusal protects against the mistake that is otherwise silent.

**Hand-built directories.** ``build`` refuses to write into a halo directory
that holds ``<sim>-L*`` run directories but no ``.pipeline/``, which means it
was built by hand. This is what stops a halo added to the registry from
overwriting somebody's existing work. ``--adopt`` overrides it, and will write
generated files into that directory, so do not pass it casually.

.. note::

   Nothing stops a halo whose levels use different redshift output lists. If
   you change the list in a template part-way through a ladder, the levels are
   no longer directly comparable -- the same ``RD`` number means a different
   redshift at each level -- so either finish a halo before changing it, or
   regenerate the lower levels afterwards.


Recovering a stalled run
------------------------

``STALLED`` is never retried automatically. Neither ``advance``, the
job-chained hook, nor the poller will touch a stalled stage, because a stall
usually means the run hit something that will stop it again, and resubmitting
into that just burns allocation. Restarting is always a decision you make.

Start by reading why it stopped. The note in ``status`` says what happened:
``walltime kill``, ``reached RD0014 but no RunFinished``, ``RunFinished but
never reached RD0014``, ``simrun.pl: in trouble``, or a PBS message such as a
node failure. Fix that cause before restarting anything, or the run will simply
stall again in the same place.

One stage
~~~~~~~~~

Resubmit it from its stage directory::

    cd $FOGGIE_ICS_DIR/halo51541/25Mpc_DM_512-L1
    qsub -koed RunScript.sh

``simrun.pl`` restarts from the last output, so nothing is recomputed. Once the
run reaches its final dump the chain resumes on its own.

Several stages, one shared cause
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes one external problem stops everything at once: the filesystem fills,
a node fails, the scheduler has an outage. Every affected stage goes
``STALLED`` for the same reason, and once that reason is fixed they can all be
restarted together::

    python3 .../ic_pipeline.py resume --dry-run   # what would restart
    python3 .../ic_pipeline.py resume             # restart it

``resume`` resubmits every ``STALLED`` stage that has a ``RunScript.sh``,
reporting where each one stopped, and skips stages whose ICs were never built.
``--halo <id>`` restricts it to one halo, which is useful when only some of
the stalled runs are worth restarting.

It is the human-initiated escape hatch, and is deliberately never called by the
poller or the hook. Run it when you know why things stalled and that the reason
is gone.

.. warning::

   Check the cause is genuinely fixed first. After a full filesystem, confirm
   there is enough space for the runs to finish, not merely enough to start:
   a stage part-way through a dense output list can still have hundreds of
   gigabytes left to write, and will stall again at the same point.

Stages that are ``READY`` or ``BUILT`` rather than ``STALLED`` are not
``resume``'s business -- ``advance`` or the next poll sweep picks those up
normally.


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

Per-level parameter edits
~~~~~~~~~~~~~~~~~~~~~~~~~

Historically, each new refinement level needed four edits to the Enzo parameter
file by hand, recorded in ``enzo-mrp-music/bds_notes``. The pipeline now makes
all four, and it is worth knowing which, both to recognise them in a generated
file and because getting one wrong produces a run that works and is wrong:

1. **Divide** ``MinimumOverDensityForRefinement`` **by 8 per level.** Rendered
   as ``8**-(L-1)``: ``1.`` at L1, ``0.125`` at L2, ``0.015625`` at L3.
2. **Add** ``8`` **to** ``CellFlaggingMethod``, the must-refine-particle method.
   Already present in the templates.
3. **Set** ``MustRefineParticlesCreateParticles = 3`` **and**
   ``MustRefineParticlesRefineToLevel`` **to the level.** Written by
   enzo-mrp-music into ``parameter_file.txt``, and set in the template.
4. **Copy the nested grid geometry** -- the ``CosmologySimulationGrid*`` lines
   -- from MUSIC's ``parameter_file.txt`` into the parameter file, and nothing
   else from that file. Substituted at build time.

``CosmologySimulationNumberOfInitialGrids`` is likewise set to ``level + 1``.

If you change a template, re-approve it::

    python3 .../ic_pipeline.py validate-templates              # check
    python3 .../ic_pipeline.py validate-templates --rebaseline # approve

The check compares against an approved baseline under
``templates_512/baseline/``, so an intentional edit is approved once while an
accidental change to a refinement or cosmology parameter still fails.

Gas physics parameters
~~~~~~~~~~~~~~~~~~~~~~

``gas-LX.enzo`` carries the star formation and feedback physics; the DM template
has none of it. A few things about it are easy to get wrong, and Enzo will not
tell you:

* ``StarParticleCreation = 2048`` selects the H2-regulated star maker
  (``star_maker_h2reg``). Its efficiency parameter is
  ``H2StarMakerEfficiency``. **StarMakerMassEfficiency is not passed to that
  routine at all** and has no effect here, despite looking like it should --
  check the call site in ``Grid_StarParticleHandler.C`` before tuning either.
* ``StarMakerMinimumMass`` is the one to set. ``H2StarMakerMinimumMass`` is
  deprecated and setting it explicitly is a hard failure; Enzo copies
  ``StarMakerMinimumMass`` into it.
* Several ``H2StarMaker*`` parameters in the template restate Enzo's defaults
  explicitly. That is deliberate -- it keeps the physics visible in the
  parameter file rather than implied by the build.
* ``gas_max_refine_level`` in the box config feeds ``MaximumRefinementLevel``,
  ``MaximumGravityRefinementLevel`` and ``MaximumParticleRefinementLevel``
  together. It is **7** while the gas path is under test, matching the
  hand-built runs; 9 is the eventual target and should be adopted deliberately.

When comparing a generated parameter file against one of the older hand-built
gas runs, expect these to differ legitimately: ``dtRestartDump`` is set here and
was not there, and ``MinimumOverDensityForRefinement`` follows the same
divide-by-8-per-level rule as the DM ladder, which the hand-built gas runs did
not apply.


Command reference
-----------------

All commands take ``--registry`` to use a registry other than the default, and
most take ``--dry-run``.

``status``
    Progress table. ``--by-halo``, ``--include-manual``, ``--no-gas``,
    ``--write``, ``--out-dir``, ``--notify``.

``advance``
    Submit the next actionable stage on the DM ladder, and independently the
    gas stage if its prerequisite exists. ``--halo <id>`` for one halo,
    otherwise every enabled halo; ``--no-gas`` to skip gas. This is what the
    hook and the poller call. Never acts on a ``STALLED`` stage.

``resume``
    Resubmit ``STALLED`` stages after fixing whatever stopped them. ``--halo``,
    ``--dry-run``. Never called automatically; see `Recovering a stalled run`_.

``build``
    Generate ICs for one stage and submit it. ``--halo``, ``--level``,
    ``--phase {DM,gas}``, ``--as-job`` (run IC generation on a compute node --
    it needs about 10 GB and must not run on a front end), ``--no-submit``,
    ``--adopt``.

``poll``
    One sweep. ``--install-at`` to start the recurring chain, ``--interval``,
    ``--notify``.

``qc``
    Diagnostic plots per level, centred on the halo, with a contamination
    table. ``--halo``, ``--levels``, ``--out``, ``--as-job``.

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
