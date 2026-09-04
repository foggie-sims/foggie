#!/bin/bash
# Idempotent multizoom ladder driver, parameterized by group.
#
#   advance_group.sh --group <name> [--halos 1,2,3,...] [--max-level 3]
#                    [--mode merge|union] [--registry <path>]
#
# Whenever the current level's run reaches z=0 it builds the next level's
# ICs; whenever ICs exist without a running Enzo job it assembles and
# submits.  Safe to call repeatedly (e.g. from cron or a Monitor loop):
# each call advances at most one step and says what it did.
#
# --halos defines the group ad hoc, so a different set of halos needs no
# registry edit; the group name is the label its directory takes.
set -u
GROUP=""; HALOS=""; MAXLEVEL=3; MODE=merge; SHIFT_OVERRIDE=""; GAS_NREF=""
BUILD_WALL="4:00:00"
REGISTRY=/nobackupnfs1/jtumlins/foggie-multizoom/runs/halo_registry_multizoom.ecsv
while [ $# -gt 0 ]; do
  case "$1" in
    --group) GROUP="$2"; shift 2;;
    --halos) HALOS="$2"; shift 2;;
    --max-level) MAXLEVEL="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --shift) SHIFT_OVERRIDE="$2"; shift 2;;
    --gas-nref) GAS_NREF="$2"; shift 2;;
    --build-walltime) BUILD_WALL="$2"; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
[ -n "$GROUP" ] || { echo "--group is required" >&2; exit 2; }
HALO_ARG=""; [ -n "$HALOS" ] && HALO_ARG="--halos $HALOS"

export FOGGIE_REPO=/nobackupnfs1/jtumlins/foggie-multizoom/foggie
export FOGGIE_ICS_DIR=/nobackupnfs1/jtumlins/25Mpc_new_cosmology
export MULTIZOOM_ICS_DIR=/nobackupnfs1/jtumlins/foggie-multizoom/runs
export MULTIZOOM_ENZO_EXE=/nobackupnfs1/jtumlins/foggie-multizoom/enzo-multizoom/src/enzo/enzo.exe
export MULTIZOOM_MUSIC_EXE_DIR=/nobackupnfs1/jtumlins/foggie-multizoom/scratch/music-shiftpatch
export MULTIZOOM_MUSIC_LD_PATH=/home1/jtumlins/local/lib:/nasa/pkgsrc/2015Q1/lib:/u/jtumlins/installs/gsl-2.4/lib:/nasa/hdf5/1.8.18_serial/lib
# One common domain shift for every run of the group (merge mode); required
# when any target's Lagrangian region sits near the periodic boundary.
[ -n "$SHIFT_OVERRIDE" ] && export MULTIZOOM_SHIFT_OVERRIDE="$SHIFT_OVERRIDE"
export PYTHONPATH=/nobackupnfs1/jtumlins/foggie-multizoom
export PATH=/nobackupnfs1/jtumlins/anaconda3/bin:$PATH

MZ=/nobackupnfs1/jtumlins/foggie-multizoom
G=$MULTIZOOM_ICS_DIR/multizoom_$GROUP
SIM=25Mpc_DM_512
FINAL=RD0014
BUILD=$G/BuildScript-L%d-$MODE.sh
cd $MZ

queued () { qstat -u jtumlins 2>/dev/null | grep -q "$1"; }

mkdir -p $G
for L in $(seq 1 $MAXLEVEL); do
  ICS=$G/$SIM-L$L; PREV=$((L-1))
  if [ ! -f $ICS/parameter_file.txt ]; then
    if queued "mz-$GROUP-L$L-b"; then echo "L$L: build queued/running"; exit 0; fi
    if [ $L -gt 1 ] && [ ! -f $G/$SIM-L$PREV/$FINAL/$FINAL ]; then
      echo "L$L: waiting for L$PREV to reach $FINAL"; exit 0; fi
    S=$(printf "$BUILD" $L)
    cat > $S <<PBS
#!/bin/bash
#PBS -N mz-$GROUP-L$L-build
#PBS -W group_list=s3128
#PBS -l select=1:ncpus=64:mpiprocs=1:model=mil_ait
#PBS -l walltime=$BUILD_WALL
#PBS -q normal
#PBS -j oe
#PBS -o build_L${L}_$MODE.log
$(env | grep -E '^(FOGGIE_|MULTIZOOM_|PYTHONPATH=)' | sed "s/^\([A-Za-z_][A-Za-z0-9_]*\)=\(.*\)$/export \1='\2'/")
export PATH=/nobackupnfs1/jtumlins/anaconda3/bin:\$PATH
cd $MZ
python3 -m foggie.initial_conditions.multizoom.pipeline_integration build \\
    --group $GROUP --level $L --mode $MODE $HALO_ARG --registry $REGISTRY
echo "exit=\$?"
PBS
    echo "L$L: submitting IC build"; cd $G && qsub $(basename $S); exit 0
  fi
  if [ ! -f $ICS/RunScript.sh ]; then
    echo "L$L: assembling Enzo run"
    python3 -m foggie.initial_conditions.multizoom.pipeline_integration assemble \
        --group $GROUP --level $L $HALO_ARG --registry $REGISTRY || exit 1
    sed -i "s/#PBS -N mz-.*-L$L\$/#PBS -N mz-$GROUP-L$L/" $ICS/RunScript.sh
    cd $ICS && qsub RunScript.sh && echo "L$L: submitted"; exit 0
  fi
  if [ ! -f $ICS/$FINAL/$FINAL ]; then
    queued "mz-$GROUP-L$L" && echo "L$L: run in progress" \
      || echo "L$L: NOT QUEUED and unfinished -- needs attention"
    exit 0
  fi
  echo "L$L: complete"
done
echo "ladder complete through L$MAXLEVEL"
