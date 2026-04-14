import yt
import numpy as np 


import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import argparse

from foggie.utils.consistency import *
from foggie.utils.get_refine_box import get_refine_box
from foggie.utils.get_halo_center import get_halo_center
from foggie.utils.get_proper_box_size import get_proper_box_size
from foggie.utils.get_run_loc_etc import get_run_loc_etc
from foggie.utils.yt_fields import *
from foggie.utils.foggie_load import *
from foggie.utils.analysis_utils import *


from foggie.clumps.clump_finder.utils_clump_finder import halo_id_to_name
from foggie.clumps.clump_finder import *


def parse_args():
    '''Parse command line arguments. Returns args object.'''

    parser = argparse.ArgumentParser(description='Makes basic kinematic plots for the disk (and CGM)')

    # Optional arguments:
    parser.add_argument('--halo', metavar='halo', type=str, action='store', \
                        help='Which halo? Default is 8508 (Tempest)')
    parser.set_defaults(halo='008508')

    parser.add_argument('--run', metavar='run', type=str, action='store', \
                        help='Which run? Default is nref11c_nref9f')
    parser.set_defaults(run='nref11c_nref9f')

    parser.add_argument('--minsnap', metavar='minsnap', type=int, action='store', \
                        help='Which snapshot to start at? Default is RD0042')
    parser.set_defaults(minsnap='967')

    parser.add_argument('--maxsnap', metavar='maxsnap', type=int, action='store', \
                        help='Which snapshot to end at? Default is RD0042')
    parser.set_defaults(maxsnap='2427')

    parser.add_argument('--snapstep', metavar='snapstep', type=int, action='store', \
                        help='Step between snapshots? Default is 1')
    parser.set_defaults(snapstep=1)

    parser.add_argument('--system', metavar='system', type=str, action='store', \
                        help='Where is the clump file to define the disk')
    parser.set_defaults(system='cameron_local')

    parser.add_argument('--pwd', metavar='pwd', type=bool, action='store', \
                        help='Use working directory in get_run_loc_etc. Default is False.')
    parser.set_defaults(pwd=False)

    parser.add_argument('--forcepath', metavar='forcepath', type=bool, action='store', \
                        help='Use forcepath in get_run_loc_etc. Default is False.')
    parser.set_defaults(forcepath=False)

    parser.add_argument('--is_rd', metavar='is_rd', type=bool, action='store', \
                        help='Are you analyzing RD snapshots instead of DD? Default is False.')
    parser.set_defaults(is_rd=False)

    parser.add_argument('--output_dir', metavar='output_dir', type=str, action='store', \
                        help='Where to save the histograms. Default is ./')
    parser.set_defaults(output_dir='/Users/ctrapp/Documents/foggie_analysis/mrr_tests/figures/')

    parser.add_argument('--snap_dir', metavar='snap_dir', type=str, action='store', \
                        help='If you want to specify a directory for the snapshot instead of using get_run_loc_etc, specify it here. Default is None.')
    parser.set_defaults(snap_dir="/Volumes/wde4tb/simulation_snapshots/foggie/halo_008508/nref11c_nref9f_ClumpMRRTests/")

    parser.add_argument('--mrr_track', metavar='mrr_track', type=str, action='store', \
                        help='Where is the mrr track file? Default is /Users/ctrapp/Documents/foggie_analysis/mrr_tests/Tempest0967_Clump63166TrackBox.txt')
    parser.set_defaults(mrr_track='/Users/ctrapp/Documents/foggie_analysis/mrr_tests/Tempest0967_Clump63166TrackBox.txt')

    args = parser.parse_args()
    return args

def LoadRedshiftsAndTimes(halo_c_v_name):
    halo_c_v = Table.read(halo_c_v_name, format='ascii')
    
    redshifts = {}
    times = {}
    xc = {}
    yc = {}
    zc = {}
    vxc = {}
    vyc = {}
    vzc = {}
    itr=0
    for row in halo_c_v:
        if itr>0:
            redshifts[row['col3']] = float(row['col2'])
            times[row['col3']] = ((float(row['col4'])))# * unyt.Myr).in_units('s').v
            xc[row['col3']] = float(row['col5']) * unyt.kpc
            yc[row['col3']] = float(row['col6']) * unyt.kpc
            zc[row['col3']] = float(row['col7']) * unyt.kpc
            vxc[row['col3']] = float(row['col8']) * unyt.km/unyt.s
            vyc[row['col3']] = float(row['col9']) * unyt.km/unyt.s
            vzc[row['col3']] = float(row['col10']) * unyt.km/unyt.s
        itr+=1

    return redshifts,times,xc,yc,zc,vxc,vyc,vzc


def ReadTrackFile(track_filename,track=None):
    track_number = []
    redshift = []
    blc = []
    trc = []

    with open(track_filename, "r") as f:
        for line in f:
            vals = line.split()
            if len(vals) < 7:
                continue
            if track is None:
                track_number.append(int(vals[0]))
            if track is None or int(vals[0])==track:
                redshift.append(float(vals[1]))
                blc.append([float(vals[2]), float(vals[3]), float(vals[4])])
                trc.append([float(vals[5]), float(vals[6]), float(vals[7])])

    if track is None:
        return track_number,redshift, blc, trc
    return redshift, blc, trc


args = parse_args()
halo_id = args.halo #008508
run = args.run #nref11c_nref9f
GalName = halo_id_to_name(halo_id)



snapshots = []
if args.is_rd: 
    snapshots.append( "RD"+str(args.minsnap).zfill(4) )
else:
    snapshots.append( "DD"+str(args.minsnap).zfill(4) )

for snap in range(args.minsnap,args.maxsnap,args.snapstep):
    if args.is_rd:
        snapshots.append( "RD"+str(snap).zfill(4) )
    else:
        snapshots.append( "DD"+str(snap).zfill(4) )

data_dir, output_path, run_loc, code_dir,trackname,halo_name,spectra_dir,infofile = get_run_loc_etc(args)
halo_c_v_name = code_dir+"/halo_infos/"+halo_id+"/nref11c_nref9f/halo_c_v"
redshifts,times,xc,yc,zc,vxc,vyc,zyc = LoadRedshiftsAndTimes(halo_c_v_name)

print("Redshift at snapshot 967=",redshifts[snapshots[0]])
print("Time at snapshot 967=",times[snapshots[0]])
import trident


for snapshot in snapshots:
    gal_name = halo_id_to_name(halo_id)
    gal_name+="_"+snapshot+"_"+run

    snap_name = data_dir + "halo_"+halo_id+"/"+run+"/"+snapshot+"/"+snapshot
    if args.snap_dir is not None:
        snap_name = args.snap_dir+snapshot+"/"+snapshot

    trackname = code_dir+"/halo_tracks/"+halo_id+"/nref11n_selfshield_15/halo_track_200kpc_nref9"

#particle_type_for_angmom = 'young_stars' ##Currently the default
    particle_type_for_angmom = 'gas' #Should be defined by gas with Temps below 1e4 K

    catalog_dir = code_dir + '/halo_infos/' + halo_id + '/'+run+'/'
#smooth_AM_name = catalog_dir + 'AM_direction_smoothed'
    smooth_AM_name = None

    clump_trackname = args.mrr_track
    track_redshifts,track_blc,track_trc = ReadTrackFile(clump_trackname,track=0)

    ds, refine_box = foggie_load(snap_name, trackfile_name=trackname, halo_c_v_name=halo_c_v_name, do_filter_particles=True,disk_relative=True,particle_type_for_angmom=particle_type_for_angmom,smooth_AM_name = smooth_AM_name)

    z_idx = np.argmin(np.abs(np.array(track_redshifts) - ds.current_redshift))
    track_blc = track_blc[z_idx]
    track_trc = track_trc[z_idx]


    clump_box = ds.box(track_blc, track_trc)

    center=[(track_blc[0]+track_trc[0])/2.,(track_blc[1]+track_trc[1])/2.,(track_blc[2]+track_trc[2])/2.]

    trident.add_ion_fields(ds, ions=['O II','O III','O IV','O V','O VI','Mg II'])


    p = yt.ProjectionPlot(ds, 'z', ('gas','density'), center=center, data_source=clump_box, width=(20,'kpc'))
    p.set_cmap(('gas','density'),density_color_map)
    p.set_unit(('gas','density'),'Msun/pc**2')
    p.hide_axes()
    p.annotate_scale(size_bar_args={'color':'white'})
    p.save(args.output_dir + gal_name+"_mrr_density_projection.png")

    p = yt.ProjectionPlot(ds, 'z', ('gas','O_p1_density'), center=center, data_source=clump_box, width=(20,'kpc'))
    p.set_cmap(('gas','O_p1_density'),density_color_map)
    p.set_unit(('gas','O_p1_density'),'Msun/pc**2')
    p.hide_axes()
    p.annotate_scale(size_bar_args={'color':'white'})
    p.save(args.output_dir + gal_name+"_mrr_OII_density_projection.png")

    p = yt.ProjectionPlot(ds, 'z', ('gas','O_p5_density'), center=center, data_source=clump_box, width=(20,'kpc'))
    p.set_cmap(('gas','O_p5_density'),density_color_map)
    p.set_unit(('gas','O_p5_density'),'Msun/pc**2')
    p.hide_axes()
    p.annotate_scale(size_bar_args={'color':'white'})
    p.save(args.output_dir + gal_name+"_mrr_OVI_density_projection.png")