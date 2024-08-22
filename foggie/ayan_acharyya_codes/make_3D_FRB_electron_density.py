#!/usr/bin/env python3

"""

    Title :      make_3D_FRB_electron_density
    Notes :      Make a 3D Fixed Resolution Buffer (FRB) for gas density and electron number density and save as a multi-extension fits file, and optionally plot along a chosen line of sight
    Output :     3D data cube as fits file, and optionally png figures
    Author :     Ayan Acharyya
    Started :    Aug 2024
    Examples :   run make_3D_FRB_electron_density.py --system ayan_pleiades --halo 8508 --res 1 --upto_kpc 50 --docomoving --do_all_sims
                 run make_3D_FRB_electron_density.py --system ayan_hd --halo 4123 --res 1 --upto_kpc 10 --output RD0038 --docomoving --clobber --plot_3d
                 run make_3D_FRB_electron_density.py --system ayan_hd --halo 8508 --res 1 --upto_kpc 200 --output RD0030,RD0042 --docomoving --clobber
"""
from header import *
from util import *
from yt.visualization.fits_image import FITSImageData
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['axes.linewidth'] = 1

start_time = datetime.now()

# -----------------------------------------------------------------------------
def get_AM_vector(ds):
    '''
    Computes the orientation vector of angular momentum of the disk, in the given dataset, considering young star particles
    Based on foggie_load()
    Returns the unit vector as a list
    '''
    start_time = datetime.now()

    print('Staring to derive angular momentum vector. This can take a while..')
    sphere = ds.sphere(ds.halo_center_kpc, (15., 'kpc'))
    L = sphere.quantities.angular_momentum_vector(use_gas=False, use_particles=True, particle_type='young_stars')
    print('Completed deriving angular momentum vector, in %s'% timedelta(seconds=(datetime.now() - start_time).seconds))
    norm_L = L / np.sqrt((L ** 2).sum())
    norm_L = np.array(norm_L.value)

    return norm_L

# --------------------------------------------------------------------------
def plot_3d_frb(data, ax, args, label=None, unit=None, clim=None,  cmap='viridis'):
    '''
    Function to make a 3D plot given a 3D numpy array in a given axis
    Returns axis handle
    '''
    z, x, y = data.nonzero()
    ax.scatter3D(z, x, y, c=np.log10(data), cmap=cmap, alpha=0.3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return ax

# --------------------------------------------------------------------------
def plot_proj_frb(data, ax, args, label='', unit='', clim=None,  cmap='viridis', hidex=False, hidey=False):
    '''
    Function to make a 2D projection plot (along one line of sight) given a 3D numpy array, in a given axis
    Returns axis handle
    '''
    data_proj = np.sum(data, axis=2)
    p = ax.imshow(np.log10(data_proj), cmap=cmap)

    # -----------making the axis labels etc--------------
    if hidex:
        ax.set_xticklabels(['' % item for item in ax.get_xticks()])
        ax.set_xlabel('')
    else:
        ax.set_xticklabels(['%.1F' % ((item - central_pixel) * args.kpc_per_pix) for item in ax.get_xticks()], fontsize=args.fontsize)
        ax.set_xlabel('Offset (kpc)', fontsize=args.fontsize)

    if hidey:
        ax.set_yticklabels(['' % item for item in ax.get_yticks()])
        ax.set_ylabel('')
    else:
        ax.set_yticklabels(['%.1F' % ((item - central_pixel) * args.kpc_per_pix) for item in ax.get_yticks()], fontsize=args.fontsize)
        ax.set_ylabel('Offset (kpc)', fontsize=args.fontsize)

    # ---------making the colorbar axis once, that will correspond to all projections--------------
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.05)
    cbar = fig.colorbar(p, orientation='horizontal', cax=cax)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=args.fontsize, width=2.5, length=5)
    cbar.set_label(f'log(LoS summed {label} ({unit}))', fontsize=args.fontsize/1.)
    cbar.set_label(f'log(LoS summed {label} ({unit}))', fontsize=args.fontsize/1.)

    # ---------------making annotations------------------------
    ax.text(0.97, 0.95, 'z = %.2F' % args.current_redshift, c='white', ha='right', va='top', transform=ax.transAxes, fontsize=args.fontsize, bbox=dict(facecolor='k', alpha=0.3, edgecolor='k'))
    ax.text(0.97, 0.85, 't = %.1F Gyr' % args.current_time, c='white', ha='right', va='top', transform=ax.transAxes, fontsize=args.fontsize, bbox=dict(facecolor='k', alpha=0.3, edgecolor='k'))

    return ax

# --------------------------------------------------------------------------
def plot_proj_frb_diskrel(box, field, box_width_kpc, norm_L, args, unit='', clim=None,  cmap='viridis'):
    '''
    Function to make a 2D projection plot along edge-on and face-on views given a dataset
    Borrowed a little from foggie_load()
    Returns figure handle
    '''
    x = np.random.randn(3)  # take a random vector
    x -= x.dot(norm_L) * norm_L  # make it orthogonal to L
    x /= np.linalg.norm(x)  # normalize it
    y = np.cross(norm_L, x)  # cross product with L

    field = ('gas', field)
    fontsize = args.fontsize

    # ---------------making face on and edge on projections------------------------
    p_faceon = yt.OffAxisProjectionPlot(box.ds, ds.arr(norm_L), field, data_source=box, width=(box_width, 'kpc'), weight_field='density', center=box.ds.halo_center_kpc, north_vector=ds.arr(x))
    p_edgeon = yt.OffAxisProjectionPlot(box.ds, ds.arr(x), field, data_source=box, width=(box_width, 'kpc'), weight_field='density', center=box.ds.halo_center_kpc, north_vector=ds.arr(norm_L))

    # ---------------setting up units, colormaps, etc------------------------
    p_faceon.set_log(field, True)
    p_faceon.set_unit(field, unit)
    p_faceon.set_zlim(field, zmin=10**clim[0], zmax=10**clim[1])
    p_faceon.set_cmap(field, cmap)

    p_edgeon.set_log(field, True)
    p_edgeon.set_unit(field, unit)
    p_edgeon.set_zlim(field, zmin=10**clim[0], zmax=10**clim[1])
    p_edgeon.set_cmap(field, cmap)

    # ------plotting onto a matplotlib figure--------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    p_faceon.plots[field].axes = axes[0]
    p_faceon._setup_plots()
    p_edgeon.plots[field].axes = axes[1]
    p_edgeon._setup_plots()
    divider = make_axes_locatable(axes[1])

    fig.subplots_adjust(right=0.87, top=0.98, bottom=0.1, left=0.1, wspace=0.1)

    # ---------------making colorbar------------------------
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(p_edgeon.plots[field].cb.mappable, orientation='vertical', cax=cax)
    cbar.ax.tick_params(labelsize=fontsize, width=2.5, length=5)
    cbar.set_label(p_edgeon.plots[field].cax.get_ylabel(), fontsize=fontsize)

    # ---------------prepping axes------------------------
    for index in range(len(axes)):
        ax = axes[index]
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xticklabels(['%.1F' % item for item in ax.get_xticks()], fontsize=fontsize)
        ax.set_xlabel('Offset (kpc)', fontsize=fontsize)
        if index == 0:
            ax.set_yticklabels(['%.1F' % item for item in ax.get_yticks()], fontsize=fontsize)
            ax.set_ylabel('Offset (kpc)', fontsize=fontsize)
        else:
            ax.set_yticklabels(['' % item for item in ax.get_yticks()])
            ax.set_ylabel('')

    # ---------------making annotations------------------------
    axes[0].text(0.97, 0.95, 'z = %.2F' % args.current_redshift, c='white', ha='right', va='top', transform=axes[0].transAxes, fontsize=fontsize, bbox=dict(facecolor='k', alpha=0.3, edgecolor='k'))
    axes[0].text(0.97, 0.85, 't = %.1F Gyr' % args.current_time, c='white', ha='right', va='top', transform=axes[0].transAxes, fontsize=fontsize, bbox=dict(facecolor='k', alpha=0.3, edgecolor='k'))

    axes[0].text(0.98, 0.02, 'Face on', c='white', ha='right', va='bottom', transform=axes[0].transAxes, fontsize=fontsize, bbox=dict(facecolor='k', alpha=0.3, edgecolor='k'))
    axes[1].text(0.98, 0.02, 'Edge on', c='white', ha='right', va='bottom', transform=axes[1].transAxes, fontsize=fontsize, bbox=dict(facecolor='k', alpha=0.3, edgecolor='k'))

    # ---------------saving fig------------------------
    outfile_rootname = '%s_%s_diskrel_%s%s.png' % (args.output, args.halo, quant_dict[quant_arr[0]][0], args.upto_text)
    if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
    figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))

    plt.savefig(figname)
    myprint('Saved figure ' + figname, args)
    plt.show()

    return fig

# -----main code-----------------
if __name__ == '__main__':
    args_tuple = parse_args('8508', 'RD0042')  # default simulation to work upon when comand line args not provided
    if type(args_tuple) is tuple: args, ds, refine_box = args_tuple # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
    else: args = args_tuple
    if not args.keep: plt.close('all')

    quant_dict = {'density':['density', 'Gas density', 'Msun/pc**3', -2.5, 2.5, 'cornflowerblue', density_color_map], 'el_density':['El_number_density', 'Electron density', 'cm**-3', -6, -1, 'cornflowerblue', e_color_map]} # for each quantity: [yt field, label in plots, units, lower limit in log, upper limit in log, color for scatter plot, colormap]
    quant_arr = ['el_density']#, 'density']

    # --------domain decomposition; for mpi parallelisation-------------
    if args.do_all_sims: list_of_sims = get_all_sims_for_this_halo(args) # all snapshots of this particular halo
    else: list_of_sims = list(itertools.product([args.halo], args.output_arr))
    total_snaps = len(list_of_sims)

    comm = MPI.COMM_WORLD
    ncores = comm.size
    rank = comm.rank
    print_master('Total number of MPI ranks = ' + str(ncores) + '. Starting at: {:%Y-%m-%d %H:%M:%S}'.format(datetime.now()), args)
    comm.Barrier() # wait till all cores reached here and then resume

    split_at_cpu = total_snaps - ncores * int(total_snaps/ncores)
    nper_cpu1 = int(total_snaps / ncores)
    nper_cpu2 = nper_cpu1 + 1
    if rank < split_at_cpu:
        core_start = rank * nper_cpu2
        core_end = (rank+1) * nper_cpu2 - 1
    else:
        core_start = split_at_cpu * nper_cpu2 + (rank - split_at_cpu) * nper_cpu1
        core_end = split_at_cpu * nper_cpu2 + (rank - split_at_cpu + 1) * nper_cpu1 - 1

    # -------------loop over snapshots-----------------
    print_mpi('Operating on snapshots ' + str(core_start + 1) + ' to ' + str(core_end + 1) + ', i.e., ' + str(core_end - core_start + 1) + ' out of ' + str(total_snaps) + ' snapshots', args)

    for index in range(core_start + args.start_index, core_end + 1):
        start_time_this_snapshot = datetime.now()
        this_sim = list_of_sims[index]
        print_mpi('Doing snapshot ' + this_sim[1] + ' of halo ' + this_sim[0] + ' which is ' + str(index + 1 - core_start) + ' out of the total ' + str(core_end - core_start + 1) + ' snapshots...', args)

        # -------loading in snapshot-------------------
        halos_df_name = args.code_path + 'halo_infos/00' + this_sim[0] + '/' + args.run + '/'
        halos_df_name += 'halo_cen_smoothed' if args.use_cen_smoothed else 'halo_c_v'
        if len(list_of_sims) > 1 or args.do_all_sims: args = parse_args(this_sim[0], this_sim[1])
        if type(args) is tuple: args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
        else: ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=True, disk_relative=False, halo_c_v_name=halos_df_name)

        norm_L =  get_AM_vector(ds) #np.array([-0.64498829, -0.5786498 , -0.49915379]) #computing disk orientation #

        # --------assigning additional keyword args-------------
        args.upto_text = '_upto%.1Fckpchinv' % args.upto_kpc if args.docomoving else '_upto%.1Fkpc' % args.upto_kpc
        args.current_redshift = ds.current_redshift
        args.current_time = ds.current_time.in_units('Gyr').tolist()
        args.res = args.res_arr[0]
        args.res_text = f'_res{args.res:.1f}kpc'
        if args.docomoving: args.res = args.res / (1 + args.current_redshift) / 0.695  # converting from comoving kcp h^-1 to physical kpc
        args.fontsize = 15

        # --------determining corresponding text suffixes and figname-------------
        #args.fig_dir = args.output_dir + 'figs/'
        args.fig_dir = '/Users/acharyya/Library/CloudStorage/GoogleDrive-ayan.acharyya@inaf.it/My Drive/FOGGIE-Curtin/plots/'
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

        #args.fits_dir = args.output_dir + 'txtfiles/'
        args.fits_dir = '/Users/acharyya/Library/CloudStorage/GoogleDrive-ayan.acharyya@inaf.it/My Drive/FOGGIE-Curtin/data/'
        Path(args.fits_dir).mkdir(parents=True, exist_ok=True)

        outfile_rootname = '%s_%s_FRB_%s%s%s.png' % (args.output, args.halo, quant_dict[quant_arr[0]][0], args.upto_text, args.res_text)
        if args.do_all_sims: outfile_rootname = 'z=*_' + outfile_rootname[len(args.output) + 1:]
        figname = args.fig_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift))
        fitsname = args.fits_dir + outfile_rootname.replace('*', '%.5F' % (args.current_redshift)).replace('.png', '.fits')

        if not os.path.exists(fitsname) or args.clobber:
            #try:
            # ------tailoring the simulation box for individual snapshot analysis--------
            if args.upto_kpc is not None:
                if args.docomoving: args.galrad = args.upto_kpc / (1 + args.current_redshift) / 0.695  # fit within a fixed comoving kpc h^-1, 0.695 is Hubble constant
                else: args.galrad = args.upto_kpc  # fit within a fixed physical kpc
            else:
                args.re = get_re_from_coldgas(args) if args.use_gasre else get_re_from_stars(ds, args)
                args.galrad = args.re * args.upto_re  # kpc

            # extract the required box
            box_width = args.galrad * 2  # in kpc
            box_width_kpc = ds.arr(box_width, 'kpc')
            args.ncells = int(box_width / args.res)
            box_center = ds.halo_center_kpc
            box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2., box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2., box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]

            central_pixel = args.ncells / 2
            args.kpc_per_pix = 2 * args.galrad / args.ncells

            # -------setting up fig--------------
            if args.plot_3d or args.plot_proj:
                fig = plt.figure(figsize=(2 + 4 * len(quant_arr), 5))
                fig.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.92, wspace=0.4 if args.plot_3d else 0.02, hspace=0.)

            # -------making and plotting the 3D FRBs--------------
            all_data = ds.arbitrary_grid(left_edge=box.left_edge, right_edge=box.right_edge, dims=[args.ncells, args.ncells, args.ncells])
            img_hdu_list = []

            for index, quant in enumerate(quant_arr):
                myprint(f'Making and plotting FRB for {quant} which is {index+1} out of {len(quant_arr)} quantities..', args)

                # --------making the 3D FRB------------
                FRB = all_data[('gas', quant_dict[quant][0])].in_units(quant_dict[quant][2]).astype(np.float32)

                # --------making the FITS ImageHDU---------------
                img_hdu = FITSImageData(FRB, ('gas', quant_dict[quant][1]))
                header = img_hdu[0].header
                for ind in range(3):
                    header[f'CDELT{ind+1}'] = args.kpc_per_pix
                    header[f'CUNIT{ind+1}'] = 'kpc'
                    header[f'NORMAL_UNIT_VECTOR{ind+1}'] = norm_L[ind]
                img_hdu_list.append(img_hdu)

                # ------making the plots-----------
                if args.plot_3d or args.plot_proj:
                    ax = fig.add_subplot(1, len(quant_arr), index + 1, projection='3d' if args.plot_3d else None)
                    if args.plot_3d: ax = plot_3d_frb(FRB, ax, args, label=quant_dict[quant][1], unit=quant_dict[quant][2], clim=[quant_dict[quant][3], quant_dict[quant][4]], cmap=quant_dict[quant][6])
                    elif args.plot_proj: ax = plot_proj_frb(FRB, ax, args, label=quant_dict[quant][1], unit=quant_dict[quant][2], clim=[quant_dict[quant][3], quant_dict[quant][4]], cmap=quant_dict[quant][6], hidey=index > 0)

                fig_diskrel = plot_proj_frb_diskrel(box, quant_dict[quant][0], box_width_kpc, norm_L, args, unit=quant_dict[quant][2], clim=[quant_dict[quant][3], quant_dict[quant][4]],  cmap=quant_dict[quant][6])

            # ------saving fits file------------------
            combined_img_hdu = FITSImageData.from_images(img_hdu_list)
            combined_img_hdu.writeto(fitsname, overwrite=args.clobber)
            myprint('Saved fits file as ' + fitsname, args)

            # ------saving fig------------------
            if args.plot_3d or args.plot_proj:
                fig.savefig(figname)
                myprint('Saved plot as ' + figname, args)

            plt.show(block=False)
            print_mpi('This snapshots completed in %s' % timedelta(seconds=(datetime.now() - start_time_this_snapshot).seconds), args)
            '''
            except Exception as e:
                print_mpi('Skipping ' + this_sim[1] + ' because ' + str(e), args)
                continue
            '''
        else:
            print('Skipping snapshot %s as %s already exists. Use --clobber_plot to remake figure.' %(args.output, fitsname))
            continue

    # -----------------------------------------------------------------------------------
    if ncores > 1: print_master('Parallely: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' cores was %s' % timedelta(seconds=(datetime.now() - start_time).seconds), args)
    else: print_master('Serially: time taken for ' + str(total_snaps) + ' snapshots with ' + str(ncores) + ' core was %s' % timedelta(seconds=(datetime.now() - start_time).seconds), args)
