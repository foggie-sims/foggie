#!/usr/bin/env python3

"""

    Title :      volume_rendering_movie
    Notes :      Attempt to volume rendering FOGGIE outputs and make movies
    Output :     volume rendered plots as png files (which can be later converted to a movie via animate_png.py)
    Author :     Ayan Acharyya
    Started :    July 2021
    Examples :
run volume_rendering_movie.py --system ayan_hd --halo 4123 --output RD0038 --do gas --imres 128 --nmovframes 10 --max_frot 0.25 --max_zoom 1.5 --move_to 0,0,0.75 --makemovie --sigma 8 --delay 1
run volume_rendering_movie.py --system ayan_hd --halo 4123 --output RD0038 --do gas --imres 1024 --nmovframes 200 --max_frot 1 --max_zoom 15 --move_to 0,0,0.75 --makemovie --sigma 8 --delay 0.1

"""
from header import *
from util import *
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.api import LineSource
start_time = time.time()

# --------------------------------------------------------------------------------
def get_metal_color_function(tfh, bounds, colormap, nlayers=8):
    '''
    Function to make color transfer function for density
    '''
    tfh.tf.add_layers(nlayers, w=0.01, alpha=np.logspace(-2.0, 0, nlayers), colormap=colormap)

    return tfh

# --------------------------------------------------------------------------------
def get_temp_color_function(tfh, bounds, colormap, nlayers=8):
    '''
    Function to make color transfer function for density
    '''
    tfh.tf.add_layers(nlayers, w=0.005, alpha=np.logspace(0, -2.5, nlayers), colormap=colormap)

    return tfh

# --------------------------------------------------------------------------------
def get_density_color_function(tfh, bounds, colormap, nlayers=8):
    '''
    Function to make color transfer function for density
    '''
    # the values below are hard-coded, corresponding to the gas density field range in Blizzard RD0038
    col_max, more_gauss1, more_gauss2 = 3e-24, 1e-23, 3e-23

    # add_layers() function will add evenly spaced isocontours along the transfer function, sampling a colormap to determine the colors of the layers
    tfh.tf.add_layers(nlayers, w=0.01, alpha=np.logspace(-2.5, 0, nlayers), colormap=colormap, ma=np.log10(col_max), mi=np.log10(bounds[0]), col_bounds=[np.log10(bounds[0]), np.log10(col_max)])

    # if you would like to add a gaussian with a customized color or no color, use add_gaussian()
    tfh.tf.add_gaussian(np.log10(more_gauss1), width=.005, height=[1, 0.35, 0.0, 2.0])  # height = [R,G,B,alpha]
    tfh.tf.add_gaussian(np.log10(more_gauss2), width=.005, height=[1, 1, 0.8, 5.0])  # height = [R,G,B,alpha]

    return tfh

# --------------------------------------------------------------------------------
def make_transfer_function(ds, args):
    '''
    Function to make yt.transferfunction
    '''
    # set the field to be plotted, with bounds
    field = field_dict[args.do]
    bounds = bounds_dict[args.do]

    tfh = TransferFunctionHelper(ds)
    tfh.set_field(field)
    tfh.set_bounds(bounds)
    tfh.set_log(True)
    tfh.build_transfer_function()
    tfh.grey_opacity = True

    color_func_dict = {'gas': get_density_color_function, 'temp': get_temp_color_function, 'metal': get_metal_color_function}

    tfh = color_func_dict[args.do](tfh, bounds, colormap_dict[args.do], nlayers=nlayers_dict[args.do])

    if args.saveplot: tfh.plot(fn = fig_dir + 'trial_transfer_function_' + field_dict[args.do][1] + '.png', profile_field=field)
    return tfh

# --------------------------------------------------------------------------------
def setup_camera(sc, domain, resolution=(1024, 1024), starting_pos=None):
    '''
    Function to setup the fiducial camera orientation
    '''
    cam = sc.add_camera(domain, lens_type='perspective')
    cam.resolution = resolution
    if starting_pos is None: starting_pos = domain.left_edge
    cam.set_position(starting_pos)
    cam.focus = domain.center
    cam.north_vector = [1, 0, 0] # positive x-axis
    cam.switch_view()

    return cam

# -----main code-----------------
if __name__ == '__main__':
    # set variables and dictionaries
    do_yaw = True
    colormap_dict = {'temp':temperature_color_map, 'metal':metal_color_map, 'gas':'viridis'} # density_color_map # 'cividis' #
    field_dict = {'gas':('gas', 'density'), 'gas_entropy':('gas', 'entropy'), 'stars':('deposit', 'stars_density'),'ys_density':('deposit', 'young_stars_density'), 'ys_age':('my_young_stars', 'age'), 'ys_mass':('deposit', 'young_stars_mass'), 'metal':('gas', 'metallicity'), 'temp':('gas', 'temperature'), 'dm':('deposit', 'dm_density'), 'vrad':('gas', 'radial_velocity_corrected')}
    bounds_dict = defaultdict(lambda: None, gas=(1e-29, 1e-22), temp=(2e3, 4e6), metal=(1e-2, 1e1)) # in g/cc, range within refine_box; hard-coded for Blizzard RD0038; but should be broadly applicable to other snaps too
    nlayers_dict = defaultdict(lambda: 8, gas=8, temp=6, metal=5)

    # parse args and load in sim
    args = parse_args('8508', 'RD0042')
    if type(args) is tuple:
        args, ds, refine_box = args  # if the sim has already been loaded in, in order to compute the box center (via utils.pull_halo_center()), then no need to do it again
        myprint('ds ' + str(ds) + ' for halo ' + str(args.halo) + ' was already loaded at some point by utils; using that loaded ds henceforth', args)
    else:
        ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=False)

    # parse the resolution, zoom, etc
    field = field_dict[args.do]
    resolution = (args.imres, args.imres)
    max_rot = 2 * np.pi * args.max_frot
    fig_dir = args.output_dir + 'figs/' + args.output + '/'
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    outfile_rootname = 'vol_rendering_%s_imres%d_max_rot_%.1Fdeg_max_zoom_%.1F_move_to_%s_sigma_clip%0.1F_frame_*.png'% (args.do, args.imres, ds.quan(max_rot, 'radian').in_units('degree').value, args.max_zoom, ','.join(args.move_to.astype(str)), args.sigma)
    starting_pos = refine_box.left_edge  # starting position for camera
    refine_box.width = refine_box.right_edge - refine_box.left_edge
    move_to = args.move_to * refine_box.width + starting_pos  # user input args.move_to is in units of domain length, and relative to starting position i.e. if args.move_to = [0,0,0] it implies final position of camera = initial position

    # compute rotation and zoom for each frame
    if args.nmovframes > 1:
        # 1 subtracted from args.nmovframes because the first frame is not rotated or zoomed, therefore the intended total rotation and total zoom is to be accomplished by n - 1 frames
        zoom_each_frame_factor = 10 ** (np.log10(args.max_zoom) / (args.nmovframes/2 - 1))  # n/2 - 1 instead of n - 1 because the total zoom-in is to be accomplished halfway, the next half being the zoom-out phase
        rotate_each_frame_by = max_rot / (args.nmovframes - 1)
        move_each_frame_by = (move_to - starting_pos) / (args.nmovframes - 1)
    else:
        zoom_each_frame_factor, rotate_each_frame_by, move_each_frame_by = 1, 0, [0, 0, 0]

    # make and assign transfer function
    tfh = make_transfer_function(refine_box.ds, args)

    # create scene
    sc = yt.create_scene(refine_box, field=field)
    source = sc.get_source()
    source.set_transfer_function(tfh.tf)

    # Draw the domain boundary
    if args.annotate_domain: sc.annotate_domain(ds, color=[1, 1, 1, 0.01])

    # Draw a coordinate axes triad
    if args.annotate_axes: sc.annotate_axes(alpha=0.01)

    # add length scale annotation
    vertices = np.array([[[2,0,0], [1,0,0]]]) * 100 * kpc # a line along x-axis from [10,0,0] kpc to [20,0,0]
    length_scale = LineSource(vertices, np.array([[1, 1, 1, 1]])) # white, opaque line
    sc.add_source(length_scale)

    # setup camera defaults
    cam = setup_camera(sc, refine_box, resolution=resolution, starting_pos=starting_pos)

    start_frame, end_frame = 0, args.nmovframes # default, for full run
    #start_frame, end_frame = 93, 95 # for debugging

    for thisframe in range(end_frame): # cannot start at start_frame because it has to loop through all camera orientations from frame 0 in order to get to the desired frame
        myprint('Doing ' + str(thisframe) + ' out of ' + str(args.nmovframes - 1), args)

        if thisframe < args.nmovframes / 2: zoom_factor = zoom_each_frame_factor  # zoom in during first half
        else: zoom_factor = 1 / zoom_each_frame_factor  # zoom out during second half

        if thisframe: # do not rotate/zoom/move the first frame
            cam.rotate(rotate_each_frame_by, rot_vector=cam.unit_vectors[1] if do_yaw else [1, 0, 0], rot_center=ds.arr(cam.focus))
            cam.zoom(zoom_factor)
            cam.set_position(cam.get_position() + move_each_frame_by)
            cam.switch_view()

        thisframe_name = fig_dir + outfile_rootname.replace('*', '%03i') % (thisframe)
        should_render = (not os.path.exists(thisframe_name) or args.clobber) and (thisframe >= start_frame)

        if args.debug:
            rot_in_deg = ds.quan(rotate_each_frame_by, 'radian').in_units('degree').value
            total_zoom_yet = zoom_factor ** thisframe if thisframe < args.nmovframes/2 else args.max_zoom * zoom_factor ** (thisframe - args.nmovframes/2 + 1)
            current_position = starting_pos + move_each_frame_by * thisframe
            print('Deb115: this rotation is by=', rot_in_deg, 'degrees; '
                  'total rotation thus far is by=',rot_in_deg * thisframe, 'degrees; '
                  'this zoom is by=', zoom_factor, 'X; '
                  'total zoom thus far is by=', total_zoom_yet, 'X; '
                  'this movement is by=', move_each_frame_by, '; '
                  'total movement thus far is up to=', current_position, '=', (current_position - refine_box.center)/refine_box.width, 'user units; '
                  'camera width=', cam.get_width())
            print('Camera object=', cam, '\n')

        if should_render: myprint('Frame does not exist (or clobber=True), will render frame..', args)
        else: myprint('Frame already exists (or outside specified frame index range). Skip rendering.', args)

        if not args.noplot and should_render:
            im = sc.render()
            sc.save(thisframe_name, sigma_clip=args.sigma, render=False)

    if args.makemovie and args.nmovframes > 1 and not args.noplot:
        myprint('Finished creating snapshots, calling animate_png.py to create movie..', args)
        subprocess.call(['python ~/Work/astro/ayan_codes/animate_png.py --inpath ' + fig_dir + ' --rootname ' + outfile_rootname + ' --delay ' + str(args.delay_frame)], shell=True)

    print('Completed in %s minutes' % ((time.time() - start_time) / 60))
