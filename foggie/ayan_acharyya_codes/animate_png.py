##!/usr/bin/env python3

"""

    Title :      animate_png
    Notes :      Makes animated gif from <ROOTNAME>%d.png files within <PATH> and saves <OUTPUTFILENAME>.gif in <PATH>
    Notes :      Adapted from https://pythonprogramming.altervista.org/png-to-gif/
    Author :     Ayan Acharyya
    Started :    May 2021
    Example :    run animate_png.py --inpath ~/Work/astro/enzo-e-outputs/ayan/ --rootname initial_ayan- --outfile ayan.gif
    OR      :    run ~/Work/astro/ayan_codes/animate_png.py --inpath /Users/acharyya/Work/astro/enzo-e-outputs/cosmology_examples/ --rootname ENZOP_DD*/mesh-*.png

"""

from PIL import Image, ImageFont, ImageDraw
import glob, argparse, time, os
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import image as mgimg
import numpy as np
import imageio
from mpi4py import MPI
import subprocess

# -------------------------------------------------------------------------------------------
def print_mpi(string):
    '''
    Function to print corresponding to each mpi thread
    '''
    comm = MPI.COMM_WORLD
    print('[' + str(comm.rank) + '] {' + subprocess.check_output(['uname -n'],shell=True)[:-1].decode("utf-8") + '} ' + string + '\n')

# -------------------------------------------------------------------
def make_anim_imageio(args):
    filenames = glob.glob(args.inpath + args.rootname)
    filenames.sort(key=str, reverse=args.reverse)
    outputfile = args.inpath + args.outfile

    loops = 1 # 0 = infinite loop
    if args.delay is not None: duration_per_frame = args.delay # sec
    elif args.duration is not None: duration_per_frame = args.duration / (len(filenames) * loops) # sec
    else: duration_per_frame = 0.1 # sec
    '''
    images = []
    for index, filename in enuemrate(filenames):
        print_mpi('Appending file ' + os.path.split(filename)[1] + ' which is ' + str(index + 1) + ' out of ' + str(len(filenames))) #
        images.append(imageio.imread(filename))
    imageio.mimsave(outputfile, images, duration=duration_per_frame, loop=loops)
    '''
    with imageio.get_writer(outputfile, mode='I', fps=int(1./duration_per_frame)) as writer:
        for index, filename in enumerate(filenames):
            print_mpi('Appending file ' + os.path.split(filename)[1] + ' which is ' + str(index + 1) + ' out of ' + str(len(filenames)))  #
            image = imageio.imread(filename)
            writer.append_data(image)

    print_mpi('Combined ' + str(len(filenames)) + ' images into ' + outputfile)
    return args

# -------------------------------------------------------------------
def make_anim_mplanimation(args):
    filenames = glob.glob(args.inpath + args.rootname)
    filenames.sort(key=str)
    fig, ax = plt.subplots()

    # initialization of animation, plot array of zeros
    def init():
        imobj.set_data(np.zeros((100, 100)))
        return imobj

    def animate(index):
        fname = args.inpath + args.rootname.replace('*', '%03d') % index # Read in picture
        print_mpi('Deb33: filename=' + '\n'.join(fname)) #
        img = mgimg.imread(fname)[-1::-1] # here I use [-1::-1], to invert the array; Otherwise it plots up-side down
        imobj.set_data(img)

        return imobj

    imobj = ax.imshow(np.zeros((100, 100)), origin='lower', alpha=1.0, zorder=1, aspect=1)

    anim = animation.FuncAnimation(fig, animate, init_func=init, repeat=True, frames=range(0, len(filenames)), interval=200, blit=True, repeat_delay=1000)

    plt.show(block=False)

    duration_per_frame = 0.1 # in seconds
    outputfile = args.inpath + args.outfile
    #anim.save(outputfile, fps=int(1/duration_per_frame))
    print_mpi('Combined ' + str(len(filenames)) + ' images into ' + outputfile)
    return anim

# -------------------------------------------------------------------
def make_anim_pil(args):
    filenames = glob.glob(args.inpath + args.rootname)
    filenames.sort(key=str)
    #print_mpi('Deb22: filenames=\n' + '\n'.join(filenames)) #
    frames = [Image.open(thisfile) for thisfile in filenames]

    loops = 1 # use loop = 0 for looping forever
    if args.duration is None:
        duration_per_frame = 0.1 # in seconds
        args.duration = len(frames) * duration_per_frame * loops

    outputfile = args.inpath + args.outfile
    frames[0].save(outputfile, format='GIF', append_images=frames[1:], save_all=True, duration=args.duration, loop=loops)

    print_mpi('Combined ' + str(len(frames)) + ' images into ' + outputfile)
    return args

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    # -------------------------------------------------------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='''dummy''')

    parser.add_argument('--inpath', metavar='inpath', type=str, action='store', default='/Users/acharyya/Work/astro/enzo-e-outputs/ayan/')
    parser.add_argument('--rootname', metavar='rootname', type=str, action='store', default='initial_ayan-')
    parser.add_argument('--outfile', metavar='outfile', type=str, action='store', default='none')
    parser.add_argument('--ext', metavar='ext', type=str, action='store', default='.mp4')
    parser.add_argument('--duration', metavar='duration', type=float, action='store', default=None)
    parser.add_argument('--delay', metavar='delay', type=float, action='store', default=None)
    parser.add_argument('--reverse', dest='reverse', action='store_true', default=False)
    args = parser.parse_args()

    if '*' not in args.rootname: args.rootname += '*'
    if args.rootname[-4:] != '.png': args.rootname += '.png'

    if args.outfile == 'none': args.outfile = args.rootname.replace('.png', '').replace('*', '').replace('/', '_') + '_anim'
    if args.outfile[-len(args.ext):] != args.ext: args.outfile += args.ext
    # -------------------------------------------------------------------

    #args = make_anim_pil(args)
    #anim = make_anim_mplanimation(args)
    args = make_anim_imageio(args)

    print_mpi('Completed in %s minutes' % ((time.time() - start_time) / 60))
