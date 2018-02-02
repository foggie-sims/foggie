'''
try to dynamically figure out relevant paths
'''

import getpass
import socket

def get_path_names():
    username = getpass.getuser()
    hostname = socket.gethostname()
    foggie_dir, output_dir = "", ""
    if username == "molly" and hostname == "dhumuha-2.local":
        foggie_dir = "/Users/molly/foggie/"
        output_dir = "/Users/molly/Dropbox/foggie-collab/"

    if username == "molly" and hostname == "oak.stsci.edu":
        foggie_dir = "/astro/simulations/FOGGIE/"
        output_dir = "/Users/molly/Dropbox/foggie-collab/"

    return foggie_dir, output_dir
