'''
try to dynamically figure out relevant paths
'''

import getpass
import socket

def get_path_names():
    username = getpass.getuser()
    hostname = socket.gethostname()
    foggie_dir, output_dir = "", ""
    if username == "molly" and "dhumuha" in hostname:
        foggie_dir = "/Users/molly/foggie/"
        output_dir = "/Users/molly/Dropbox/foggie-collab/"

    if username == "molly" and "oak" in hostname:
        foggie_dir = "/astro/simulations/FOGGIE/"
        output_dir = "/Users/molly/Dropbox/foggie-collab/"

    if username == "tumlinson": 
        foggie_dir = "/astro/simulations/FOGGIE/"
        output_dir = "/Users/molly/Dropbox/FOGGIE/collab/"

    return foggie_dir, output_dir
