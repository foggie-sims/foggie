# -*- coding: utf-8 -*-
"""
Functions related to post-rockstar consistent trees.
JT Feb 2018
"""
from astropy.table import Table

def import_tree(filename):
    """
    Read in the consistent tree and return a convenient representation of it in as astropy table
    """

    a = Table.read(filename, format='ascii.commented_header')

    tree = Table([ a['scale(0)'], a['id(1)'], a['desc_scale(2)'], a['desc_id(3)'],
                   a['Mvir(10)'], a['Rvir(11)'],
                   a['x(17)'], a['y(18)'], a['z(19)'],
                   a['Tree_root_ID(29)'], a['Orig_halo_ID(30)'], a['Snap_idx(31)'] ],
                   names=['ascale', 'id','desc_scale','desc_id','mvir','rvir','x','y','z','rootID','origID','snapidx'])

    return tree

def select_root(tree, rootindex):
    """
    Select out the halos with a root of the given index and return the new, pared back tree.
    """

    new = tree[tree['rootID'] == rootindex]

    return new 
