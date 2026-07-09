#!/usr/bin/env python
# coding: utf-8

import yt, unyt, numpy as np, os, matplotlib.pyplot as plt, h5py
from dataclasses import dataclass, field
from datetime import datetime
from foggie.utils.foggie_load import *
from foggie.utils.consistency import *
from yt.funcs import mylog
import glob
from astropy.table import Table, vstack, Column


def catalogs_to_table(run, prefix='DD'):
    """Stack the per-output halo catalogs for a run into one redshift-sorted table."""

    halo_tables = glob.glob(run+'/halo_catalogs/*/'+prefix+'????.0.fits')

    tables = []
    for tt in halo_tables:
        this_table = Table.read(tt.split('.')[0]+'.0.fits')
        with h5py.File(tt.split('.')[0]+'.0.h5', 'r') as f:
            this_table['z'] = f.attrs['CosmologyCurrentRedshift']
        tables.append(this_table[0])

    big_table = vstack(tables)
    big_table.sort('z', reverse=True)

    return big_table


# ---------------------------------------------------------------------------
# Run registry: one line per feedback run. Identity (path, label, color,
# linestyle) lives here only; every figure is driven from this list.  A label
# beginning with '_' is hidden from legends (matplotlib convention).
# ---------------------------------------------------------------------------

@dataclass
class Run:
    key: str
    path: str
    label: str
    color: str
    linestyle: str = 'solid'
    table: Table = field(default=None, repr=False)


RUNS = [
    Run('pr63',        '/u/jtumlins/nobackup/pr63/H2radtest', 'PR63 Rad',     'cyan'),
    Run('therm',       'H2therm_ff',                          'Thermal',      'blue'),
    Run('mech',        'H2mech_tab_cont_ff',                  'Mech',         'orange', 'dashed'),
    Run('rad',         'H2radtest100',                        'Rad100',       'green',  'dashed'),
    Run('default',     'H2mech_tab_cont_cassi',               '_default',     'orange'),
    Run('mom5',        'H2mech_tab_cont_5mom_ff',             'Mom x5',       'darkslateblue'),
    Run('mom_rad',     'H2mech_tab_cont_mom3x_rad100_ff',     'Mom+Rad',      'pink'),
    Run('radius3',     'H2mech_tab_cont_radius3_ff',          'Radius=3',     'red'),
    Run('rad3_rad100', 'H2mech_tab_cont_radius3_rad100_ff',   'Rad=3,Rad100', 'purple'),
    Run('numerical',   'H2numerical',                         'Numerical',    'gray'),
]

BY_KEY = {r.key: r for r in RUNS}

# Which runs appear in which class of figure. pr63 is now in all of them.
WITH_DEFAULT = ['therm', 'mech', 'rad', 'default', 'mom5', 'mom_rad', 'radius3', 'rad3_rad100', 'pr63']
NO_DEFAULT   = ['therm', 'mech', 'rad', 'mom5', 'mom_rad', 'radius3', 'rad3_rad100', 'pr63']
CGM_KEYS     = ['therm', 'mech', 'mom5', 'mom_rad', 'radius3', 'rad3_rad100', 'pr63']


def load_runs(runs=RUNS):
    for r in runs:
        r.table = catalogs_to_table(r.path)
        print(f'Read the {r.label} table')


# ---------------------------------------------------------------------------
# Generic single-panel figure. yfunc (and optional xfunc) map a run's table to
# the plotted arrays, so per-figure differences (.to(), /1e8, log10, ratios,
# cumsum) live in one lambda instead of one hand-written line per run.
# ---------------------------------------------------------------------------

def plot_lines(yfunc, ylabel, title, filename, keys,
               xfunc=lambda t: t['z'], xlabel='Redshift',
               xlim=(3.2, 0.0), ylim=None, kind='plot', linestyle=None,
               legend_loc='best', texts=None, save_kw=None, **plot_kw):

    plt.figure()
    for key in keys:
        r = BY_KEY[key]
        x, y = xfunc(r.table), yfunc(r.table)
        ls = linestyle if linestyle is not None else r.linestyle
        if kind == 'scatter':
            plt.scatter(x, y, s=7, label=r.label, color=r.color, **plot_kw)
        else:
            plt.plot(x, y, label=r.label, color=r.color, linestyle=ls, **plot_kw)

    if xlim: plt.xlim(*xlim)
    if ylim: plt.ylim(*ylim)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    for tx, ty, ts in (texts or []):
        plt.text(tx, ty, ts)
    plt.legend(loc=legend_loc)
    plt.savefig(filename, **(save_kw or {}))
    print(filename)
    plt.close()


def plot_rvir_pair(field_name, scale, ylabel, titles, filename, keys,
                   xlim=(3.2, 0.0), ylim=(0, 4)):
    """Two-panel figure: field within Rvir (left) and 2Rvir (right)."""

    _, axes = plt.subplots(1, 2, figsize=(10, 6))
    for ax, suffix in zip(axes, ['', '_2rvir']):
        for key in keys:
            r = BY_KEY[key]
            ax.plot(r.table['z'], r.table[field_name + suffix].to('Msun') / scale,
                    color=r.color, label=r.label)
        ax.set_box_aspect(1)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel('Redshift')
        ax.legend()
    axes[0].set_title(titles[0]); axes[1].set_title(titles[1])
    axes[0].set_ylabel(ylabel)
    plt.savefig(filename)
    print(filename)
    plt.close()


# ---------------------------------------------------------------------------
# Baryon budget: stacked phase bands, outermost -> innermost. Each entry lists
# the mass components summed for that band's top edge, its color/label, and the
# single component whose fraction is annotated (None for the outer Hot band).
# ---------------------------------------------------------------------------

_MASS_COL = {
    'warm': 'total_warm_cgm_gas_mass', 'hot': 'total_hot_cgm_gas_mass',
    'cold': 'total_cold_cgm_gas_mass', 'cool': 'total_cool_cgm_gas_mass',
    'ism':  'total_ism_gas_mass',      'star': 'total_star_mass',
}

BARYON_PHASES = [
    (['warm', 'hot', 'cold', 'cool', 'ism', 'star'], '#f2dc61', 'Hot CGM',  None),
    (['warm', 'cold', 'cool', 'ism', 'star'],        '#659B4d', 'Warm CGM', 'warm'),
    (['cool', 'cold', 'ism', 'star'],                '#6f427b', 'Cool CGM', 'cool'),
    (['cold', 'ism', 'star'],                        '#C66D64', 'Cold CGM', 'cold'),
    (['ism', 'star'],                                '#4a6091', 'ISM',      'ism'),
    (['star'],                                       '#9e302c', 'Stars',    'star'),
]


def baryons_vs_z(table, zrange=[3, 1.5], title='Baryon Budget', filename='a.png', suffix=''):
    """Stacked baryon budget vs redshift.

    suffix='' sums the phase masses within Rvir; suffix='_2rvir' sums the 2Rvir
    columns instead. In both cases fractions are normalized by the halo virial
    mass (total_mass), which has no 2Rvir counterpart in the catalogs.
    """

    plt.figure()
    plt.xlim(zrange[0], zrange[1])
    plt.ylim(0, 0.3)
    plt.xlabel('Redshift')
    plt.ylabel('Fraction of Halo Total Mass')

    z = table['z']
    total_mass = table['total_mass'].to('Msun')

    for comps, color, label, annot in BARYON_PHASES:
        band = sum(table[_MASS_COL[c] + suffix].to('Msun') for c in comps)
        frac = band / total_mass
        plt.plot(z, frac, color=color, linewidth=2)
        plt.fill_between(z, frac, color=color, label=label)
        if annot:
            comp_frac = table[_MASS_COL[annot] + suffix].to('Msun') / total_mass
            plt.text(z[-1], frac[-1], str(comp_frac[-1] * 100.)[0:3] + '%', color=color)

    plt.plot([0, 7], [0.0461 / 0.285, 0.0461 / 0.285], linestyle='dashed', color='orange')
    plt.title(title)
    plt.legend(frameon=0, loc='upper right', ncols=3)
    plt.savefig(filename)
    plt.close()

    return


# ---------------------------------------------------------------------------
# Figures whose x-axis is not simply z, or that draw two curves per run, are
# small dedicated blocks below.
# ---------------------------------------------------------------------------

def plot_ism_HI_H2(keys, filename='total_ism_HI_H2_gas_mass.png'):
    """HI (dotted) and H2 (dashed) ISM mass for each run."""

    plt.figure()
    for key in keys:
        r = BY_KEY[key]
        plt.plot(r.table['z'], r.table['total_ism_HI_mass'].to('Msun') / 1e10,
                 color=r.color, linestyle='dotted')
    for key in keys:
        r = BY_KEY[key]
        plt.plot(r.table['z'], r.table['total_ism_H2_mass'].to('Msun') / 1e10,
                 label=r.label, color=r.color, linestyle='dashed')

    plt.xlabel('Redshift')
    plt.ylabel('M_HI or M_H2 / 1e10')
    plt.text(3.0, 2.25, 'dotted = HI')
    plt.text(3.0, 2.15, 'dashed = H2')
    plt.legend(loc='upper left', ncols=3)
    plt.title('ISM Mass vs. Feedback Scheme')
    plt.xlim(3.2, 0.0)
    plt.ylim(0, 3)
    plt.savefig(filename)
    print(filename)
    plt.close()


def plot_tacconi(keys, filename='Mmol_over_Mstar.png'):
    """Molecular-to-stellar mass ratio vs log(1+z), on a transparent canvas."""

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    for key in keys:
        r = BY_KEY[key]
        y = np.log10(r.table['total_ism_H2_mass'].to('Msun').value
                     / r.table['total_star_mass'].to('Msun').value)
        ax.plot(np.log10(1. + r.table['z']), y, label=r.label, color=r.color)

    ax.set_xlim(-0.1, 0.9)
    ax.set_ylim(-2, 1)
    ax.set_xlabel('log z')
    ax.set_ylabel('log [M_mol / Mstar]')
    ax.set_title('Molecular Mass to Stellar Mass')
    ax.legend(loc='lower right')
    plt.savefig(filename, dpi=200, bbox_inches='tight', transparent=True)
    print(filename)
    plt.close()


def plot_mstar_ratio(keys, filename='Mstar_ratio.png'):
    """Stellar mass of each run relative to the thermal run at matching redshift.

    Runs may sample z differently, so we interpolate the thermal run's stellar
    mass onto each run's redshifts rather than aligning by array index.
    """

    plt.figure()
    therm = BY_KEY['therm'].table
    therm_z = np.asarray(therm['z'])
    therm_mstar = therm['total_star_mass'].to('Msun').value
    # np.interp requires monotonically increasing sample points
    order = np.argsort(therm_z)
    therm_z, therm_mstar = therm_z[order], therm_mstar[order]

    for key in keys:
        r = BY_KEY[key]
        z = np.asarray(r.table['z'])
        mstar = r.table['total_star_mass'].to('Msun').value
        # only compare where the run's z overlaps the thermal run's z range
        in_range = (z >= therm_z[0]) & (z <= therm_z[-1])
        therm_at_z = np.interp(z[in_range], therm_z, therm_mstar)
        plt.plot(z[in_range], mstar[in_range] / therm_at_z,
                 color=r.color, label=r.label)

    plt.legend(loc='upper right', ncols=3)
    plt.xlim(3.2, 0.0)
    plt.ylim(0.0, 1.2)
    plt.xlabel('Redshift')
    plt.ylabel('Stellar Mass Ratio to Thermal')
    plt.savefig(filename)
    print(filename)
    plt.close()


# ===========================================================================
# Driver
# ===========================================================================

if __name__ == '__main__':

    load_runs()

    # Outflow mass (>500 km/s) vs. SFR7
    plot_lines(
        yfunc=lambda t: np.log10(t['outflow_mass_500'].to('Msun').value),
        xfunc=lambda t: np.log10(t['sfr7'].to('Msun/yr').value),
        xlabel='log SFR7', ylabel=' Outflow Mass (> 500 km/s)',
        title='Outflow Mass > 500 vs. SFR7', filename='outflow_mass_500_vs_sfr7.png',
        keys=NO_DEFAULT, kind='scatter', xlim=(0, 1.5), ylim=(3, 9))

    # SFR7 vs redshift
    plot_lines(
        yfunc=lambda t: t['sfr7'].to('Msun/yr'),
        ylabel='SFR [Msun/yr]', title='SFR and Outflows vs. Feedback Scheme',
        filename='SFR7_vs_redshift.png', keys=WITH_DEFAULT, ylim=(-0.02, 50))

    # Cumulative >500 km/s outflow mass
    plot_lines(
        yfunc=lambda t: t['outflow_mass_500'].cumsum().to('Msun') / 1e11,
        ylabel='Cumulative Outflow Mass (>500) / 1e11',
        title='Cumulative Outflow Mass vs Feedback Scheme',
        filename='Outflow500_cumulative.png', keys=NO_DEFAULT,
        ylim=(-0.02, 1), linestyle='dashed', legend_loc='upper left')

    # >500 km/s outflow mass vs z
    plot_lines(
        yfunc=lambda t: t['outflow_mass_500'].to('Msun') / 1e8,
        ylabel='Outflow Mass 500 / 1e8', title='Outflows > 500 km/s vs. Feedback Scheme',
        filename='outflow500_vs_z.png', keys=WITH_DEFAULT, ylim=(-0.02, 5))

    # >300 km/s outflow mass vs z
    plot_lines(
        yfunc=lambda t: t['outflow_mass_300'].to('Msun') / 1e9,
        ylabel='Outflow Mass (300) / 1e9', title='Outflows vs. Feedback Scheme',
        filename='outflow_mass_300.png', keys=WITH_DEFAULT, ylim=(-0.02, 5))

    # Stellar mass vs z
    plot_lines(
        yfunc=lambda t: t['total_star_mass'].to('Msun') / 1e10,
        ylabel='Mstar / 1e10', title='Stellar Mass vs. Feedback Scheme',
        filename='total_star_mass.png', keys=WITH_DEFAULT, ylim=(0, 10),
        legend_loc='upper left', texts=[(0.8, 0.6, 'Linear scale')])

    # CGM gas mass, Rvir and 2Rvir
    plot_rvir_pair('total_cgm_gas_mass', 1e10, 'CGM Total Mass in Msun / 1e10',
                   ('CGM Mass R < Rvir', 'CGM Mass R < 2Rvir'),
                   'total_cgm_gas_mass.png', keys=CGM_KEYS, ylim=(0, 4))

    # Warm CGM gas mass, Rvir and 2Rvir
    plot_rvir_pair('total_warm_cgm_gas_mass', 1e10, 'CGM Warm Mass in Msun / 1e10',
                   ('CGM Warm Mass R < Rvir', 'CGM Warm Mass R < 2Rvir'),
                   'total_warm_cgm_gas_mass.png', keys=CGM_KEYS, ylim=(0, 1.2))

    # Hot CGM gas mass, Rvir and 2Rvir
    plot_rvir_pair('total_hot_cgm_gas_mass', 1e9, 'CGM Hot Mass in Msun / 1e9',
                   ('CGM Hot Mass R < Rvir', 'CGM Hot Mass R < 2Rvir'),
                   'total_hot_cgm_gas_mass.png', keys=CGM_KEYS,
                   xlim=(3.2, 0.4), ylim=(0, 1.2))

    # ISM gas mass vs z
    plot_lines(
        yfunc=lambda t: t['total_ism_gas_mass'].to('Msun') / 1e10,
        ylabel='M_ISM / 1e10', title='ISM Mass vs. Feedback Scheme',
        filename='total_ism_gas_mass.png', keys=WITH_DEFAULT, ylim=(0, 4))

    # ISM HI and H2 mass vs z
    plot_ism_HI_H2(keys=WITH_DEFAULT)

    # Molecular mass fraction vs z
    plot_lines(
        yfunc=lambda t: t['total_ism_H2_mass'] / (t['total_ism_HI_mass'] + t['total_ism_H2_mass']),
        ylabel='Molecular Mass Fraction', title='Molecular Mass Fraction vs. Feedback Scheme',
        filename='molecular_fraction.png', keys=NO_DEFAULT, ylim=(0, 1))

    # ISM metallicity vs z
    plot_lines(
        yfunc=lambda t: np.log10(t['average_metallicity']),
        ylabel='Average Metallicity', title='ISM Metallicity vs. Feedback Scheme',
        filename='average_metallicity.png', keys=WITH_DEFAULT, ylim=None)

    # SFR7 vs stellar mass
    plot_lines(
        xfunc=lambda t: np.log10(t['total_star_mass'].to('Msun').value),
        yfunc=lambda t: np.log10(t['sfr7'].to('Msun/yr').value),
        xlabel='log Mstar [Msun]', ylabel='SFR [Msun/yr]',
        title='SFR and Outflows vs. Feedback Scheme', filename='SFR_vs_Mstar.png',
        keys=WITH_DEFAULT, xlim=(7.8, 11.9), ylim=(-1.4, 3.4))

    # Tacconi-style molecular-to-stellar ratio
    plot_tacconi(keys=WITH_DEFAULT)

    # Stellar mass ratio to thermal run
    plot_mstar_ratio(keys=NO_DEFAULT)

    # Baryon budgets, one figure per run. Each entry: (run key, title stem, file stem)
    BARYON_BUDGETS = [
        ('therm',       'Thermal Feedback',                'thermal'),
        ('mech',        'Mechanical Feedback',             'mechanical'),
        ('rad',         '100x Rad & Thermal Feedback',     'rad100'),
        ('mom5',        'Momemtum x5 Feedback',            'mom5x'),
        ('mom_rad',     'Momemtum x3 + Rad100 Feedback',   'mom3x_rad'),
        ('radius3',     'Momemtum x5 + Radius=3 ',         'mom5x_radius3'),
        ('rad3_rad100', 'Mom x5,Radius=3,Rad100 ',         'mom5x_radius3_rad100'),
        ('numerical',   'H2numerical nref11n ',            'H2numerical'),
        ('pr63',        'PR63 Radiation',                  'PR63'),
    ]

    for key, stem, fstem in BARYON_BUDGETS:
        # Rvir budget (unchanged)
        baryons_vs_z(BY_KEY[key].table, zrange=[3, 0.0],
                     title=f'Tempest Baryon Budget with {stem}',
                     filename=f'baryon_budget_{fstem}.png')
        # parallel 2Rvir budget (numerators summed within 2Rvir, normalized by Mvir)
        baryons_vs_z(BY_KEY[key].table, zrange=[3, 0.0], suffix='_2rvir',
                     title=f'Tempest 2Rvir Baryon Budget with {stem}',
                     filename=f'baryon_budget_{fstem}_2rvir.png')
