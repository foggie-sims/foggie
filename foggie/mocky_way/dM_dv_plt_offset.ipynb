{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import sys\n",
    "from astropy.io import fits\n",
    "from astropy.table import Table\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib as mpl\n",
    "mpl.rcParams['font.family'] = 'stixgeneral'\n",
    "\n",
    "sim_name = 'nref11n_nref10f'\n",
    "dd_name = 'DD2175'\n",
    "\n",
    "# get dM/dv\n",
    "file_gc = 'figs/dM_dv/fits/nref11n_nref10f_DD2175_dMdv_cgm_halo_center.fits'\n",
    "data_gc = Table.read(file_gc, format='fits')\n",
    "dv_bins_gc = data_gc['v (km/s)']\n",
    "dM_all_gc = data_gc['dM (Msun/km/s)']\n",
    "dM_cold_gc = data_gc['dM_cold (Msun/km/s)']\n",
    "dM_cool_gc = data_gc['dM_cool (Msun/km/s)']\n",
    "dM_warm_gc = data_gc['dM_warm (Msun/km/s)']\n",
    "dM_hot_gc = data_gc['dM_hot (Msun/km/s)']\n",
    "\n",
    "# get dM/dv wrt to observer \n",
    "file_sun = 'figs/dM_dv/fits/nref11n_nref10f_DD2175_dMdv_cgm_offcenter_location.fits'\n",
    "data_sun = Table.read(file_sun, format='fits')\n",
    "dv_bins_sun = data_sun['v (km/s)']\n",
    "dM_all_sun = data_sun['dM (Msun/km/s)']\n",
    "dM_cold_sun = data_sun['dM_cold (Msun/km/s)']\n",
    "dM_cool_sun = data_sun['dM_cool (Msun/km/s)']\n",
    "dM_warm_sun = data_sun['dM_warm (Msun/km/s)']\n",
    "dM_hot_sun = data_sun['dM_hot (Msun/km/s)']\n",
    "\n",
    "# figure out the offset per phase \n",
    "offset_cold = dM_cold_sun - dM_cold_gc\n",
    "offset_cool = dM_cool_sun - dM_cool_gc \n",
    "offset_warm = dM_warm_sun - dM_warm_gc \n",
    "offset_hot = dM_hot_sun - dM_hot_gc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [],
   "source": [
    "fs=14\n",
    "from foggie.utils import consistency\n",
    "cmap = consistency.temperature_discrete_cmap\n",
    "c_all = plt.cm.Greys(0.7)\n",
    "c_cold = cmap(0.05)\n",
    "c_cool = cmap(0.25)\n",
    "c_warm = cmap(0.6)\n",
    "c_hot = cmap(0.9)\n",
    "\n",
    "sun_ls = '-'\n",
    "sun_lw = 3\n",
    "\n",
    "gc_ls = '--'\n",
    "gc_lw = 2\n",
    "\n",
    "vmin = -400\n",
    "vmax = 400\n",
    "\n",
    "fs = 14 \n",
    "\n",
    "ymin1 = 2e3\n",
    "ymax1 = 8e8\n",
    "\n",
    "ymin2 = -8e8 \n",
    "ymax2 = 8e8 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"\\nvel = dv_bins_sun \\nsun_flux = dM_hot_sun \\ngc_flux = dM_hot_gc \\ncolor = c_hot\\noffset_flux = offset_hot\\ntag = 'hot'\\n\""
      ]
     },
     "execution_count": 123,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''\n",
    "vel = dv_bins_sun \n",
    "sun_flux = dM_cold_sun \n",
    "gc_flux = dM_cold_gc \n",
    "color = c_cold \n",
    "offset_flux = offset_cold\n",
    "tag = 'cold'\n",
    "'''\n",
    "\n",
    "'''\n",
    "vel = dv_bins_sun \n",
    "sun_flux = dM_cool_sun \n",
    "gc_flux = dM_cool_gc \n",
    "color = c_cool \n",
    "offset_flux = offset_cool\n",
    "tag = 'cool'\n",
    "'''\n",
    "\n",
    "vel = dv_bins_sun \n",
    "sun_flux = dM_warm_sun \n",
    "gc_flux = dM_warm_gc \n",
    "color = c_warm \n",
    "offset_flux = offset_warm\n",
    "tag = 'warm'\n",
    "\n",
    "'''\n",
    "vel = dv_bins_sun \n",
    "sun_flux = dM_hot_sun \n",
    "gc_flux = dM_hot_gc \n",
    "color = c_hot\n",
    "offset_flux = offset_hot\n",
    "tag = 'hot'\n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize=(4, 6))\n",
    "ax1 = fig.add_axes([0.2, 0.57, 0.7, 0.4])\n",
    "ax2 = fig.add_axes([0.2, 0.12, 0.7, 0.43])\n",
    "\n",
    "ax1.step(vel, sun_flux, color=color, linestyle=sun_ls, linewidth=sun_lw, label='wrt gc')\n",
    "ax1.step(vel, gc_flux, color=color, linestyle=gc_ls, linewidth=gc_lw, label='wrt sun')\n",
    "ax1.set_yscale('log')\n",
    "ax1.set_xlim(vmin, vmax)\n",
    "ax1.set_xticklabels([])\n",
    "ax1.set_ylim(ymin1, ymax1)\n",
    "ax1.minorticks_on()\n",
    "ax1.set_ylabel(r'dM/dv  [M$_\\odot$/(km s$^{-1}$)]', fontsize=fs)\n",
    "ax1.legend(fontsize=fs-2)\n",
    "ax1.fill_between([-100, 100], ymin1, ymax1, color=plt.cm.Greys(0.6), alpha=0.4)\n",
    "\n",
    "ax2.step(vel, offset_flux, color=color)\n",
    "# ax2.set_ylim(ymin, ymax)\n",
    "ax2.set_yscale('symlog')\n",
    "ax2.set_yticks([-1e8, -1e7, -1e6, -1e5, -1e4, -1e3, -1e2, \n",
    "                0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])\n",
    "ax2.set_yticklabels([r'-10$^8$', '', r'-10$^6$', '', r'-10$^4$', '', r'-10$^2$', 0, \n",
    "                     r'10$^2$', '', r'10$^4$', '', r'10$^6$', '',  r'10$^8$'])\n",
    "ax2.set_ylim(ymin2, ymax2)\n",
    "ax2.set_xlabel(r'Velocity (km s$^{-1}$)', fontsize=fs)\n",
    "ax2.set_ylabel(r'Flux Offset (sun - gc)', fontsize=fs)\n",
    "ax2.fill_between([-100, 100], ymin2, ymax2, color=plt.cm.Greys(0.6), alpha=0.4)\n",
    "\n",
    "for ax in [ax1, ax2]: \n",
    "    ax.minorticks_on()\n",
    "    ax.set_xlim(vmin, vmax)\n",
    "    ax.grid(linestyle=':', alpha=0.7, color=plt.cm.Greys(0.8))\n",
    "    for tick in ax.xaxis.get_major_ticks(): \n",
    "        tick.label.set_fontsize(fs)\n",
    "    for tick in ax.yaxis.get_major_ticks():\n",
    "        tick.label.set_fontsize(fs)\n",
    "        \n",
    "fig.savefig('figs/dM_dv/nref11n_nref10f_DD2175_dMdv_cgm_offset_%s.pdf'%(tag))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
