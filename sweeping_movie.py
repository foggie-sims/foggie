import astropy
from astropy.io import fits
import yt
from yt import YTArray
plt.ioff()
plt.close('all')
if False:
    print 'Loading nref11n_selfshield_z15...'
    ds_1 = yt.load('~/Dropbox/rcs_foggie/data/halo_008508/nref11n_selfshield_z15/RD0018/RD0018')

    print 'Loading nref11n_nref10f_selfshield_z6...'
    ds_2 = yt.load('~/Dropbox/rcs_foggie/data/halo_008508/nref11n_nref10f_selfshield_z6/RD0018/RD0018')
    ad_1 = ds_1.all_data()
    ad_2 = ds_2.all_data()




def find_center(ad):
    x = histogram(ad['particle_position_x'].value, bins = linspace(0.45, 0.55, 10000))
    good_argmax_x = argmax(x[0])
    x_cen = (x[1][good_argmax_x] + x[1][good_argmax_x+1])/2.

    y = histogram(ad['particle_position_y'].value, bins = linspace(0.45, 0.55, 20000))
    good_argmax_y = argmax(y[0])
    y_cen = (y[1][good_argmax_y] + y[1][good_argmax_y+1])/2.

    z = histogram(ad['particle_position_z'].value, bins = linspace(0.45, 0.55, 10000))
    good_argmax_z = argmax(z[0])
    z_cen = (z[1][good_argmax_z] + z[1][good_argmax_z+1])/2.



    return [x_cen,y_cen,z_cen]

if True:
    cen_1 = find_center(ad_1)
    cen_2 = find_center(ad_2)


N = 400

zlim_max = log10(1.e-4)
zlim_min = log10(10.)

for i in arange(1, 2):
    print i, 1.*cos(pi*(i)/100.),1*sin(pi*(i)/100.)

    #L = [max(1*cos(pi*(i)/100.), 1.e-1),0, max(1*sin(pi*(i)/100.), 1.e-1)] # vector normal to cutting plane


    L =[-0.37085436,  0.14802026,  0.91681898]
    L = [ 0.8390849 , -0.46594005,  0.2807782 ]

    north_vector = [0,1,0]
    
    W_kpc_initial = 100
    W_kpc_final = 40

    if i > N/2.:
        cn = 1.*(i - N/2.)
        x_w = max(W_kpc_initial - cn, W_kpc_final)
        y_w = max(W_kpc_initial - cn, W_kpc_final)
        z_w = max(W_kpc_initial - cn, W_kpc_final)
    else:
        x_w = W_kpc_initial
        y_w = W_kpc_initial
        z_w = W_kpc_initial


    W = YTArray([x_w, y_w, z_w], 'kpc')

    print W, L

    N = 512
    image1 = yt.off_axis_projection(ds_1, cen_1, L, W, N, ('gas', 'density'), north_vector =  north_vector)
    image2 = yt.off_axis_projection(ds_2, cen_2, L, W, N, ('gas', 'density'), north_vector =  north_vector)
    #image1 = yt.off_axis_projection(ds_1, cen_1, L, W, N, "density", north_vector =  north_vector)
    #image2 = yt.off_axis_projection(ds_2, cen_2, L, W, N, "density", north_vector =  north_vector)
    #im1 = yt.write_image(np.log10(image1), "../figures/foggie/movie/temp/im1_%i_offaxis_projection.png" % i)  
    #im2 = yt.write_image(np.log10(image2), "../figures/foggie/movie/temp/im2_%i_offaxis_projection.png" % i)  
    fig, axes = subplots(1,2, figsize = (10.8, 5))


    image1 = image1.in_units('Msun * kpc**-2')
    image2 = image2.in_units('Msun * kpc**-2')

    im1 = axes[0].imshow(np.log10(image1), vmin = 4.5, vmax = 9.5)
    im2 = axes[1].imshow(np.log10(image2), vmin = 4.5, vmax = 9.5)
    
    bar_len_kpc = 10.
    bar_len_pix = 1.*N/x_w * bar_len_kpc

    y_bar_start_pix = 0.5*N
    y_bar_end_pix = y_bar_start_pix + bar_len_pix

    x_bar_pix = 0.9*N


    axes[0].plot([x_bar_pix, x_bar_pix], [y_bar_start_pix, y_bar_end_pix], color = 'white', linewidth = 3)
    axes[0].annotate("10 kpc", (x_bar_pix-0.06*N, y_bar_start_pix- 0.02*N), color = 'white', fontsize = 15, rotation = 0)


    axes[0].annotate('Halo 008508\nz = 2.56', (0.07, 0.85), xycoords = 'axes fraction', color = 'white', fontweight = 'bold', fontsize = 20)
    axes[0].annotate('nref11n_selfshield_z15', (0.07, 0.03), xycoords = 'axes fraction', color = 'white', fontweight = 'bold', fontsize = 18)
    axes[1].annotate('nref11n_nref10f_selfshield_z6', (0.07, 0.03), xycoords = 'axes fraction', color = 'white', fontweight = 'bold', fontsize = 18)


    for ax in axes:
        ax.axis('off')

    cax = fig.add_axes([0.915, 0.0, 0.02, 1.0])
    cbr = fig.colorbar(im1, cax=cax,orientation="vertical")
    cbr.set_label('projected gas density (M$_{\odot}$ kpc$^{-2}$)', fontsize = 15)
    
    fig.subplots_adjust(left = 0.0, right = 0.92, top =1.0, bottom = 0.0, hspace = 0.0, wspace = 0.0)
    fig.savefig("../figures/foggie/movie/comb_%i_offaxis_projection.png"%i, dpi = 300)

    plt.close('all')


    '''
    '''