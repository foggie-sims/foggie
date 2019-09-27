import astropy
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve_fft


plt.ioff()
plt.close('all')
data_dir = '/Users/rsimons/Dropbox/rcs_foggie/outputs/momentum_fits'
fig_dir = '/Users/rsimons/Dropbox/rcs_foggie/figures'

def add_at(ax, t, loc=2):
    fp = dict(size=10)
    _at = AnchoredText(t, loc=loc, prop=fp)
    ax.add_artist(_at)
    return _at

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, math.sqrt(variance))


class camera:
    ###This class has been adopted from Greg Snyder

    # The camera class performs the basic operations.
    # It takes as input 10 parameters from the CAMERAX-PARAMETERS HDUs created by Sunrise
    # The position and FOV units are in KPC
    # It returns an object containing these data plus methods for converting generic
    #   x,y,z coordinates from the simulation frame (in Physical kpc!!) into a camera-based coordinate system.
    # The camera coordinates are defined such that the axis ranges are [-1,1].
    #     The return coordinates can be modified to use a pixel-based grid instead, but this more generic function can be used for both the 
    #     CANDELized and perfect images (i.e., on the same axis extent, in matplotlib terms)
    # There is one remaining uncertainty -- the sense of the rows and columns in the stored images (and how they are read into a given language).
    #     In Python:pyfits/astropy, given simulation coordinates, the returned pixel values correspond to the location on the image given the following assumptions:
    #     The "imshow" command was run with origin='lower' and extent=(-1,1,-1,1)
    #     The images returned by pyfits from the broadband.fits or _candelized_noise.fits must be **TRANSPOSED** first
    #     Presumably there are other iterations of these two settings (or other ways to manipulate the images) that will be satisfactory (or better, if the reason they work is known).
    #      -Greg Snyder, 8/21/2014

    def __init__(self,x,y,z,dirx,diry,dirz,upx,upy,upz,fov):
        self.x=x
        self.y=y
        self.z=z
        self.dirx=dirx  #x3 unit vector w/ ref to lab frame
        self.diry=diry
        self.dirz=dirz
        self.upx=upx  #x2 unit vector
        self.upy=upy
        self.upz=upz
        self.fov = fov
        #These vectors are defined following the convention at http://en.wikipedia.org/wiki/Pinhole_camera_model
        self.x3vector = np.asarray([self.dirx,self.diry,self.dirz])
        self.x2vector = np.asarray([self.upx,self.upy,self.upz])
        self.x1vector = np.cross(self.x2vector,self.x3vector)
        self.x1vector = self.x1vector/np.linalg.norm(self.x1vector)

        # This is the heart of the matter.  The idea is to express the object's coordinates in the frame of the camera model defined in __init__.
        # Let the object's position expressed in the original frame be A, and the unit vectors i1, i2, i3 be those along the simulation axes.
        # Let the object's position defined in the camera's reference (without shifting the origin yet) be A'.
        # Then, with linear operators, A' = M A, where M is constructed by taking dot products of the camera's unit vectors i' with the original unit vectors i.
        # When the original frame is standard cartesian coords, this equation reduces to the algebra below.
    def express_in_camcoords(self,x,y,z):
        new_x = x*self.x1vector[0] + y*self.x1vector[1] + z*self.x1vector[2]
        new_y = x*self.x2vector[0] + y*self.x2vector[1] + z*self.x2vector[2]
        new_z = x*self.x3vector[0] + y*self.x3vector[1] + z*self.x3vector[2]
        return np.asarray([new_x,new_y,new_z])

    # vel is the velocity vector in the oringinal frame
    def xyzvel_to_los(self, vel):
        camvec=self.express_in_camcoords(vel[:,0], vel[:,1], vel[:,2])
        return camvec[2]

    #Wrapper that reconstructs the Sunrise pinhole camera model, expresses the object's position in the camera frame, and computes its position in the image plane.
    def xyz_to_pixelvals(self,x,y,z):
        camdist = (self.x**2 + self.y**2 + self.z**2)**0.5
        camvec = self.express_in_camcoords(x,y,z)
        #define focal length such that image values span from -1 to 1.
        f = camdist/(0.5*self.fov)
        #See guidance at http://en.wikipedia.org/wiki/Pinhole_camera_model
        y1 = (f/camdist)*camvec[0]*1.0
        y2 = (f/camdist)*camvec[1]*1.0

        return y1,y2

def set_camobj_from_hdu(camparhdu_1):
    #set_camobj_from_hdu:  Helper function taking a pyfits HDU, the hdu['CAMERAX-PARAMETERS'], and returning an initialized camera object.    

    camposx = camparhdu_1.header.get('CAMPOSX')
    camposy = camparhdu_1.header.get('CAMPOSY')
    camposz = camparhdu_1.header.get('CAMPOSZ')
    camdirx = camparhdu_1.header.get('CAMDIRX')
    camdiry = camparhdu_1.header.get('CAMDIRY')
    camdirz = camparhdu_1.header.get('CAMDIRZ')
    camupx = camparhdu_1.header.get('CAMUPX')
    camupy = camparhdu_1.header.get('CAMUPY')
    camupz = camparhdu_1.header.get('CAMUPZ')
    fov_kpc = camparhdu_1.header.get('linear_fov')
    camobj_1 = camera(camposx,camposy,camposz,camdirx,camdiry,camdirz,camupx,camupy,camupz,fov_kpc)
    return camobj_1



def make_heatmap(ax, y1_1, y2_1, min_x, max_x, min_y, max_y, bins_n, weights, good, vmn_scale, vmx_scale, cmap, kerns):
    if good:        
        y1_1 = y1_1[good]
        y2_1 = y2_1[good]
        weights = weights[good]
    l = where((y1_1 < max_x) & (y1_1 > max_x))[0]

    heatmap, xedges, yedges = np.histogram2d(y1_1, y2_1, bins=[linspace(min_x, max_x,bins_n), linspace(min_x, max_x,bins_n),], weights = weights)
    heatmap[heatmap <= 0.0] = 0.0001
    htmp_rav = heatmap.ravel()
    htmp_rav = htmp_rav[~isnan(htmp_rav)]
    sorted_heatmap = argsort(htmp_rav)
    srt_heatmap = htmp_rav[sorted_heatmap]
    vmn = log10(srt_heatmap[int(vmn_scale*len(srt_heatmap))])#*0.0001
    vmx = log10(srt_heatmap[int(vmx_scale*len(srt_heatmap))])#*1.
    #print vmn, vmx
    #vmx = 8.3
    #vmn = -4

    #vmn = -vmx
    #heatmap_asinh = img_scale.log(heatmap, scale_min=vmn, scale_max=vmx)
    #kern = Gaussian2DKernel(0.1)
    #heatmap = convolve_fft(heatmap, kern)
    kern = Gaussian2DKernel(kerns)
    heatmap = convolve_fft(heatmap, kern)

    heatmap = log10(heatmap)
    #heatmap[heatmap > (vmx)/2.] = (vmx)/2.1

    #cmap = matplotlib.cm.Blues_r
    #cmap = matplotlib.cm.jet

    cmap.set_bad('black',1.)
    msk_heatmap = np.ma.array(heatmap,   mask = np.isnan(heatmap))


    ax.imshow(heatmap, interpolation = 'nearest', origin = 'lower', vmin = vmn, vmax = vmx,cmap=cmap)


    ax.set_xticklabels([''])
    ax.set_yticklabels([''])
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.tick_params(axis="both", which='major', color='black', labelcolor='black',labelsize=10, size=5, width=1.5)
    ax.tick_params(axis="both", which='minor', color='black', labelcolor='black',labelsize=10, size=3, width=1.5)

    return ax





def make_star_maps(ax, x_pos, y_pos, z_pos, Lbol, age, cam_header, min_x, max_x, min_y, max_y, cmap, vmns, vmxs, kerns, n = False):




    x = cam_header['CAMPOSX']
    y = cam_header['CAMPOSY']
    z = cam_header['CAMPOSZ']
    dirx = cam_header['CAMDIRX']
    diry = cam_header['CAMDIRY']
    dirz = cam_header['CAMDIRZ']
    upx = cam_header['CAMUPX']
    upy = cam_header['CAMUPY']
    upz = cam_header['CAMUPZ']
    upz = cam_header['CAMUPZ']
    fov = cam_header['FOV']

    camobj_1 = camera(x,y,z,dirx,diry,dirz,upx,upy,upz,fov)



    #These are the projected coordinates. Their values are scaled to the field of view.
    y1_1,y2_1       = camobj_1.xyz_to_pixelvals(x_pos,y_pos,z_pos)

    #To get the projected coordinates in a physical unit
    y1_1 = y1_1 * camobj_1.fov
    y2_1 = y2_1 * camobj_1.fov


    #Close figures and turn off plot interactive mode
    
    #Make a plot only for the young stars (<1.e9)

    n_pix           = 4000.

    ax = make_heatmap(ax, y1_1, y2_1, min_x = min_x, max_x = max_x, min_y = min_y, max_y = max_y, bins_n = n_pix, weights = Lbol, 
                      good = where(age < 2.e100), vmn_scale = vmns, vmx_scale = vmxs, cmap = cmap, kerns = kerns)

    xlim_orig = ax.get_xlim()
    ylim_orig = ax.get_xlim()


    y = 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]) + ax.get_ylim()[0]
    xmid = 0.85*(ax.get_xlim()[1]-ax.get_xlim()[0]) + ax.get_xlim()[0]
    xstart = xmid - (5./(max_x - min_x))*ax.get_xlim()[1]-ax.get_xlim()[0]
    xend   = xmid + (5./(max_x - min_x))*ax.get_xlim()[1]-ax.get_xlim()[0]


    if n == True: ax.annotate("10 kpc", (0.90*xmid, y*1.5), fontsize = 15, fontweight = 'bold', color = 'grey')
    ax.plot([xstart, xend], [y, y],  linewidth = 6, color = 'grey')

    ax.set_xlim(xlim_orig)
    ax.set_ylim(ylim_orig)


    ylab = 0.92*(ax.get_ylim()[1]-ax.get_ylim()[0]) + ax.get_ylim()[0]
    xlab = 0.06*(ax.get_xlim()[1]-ax.get_xlim()[0]) + ax.get_xlim()[0]
    xlab2 = 0.70*(ax.get_xlim()[1]-ax.get_xlim()[0]) + ax.get_xlim()[0]


    #ax.annotate("young stars", (xlab, ylab), fontsize = 30, fontweight = 'bold', color = 'lightskyblue', alpha = 1.0)
    #ax.annotate("(age < 20 Myr)", (xlab, 0.95*ylab), fontsize = 20, fontweight = 'bold', color = 'lightskyblue', alpha = 1.0)
    #ax.annotate("young stars", (xlab, ylab), fontsize = 30, fontweight = 'bold', color = 'grey', alpha = 1.0)
    #ax.annotate("(age < 20 Myr)", (xlab, 0.95*ylab), fontsize = 20, fontweight = 'bold', color = 'grey', alpha = 1.0)



    #ax.annotate("z = 1.5", (xlab2, 0.98*ylab), fontsize = 45, fontweight = 'bold', color = 'white', alpha = 1.0)


    return ax












DD = 900.


if True:
    simname = 'nref11n_selfshield_z15'
    #fitsname = 'nref11n_nref10f_selfshield_z6_DD%.4i_momentum.fits'%DD
    fitsname = '%s_DD%.4i_momentum.fits'%(simname, DD)

    mom_data = fits.open(data_dir + '/' + fitsname)





fig, axes = plt.subplots(1,2, figsize = (8, 4))



for ax in axes:
    ax.axis('off')


axes[0].annotate("Dark Matter", (0.5, 0.9), xycoords = 'axes fraction', fontweight = 'bold', color = 'white', fontsize = 20, ha = 'center', va = 'center')
axes[1].annotate("Stars", (0.5, 0.9), xycoords = 'axes fraction', fontweight = 'bold', color = 'white', fontsize = 20, ha = 'center', va = 'center')




cam_header = {}





for i in arange(1, 15):
    cam_header['CAMPOSX'] = 0.
    cam_header['CAMPOSY'] = 1*cos(pi*i/25.)
    cam_header['CAMPOSZ'] = 1*sin(pi*i/25.)
    cam_header['CAMDIRX'] = 0.
    cam_header['CAMDIRY'] = -cam_header['CAMPOSY']
    cam_header['CAMDIRZ'] = -cam_header['CAMPOSZ']
    cam_header['CAMUPX'] = 1.
    cam_header['CAMUPY'] = 0.
    cam_header['CAMUPZ'] = 0.
    cam_header['FOV'] = 100.



    sx_pos = mom_data['STARS_GAL_POSITION'].data[0]
    sy_pos = mom_data['STARS_GAL_POSITION'].data[1]
    sz_pos = mom_data['STARS_GAL_POSITION'].data[2]
    smass = mom_data['STAR_MASS'].data
    sage = mom_data['STAR_AGE'].data


    dx_pos = mom_data['DARK_GAL_POSITION'].data[0]
    dy_pos = mom_data['DARK_GAL_POSITION'].data[1]
    dz_pos = mom_data['DARK_GAL_POSITION'].data[2]
    dmass = mom_data['DARK_MASS'].data
    dage = mom_data['DARK_AGE'].data


    m = 75

    axes[0] = make_star_maps(ax = axes[0], x_pos = sx_pos, y_pos = sy_pos, z_pos = sz_pos, Lbol = smass, age = sage, cam_header = cam_header, min_x = -m, max_x = m, min_y = -m, max_y = m, cmap = cm.gist_heat, vmns = 0.70, vmxs = 0.9999, kerns = 0.5, n = True)
    axes[1] = make_star_maps(ax = axes[1], x_pos = dx_pos, y_pos = dy_pos, z_pos = dz_pos, Lbol = dmass, age = dage, cam_header = cam_header, min_x = -m, max_x = m, min_y = -m, max_y = m, cmap = cm.Blues_r, vmns = 0.70, vmxs = 0.9999999, kerns = 0.5)



    fig.subplots_adjust(left = 0, right = 1.0, top = 1.0, bottom = 0.0, hspace = 0, wspace = 0.0)

    fig.savefig(fig_dir + '/%s_%.4i_stars_dm_%i.png'%(simname, DD, i), dpi = 300.)


    plt.close('all')
































