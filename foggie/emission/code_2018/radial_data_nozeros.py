def radial_data(data,annulus_width=1,working_mask=None,x=None,y=None,rmax=None):
    """
    r = radial_data(data,annulus_width,working_mask,x,y)
    
    A function to reduce an image to a radial cross-section.
    
    INPUT:
    ------
    data   - whatever data you are radially averaging.  Data is
            binned into a series of annuli of width 'annulus_width'
            pixels.
    annulus_width - width of each annulus.  Default is 1.
    working_mask - array of same size as 'data', with zeros at
                      whichever 'data' points you don't want included
                      in the radial data computations.
      x,y - coordinate system in which the data exists (used to set
             the center of the data).  By default, these are set to
             integer meshgrids
      rmax -- maximum radial value over which to compute statistics
    
     OUTPUT:
     -------
      r - a data structure containing the following
                   statistics, computed across each annulus:
          .r      - the radial coordinate used (outer edge of annulus)
          .mean   - mean of the data in the annulus
          .std    - standard deviation of the data in the annulus
          .median - median value in the annulus
          .max    - maximum value in the annulus
          .min    - minimum value in the annulus
          .numel  - number of elements in the annulus
	  .q25    - 25th quartile for the annulus
	  .q75    - 75th quartile for the annulus
    """
    
# 2010-03-10 19:22 IJC: Ported to python from Matlab
# 2005/12/19 Added 'working_region' option (IJC)
# 2005/12/15 Switched order of outputs (IJC)
# 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
# 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory
 
    import numpy as np

    class radialDat:
        """Empty object container.
        """
        def __init__(self): 
	    self.q75 = None
	    self.q25 = None
            self.mean = None
            self.std = None
            self.median = None
            self.numel = None
            self.max = None
            self.min = None
            self.r = None
	    self.fractionAbove = None
	    self.fractionAboveidx = None
    #---------------------
    # Set up input parameters
    #---------------------
    data = np.array(data)
    
    #if working_mask==None:
    if working_mask is  None:
        working_mask = np.ones(data.shape,bool)
    
    npix, npiy = data.shape
    if x is None or y is None:
        x1 = np.arange(-npix/2.,npix/2.)
        y1 = np.arange(-npiy/2.,npiy/2.)
        x,y = np.meshgrid(y1,x1)

    r = abs(x+1j*y)

    if rmax is None:
        rmax = r[working_mask].max()

    #---------------------
    # Prepare the data container
    #---------------------
    dr = np.abs([x[0,0] - x[0,1]]) * annulus_width
    radial = np.arange(rmax/dr)*dr + dr/2.
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.q25 = np.zeros(nrad)
    radialdata.q75 = np.zeros(nrad)
    radialdata.mean = np.zeros(nrad)
    radialdata.std = np.zeros(nrad)
    radialdata.median = np.zeros(nrad)
    radialdata.numel = np.zeros(nrad)
    radialdata.max = np.zeros(nrad)
    radialdata.min = np.zeros(nrad)
    radialdata.r = radial
    radialdata.fractionAboveidx = []
    radialdata.fractionAbove = np.zeros(nrad)
    
    #---------------------
    # Loop through the bins
    #---------------------
    for irad in range(nrad): #= 1:numel(radial)
      minrad = irad*dr
      maxrad = minrad + dr
      thisindex = (r>=minrad) * (r<maxrad) * working_mask
      if not thisindex.ravel().any():
	radialdata.q25[irad] = np.nan
	radialdata.q75[irad] = np.nan
        radialdata.mean[irad] = np.nan
        radialdata.std[irad]  = np.nan
        radialdata.median[irad] = np.nan
        radialdata.numel[irad] = np.nan
        radialdata.max[irad] = np.nan
        radialdata.min[irad] = np.nan
	radialdata.fractionAbove[irad] = np.nan
      else:
        datanow = data[thisindex]
	idx = np.isinf(datanow)
	idx = [not i for i in idx]
        if len(idx) > 0:
		datanow = datanow[idx]
		#print 'DELETE ALL THE THINGS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
	if len(datanow) != 0:
		radialdata.q25[irad] = np.percentile(datanow,25)
		radialdata.q75[irad] = np.percentile(datanow,75)
        	radialdata.mean[irad] = datanow.mean()
        	radialdata.std[irad]  = datanow.std()
        	radialdata.median[irad] = np.median(datanow)
        	radialdata.numel[irad] = datanow.size
        	radialdata.max[irad] = datanow.max()
        	radialdata.min[irad] = datanow.min()
    		radialdata.fractionAbove[irad] = (len(np.where(datanow > datanow.mean())[0])/float(len(datanow)))
    	else:
		#print 'ALL ZEROS 000000000000000000000000000000000000000000000000000000'
		radialdata.q25[irad] = np.nan
        	radialdata.q75[irad] = np.nan
        	radialdata.mean[irad] = np.nan
       		radialdata.std[irad]  = np.nan
        	radialdata.median[irad] = np.nan
        	radialdata.numel[irad] = np.nan
        	radialdata.max[irad] = np.nan
        	radialdata.min[irad] = np.nan
        	radialdata.fractionAbove[irad] = np.nan
    #---------------------
    # Return with data
    #---------------------
    
    return radialdata
