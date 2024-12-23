from astropy.io import fits
import numpy as np

root_directory = "./"

class beam:
    def __init__(self):
        self.xoffset=None
        self.yoffset=None
        
        self.sensitivity_response = None
        self.sensitivity_wavelength = None
        self.sensitivity_err = None

        self.dydx_0 = None
        self.dydx_1 = None
        
        self.dldp_0 = None
        self.dldp_1 = None
     
        self.dpdl_0 = None
        self.dpdl_1 = None
        
        self.dydx_order = None
        self.disp_order = None
        
        
    def load_filter_sensitivity(self,filename):

        filedir = root_directory + "conf/"+filename
    
        data = fits.open(filedir)
        sensitivity_data = data[1].data
        ns = np.size(sensitivity_data)
        
        self.sensitivity_response = np.zeros((ns))
        self.sensitivity_wavelength = np.zeros((ns))
        self.sensitivity_err = np.zeros((ns))
    
        for i in range(0,ns):
            self.sensitivity_wavelength[i] = sensitivity_data[i][0]
            self.sensitivity_response[i] = sensitivity_data[i][1]
            self.sensitivity_err[i] = sensitivity_data[i][2]
            
    def disp_x(self,i,j,dp):
        if self.dydx_1 is None: return dp #Zeroth moment
        b_10,b_11,b_12,b_13,b_14,b_15 = self.dydx_1
        b_1_ij = b_10 + b_11*i + b_12*j + b_13*i*i + b_14*i*j + b_15*j*j
        dx = dp/np.sqrt(1+b_1_ij*b_1_ij)
        return dx #TODO - IS THIS CORRECT?
        
    
    def disp_y(self,i,j,dx):
        b_00,b_01,b_02,b_03,b_04,b_05 = self.dydx_0
        if self.dydx_1 is not None:
            b_10,b_11,b_12,b_13,b_14,b_15 = self.dydx_1
            b_1_ij = b_10 + b_11*i + b_12*j + b_13*i*i + b_14*i*j + b_15*j*j
        else:
            b_1_ij = 0.0 #Zeroth Moment

        b_0_ij = b_00 + b_01*i + b_02*j + b_03*i*i + b_04*i*j + b_05*j*j

        dy = b_0_ij + dx*b_1_ij
        return dy
        
    
    
    def inv_disp_lambda(self,i,j,wavelength):
        beta_00,beta_01,beta_02,beta_03,beta_04,beta_05 = self.dldp_0
        beta_10,beta_11,beta_12,beta_13,beta_14,beta_15 = self.dldp_1

        beta_0_ij = beta_00 + beta_01*i + beta_02*j + beta_03*i*i + beta_04*i*j + beta_05*j*j
        beta_1_ij = beta_10 + beta_11*i + beta_12*j + beta_13*i*i + beta_14*i*j + beta_15*j*j


        dp = (wavelength - beta_0_ij) / beta_1_ij
        #return some dimensionless trace value
        return dp
    
    


class _instrument:
    def __init__(self,args):
         self.name = args.instrument
         self.filter = args.filter
         self.dfilter = args.dfilter

         self.xrange = None #left and right coordinates of FOV
         self.yrange = None #Lower and upper coordinates of FOV

         self.beam_a = beam() #1st order
         self.beam_b = beam() #0th order
         self.beam_c = beam() #2nd order
         self.beam_e = beam() #-1st order
         
         if args.instrument=="WFC3":
            self.radius = 2.4 #HST, meters
            self.gain = 1.5
            self.electrons_per_photon = 1;

            if self.filter == "G280": self.channel = "UVIS"
            elif self.filter == "G102": self.channel = "NIR"
            elif self.filter == "G141": self.channel = "NIR"
            else: self.channel="NIR"

            
            if self.channel=="NIR":
                self.pixel_size_arcsec = 0.13
            elif self.channel=="UVIS":
                self.pixel_size_arcsec = 0.04
                
            self.dark_current = 0.048 #electrons/pix/sec
            
            if self.dfilter in ["F105W","F098M","F127M","F126N","F128N","F130N","F132N"]:
                self.thermal_background = 0.051 #electrons/pix/sec
            elif self.dfilter in ["F110W","F125W","F139M"]:
                self.thermal_background = 0.052 #electrons/pix/sec
            elif self.dfilter in ["F140W"]:
                self.thermal_background = 0.070 #electrons/pix/sec
            elif self.dfilter in ["F160W"]:
                self.thermal_background = 0.134 #electrons/pix/sec
            elif self.dfilter in ["F153M"]:
                self.thermal_background = 0.060 #electrons/pix/sec
            elif self.dfilter in ["F164N"]:
                self.thermal_background = 0.065 #electrons/pix/sec
            elif self.dfilter in ["F167N"]:
                self.thermal_background = 0.071 #electrons/pix/sec
            else:
                self.thermal_background = 0.051 #default to minimum
                
                
            if self.dfilter=="F105W":
                self.zodiacal_background = 0.774 
                self.earthshine = 0.238
            else:
                self.zodiacal_background = 0.774
                self.earthshine = 0.238
                
                
            self.zodiacal_background *= args.pZodiacal
            self.earthshine *= args.pEarthshine
            self.sky_background = self.zodiacal_background+self.earthshine
            #print("Warning: Cannot identify instrument, setting aperture radius to 1 m")
         
         
         
        



    def load_conf_file(self,instrument,filter,dfilter):
        filedir=None
        if instrument=="WFC3":
            filedir = root_directory+"/conf/"+filter+"."+dfilter+".v4.32.conf"
        if filedir is None:
            print("Warning: could not find configuration file for:",instrument,filter,dfilter)
        
        with open(filedir, 'r') as file:
            for line in file:
                values = line.strip().split()
                try: param = values[0]
                except: continue
                
                if param=="YOFF_A":
                    self.beam_a.yoffset=float(values[1])
                if param=="XOFF_A":
                    self.beam_a.xoffset=float(values[1])
                if param=="DYDX_ORDER_A":
                    self.beam_a.dydx_order = float(values[1])
                if param=="DYDX_A_0":
                    self.beam_a.dydx_0=np.array(values[1:]).astype(float)
                if param=="DYDX_A_1":
                    self.beam_a.dydx_1=np.array(values[1:]).astype(float)
                if param=="DISP_ORDER_A":
                    self.beam_a.disp_order = float(values[1])
                if param=="DLDP_A_0":
                    self.beam_a.dldp_0=np.array(values[1:]).astype(float)
                if param=="DLDP_A_1":
                    self.beam_a.dldp_1=np.array(values[1:]).astype(float)
                if param=="DPDL_A_0":
                    self.beam_a.dpdl_0=np.array(values[1:]).astype(float)
                if param=="DPDL_A_1":
                    self.beam_a.dpdl_1=np.array(values[1:]).astype(float)
                if param=="SENSITIVITY_A":
                    self.beam_a.load_filter_sensitivity(values[1])
                    
                if param=="YOFF_B":
                    self.beam_b.yoffset=float(values[1])
                if param=="XOFF_B":
                    self.beam_b.xoffset=float(values[1])
                if param=="DYDX_ORDER_B":
                    self.beam_b.dydx_order = float(values[1])
                if param=="DYDX_B_0":
                    self.beam_b.dydx_0=np.array(values[1:]).astype(float)
                if param=="DYDX_B_1":
                    self.beam_b.dydx_1=np.array(values[1:]).astype(float)
                if param=="DISP_ORDER_B":
                    self.beam_b.disp_order = float(values[1])
                if param=="DLDP_B_0":
                    self.beam_b.dldp_0=np.array(values[1:]).astype(float)
                if param=="DLDP_B_1":
                    self.beam_b.dldp_1=np.array(values[1:]).astype(float)
                if param=="DPDL_B_0":
                    self.beam_b.dpdl_0=np.array(values[1:]).astype(float)
                if param=="DPDL_B_1":
                    self.beam_b.dpdl_1=np.array(values[1:]).astype(float)
                if param=="SENSITIVITY_B":
                    self.beam_b.load_filter_sensitivity(values[1])
                    
                if param=="YOFF_C":
                    self.beam_c.yoffset=float(values[1])
                if param=="XOFF_C":
                    self.beam_c.xoffset=float(values[1])
                if param=="DYDX_C_0":
                    self.beam_c.dydx_0=np.array(values[1:]).astype(float)
                if param=="DYDX_ORDER_C":
                    self.beam_c.dydx_order = float(values[1])
                if param=="DYDX_C_1":
                    self.beam_c.dydx_1=np.array(values[1:]).astype(float)
                if param=="DISP_ORDER_C":
                    self.beam_c.disp_order = float(values[1])
                if param=="DLDP_C_0":
                    self.beam_c.dldp_0=np.array(values[1:]).astype(float)
                if param=="DLDP_C_1":
                    self.beam_c.dldp_1=np.array(values[1:]).astype(float)
                if param=="DPDL_C_0":
                    self.beam_c.dpdl_0=np.array(values[1:]).astype(float)
                if param=="DPDL_C_1":
                    self.beam_c.dpdl_1=np.array(values[1:]).astype(float)
                if param=="SENSITIVITY_C":
                    self.beam_c.load_filter_sensitivity(values[1])
            
                if param=="YOFF_E":
                    self.beam_e.yoffset=float(values[1])
                if param=="XOFF_E":
                    self.beam_e.xoffset=float(values[1])
                if param=="DYDX_E_0":
                    self.beam_e.dydx_0=np.array(values[1:]).astype(float)
                if param=="DYDX_ORDER_E":
                    self.beam_e.dydx_order = float(values[1])
                if param=="DYDX_E_1":
                    self.beam_e.dydx_1=np.array(values[1:]).astype(float)
                if param=="DISP_ORDER_E":
                    self.beam_e.disp_order = float(values[1])
                if param=="DLDP_E_0":
                    self.beam_e.dldp_0=np.array(values[1:]).astype(float)
                if param=="DLDP_E_1":
                    self.beam_e.dldp_1=np.array(values[1:]).astype(float)
                if param=="DPDL_E_0":
                    self.beam_e.dpdl_0=np.array(values[1:]).astype(float)
                if param=="DPDL_E_1":
                    self.beam_e.dpdl_1=np.array(values[1:]).astype(float)
                if param=="SENSITIVITY_E":
                    self.beam_e.load_filter_sensitivity(values[1])
                    
                if param=="XRANGE": self.xrange = np.array([values[1],values[2]]).astype(float)
                if param=="YRANGE": self.yrange = np.array([values[1],values[2]]).astype(float)
                    
                    

def load_instrument(args):
    if args.lambda_min < 0 or args.lambda_max < 0:
        print("Setting bandwidth to default for",args.instrument,"with filter",args.filter)
        if args.instrument == "WFC3":
            if args.filter == "G280":
                args.lambda_min = 1900.
                args.lambda_max = 8000.
            elif args.filter == "G102":
                args.lambda_min = 8000.
                args.lambda_max = 11500.
            elif args.filter == "G141":
                args.lambda_min = 10750.
                args.lambda_max = 17000.
                
    if args.dispersion < 0:
        print("Setting dispersion to default for",args.instrument,"with filter",args.filter)
        if args.instrument == "WFC3":
            if args.filter == "G280":
                args.dispersion = 13.
            elif args.filter == "G102":
                args.dispersion = 24.5
            elif args.filter == "G141":
                args.dispersion = 46.5
                
                
    if args.xoffset is None or args.yoffset is None:
        print("Setting first order offsets to default for",args.instrument,"with filter",args.filter)
        if args.instrument == "WFC3":
            if args.filter == "G102":
                args.xoffset = 252.55
                args.yoffset = 4.2

            elif args.filter == "G141":
                args.xoffset = 187.6
                args.yoffset = 0.25   

    if args.effective_exposure is None: args.effective_exposure = args.exposure 
    
    instrument = _instrument(args)
    instrument.load_conf_file(args.instrument,args.filter,args.dfilter)
    
    return args, instrument

