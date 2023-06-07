import numpy as np
from scipy.optimize import curve_fit


def Ring(xdata, *args):
    ''' Gaussian ring to estimate center
    
        == Params ==
        xdata : array
            Spatial coordinates formated from 2D meshgrid as:
            xdata[:,0] = xx.flatten()
            xdata[:,1] = yy.flatten()
        *args : np.array
            arguments for the ring (x_center, y_center, radius, amplitude, width)
            
        == Returns ==
        array
            Ring reshaped into a 1D array
    '''
    u = xdata[:, 0]
    v = xdata[:, 1]
    return args[3]*np.exp(-((np.sqrt((u-args[0])**2+(v-args[1])**2)-np.abs(args[2]))**2)/(2*args[4]))

def Paraboloid(xdata, *args):
    ''' Paraboloid function
    
        == Params ==
        xdata : array
            Spatial coordinates formated from 2D meshgrid as:
            xdata[:,0] = xx.flatten()
            xdata[:,1] = yy.flatten()
        *args : np.array
            arguments for the paraboloid (x_center, y_center, x_axis, y_axis, z_shift)
            
        == Returns ==
        array
            Paraboloid reshaped into a 1D array
    '''
    u = xdata[:, 0]
    v = xdata[:, 1]
    return -(u-args[0])**2/args[2]-(v-args[1])**2/args[3]+args[4]

def Shell_Func(x, y, x0, y0, q, shell_width):
    ''' Shell to mask data
    
        == Params ==
        x, y : array
            coordinate arrays in [mm]
        x0, y0 : scalar
            coordinate of the scattering center in [mm]
        q : array
            1D q array
        shell_width : scalar
            With of the mask shell
            
        == Returns ==
        array
            Mask in 2D
    '''
    dist_from_center = np.sqrt((x-x0)**2 + (y-y0)**2)
    mask_in = dist_from_center >= q - shell_width
    mask_out = dist_from_center <= q + shell_width
    return np.logical_and(mask_in,mask_out)

def ExtractCenter(xx,yy,ax_size,image,block):
    ''' Estimate the center of the scattering from static measurement
    
        == Params ==
        xx, yy : array
            meshgrid of coordinates in [mm]
        ax_size : array
            dimensions of image
        image : array
            data to fit
        block : scalar
            radius to block. If set to zero, then it is not computed.
            
        == Returns ==
        scalars
            x and y center positions [mm]
    '''
    # Guess
    x0 = 0
    y0 = 0
    q = 10
    A = 1
    sigma = 10
    p0 = np.array([x0, y0, q, A, sigma])
    
    # Allocate axis
    data = np.empty((ax_size[0]*ax_size[1],2))
    data[:,0] = xx.flatten()
    data[:,1] = yy.flatten()
    
    # Fit from maximum scattering
    # Flatten image and remove NaNs
    image2 = image.flatten()
    image2[np.isnan(image2)]=0

    # Fit model
    popt, pcov = curve_fit(Ring, data, image2, p0)
    
    # Fit by blocking a low-q signal
    if block != 0:
        # Block and fit again
        dist_from_center = np.sqrt((xx-popt[0])**2 + (yy-popt[1])**2)
        mask = dist_from_center >= block
        image3 = image*mask
        image3 = image3.flatten()
        image3[np.isnan(image3)]=0
        p0 = np.array([popt[0], popt[1], block*1.1, A, sigma])
        popt, pcov = curve_fit(Ring, data, image3, p0)
    
    # Compute error
    perr = np.sqrt(np.diag(pcov))

    # Return fitted center
    return popt[0], popt[1], perr[0], perr[1]

def Integrator(xx,yy,q,image,Error,ax_size,x0,y0,shell_width,angle_range,symmetry_flag):
    ''' Azimuthal average and integration
    
        == Params ==
        xx, yy : array
            meshgrid of coordinates in [mm]
        q : array
            1D q array
        image : array
            data to integate
        Error : array
            Error estimate based on pixel size and shift
        x0, y0 : scalar
            coordinate of the scattering center in [mm]
        shell_width : scalar
            With of the mask shell
        angle_range : array
            A range of angles to mask. First angle < Second angle
        symmetry_flag : flag
            1 for axially symmetric, 0 for just the angle range provided
            
        == Returns ==
        scalars
            x and y center positions [mm]
            x and y center positions error [mm]
    '''
    # Mask angle range
    if np.sum(angle_range) == 0:
        MaskImage = np.ones(ax_size)
    else:
        '''
        angle_range = angle_range + 1e-10*np.ones(np.size(angle_range))
        line1 = np.tan(np.pi*(angle_range[0])/180)*xx
        line2 = np.tan(np.pi*(angle_range[1])/180)*xx
        mask1 = yy >= line1
        mask2 = yy <= line2
        if symmetry_flag == 1:
            MaskImage = 1-np.logical_or(mask1,mask2)+np.logical_and(mask1,mask2)
        else:
            MaskImage = np.logical_and(mask1,mask2)
        '''
        phase_map = np.arctan2(yy,xx)
        phase_map = (phase_map + 2 * np.pi) % (2 * np.pi)
        angle_range = angle_range + (1e-10)*np.ones(np.size(angle_range))
        angle_range = angle_range * np.pi / 180
        mask1 = np.logical_and(phase_map >= angle_range[0]*np.ones(phase_map.shape),phase_map <= (angle_range[0]+np.pi)*np.ones(phase_map.shape))
        mask2 = np.logical_and(phase_map >= angle_range[1]*np.ones(phase_map.shape),phase_map <= (angle_range[1]+np.pi)*np.ones(phase_map.shape))
        if symmetry_flag == 1:
            MaskImage = np.logical_xor(mask1,mask2)
        else:
            MaskImage = np.logical_and(mask1,np.logical_not(mask2))
    image = image * MaskImage
    
    # Store NaN mask
    Mask = np.zeros(ax_size)
    Mask[np.isnan(image)]=1
    
    # Allocate Intensity and error
    Intensity = np.zeros(np.size(q))
    q_Error = np.zeros(np.size(q))

    # Loop the radius
    for idR,R in np.ndenumerate(q):
        # Define shell
        Shell = Shell_Func(xx, yy, x0, y0, R, shell_width)

        # Data
        Probe = image * Shell
        
        # Error
        Probe_error = Error * Shell

        # Mask
        Shell_mask = Mask * Shell

        # Compute average intensity
        Shell_sum = np.sum(Shell.flatten())
        Shell_mask_sum = np.sum(Shell_mask.flatten())
        Intensity[idR] = np.nansum(Probe.flatten()/(Shell_sum-Shell_mask_sum))
        q_Error[idR] = np.nansum(Probe_error.flatten()/Shell_sum)

    # Get integrated intensity
    return Intensity, Intensity*(2*np.pi*q), q_Error, image

def RefineCenter(qxx,qyy,q,image,ax_size,shell_width,det_dis,wavelen,block,xx,yy):
    ''' Estimate the error in the determination of the scattering center
    
        == Params ==
        qxx, qyy : array
            meshgrid of q vector [nm^-1]
        q : array
            1D q array
        image : array
            data to integate
        ax_size : array
            dimensions of image
        shell_width : scalar
            With of the mask shell
        det_dis : scalar
            Detector distance in [mm]
        wavelen : scalar
            wavelength in [nm]
        
            
        == Returns ==
        Average intensity : 1D array
        Integrated intensity : 1D array
        Error of pixel position in q : 1D array
    '''
    # Create a matrix of "centers" in mm
    r00 = np.arange(-2, 2, 0.5)

    # Allocate output
    Map = np.zeros((np.size(r00),np.size(r00)))
    
    if block != 0:
        # Block and fit again
        dist_from_center = np.sqrt(xx**2 + yy**2)
        mask = dist_from_center >= block
        image2 = image*mask
        image2[np.isnan(image2)]=0

    # Loop the centers
    for idX,X in np.ndenumerate(r00):
        for idY,Y in np.ndenumerate(r00):
            # Convert to q space
            qx00 = 2 * np.pi * np.sin(np.arctan(X / det_dis)) / wavelen
            qy00 = 2 * np.pi * np.sin(np.arctan(Y / det_dis)) / wavelen

            # Integrate intensity
            _, Intensity_Int_test, _, _ = Integrator(qxx,qyy,q,image2,np.zeros(ax_size),ax_size,qx00,qy00,shell_width,0,0)

            # Find maximum
            Map[idX,idY] = np.max(Intensity_Int_test)
            
    # Fit a paraboloid
    # Starting guess
    x0 = 0
    y0 = 0
    sx = 1
    sy = 1
    A = 0.1
    p0 = np.array([x0, y0, sx, sy, A])

    # Allocate axis
    x00, y00 = np.meshgrid(r00,r00)
    data = np.empty((np.size(r00)**2,2))
    data[:,0] = x00.flatten()
    data[:,1] = y00.flatten()

    # Flatten data
    Map2 = Map.flatten()

    # Fit model, first pass
    popt, pcov = curve_fit(Paraboloid, data, Map2, p0)
    perr = np.sqrt(np.diag(pcov))
    
    return popt[0], popt[1], perr[0], perr[1]