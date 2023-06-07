import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Bimodal
def LinLorentzian_Bimodal(xdata, *args):
    return (args[0]+args[3]/((xdata-args[1])**2/(args[2])**2+1)\
                   +args[6]/((xdata-args[4])**2/(args[5])**2+1))**2
            
def Lorentzian_Bimodal(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2])**2+1)\
                           +args[6]/((xdata-args[4])**2/(args[5])**2+1))

def LinAsymLorentzian(xdata, *args):
    return (args[0]+args[3]/((xdata-args[1])**2/(args[2]/(1+np.exp(args[4]*(xdata-args[1]))))**2+1))**2
            
            
#Lorentzian
def LinLorentzian_1(xdata, *args):
    return (args[0]+args[3]/((xdata-args[1])**2/(args[2])**2+1))**2

def LinLorentzian_3(xdata, *args):
    return (args[0]+args[3]/((xdata-args[1])**2/(args[2])**2+1)\
                   +args[4]/((xdata-3*args[1])**2/(3*args[2])**2+1))**2

def LinLorentzian_5(xdata, *args):
    return (args[0]+args[3]/((xdata-args[1])**2/(args[2])**2+1)\
                   +args[4]/((xdata-3*args[1])**2/(3*args[2])**2+1)\
                   +args[5]/((xdata-5*args[1])**2/(5*args[2])**2+1))**2

def Lorentzian_1(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2])**2+1))

def Lorentzian_3(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2])**2+1)\
                           +args[4]/((xdata-3*args[1])**2/(3*args[2])**2+1))

def Lorentzian_5(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2])**2+1)\
                           +args[4]/((xdata-3*args[1])**2/(3*args[2])**2+1)\
                           +args[5]/((xdata-5*args[1])**2/(5*args[2])**2+1))

def Lorentzian_1_FormFac(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2])**2+1))\
           -(2*xdata/args[4])

def Lorentzian_3_FormFac(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2])**2+1)\
                           +args[5]/((xdata-3*args[1])**2/(3*args[2])**2+1))\
           -(2*xdata/args[4])

def Lorentzian_5_FormFac(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2])**2+1)\
                           +args[5]/((xdata-3*args[1])**2/(3*args[2])**2+1)\
                           +args[6]/((xdata-5*args[1])**2/(5*args[2])**2+1))\
           -(2*xdata/args[4])

def Lorentzian_3_bkg(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2])**2+1)\
                           +args[4]/((xdata-3*args[1])**2/(3*args[2])**2+1))+args[5]


# Assymetric Lorentzian
def AsymLorentzian_1(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2]/(1+np.exp(args[4]*(xdata-args[1]))))**2+1))

def AsymLorentzian_3(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2]/(1+np.exp(args[4]*(xdata-args[1]))))**2+1)\
                           +args[5]/((xdata-3*args[1])**2/(3*args[2]/(1+np.exp(args[4]*(xdata-3*args[1]))))**2+1))

def AsymLorentzian_5(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2]/(1+np.exp(args[4]*(xdata-args[1]))))**2+1)\
                           +args[5]/((xdata-3*args[1])**2/(3*args[2]/(1+np.exp(args[4]*(xdata-3*args[1]))))**2+1)\
                           +args[6]/((xdata-5*args[1])**2/(5*args[2]/(1+np.exp(args[4]*(xdata-5*args[1]))))**2+1))

def AsymLorentzian_1_FormFac(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2]/(1+np.exp(args[4]*(xdata-args[1]))))**2+1))\
           -(2*xdata/args[5])

def AsymLorentzian_3_FormFac(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2]/(1+np.exp(args[4]*(xdata-args[1]))))**2+1)\
                           +args[6]/((xdata-3*args[1])**2/(3*args[2]/(1+np.exp(args[4]*(xdata-3*args[1]))))**2+1))\
           -(2*xdata/args[5])

def AsymLorentzian_5_FormFac(xdata, *args):
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2]/(1+np.exp(args[4]*(xdata-args[1]))))**2+1)\
                           +args[6]/((xdata-3*args[1])**2/(3*args[2]/(1+np.exp(args[4]*(xdata-3*args[1]))))**2+1)\
                           +args[7]/((xdata-5*args[1])**2/(5*args[2]/(1+np.exp(args[4]*(xdata-5*args[1]))))**2+1))\
           -(2*xdata/args[5])

# Gaussian
def Gaussian_1(xdata, *args):
    return 2*np.log(args[0]+args[3]*np.exp(-(xdata-args[1])**2/(2*args[2]**2)))

def Gaussian_3(xdata, *args):
    return 2*np.log(args[0]+args[3]*np.exp(-(xdata-args[1])**2/(2*args[2]**2))\
                           +args[4]*np.exp(-(xdata-3*args[1])**2/(2*(3*args[2])**2)))

def Gaussian_5(xdata, *args):
    return 2*np.log(args[0]+args[3]*np.exp(-(xdata-args[1])**2/(2*args[2]**2))\
                           +args[4]*np.exp(-(xdata-3*args[1])**2/(2*(3*args[2])**2))\
                           +args[5]*np.exp(-(xdata-5*args[1])**2/(2*(5*args[2])**2)))

def Gaussian_1_FormFac(xdata, *args):
    return 2*np.log(args[0]+args[3]*np.exp(-(xdata-args[1])**2/(2*args[2]**2)))\
           -(2*xdata/args[4])

def Gaussian_3_FormFac(xdata, *args):
    return 2*np.log(args[0]+args[3]*np.exp(-(xdata-args[1])**2/(2*args[2]**2))\
                           +args[5]*np.exp(-(xdata-3*args[1])**2/(2*(3*args[2])**2)))\
           -(2*xdata/args[4])

def Gaussian_5_FormFac(xdata, *args):
    return 2*np.log(args[0]+args[3]*np.exp(-(xdata-args[1])**2/(2*args[2]**2))\
                           +args[5]*np.exp(-(xdata-3*args[1])**2/(2*(3*args[2])**2))\
                           +args[6]*np.exp(-(xdata-5*args[1])**2/(2*(5*args[2])**2)))\
           -(2*xdata/args[4])

# Voigt
def Voigt_1(xdata, *args):
    cl = (0.68188 + 0.61239*args[4] - 0.18384*args[4]**2 - 0.11568*args[4]**3)
    cg = (0.32460 - 0.61825*args[4] + 0.17681*args[4]**2 + 0.12109*args[4]**3)
    return 2*np.log(args[0]+( args[5]*cl/((xdata-args[1])**2/(args[2])**2+1)\
                             +args[6]*cg*np.exp(-(xdata-args[1])**2/(2*args[3]**2))))

def Voigt_3(xdata, *args):
    cl = (0.68188 + 0.61239*args[4] - 0.18384*args[4]**2 - 0.11568*args[4]**3)
    cg = (0.32460 - 0.61825*args[4] + 0.17681*args[4]**2 + 0.12109*args[4]**3)
    return 2*np.log(args[0]+( args[5]*cl/((xdata-args[1])**2/(args[2])**2+1)\
                             +args[6]*cg*np.exp(-(xdata-args[1])**2/(2*args[3]**2)))\
                             +args[7]*cl/((xdata-3*args[1])**2/(3*args[2])**2+1)\
                             +args[8]*cg*np.exp(-(xdata-3*args[1])**2/(2*(3*args[3])**2)))

def Voigt_5(xdata, *args):
    cl = (0.68188 + 0.61239*args[4] - 0.18384*args[4]**2 - 0.11568*args[4]**3)
    cg = (0.32460 - 0.61825*args[4] + 0.17681*args[4]**2 + 0.12109*args[4]**3)
    return 2*np.log(args[0]+( args[5]*cl/((xdata-args[1])**2/(args[2])**2+1)\
                             +args[6]*cg*np.exp(-(xdata-args[1])**2/(2*args[3]**2)))\
                             +args[7]*cl/((xdata-3*args[1])**2/(3*args[2])**2+1)\
                             +args[8]*cg*np.exp(-(xdata-3*args[1])**2/(2*(3*args[3])**2))\
                             +args[9]*cl/((xdata-5*args[1])**2/(5*args[2])**2+1)\
                             +args[10]*cg*np.exp(-(xdata-5*args[1])**2/(2*(5*args[3])**2)))

def Voigt_1_FormFac(xdata, *args):
    cl = (0.68188 + 0.61239*args[4] - 0.18384*args[4]**2 - 0.11568*args[4]**3)
    cg = (0.32460 - 0.61825*args[4] + 0.17681*args[4]**2 + 0.12109*args[4]**3)
    return 2*np.log(args[0]+( args[6]*cl/((xdata-args[1])**2/(args[2])**2+1)\
                             +args[7]*cg*np.exp(-(xdata-args[1])**2/(2*args[3]**2))))\
           -(2*xdata/args[5])

def Voigt_3_FormFac(xdata, *args):
    cl = (0.68188 + 0.61239*args[4] - 0.18384*args[4]**2 - 0.11568*args[4]**3)
    cg = (0.32460 - 0.61825*args[4] + 0.17681*args[4]**2 + 0.12109*args[4]**3)
    return 2*np.log(args[0]+( args[6]*cl/((xdata-args[1])**2/(args[2])**2+1)\
                             +args[7]*cg*np.exp(-(xdata-args[1])**2/(2*args[3]**2)))\
                             +args[8]*cl/((xdata-3*args[1])**2/(3*args[2])**2+1)\
                             +args[9]*cg*np.exp(-(xdata-3*args[1])**2/(2*(3*args[3])**2)))\
           -(2*xdata/args[5])

def Voigt_5_FormFac(xdata, *args):
    cl = (0.68188 + 0.61239*args[4] - 0.18384*args[4]**2 - 0.11568*args[4]**3)
    cg = (0.32460 - 0.61825*args[4] + 0.17681*args[4]**2 + 0.12109*args[4]**3)
    return 2*np.log(args[0]+( args[6]*cl/((xdata-args[1])**2/(args[2])**2+1)\
                             +args[7]*cg*np.exp(-(xdata-args[1])**2/(2*args[3]**2)))\
                             +args[8]*cl/((xdata-3*args[1])**2/(3*args[2])**2+1)\
                             +args[9]*cg*np.exp(-(xdata-3*args[1])**2/(2*(3*args[3])**2))\
                             +args[10]*cl/((xdata-5*args[1])**2/(5*args[2])**2+1)\
                             +args[11]*cg*np.exp(-(xdata-5*args[1])**2/(2*(5*args[3])**2)))\
           -(2*xdata/args[5])

# Pinholes
def Pinholes(xdata, *args):
    '''
    return 2*np.log(args[0]+args[3]/((xdata-args[1])**2/(args[2])**2+1)\
                           +args[4]/((xdata-3*args[1])**2/(args[2])**2+1)\
                           +args[5]/((xdata-5*args[1])**2/(args[2])**2+1)\
                           +args[6]/((xdata-7*args[1])**2/(args[2])**2+1)\
                           +args[7]/((xdata-9*args[1])**2/(args[2])**2+1))
                           #+args[8]/((xdata-11*args[1])**2/(args[2])**2+1))
                           '''
    return args[8]*np.log(args[0]+args[3]*np.exp(-(xdata-args[1])**2/(2*args[2]**2))\
                           +args[4]*np.exp(-(xdata-3*args[1])**2/(2*args[2]**2))\
                           +args[5]*np.exp(-(xdata-5*args[1])**2/(2*args[2]**2))\
                           +args[6]*np.exp(-(xdata-7*args[1])**2/(2*args[2]**2))\
                           +args[7]*np.exp(-(xdata-9*args[1])**2/(2*args[2]**2)))
                           #+args[8]/((xdata-11*args[1])**2/(args[2])**2+1))

def Background(xdata, *args):
    return -(xdata/args[0])+args[1]



# Fitting routine
def Fit_1D_Data(model, x, y, index, q0, qmin):
    ''' Fit model to y(x), only at values selected by index.
    
        == Params ==
        model : string
            Model fctn that will be adjusted to fit data (one of 
            several listed below)
        x, y : ndarray
            Independent and dependent variables describing data to be fit
        index : boolean
            Only data values where 'index' = True will be used for fit
        q0 : numeric
            Initial value for peak position
        qmin : numeric
            Fit constraint. Peak position cannot be smaller than qmin.
            
        == Returns ==
        params : dict
            Params describing best fit for specified model fctn
        fit : array
            Fitted curve, i.e. model evaluated for params in fit1D
    '''
    # Properties of box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # Preallocate the form factor
    Q = 0
    dQ = 0
    
    # Plotting axis
    r = np.arange(0,np.max(x),x[1]/10)
    
    # Plot data
    fig, ax = plt.subplots()
    ax.plot(x[index],y[index],linewidth=2)
    
    # Fit
    if model == 'LinLorentzian_1':
        p0 = np.array([1, q0, 0.1, 1])
        B = ([0., qmin, 0., 0.], [1000., 1., 100., 1000.])
        popt, pcov = curve_fit(LinLorentzian_1, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3]])
        dA = np.array([perr[3]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = LinLorentzian_1(x,*popt)
        ax.plot(r,LinLorentzian_1(r,*popt),linewidth=2)
    elif model == 'LinLorentzian_3':
        p0 = np.array([1, q0, 0.1, 1, 1])
        B = ([0., qmin, 0., 0., 0.], [1000., 1., 100., 1000., 1000.])
        popt, pcov = curve_fit(LinLorentzian_3, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[4]])
        dA = np.array([perr[3], perr[4]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = LinLorentzian_3(x,*popt)
        ax.plot(r,LinLorentzian_3(r,*popt),linewidth=2)
    elif model == 'LinLorentzian_5':
        p0 = np.array([1, q0, 0.1, 1, 1, 1])
        B = ([0., qmin, 0., 0., 0., 0.], [1000., 1., 100., 1000., 1000., 1000.])
        popt, pcov = curve_fit(LinLorentzian_5, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[4], popt[5]])
        dA = np.array([perr[3], perr[4], perr[5]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = LinLorentzian_5(x,*popt)
        ax.plot(r,LinLorentzian_5(r,*popt),linewidth=2)
    elif model == 'Lorentzian_1':
        p0 = np.array([1, q0, 0.1, 1])
        B = ([-1000., qmin, 0., 0.], [1000., 1., 100., 1000.])
        popt, pcov = curve_fit(Lorentzian_1, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3]])
        dA = np.array([perr[3]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = Lorentzian_1(x,*popt)
        ax.plot(r,Lorentzian_1(r,*popt),linewidth=2)
    elif model == 'Lorentzian_2':
        p0 = np.array([1, q0, 0.1, 1, 1])
        B = ([0., qmin, 0., 0., 0.], [1000., 1., 100., 1000., 1000.])
        popt, pcov = curve_fit(Lorentzian_2, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[4]])
        dA = np.array([perr[3], perr[4]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit=Lorentzian_2(x,*popt)
        ax.plot(r,Lorentzian_2(r,*popt),linewidth=2)
    elif model == 'Lorentzian_3':
        p0 = np.array([1, q0, 0.1, 1, 1])
        B = ([0., qmin, 0., 0., 0.], [1000., 1., 100., 1000., 1000.])
        popt, pcov = curve_fit(Lorentzian_3, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[4]])
        dA = np.array([perr[3], perr[4]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit=Lorentzian_3(x,*popt)
        ax.plot(r,Lorentzian_3(r,*popt),linewidth=2)
    elif model == 'Lorentzian_3_bkg':
        p0 = np.array([1, q0, 0.1, 1, 1, -10])
        B = ([0., qmin, 0., 0., 0., -100], [1000., 1., 100., 1000., 1000., 0])
        popt, pcov = curve_fit(Lorentzian_3_bkg, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[4]])
        dA = np.array([perr[3], perr[5]])
        Other = np.array([popt[0], popt[5]])
        dOther = np.array([perr[0], perr[4]])
        # Plot
        fit=Lorentzian_3_bkg(x,*popt)
        ax.plot(r,Lorentzian_3_bkg(r,*popt),linewidth=2)
    elif model == 'Lorentzian_5':
        p0 = np.array([1, q0, 0.1, 1, 1, 1])
        B = ([0., qmin, 0., 0., 0., 0.], [1000., 1., 100., 1000., 1000., 1000.])
        popt, pcov = curve_fit(Lorentzian_5, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[4], popt[5]])
        dA = np.array([perr[3], perr[4], perr[5]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = Lorentzian_5(x,*popt)
        ax.plot(r,Lorentzian_5(r,*popt),linewidth=2)
    elif model == 'Lorentzian_1_FormFac':
        p0 = np.array([1, q0, 0.1, 1, 0.1])
        B = ([0., qmin, 0., 0., 0.], [1000., 1., 100., 1000., 100.])
        popt, pcov = curve_fit(Lorentzian_1_FormFac, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3]])
        dA = np.array([perr[3]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = Lorentzian_1_FormFac(x,*popt)
        ax.plot(r,Lorentzian_1_FormFac(r,*popt),linewidth=2)
    elif model == 'Lorentzian_3_FormFac':
        p0 = np.array([1, q0, 0.1, 1, 0.1, 1])
        B = ([0., qmin, 0., 0., 0., 0.], [1000., qmin*1.3, 100., 1000., 1., 1000.])
        popt, pcov = curve_fit(Lorentzian_3_FormFac, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[5]])
        dA = np.array([perr[3], perr[5]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = Lorentzian_3_FormFac(x,*popt)
        ax.plot(r,Lorentzian_3_FormFac(r,*popt),linewidth=2)
    elif model == 'Lorentzian_5_FormFac':
        p0 = np.array([1, q0, 0.1, 1, 0.1, 1, 1])
        B = ([0., qmin, 0., 0., 0., 0., 0.], [1000., 1., 100., 1000., 100., 1000., 1000.])
        popt, pcov = curve_fit(Lorentzian_5_FormFac, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[5], popt[6]])
        dA = np.array([perr[3], perr[5], perr[6]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = Lorentzian_5_FormFac(x,*popt)
        ax.plot(r,Lorentzian_5_FormFac(r,*popt),linewidth=2)
    elif model == 'AsymLorentzian_1':
        p0 = np.array([1, q0, 0.1, 1, 0])
        B = ([0., qmin, 0., 0., -1000.], [1000., 1., 100., 1000., 1000.])
        popt, pcov = curve_fit(AsymLorentzian_1, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3]])
        dA = np.array([perr[3]])
        Other = np.array([popt[0], popt[4]])
        dOther = np.array([perr[0], perr[4]])
        # Plot
        fit = AsymLorentzian_1(x,*popt)
        ax.plot(r,AsymLorentzian_1(r,*popt),linewidth=2)
    elif model == 'AsymLorentzian_3':
        p0 = np.array([1, q0, 0.1, 1, 0, 1])
        B = ([0., qmin, 0., 0., -1000., 0.], [1000., 1., 100., 1000., 1000., 1000.])
        popt, pcov = curve_fit(AsymLorentzian_3, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[5]])
        dA = np.array([perr[3], perr[5]])
        Other = np.array([popt[0], popt[4]])
        dOther = np.array([perr[0], perr[4]])
        # Plot
        fit = AsymLorentzian_3(x,*popt)
        ax.plot(r,AsymLorentzian_3(r,*popt),linewidth=2)
    elif model == 'AsymLorentzian_5':
        p0 = np.array([1, q0, 0.1, 1, 0, 1, 1])
        B = ([0., qmin, 0., 0., -1000., 0., 0.], [1000., 1., 100., 1000., 1000., 1000., 1000.])
        popt, pcov = curve_fit(AsymLorentzian_5, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[5], popt[6]])
        dA = np.array([perr[3], perr[5], perr[6]])
        Other = np.array([popt[0], popt[4]])
        dOther = np.array([perr[0], perr[4]])
        # Plot
        fit = AsymLorentzian_5(x,*popt)
        ax.plot(r,AsymLorentzian_5(r,*popt),linewidth=2)
    elif model == 'AsymLorentzian_1_FormFac':
        p0 = np.array([1, q0, 0.1, 1, 0, 0.1])
        B = ([0., qmin, 0., 0., -1000., -1000.], [1000., 1., 100., 1000., 1000., 1000.])
        popt, pcov = curve_fit(AsymLorentzian_1_FormFac, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3]])
        dA = np.array([perr[3]])
        Other = np.array([popt[0], popt[4]])
        dOther = np.array([perr[0], perr[4]])
        # Plot
        fit = AsymLorentzian_1_FormFac(x,*popt)
        ax.plot(r,AsymLorentzian_1_FormFac(r,*popt),linewidth=2)
    elif model == 'AsymLorentzian_3_FormFac':
        p0 = np.array([1, q0, 0.1, 1, 0, 0.1, 1])
        B = ([0., qmin, 0., 0., -1000., -1000., -1000.], [1000., 1., 100., 1000., 1000., 1000., 1000.])
        popt, pcov = curve_fit(AsymLorentzian_3_FormFac, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[6]])
        dA = np.array([perr[3], perr[6]])
        Other = np.array([popt[0], popt[4]])
        dOther = np.array([perr[0], perr[4]])
        # Plot
        fit = AsymLorentzian_3_FormFac(x,*popt)
        ax.plot(r,AsymLorentzian_3_FormFac(r,*popt),linewidth=2)
    elif model == 'AsymLorentzian_5_FormFac':
        p0 = np.array([1, q0, 0.1, 1, 0, 0.1, 1, 1])
        B = ([0., qmin, 0., 0., -1000., -1000., -1000., -1000.], [1000., 1., 100., 1000., 1000., 1000., 1000., 1000.])
        popt, pcov = curve_fit(AsymLorentzian_5_FormFac, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[6], popt[7]])
        dA = np.array([perr[3], perr[6], perr[7]])
        Other = np.array([popt[0], popt[4]])
        dOther = np.array([perr[0], perr[4]])
        # Plot
        fit = AsymLorentzian_5_FormFac(x,*popt)
        ax.plot(r,AsymLorentzian_5_FormFac(r,*popt),linewidth=2)
    elif model == 'Gaussian_1':
        p0 = np.array([1, q0, 0.1, 1])
        B = ([0., qmin, 0., 0.], [1000., 1., 100., 1000.])
        popt, pcov = curve_fit(Gaussian_1, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3]])
        dA = np.array([perr[3]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = Gaussian_1(x,*popt)
        ax.plot(r,Gaussian_1(r,*popt),linewidth=2)
    elif model == 'Gaussian_3':
        p0 = np.array([1, q0, 0.1, 1, 1])
        B = ([0., qmin, 0., 0., 0.], [1000., 1., 100., 1000., 1000.])
        popt, pcov = curve_fit(Gaussian_3, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[4]])
        dA = np.array([perr[3], perr[4]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = Gaussian_3(x,*popt)
        ax.plot(r,Gaussian_3(r,*popt),linewidth=2)
    elif model == 'Gaussian_5':
        p0 = np.array([1, q0, 0.1, 1, 1, 1])
        B = ([0., qmin, 0., 0., 0., 0.], [1000., 1., 100., 1000., 1000., 1000.])
        popt, pcov = curve_fit(Gaussian_5, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[4], popt[5]])
        dA = np.array([perr[3], perr[4], perr[5]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = Gaussian_5(x,*popt)
        ax.plot(r,Gaussian_5(r,*popt),linewidth=2)
    elif model == 'Gaussian_1_FormFac':
        p0 = np.array([1, q0, 0.1, 1, 0.1])
        B = ([0., qmin, 0., 0., 0.], [1000., 1., 100., 1000., 100.])
        popt, pcov = curve_fit(Gaussian_1_FormFac, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3]])
        dA = np.array([perr[3]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = Gaussian_1_FormFac(x,*popt)
        ax.plot(r,Gaussian_1_FormFac(r,*popt),linewidth=2)
    elif model == 'Gaussian_3_FormFac':
        p0 = np.array([1, q0, 0.1, 1, 0.1, 1])
        B = ([0., qmin, 0., 0., 0., 0.], [1000., 1., 100., 1000., 100., 1000.])
        popt, pcov = curve_fit(Gaussian_3_FormFac, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[5]])
        dA = np.array([perr[3], perr[5]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = Gaussian_3_FormFac(x,*popt)
        ax.plot(r,Gaussian_3_FormFac(r,*popt),linewidth=2)
    elif model == 'Gaussian_5_FormFac':
        p0 = np.array([1, q0, 0.1, 1, 0.1, 1, 1])
        B = ([0., qmin, 0., 0., 0., 0., 0.], [1000., 1., 100., 1000., 100., 1000., 1000.])
        popt, pcov = curve_fit(Gaussian_5_FormFac, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[5], popt[6]])
        dA = np.array([perr[3], perr[5], perr[6]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = Gaussian_5_FormFac(x,*popt)
        ax.plot(r,Gaussian_5_FormFac(r,*popt),linewidth=2)
    elif model == 'Voigt_1':
        p0 = np.array([1, q0, 0.1, 0.1, 1, 0.9])
        B = ([0., qmin, 0., 0., 0., -1.], [1000., 1., 100., 100., 1000., 1.])
        popt, pcov = curve_fit(Voigt_1, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[5], popt[6]])
        dA = np.array([perr[5], popt[6]])
        Other = np.array([popt[0], popt[3], popt[4]])
        dOther = np.array([perr[0], perr[3], perr[4]])
        # Plot
        fit = Voigt_1(x,*popt)
        ax.plot(r,Voigt_1(r,*popt),linewidth=2)
    elif model == 'Voigt_3':
        p0 = np.array([1, q0, 0.1, 0.1, 1, 0.9, 1])
        B = ([0., qmin, 0., 0., 0., -1., 0.], [1000., 1., 100., 100., 1000., 1., 1000.])
        popt, pcov = curve_fit(Voigt_3, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[5], popt[6], popt[7], popt[8]])
        dA = np.array([perr[5], perr[6], perr[7], perr[8]])
        Other = np.array([popt[0], popt[3], popt[4]])
        dOther = np.array([perr[0], perr[3], perr[4]])
        # Plot
        fit = Voigt_3(x,*popt)
        ax.plot(r,Voigt_3(r,*popt),linewidth=2)
    elif model == 'Voigt_5':
        p0 = np.array([1, q0, 0.1, 0.1, 1, 0.9, 1, 1])
        B = ([0., qmin, 0., 0., 0., -1., 0., 0.], [1000., 1., 100., 100., 1000., 1., 1000., 1000.])
        popt, pcov = curve_fit(Voigt_5, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[5], popt[6], popt[7], popt[8], popt[9], popt[10]])
        dA = np.array([perr[5], perr[6], perr[7], perr[8], perr[9], perr[10]])
        Other = np.array([popt[0], popt[4], popt[5]])
        dOther = np.array([perr[0], perr[4], perr[5]])
        # Plot
        fit = Voigt_5(x,*popt)
        ax.plot(r,Voigt_5(r,*popt),linewidth=2)
    elif model == 'Voigt_1_FormFac':
        p0 = np.array([1, q0, 0.1, 0.1, 1, 0.9, 0.1])
        B = ([0., qmin, 0., 0., 0., -1., 0.], [1000., 1., 100., 100., 1000., 1., 100.])
        popt, pcov = curve_fit(Voigt_1_FormFac, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[6], popt[7]])
        dA = np.array([perr[6], perr[7]])
        Other = np.array([popt[0], popt[4], popt[5]])
        dOther = np.array([perr[0], perr[4], perr[5]])
        # Plot
        fit = Voigt_1_FormFac(x,*popt)
        ax.plot(r,Voigt_1_FormFac(r,*popt),linewidth=2)
    elif model == 'Voigt_3_FormFac':
        p0 = np.array([1, q0, 0.1, 0.1, 1, 0.9, 0.1, 1])
        B = ([0., qmin, 0., 0., 0., -1., 0., 0.], [1000., 1., 100., 100., 1000., 1., 100., 1000.])
        popt, pcov = curve_fit(Voigt_3_FormFac, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[6], popt[7], popt[8], popt[9]])
        dA = np.array([perr[6], perr[7], perr[8], perr[9]])
        Other = np.array([popt[0], popt[4], popt[5]])
        dOther = np.array([perr[0], perr[4], perr[5]])
        # Plot
        fit = Voigt_3_FormFac(x,*popt)
        ax.plot(r,Voigt_3_FormFac(r,*popt),linewidth=2)
    elif model == 'Voigt_5_FormFac':
        p0 = np.array([1, q0, 0.1, 0.1, 1, 0.9, 0.1, 1, 1])
        B = ([0., qmin, 0., 0., 0., -1., 0., 0., 0.], [1000., 1., 100., 100., 1000., 1., 100., 1000., 1000.])
        popt, pcov = curve_fit(Voigt_5_FormFac, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[6], popt[7], popt[8], popt[9], popt[10], popt[11]])
        dA = np.array([perr[6], perr[7], perr[8], perr[9], perr[10], perr[11]])
        Other = np.array([popt[0], popt[3], popt[5]])
        dOther = np.array([perr[0], perr[3], perr[5]])
        # Plot
        fit = Voigt_5_FormFac(x,*popt)
        ax.plot(r,Voigt_5_FormFac(r,*popt),linewidth=2)
    elif model == 'Pinholes':
        p0 = np.array([1., q0, 1e-1, 1., 1., 1., 1., 1., 2])
        B = ([0., qmin, 0., 0, 0, 0., 0., 0., 0.], [1000., 1., 10, 1000., 1000., 1000., 1000., 1000., 3])
        popt, pcov = curve_fit(Pinholes, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[4], popt[5], popt[6]])
        dA = np.array([perr[3], perr[4], perr[5], perr[6]])
        Other = 0
        dOther = 0
        # Plot
        fit = Pinholes(x,*popt)
        ax.plot(r,Pinholes(r,*popt),linewidth=2)
    elif model == 'Background':
        p0 = np.array([1., 1.])
        B = ([0., -1000.], [10., 1000.])
        popt, pcov = curve_fit(Background, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        # Plot
        fit = Background(x,*popt)
        ax.plot(r,Background(r,*popt),linewidth=2)
    elif model == 'LinLorentzian_Bimodal':
        p0 = np.array([1, q0, 0.1, 1, q0*0.8, 0.1, 1])
        B = ([0., qmin, 0., 0., qmin, 0., 0.], [1000., 1., 100., 1000., 1., 100., 1000.])
        popt, pcov = curve_fit(LinLorentzian_Bimodal, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[6]])
        dA = np.array([perr[3], perr[6]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = LinLorentzian_Bimodal(x,*popt)
        ax.plot(r,LinLorentzian_Bimodal(r,*popt),linewidth=2)
    elif model == 'Lorentzian_Bimodal':
        p0 = np.array([1, q0, 0.1, 1, q0*0.8, 0.1, 1])
        B = ([0., qmin, 0., 0., qmin, 0., 0.], [1000., 1., 100., 1000., 1., 100., 1000.])
        popt, pcov = curve_fit(Lorentzian_Bimodal, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3], popt[6]])
        dA = np.array([perr[3], perr[6]])
        Other = popt[0]
        dOther = perr[0]
        # Plot
        fit = LinLorentzian_Bimodal(x,*popt)
        ax.plot(r,LinLorentzian_Bimodal(r,*popt),linewidth=2)
    elif model == 'LinAsymLorentzian':
        p0 = np.array([1, q0, 0.1, 1, 0])
        B = ([0., qmin, 0., 0., -1000.], [1000., 1., 100., 1000., 1000.])
        popt, pcov = curve_fit(LinAsymLorentzian, x[index], y[index], p0, bounds=B)
        perr = np.sqrt(np.diag(pcov))
        A = np.array([popt[3]])
        dA = np.array([perr[3]])
        Other = np.array([popt[0], popt[4]])
        dOther = np.array([perr[0], perr[4]])
        # Plot
        fit = LinAsymLorentzian(x,*popt)
        ax.plot(r,LinAsymLorentzian(r,*popt),linewidth=2)
            
            
    # Compute error
    perr = np.sqrt(np.diag(pcov))
    
    # Style plot
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$q$ (nm$^{-1}$)',fontsize=20)
    plt.ylabel('Integrated intensity (arb. units)',fontsize=20)
    if model=='Lorentzian_1_FormFac' or model=='Lorentzian_3_FormFac' or model=='Lorentzian_5_FormFac':
        ax.text(0.5, 0.75, ('$q_0=$%f nm$^{-1}\pm$%f nm$^{-1}$\n$\Gamma=$%f nm$^{-1}\pm$%f nm$^{-1}$\n$Q=$%f nm$^{-1}\pm$%f nm$^{-1}$' % (popt[1],perr[1],popt[2],perr[2],popt[4],perr[4])),
                transform=ax.transAxes, fontsize=14, bbox=props)
        Q = popt[4]
        dQ = perr[4]
    elif model=='AsymLorentzian_1_FormFac' or model=='AsymLorentzian_3_FormFac' or model=='AsymLorentzian_5_FormFac':
        ax.text(0.5, 0.75, ('$q_0=$%f nm$^{-1}\pm$%f nm$^{-1}$\n$\Gamma=$%f nm$^{-1}\pm$%f nm$^{-1}$\n$Q=$%f nm$^{-1}\pm$%f nm$^{-1}$' % (popt[1],perr[1],popt[2],perr[2],popt[5],perr[5])),
                transform=ax.transAxes, fontsize=14, bbox=props)
        Q = popt[5]
        dQ = perr[5]
    elif model=='Gaussian_1_FormFac' or model=='Gaussian_3_FormFac' or model=='Gaussian_5_FormFac':
        ax.text(0.5, 0.75, ('$q_0=$%f nm$^{-1}\pm$%f nm$^{-1}$\n$\Gamma=$%f nm$^{-1}\pm$%f nm$^{-1}$\n$Q=$%f nm$^{-1}\pm$%f nm$^{-1}$' % (popt[1],perr[1],popt[2],perr[2],popt[4],perr[4])),
                transform=ax.transAxes, fontsize=14, bbox=props)#
        Q = popt[4]
        dQ = perr[4]
    elif model=='Voigt_1_FormFac' or model=='Voigt_3_FormFac' or model=='Voigt_5_FormFac':
        ax.text(0.5, 0.75, ('$q_0=$%f nm$^{-1}\pm$%f nm$^{-1}$\n$\Gamma=$%f nm$^{-1}\pm$%f nm$^{-1}$\n$Q=$%f nm$^{-1}\pm$%f nm$^{-1}$' % (popt[1],perr[1],popt[2],perr[2],popt[6],perr[6])),
                transform=ax.transAxes, fontsize=14, bbox=props)
        Q = popt[5]
        dQ = perr[5]
    elif model=='Background':
        ax.text(0.5, 0.75, ('$Q=$%f nm$^{-1}\pm$%f nm$^{-1}$\n$B=$%f \pm%f' % (popt[0],perr[0],popt[1],perr[1])),
                transform=ax.transAxes, fontsize=14, bbox=props)
    elif model == 'Pinholes':
        ax.text(0.5, 0.75, ('$q_0=$%f nm$^{-1}\pm$%f nm$^{-1}$\n$\Gamma=$%f nm$^{-1}\pm$%f nm$^{-1}$' % (popt[1],perr[1],popt[2],perr[2])),
                transform=ax.transAxes, fontsize=14, bbox=props)
        Q = popt[8]
        dQ = perr[8]
    else:
        ax.text(0.5, 0.75, ('$q_0=$%f nm$^{-1}\pm$%f nm$^{-1}$\n$\Gamma=$%f nm$^{-1}\pm$%f nm$^{-1}$' % (popt[1],perr[1],popt[2],perr[2])),
                transform=ax.transAxes, fontsize=14, bbox=props)
    plt.show()
    if model=='LinLorentzian_Bimodal' or model=='Lorentzian_Bimodal':
        q0 = np.array([popt[1], popt[4]])
        dq0 = np.array([perr[1], perr[4]])
        Gamma = np.array([popt[2], popt[5]])
        dGamma = np.array([perr[2], perr[5]])
    else:
        q0 = popt[1]
        dq0 = perr[1]
        Gamma = popt[2]
        dGamma = perr[2]
    
    # Put results in dictionary
    if model=='Background':
        params = {
        'Q' : popt[0],
        'dQ' : perr[0],
        'B' : popt[1],
        'dB' : perr[1],
        }
    else:
        params = {
        'qpk' : q0,
        'dqpk' : dq0,
        'Gamma' : Gamma,
        'dGamma' : dGamma,
        'Q' : Q,
        'dQ' : dQ,
        'Apk' : A,
        'dApk' : dA,
        'Other' : Other,
        'dOther' : dOther,
        }
    
    return params, fit
    
    # Old version w/o dict
#     q0 = popt[1]
#     Gamma = popt[2]
#     dq0 = perr[1]
#     dGamma = perr[2]
#     return q0, Gamma, dq0, dGamma, Q, dQ, A, dA, fit