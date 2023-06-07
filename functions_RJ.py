# -*- coding: utf-8 -*-
''' Function for analysis of data from FERMI DiProI endstation writen by Rahul Jangid'''

from sys import platform
import numpy as np
import matplotlib.pyplot as plt
import sys
import imageio
import os
import re
import math
import time
import glob
from PIL import Image
import pandas as pd

def concat_type():
    """_summary_

    Returns:
        _type_: _description_
    """
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        concat_3 = "{0}/{1}/{2}"
        concat_2 = "{0}/{1}"
        concat_1 = "/"
    elif platform == "win64" or platform == "win32":
        concat_3 = "{0}\\{1}\\{2}"
        concat_2 = "{0}\\{1}"
        concat_1 = "\\"
    else:
        concat_3 = "{0}\\{1}\\{2}"
        concat_2 = "{0}\\{1}"
        concat_1 = "\\"
    return concat_1, concat_2, concat_3

def concat(x):
    ''' Identify system OS to set formatting for path strings

        == Params ==
        x : integer from set {1, 2, 3, 4}

        == Returns ==
        x = 1 will return concat 1 depending on the platform
        x = 2 will return concat 2 depending on the platform
        x = 3 will return concat 3 depending on the platform

    '''
    if x in [1, 2, 3, 4]:
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            if x == 1:
                concat = "/"
            elif x == 2:
                concat = "{0}/{1}"
            elif x == 3:
                concat = "{0}/{1}/{2}"
            elif x == 4:
                concat = "{0}/{1}/{2}/rawdata"
        elif platform == "win64" or platform == "win32":
            if x == 1:
                concat = "\\"
            elif x == 2:
                concat = "{0}\\{1}"
            elif x == 3:
                concat = "{0}\\{1}\\{2}"
            elif x == 4:
                concat = "{0}\\{1}\\{2}\\rawdata"
        else:
            print("System detection failed. Using defaults")
            if x == 1:
                concat_1 = "\\"
            elif x == 2:
                concat_2 = "{0}\\{1}"
            elif x == 3:
                concat_3 = "{0}\\{1}\\{2}"
            elif x == 4:
                concat_3 = "{0}\\{1}\\{2}\\rawdata"
    elif x not in [1, 2, 3, 4]:
        sys.exit("Give integer from set [1, 2, 3, 4]")

    return concat

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def binPixSize(im, det_size, pix_size):
    ''' Identifies the binning for a detector image.

        == Params ==
        im : 2D matrix which is a single image from detector 
        det_size : One integer giving one dimention of the square detector
        pix_size : Physical pixel size [mm]

        == Returns ==
        eff_pix_size = effective pixel size after binning [mm]
        factor = factor of binning

        Note: Only works for square detectors
    '''
    if np.shape(im)[0] != det_size:
        factor = det_size/np.shape(im)[0]
        #remainder = 2048%np.shape(im)[0];
        try:
            det_size % np.shape(im)[0] == 0
        except:
            print('Number of pixels is not a factor of 2048 !!')
        eff_pix_size = factor*pix_size

        return eff_pix_size, factor
    elif np.shape(im)[0] == det_size:
        factor = det_size/np.shape(im)[0]
        eff_pix_size = pix_size

        return eff_pix_size, factor


def uJ_to_mJ_cm2(energy, spot_size):
    ''' Converts the pump energy to fluence given spot size and energy

        == Params ==
        energy : 1D matrix with pump energy in [uJ]
        spot_size : pump spot size in [um]

        == Returns ==
        fluence = fluence in [mJ/cm2]

        Note: Be careful with the uints of the input energy and spot size
    '''
    energy = energy*10**(-3)
    spot_size = spot_size*10**(-4)  # converting um in cm
    spot_area = np.pi*(spot_size/2)**2  # calculating the area of the pump spot
    fluence_mj_cm = np.round(energy/spot_area, 1)  # in mj/cm2
    return fluence_mj_cm


def colorFrmCmap(colormap, counter_1, counter_2 = 1):
    ''' Returns a single color from a color map for plotting

        == Params ==
        colormap : string with the name of colormap
        counter : number to select a perticular color

        == Returns ==
        color : returns the color that you need
    '''
    cmap = plt.get_cmap(colormap)
    color = cmap(float(counter_1)/counter_2)
    return color

def colorFrmCmap2(colormap, counter):
    ''' Returns a single color from a color map for plotting
    
        == Params ==
        colormap : string with the name of colormap
        counter : number to select a perticular color
        
        == Returns ==
        color : returns the color that you need
    '''
    cmap = plt.get_cmap(colormap)
    counter1 = 3.5+counter
    color = cmap(float(counter1)/8/2)
    return color

def plot_errorbars(x, y, yerr, **kwargs):
    ''' Returns a nice plot with fill between +- errorbar

        == Params ==
        x : array with x data 
        y : array with y data 
        yerr : array with y error
        kwargs : extra key word arguments

        == Returns ==
        p : returns the plot
    '''
    p = plt.plot(x, y, **kwargs)
    pfb = plt.fill_between(x, y - yerr, y + yerr, alpha=0.3)
    return p


def plot_errorbars1(x, y, yerr, color_temp, **kwargs):
    ''' Returns a nice plot with fill between +- errorbar

        == Params ==
        x : array with x data 
        y : array with y data 
        yerr : array with y error
        kwargs : extra key word arguments

        == Returns ==
        p : returns the plot
    '''
    color = color_temp
    p = plt.plot(x, y, **kwargs)
    pfb = plt.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.3)
    return p


def parb(x, *args):
    ''' Returns a parabola

        == Params ==
        x : array with x data 
        args : arguments for the parabola

        == Returns ==
        returns a parabola
    '''
    return args[0]*x**2


def parb_plot(x, *args):
    ''' Returns a parabola

        == Params ==
        x : array with x data 
        args : arguments for the parabola

        == Returns ==
        returns a parabola
    '''
    y = args[0]*x**2
    return y


def gaussian(x, x0, sigma):
    """
    Gaussian with amplitude at 1
    """
    return np.exp(-(x-x0)**2/(2*sigma**2))


def gaussian_convolution(x, y, sigma):
    """
    Convolves y with a gaussian on interval t 
    """
    conv = np.zeros(x.shape)
    for i in range(len(x)):
        conv[i] = np.sum(gaussian(x, x[i], sigma)*y) / \
            np.sum(gaussian(x, x[i], sigma))
    return conv


def vec_gaussian_convolution(x, y, sigma):
    """
    Convolves y with a gaussian on interval t 
    Vectorized so at least 5 times faster
    """
    #conv = np.zeros(x.shape)
    x = np.reshape(x, (-1, 1))
    g = np.exp(-(x - x.T)**2/(2*sigma**2))
    gnorm = np.sum(g, axis=0)
    return g.dot(y) / gnorm


def jumpdecay_VUV_neg(x, A1, A2, t0,  tm, tr, tr2):
    """
    decay function from Viveks paper
    """
    return - np.heaviside(x-t0, 0) * \
        (((A1*tr-A2*tm)/(tr-tm))*np.exp(-(x-t0) / tm) -
         (tr*(A1-A2)/(tr-tm))*np.exp(-(x-t0) / tr) -
         A2 / np.sqrt((x - t0) / tr2 + 1))


def jumpdecay_VUV(x, A1, A2, t0,  tm, tr, tr2):
    """
    decay function from Viveks paper
    """
    return np.heaviside(x-t0, 0) * \
        (((A1*tr-A2*tm)/(tr-tm))*np.exp(-(x-t0) / tm) -
         (tr*(A1-A2)/(tr-tm))*np.exp(-(x-t0) / tr) -
         A2 / np.sqrt((x - t0) / tr2 + 1))


def jumpdecay(x, A, B, C, t0, tm, tr):
    """
    decay function with no tr2
    """
    return (C + np.heaviside(x-t0, 1) * \
        (A * np.exp(-(x - t0) / tm) -
         B * np.exp(-(x - t0) / tr) + (B-A)))


def jumpdecay1(x, A, B, C, t0, tm, tr, tr2):
    """
    decay function with tr2
    """
    return (C + np.heaviside(x-t0, 1) * \
        (A * np.exp(-(x - t0) / tm) -
         B * np.exp(-(x - t0) / tr) +
         (B - A) / np.sqrt(np.abs(x - t0) / tr2 + 1)))


def jumpdecay2(x, A, B, C, t0, tm, tr, tr2):
    """
    decay function with binomial expansion for tr2
    """
    return C + np.heaviside(x-t0, 0.5) * \
        (A * np.exp(-(x - t0) / tm) -
         B * np.exp(-(x - t0) / tr) +
         (B - A) * (1 - 0.5 * (x-t0) / tr2 + (3/8) * ((x - t0) / tr2)**2 - (5/16) * ((x-t0) / tr2)**3))


def convolve_with_tr2(x, A, B, C, t0, tm, tr, tr2, sigma):
    """
    decay function (with tr2) convolved with Gaussian 
    """
    y = jumpdecay1(x, A, B, C, t0, tm, tr, tr2)
    # y = jumpdecay2(x, A, B, C, t0, tm, tr, tr2) # with the binomial expansion
    result = vec_gaussian_convolution(x, y, sigma)
    return result


def convolve_with_tr2_binomial(x, A, B, C, t0, tm, tr, tr2, sigma):
    """
    decay function (with tr2 binomial expansion) convolved with Gaussian 
    """
    #y = jumpdecay1(x, A, B, C, t0, tm, tr, tr2)
    y = jumpdecay2(x, A, B, C, t0, tm, tr, tr2)  # with the binomial expansion
    result = vec_gaussian_convolution(x, y, sigma)
    return result


def convolved_no_tr2(x, A, B, C, t0, tm, tr, sigma):
    """
    decay function (no tr2) convolved with Gaussian 
    """
    y = jumpdecay(x, A, B, C, t0, tm, tr)
    result = vec_gaussian_convolution(x, y, sigma)
    return result


def variance(data, ddof=0):
    ''' Calculates the variance of a numpy array

        == Params ==
        data : array with the data

        == Returns ==
        variance : returns the variance of data
    '''
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - ddof)


def stdev(data):
    ''' Calculates the standard deviation of a numpy array

        == Params ==
        data : array with data

        == Returns ==
        std_dev : returns the standard deviation of data
    '''
    var = variance(data)
    std_dev = math.sqrt(var)
    return std_dev

# def data(datapath, run_nr, state):
#     time = genfromtxt(datapath + run_nr + '_time.txt')
#     amplitude = genfromtxt(datapath + run_nr + '_' + state + '.txt')
#     return time, amplitude

def get_sorted_files_path(dir_path, extension):
    # Get a list of all .png files in the directory
    png_files = glob.glob(os.path.join(dir_path, ("*"+extension)))
    # Define a custom sorting function
    def sort_key(file_name):
        # Split the file name by the _ character and take the second to last element
        num = file_name.rsplit('_', 3)[-1]
        num = os.path.splitext(num)[0]
        # Convert the numeric part to an integer and return it
        return int(num)
    # Sort the list of .png files using the custom sorting function
    sorted_png_files = sorted(png_files, key=sort_key)
    return sorted_png_files

def gen_gif(img_files_path, gif_save_dir, gif_name, fps_num):
    ''' Generates a gif from images

        == Params ==
        img_file_path : path to the folder containing all the images
        gif_save_dir : path to the folder where you want to save the gif
        gif_name : name of the gif file
        fps_num : frames per second (fps) for the gif

        == Returns ==
        gif : generates a gif file

    '''
    # filenames = (os.listdir(img_files_path))
    # filenames.sort(key=lambda f: int(re.sub('\d', '', f)))
    filenames = get_sorted_files_path(img_files_path, '.png')
    final_gif_name_save = gif_save_dir + gif_name
    with imageio.get_writer(final_gif_name_save, mode='I', fps=fps_num) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        print('GIF exported')


def moving_average(y, w):
    ''' Does moving avg when given an array x over w number of points

        == Params ==
        y : 1D array with the data to be moving averaged
        w : Number of points for moving avg

        == Returns ==
        move_avg : returns a new 1D array of size = (size(y))/w with the moving avg

    '''
    move_avg = np.convolve(y, np.ones(w), 'valid') / w
    return move_avg


def pix2q(img, det_size, pix_size, det_dis, wavelen, center_off_pix):
    ''' Converts the pixel positions on the detector to qx and qy

        == Params ==
        img : 2D numpy array which you want to calculate qx & qy.
        det_size : real size of the detector in pixels
        det_dis : Distance of the detector from the sample in [mm]
        pix_size : pix size for the det in [mm]
        center_off_pix : Dictonary containing the center offset

        == Returns ==
        x = 
        y = 
        qx = qx coordinates
        qy = qy coordinates
        center_X_pixel = 
        center_Y_pixel = 
        center_X_mm = 
        center_Y_mm = 

        Note: Only works for square detectors.
    '''
    # Using a previous function to get the effective pixel size and the bin size
    eff_pix_size, factor = binPixSize(img, det_size, pix_size)

    center_X_pixel = det_size/factor/2-center_off_pix["X"]
    center_Y_pixel = det_size/factor/2-center_off_pix["Y"]

    ax_size = img.shape

    # Create X and Y arrays with pixel positions
    x = np.arange(-ax_size[1]/2, ax_size[1]/2, 1) * eff_pix_size
    y = np.arange(-ax_size[0]/2, ax_size[0]/2, 1) * eff_pix_size

    center_X_mm = (-np.shape(img)[0]/2 + center_X_pixel)*eff_pix_size
    center_Y_mm = (-np.shape(img)[1]/2 + center_Y_pixel)*eff_pix_size

    center_X = 2 * np.pi * np.sin(np.arctan(center_X_mm / det_dis)) / wavelen
    center_Y = 2 * np.pi * np.sin(np.arctan(center_Y_mm / det_dis)) / wavelen

    # Apply fitted shift so that diffraction pattern is centered at (0,0)
    x = x - center_X_mm
    y = y - center_Y_mm

    # Convert pixel positions to q values [1/nm]
    qx = 2 * np.pi * np.sin(np.arctan(x / det_dis)) / wavelen
    qy = 2 * np.pi * np.sin(np.arctan(y / det_dis)) / wavelen
    # Generate Cartesian q coordinates (qxx,qyy) for every pixel in the detector [1/nm]
    #qxx, qyy = np.meshgrid(qx, qy)

    return x, y, qx, qy, center_X_pixel, center_Y_pixel, center_X_mm, center_Y_mm


def gen_x_mask(img, det_size, pix_size, mask_width, mask_offset):
    ''' Generates a cross mask for images with beam block shadow

        == Params ==
        img = 2D numpy array which you want to generate mask for
        det_size : real size of the detector in pixels (Give one integer)
        pix_size : pix size for the det in [mm]
        mask_width = integer value defining the width of the mask
        mask_offset = a dictionary containing the width of mask

        == Returns ==
        mask = 2D matrix where masked elements are 0
        mask_nan = 2D matrix where masked elements are NaN's
    '''
    _, factor = binPixSize(img, det_size, pix_size)

    mask_offset_X = mask_offset["X"]
    mask_offset_Y = mask_offset["Y"]

    x_start = factor*(np.shape(img)[0]/2) - mask_width/2
    x_end = factor*(np.shape(img)[0]/2) + mask_width/2

    y_start = factor*(np.shape(img)[0]/2) - mask_width/2
    y_end = factor*(np.shape(img)[0]/2) + mask_width/2

    mask = np.zeros((np.shape(img)[0], np.shape(img)[1])) + 1
    mask = mask.astype("float")

    mask_nan = mask

    mask[:, round((x_start + mask_offset_X)/factor):round((x_end + mask_offset_X)/factor)] = 0
    mask[round((y_start - mask_offset_Y)/factor):round((y_end - mask_offset_Y)/factor), :] = 0

    mask_nan[:, round((x_start + mask_offset_X)/factor):round((x_end + mask_offset_X)/factor)] = np.nan
    mask_nan[round((y_start - mask_offset_Y)/factor):round((y_end - mask_offset_Y)/factor), :] = np.nan

    return mask, mask_nan


def gen_cir_mask(im, inner_r, outer_r=1000, type="disk", center_X_pixel=256, center_Y_pixel=256):
    ''' Generates a cross mask to mask images with beam block

        == Params ==
        img = 2D numpy array which you want to generate mask for
        inner_r : real size of the detector in pixels (Give one integer)
        outer_r : pix size for the det in [mm]
        type : integer value defining the width of the mask
        center_X_pixel : a dictionary containing the width of mask
        center_Y_pixel : 

        == Returns ==
        variance : returns the variance of data

        Note: Only works for square detectors.
    '''
    ring_mask_size_x = np.shape(im)[0]
    ring_mask_size_y = np.shape(im)[1]

    ring_xx, ring_yy = np.meshgrid(
        np.arange(0, ring_mask_size_x, 1), np.arange(0, ring_mask_size_y, 1))

    array2D = (ring_yy - center_Y_pixel)**2 + (ring_xx - center_X_pixel)**2
    ringPixels_logical_1 = array2D >= inner_r**2
    ringPixels_logical_2 = array2D <= outer_r**2
    if type == "disk":
        ring_mask = ringPixels_logical_1
    elif type == "ring":
        ring_mask = ringPixels_logical_1*ringPixels_logical_2

    ring_mask = ring_mask.astype('float')
    ring_mask[ring_mask == 0] = np.nan  # or use np.nan

    return ring_mask

import numpy as np

def downsample_2D_sqr_mat(matrix = np.array, block_size = int):
    """Downsamples a 2D square matrix by black size

    Args:
        matrix (numpy array): _description_
        block_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # Get the dimensions of the matrix
    n, m = matrix.shape
    # Compute the number of blocks in each dimension
    n_blocks = n // block_size
    m_blocks = m // block_size
    # Initialize the downsampled matrix
    downsampled = np.zeros((n_blocks, m_blocks))
    # Iterate over the blocks
    for i in range(n_blocks):
        for j in range(m_blocks):
            # Compute the indices of the block
            i_start = i * block_size
            i_end = i_start + block_size
            j_start = j * block_size
            j_end = j_start + block_size
            # Compute the mean of the block
            downsampled[i, j] = matrix[i_start:i_end, j_start:j_end].mean()
    return downsampled



def downsample_2D_mat_vectorized(matrix, block_size):
    # Get the dimensions of the matrix
    n, m = matrix.shape
    # Compute the size of the blocks
    block_n, block_m = block_size
    # Compute the number of blocks in each dimension
    n_blocks = n // block_n
    m_blocks = m // block_m
    # Initialize the downsampled matrix
    downsampled = np.zeros((n_blocks, m_blocks))
    # Reshape the matrix into a 2D array of blocks
    blocks = matrix.reshape(n_blocks, block_n, m_blocks, block_m)
    # Compute the mean of each block
    downsampled = blocks.mean(axis=(1, 3))
    return downsampled

def get_creation_time(file_path):
    # Get information about the file
    file_stats = os.stat(file_path)
    # Return the creation time of the file
    return time.ctime(file_stats.st_ctime)

def three_XRMS_subfigures(qx, qy, raw_data, fit_data, title_num, save_dir):
    # %matplotlib inline
    plt.rcParams['figure.figsize'] = (22, 6)
    plt.rcParams['figure.dpi']= 300

    # Create a figure with three subplots
    font_tick = 12
    font_label = 15
    font_title = 15
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # plt.subplots(figsize=(20, 5))
    # Plot the first subfigure
    im1 = ax1.pcolormesh(qx, qy, raw_data, vmin=0, vmax=3.0, shading='auto')
    # ax1.colorbar().set_label(label='Normalized Intensity', size=15)
    ax1.set_aspect('equal', 'box')
    xticks1 = ['{:,.2f}'.format(x) for x in ax1.get_xticks()]
    yticks1 = ['{:,.2f}'.format(y) for y in ax1.get_yticks()]
    ax1.set_xticklabels(xticks1, rotation = 90, fontsize=font_tick)
    ax1.set_yticklabels(yticks1, fontsize=font_tick)
    ax1.set_xlabel('$q_x$ [1/nm]',fontsize=font_label)
    ax1.set_ylabel('$q_y$ [1/nm]',fontsize=font_label)
    ax1.set_title(f'Data {round(title_num, 1)} min', fontsize = font_title)
    cbar1 = fig.colorbar(im1, ax=ax1)
    
    # Plot the second subfigure
    im2 = ax2.pcolormesh(qx, qy, fit_data, vmin=0, vmax=3.0, shading='auto')
    # ax1.colorbar().set_label(label='Normalized Intensity', size=15)
    ax2.set_aspect('equal', 'box')
    xticks2 = ['{:,.2f}'.format(x) for x in ax2.get_xticks()]
    yticks2 = ['{:,.2f}'.format(y) for y in ax2.get_yticks()]
    ax2.set_xticklabels(xticks2, rotation = 90, fontsize=font_tick)
    ax2.set_yticklabels(yticks2, fontsize=font_tick)
    ax2.set_xlabel('$q_x$ [1/nm]',fontsize=font_label)
    ax2.set_ylabel('$q_y$ [1/nm]',fontsize=font_label)
    ax2.set_title(f'2D fit {round(title_num, 1)} min', fontsize = font_title)
    cbar2 = fig.colorbar(im2, ax=ax2)
    
    # Plot the third subfigure
    im3 = ax3.pcolormesh(qx, qy, raw_data-fit_data, vmin=-1.0, vmax=1.0, shading='auto')
    # ax1.colorbar().set_label(label='Normalized Intensity', size=15)
    ax3.set_aspect('equal', 'box')
    xticks3 = ['{:,.2f}'.format(x) for x in ax3.get_xticks()]
    yticks3 = ['{:,.2f}'.format(y) for y in ax3.get_yticks()]
    ax3.set_xticklabels(xticks3, rotation = 90, fontsize=font_tick)
    ax3.set_yticklabels(yticks3, fontsize=font_tick)
    ax3.set_xlabel('$q_x$ [1/nm]',fontsize=font_label)
    ax3.set_ylabel('$q_y$ [1/nm]',fontsize=font_label)
    ax3.set_title(f'Residual {round(title_num, 1)} min', fontsize = font_title)
    cbar3 = fig.colorbar(im3, ax=ax3)
    cbar3.set_label(label='Normalized Intensity', size=15)
    plt.savefig(save_dir, bbox_inches='tight')
    plt.close()
    

def png_to_binary(file_path, threshold=128):
    # Open image and convert to grayscale
    img = Image.open(file_path).convert('L')

    # Convert image to numpy array
    img_array = np.array(img)

    # Apply threshold to convert to binary
    img_binary = img_array > threshold

    return img_binary

def export_dataframe_excel(data_dict, file_path):
    """
    Description:
    This function takes a dictionary of data sets and a file path as input and exports each data set to a separate sheet in an Excel file. The data sets are assumed to be in the form of a Pandas DataFrame, with the data and column names provided as keys in the dictionary.

    Syntax: export_dataframe_excel(data_dict, file_path)

    Input Parameters:
    data_dict: A dictionary containing the data sets to be exported to Excel.
    file_path: A string representing the file path where the Excel file will be saved.

    Output: None

    Example:
    data_dict = {'Sheet1': {'data': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    'column_names': ['A', 'B', 'C']},
    'Sheet2': {'data': [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    'column_names': ['D', 'E', 'F']}}
    file_path = 'data.xlsx'
    export_dataframe_excel(data_dict, file_path)

    Note: This function uses the xlsxwriter engine to write Excel files, so the xlsxwriter package must be installed to use this function.
    """
    
    # Create an Excel writer object
    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    
    # Loop over the data sets and write each one to a separate sheet
    for sheet_name, data in data_dict.items():
        # Convert the data to a Pandas DataFrame
        df = pd.DataFrame(data['data'], columns=data['column_names'])
        
        # Remove invalid characters from sheet names
        sheet_name = re.sub(r'[\\/*?:\[\]]', '', sheet_name)
        
        # Export the DataFrame to a new sheet in the Excel file
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Save the Excel file
    writer.save()


# With finite radius

# Voigt function
def Voigt_2D_ring_fit(xyflat, *args):
    u = xyflat[:,0]
    v = xyflat[:,1]
    
    b = args[0]
    q_0 = args[1]
    G_v = args[2]
    I_v = args[3]
    d = args[4]
    q_x_cent = args[5]
    q_y_cent = args[6]
    
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    c_L = 0.68188+0.61293*d-0.18384*d**2-0.11568*d**3
    c_G = 0.32460-0.61825*d+0.17681*d**2+0.12109*d**3
    
    A_term = c_L*(1/np.pi)*(G_v/((q-q_0)**2+G_v**2))
    B_term = c_G*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*G_v))*np.exp(-(np.log(2)*(q-q_0)**2)/G_v**2)
    
    voigt_2d_ring = b+I_v*(A_term+B_term)**2
    return voigt_2d_ring
    
# Voigt function
def Voigt_2D_ring_plot(qxx, qyy, *args):
    u = qxx
    v = qyy
    
    b = args[0]
    q_0 = args[1]
    G_v = args[2]
    I_v = args[3]
    d = args[4]
    q_x_cent = args[5]
    q_y_cent = args[6]
    
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    c_L = 0.68188+0.61293*d-0.18384*d**2-0.11568*d**3
    c_G = 0.32460-0.61825*d+0.17681*d**2+0.12109*d**3
    
    A_term = c_L*(1/np.pi)*(G_v/((q-q_0)**2+G_v**2))
    B_term = c_G*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*G_v))*np.exp(-(np.log(2)*(q-q_0)**2)/G_v**2)
    
    voigt_2d_ring = b+I_v*(A_term+B_term)**2
    return voigt_2d_ring

# Voigt function for 2D fits of charge scattering q_0 = 0
def Voigt_2D_charge_fit(xyflat, *args):
    u = xyflat[:,0]
    v = xyflat[:,1]
    
    b = args[0]
    q_0 = 0
    G_v = args[1]
    I_v = args[2]
    d = args[3]
    q_x_cent = args[4]
    q_y_cent = args[5]
    
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    c_L = 0.68188+0.61293*d-0.18384*d**2-0.11568*d**3
    c_G = 0.32460-0.61825*d+0.17681*d**2+0.12109*d**3
    
    A_term = c_L*(1/np.pi)*(G_v/((q-q_0)**2+G_v**2))
    B_term = c_G*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*G_v))*np.exp(-(np.log(2)*(q-q_0)**2)/G_v**2)
    
    voigt_2d_charge = b+I_v*(A_term+B_term)**2
    return voigt_2d_charge
    
# Voigt function for 2D fits of charge scattering q_0 = 0
def Voigt_2D_charge_plot(qxx, qyy, *args):
    u = qxx
    v = qyy
    
    b = args[0]
    q_0 = 0
    G_v = args[1]
    I_v = args[2]
    d = args[3]
    q_x_cent = args[4]
    q_y_cent = args[5]
    
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    c_L = 0.68188+0.61293*d-0.18384*d**2-0.11568*d**3
    c_G = 0.32460-0.61825*d+0.17681*d**2+0.12109*d**3
    
    A_term = c_L*(1/np.pi)*(G_v/((q-q_0)**2+G_v**2))
    B_term = c_G*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*G_v))*np.exp(-(np.log(2)*(q-q_0)**2)/G_v**2)
    
    voigt_2d_charge = b+I_v*(A_term+B_term)**2
    return voigt_2d_charge

def Voigt_2D_charge_p_ring_fit(xyflat, *args):
    u = xyflat[:,0]
    v = xyflat[:,1]
    
    # 'B'  'q0_R'  'Gamma_R'   'A0_R'  'd_R'   'Gamma_C'   'A0_C'  'd_C'   'X0'  'Y0'
    
    b = args[0]
    q0_R = args[1]
    G_v_R = args[2]
    I_v_R = args[3]
    d_R = args[4]
    q0_C = 0
    G_v_C = args[5]
    I_v_C = args[6]
    d_C = args[7]
    q_x_cent = args[8]
    q_y_cent = args[9]
    
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    c_L_C = 0.68188+0.61293*d_C-0.18384*d_C**2-0.11568*d_C**3
    c_G_C = 0.32460-0.61825*d_C+0.17681*d_C**2+0.12109*d_C**3
    
    A_term_C = c_L_C*(1/np.pi)*(G_v_C/((q-q0_C)**2+G_v_C**2))
    B_term_C = c_G_C*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*G_v_C))*np.exp(-(np.log(2)*(q-q0_C)**2)/G_v_C**2)
    
    voigt_2d_charge = I_v_C*(A_term_C+B_term_C)**2
    
    c_L_R = 0.68188+0.61293*d_R-0.18384*d_R**2-0.11568*d_R**3
    c_G_R = 0.32460-0.61825*d_R+0.17681*d_R**2+0.12109*d_R**3
    
    A_term_R = c_L_R*(1/np.pi)*(G_v_R/((q-q0_R)**2+G_v_R**2))
    B_term_R = c_G_R*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*G_v_R))*np.exp(-(np.log(2)*(q-q0_R)**2)/G_v_R**2)
    
    voigt_2d_ring = I_v_R*(A_term_R+B_term_R)**2
    
    voigt_2D_charge_p_ring = b+voigt_2d_charge+voigt_2d_ring
    
    return voigt_2D_charge_p_ring

def Voigt_2D_charge_p_ring_plot(qxx, qyy, *args):
    u = qxx
    v = qyy
    
    # 'B'  'q0_R'  'Gamma_R'   'A0_R'  'd_R'   'Gamma_C'   'A0_C'  'd_C'   'X0'  'Y0'
    
    b = args[0]
    q0_R = args[1]
    G_v_R = args[2]
    I_v_R = args[3]
    d_R = args[4]
    q0_C = 0
    G_v_C = args[5]
    I_v_C = args[6]
    d_C = args[7]
    q_x_cent = args[8]
    q_y_cent = args[9]
    
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    c_L_C = 0.68188+0.61293*d_C-0.18384*d_C**2-0.11568*d_C**3
    c_G_C = 0.32460-0.61825*d_C+0.17681*d_C**2+0.12109*d_C**3
    
    A_term_C = c_L_C*(1/np.pi)*(G_v_C/((q-q0_C)**2+G_v_C**2))
    B_term_C = c_G_C*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*G_v_C))*np.exp(-(np.log(2)*(q-q0_C)**2)/G_v_C**2)
    
    voigt_2d_charge = I_v_C*(A_term_C+B_term_C)**2
    
    c_L_R = 0.68188+0.61293*d_R-0.18384*d_R**2-0.11568*d_R**3
    c_G_R = 0.32460-0.61825*d_R+0.17681*d_R**2+0.12109*d_R**3
    
    A_term_R = c_L_R*(1/np.pi)*(G_v_R/((q-q0_R)**2+G_v_R**2))
    B_term_R = c_G_R*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*G_v_R))*np.exp(-(np.log(2)*(q-q0_R)**2)/G_v_R**2)
    
    voigt_2d_ring = I_v_R*(A_term_R+B_term_R)**2
    
    voigt_2D_charge_p_ring = b+voigt_2d_charge+voigt_2d_ring
    
    return voigt_2D_charge_p_ring


# Voigt function for 2D lobes
def Voigt_2D_lobe_fit(xyflat, *args):
    u = xyflat[:,0]
    v = xyflat[:,1]
    
    # 'B'  'q0_L'  'Gamma_L'   'I_v_2'  'I_v_4'  'd_L'   'phi'   'X0'  'Y0'
    
    negative = False
    b = args[0]
    q_0 = args[1]
    G_v = args[2]
    I_v_2 = args[3]
    I_v_4 = args[4]
    d = args[5]
    phi = args[6]
    q_x_cent = args[7]
    q_y_cent = args[8]
    
    theta = np.arctan2(u-q_x_cent,v-q_y_cent)
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    c_L = 0.68188+0.61293*d-0.18384*d**2-0.11568*d**3
    c_G = 0.32460-0.61825*d+0.17681*d**2+0.12109*d**3
    
    A_term = c_L*(1/np.pi)*(G_v/((q-q_0)**2+G_v**2))
    B_term = c_G*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*G_v))*np.exp(-(np.log(2)*(q-q_0)**2)/G_v**2)
    
    if negative == True:
        I_2nd = (abs(I_v_2)/2*(np.cos(2*theta-2*phi)))
        I_4th = (abs(I_v_4)/2*(np.cos(4*theta-4*phi)))
    else:
        I_2nd = (abs(I_v_2)/2*(np.cos(2*theta-2*phi)+1))
        I_4th = (abs(I_v_4)/2*(np.cos(4*theta-4*phi)+1))
    
    I_v = (I_2nd + I_4th)
    voigt_2d_lobe = b+I_v*(A_term+B_term)**2
    return voigt_2d_lobe
    
# Voigt function for 2D lobes
def Voigt_2D_lobe_plot(qxx, qyy,  *args):
    u = qxx
    v = qyy
    
    # 'B'  'q0_L'  'Gamma_L'   'I_v_2'  'I_v_4'  'd_L'   'phi'   'X0'  'Y0'
    
    negative = False
    b = args[0]
    q_0 = args[1]
    G_v = args[2]
    I_v_2 = args[3]
    I_v_4 = args[4]
    d = args[5]
    phi = args[6]
    q_x_cent = args[7]
    q_y_cent = args[8]
    
    theta = np.arctan2(u-q_x_cent,v-q_y_cent)
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    c_L = 0.68188+0.61293*d-0.18384*d**2-0.11568*d**3
    c_G = 0.32460-0.61825*d+0.17681*d**2+0.12109*d**3
    
    A_term = c_L*(1/np.pi)*(G_v/((q-q_0)**2+G_v**2))
    B_term = c_G*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*G_v))*np.exp(-(np.log(2)*(q-q_0)**2)/G_v**2)
    
    if negative == True:
        I_2nd = (abs(I_v_2)/2*(np.cos(2*theta-2*phi)))
        I_4th = (abs(I_v_4)/2*(np.cos(4*theta-4*phi)))
    else:
        I_2nd = (abs(I_v_2)/2*(np.cos(2*theta-2*phi)+1))
        I_4th = (abs(I_v_4)/2*(np.cos(4*theta-4*phi)+1))
    
    I_v = (I_2nd + I_4th)
    voigt_2d_lobe = b+I_v*(A_term+B_term)**2
    return voigt_2d_lobe
    
# Voigt function for 2D lobes
def Voigt_2D_neg_lobe_fit(xyflat, *args):
    u = xyflat[:,0]
    v = xyflat[:,1]
    
    # 'B'  'q0_L'  'Gamma_L'   'I_v_2'  'I_v_4'  'd_L'   'phi'   'X0'  'Y0'
    
    negative = True
    b = args[0]
    q_0 = args[1]
    G_v = args[2]
    I_v_2 = args[3]
    I_v_4 = args[4]
    d = args[5]
    phi = args[6]
    q_x_cent = args[7]
    q_y_cent = args[8]
    
    theta = np.arctan2(u-q_x_cent,v-q_y_cent)
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    c_L = 0.68188+0.61293*d-0.18384*d**2-0.11568*d**3
    c_G = 0.32460-0.61825*d+0.17681*d**2+0.12109*d**3
    
    A_term = c_L*(1/np.pi)*(G_v/((q-q_0)**2+G_v**2))
    B_term = c_G*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*G_v))*np.exp(-(np.log(2)*(q-q_0)**2)/G_v**2)
    
    if negative == True:
        I_2nd = (abs(I_v_2)/2*(np.cos(2*theta-2*phi)))
        I_4th = (abs(I_v_4)/2*(np.cos(4*theta-4*phi)))
    else:
        I_2nd = (abs(I_v_2)/2*(np.cos(2*theta-2*phi)+1))
        I_4th = (abs(I_v_4)/2*(np.cos(4*theta-4*phi)+1))
    
    I_v = (I_2nd + I_4th)
    voigt_2d_lobe = b+I_v*(A_term+B_term)**2
    return voigt_2d_lobe
    
# Voigt function for 2D lobes
def Voigt_2D_neg_lobe_plot(qxx, qyy,  *args):
    u = qxx
    v = qyy
    
    # 'B'  'q0_L'  'Gamma_L'   'I_v_2'  'I_v_4'  'd_L'   'phi'   'X0'  'Y0'
    
    negative = True
    b = args[0]
    q_0 = args[1]
    G_v = args[2]
    I_v_2 = args[3]
    I_v_4 = args[4]
    d = args[5]
    phi = args[6]
    q_x_cent = args[7]
    q_y_cent = args[8]
    
    theta = np.arctan2(u-q_x_cent,v-q_y_cent)
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    c_L = 0.68188+0.61293*d-0.18384*d**2-0.11568*d**3
    c_G = 0.32460-0.61825*d+0.17681*d**2+0.12109*d**3
    
    A_term = c_L*(1/np.pi)*(G_v/((q-q_0)**2+G_v**2))
    B_term = c_G*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*G_v))*np.exp(-(np.log(2)*(q-q_0)**2)/G_v**2)
    
    if negative == True:
        I_2nd = (abs(I_v_2)/2*(np.cos(2*theta-2*phi)))
        I_4th = (abs(I_v_4)/2*(np.cos(4*theta-4*phi)))
    else:
        I_2nd = (abs(I_v_2)/2*(np.cos(2*theta-2*phi)+1))
        I_4th = (abs(I_v_4)/2*(np.cos(4*theta-4*phi)+1))
    
    I_v = (I_2nd + I_4th)
    voigt_2d_lobe = b+I_v*(A_term+B_term)**2
    return voigt_2d_lobe
    
    
# Voigt function for 2D full without odd har
def Voigt_2D_charge_p_ring_p_lobe_fit(xyflat, *args):
    u = xyflat[:,0]
    v = xyflat[:,1]
    
    # 'B'   'Gamma_C'   'I_C'  'd_C'  'q_R'  'Gamma_R'  'I_R'   'd_R'  'q_L'  'Gamma_L'   'I_L_2'  'I_L_4'  'd_L'  'phi_L'   'X0'  'Y0'
    
    # Use this to allow the lobe oscilation to go below zero in I
    negative_I = False
    
    # Background
    b = args[0]
    
    # Charge scattering
    q_C = 0
    g_C = args[1]
    i_C = args[2]
    d_C = args[3]
    
    # Ring scattering
    q_R = args[4]
    g_R = args[5]
    i_R = args[6]
    d_R = args[7]
    
    # Lobe scattering
    q_L = args[8]
    g_L = args[9]
    i_L_2 = args[10]
    i_L_4 = args[11]
    d_L = args[12]
    phi_L = args[13]
    q_x_cent = args[14]
    q_y_cent = args[15]
    
    # Defining theta and q
    theta = np.arctan2(u-q_x_cent,v-q_y_cent)
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    # Charge scattering function
    c_L_C = 0.68188+0.61293*d_C-0.18384*d_C**2-0.11568*d_C**3
    c_G_C = 0.32460-0.61825*d_C+0.17681*d_C**2+0.12109*d_C**3

    a_term_C = c_L_C*(1/np.pi)*(g_C/((q-q_C)**2+g_C**2))
    b_term_C = c_G_C*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_C))*np.exp(-(np.log(2)*(q-q_C)**2)/g_C**2)
    
    voigt_2d_charge = i_C*(a_term_C+b_term_C)**2
    
    # Ring function
    c_L_R = 0.68188+0.61293*d_R-0.18384*d_R**2-0.11568*d_R**3
    c_G_R = 0.32460-0.61825*d_R+0.17681*d_R**2+0.12109*d_R**3

    a_term_R = c_L_R*(1/np.pi)*(g_R/((q-q_R)**2+g_R**2))
    b_term_R = c_G_R*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_R))*np.exp(-(np.log(2)*(q-q_R)**2)/g_R**2)
    
    voigt_2d_ring = i_R*(a_term_R+b_term_R)**2
    
    # Lobes function
    c_L_L = 0.68188+0.61293*d_L-0.18384*d_L**2-0.11568*d_L**3
    c_G_L = 0.32460-0.61825*d_L+0.17681*d_L**2+0.12109*d_L**3

    a_term_L = c_L_L*(1/np.pi)*(g_L/((q-q_L)**2+g_L**2))
    b_term_L = c_G_L*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_L))*np.exp(-(np.log(2)*(q-q_L)**2)/g_L**2)
    
    if negative_I == True:
        i_L_2nd = (abs(i_L_2)/2*(np.cos(2*theta-2*phi_L)))
        i_L_4th = (abs(i_L_4)/2*(np.cos(4*theta-4*phi_L)))
    else:
        i_L_2nd = (abs(i_L_2)/2*(np.cos(2*theta-2*phi_L)+1))
        i_L_4th = (abs(i_L_4)/2*(np.cos(4*theta-4*phi_L)+1))
    
    i_term_L = (i_L_2nd + i_L_4th)
    voigt_2d_lobe = i_term_L*(a_term_L+b_term_L)**2
    
    return (b + voigt_2d_charge + voigt_2d_ring + voigt_2d_lobe)

def Voigt_2D_charge_p_ring_p_lobe_plot(qxx, qyy, *args):
    u = qxx
    v = qyy
    
    # 'B'  'q0_L'  'Gamma_L'   'I_v_2'  'I_v_4'  'd_L'   'phi'   'X0'  'Y0'
    
    # Use this to allow the lobe oscilation to go below zero in I
    negative_I = False
    
    # Background
    b = args[0]
    
    # Charge scattering
    q_C = 0
    g_C = args[1]
    i_C = args[2]
    d_C = args[3]
    
    # Ring scattering
    q_R = args[4]
    g_R = args[5]
    i_R = args[6]
    d_R = args[7]
    
    # Lobe scattering
    q_L = args[8]
    g_L = args[9]
    i_L_2 = args[10]
    i_L_4 = args[11]
    d_L = args[12]
    phi_L = args[13]
    q_x_cent = args[14]
    q_y_cent = args[15]
    
    # Defining theta and q
    theta = np.arctan2(u-q_x_cent,v-q_y_cent)
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    # Charge scattering function
    c_L_C = 0.68188+0.61293*d_C-0.18384*d_C**2-0.11568*d_C**3
    c_G_C = 0.32460-0.61825*d_C+0.17681*d_C**2+0.12109*d_C**3

    a_term_C = c_L_C*(1/np.pi)*(g_C/((q-q_C)**2+g_C**2))
    b_term_C = c_G_C*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_C))*np.exp(-(np.log(2)*(q-q_C)**2)/g_C**2)
    
    voigt_2d_charge = i_C*(a_term_C+b_term_C)**2
    
    # Ring function
    c_L_R = 0.68188+0.61293*d_R-0.18384*d_R**2-0.11568*d_R**3
    c_G_R = 0.32460-0.61825*d_R+0.17681*d_R**2+0.12109*d_R**3

    a_term_R = c_L_R*(1/np.pi)*(g_R/((q-q_R)**2+g_R**2))
    b_term_R = c_G_R*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_R))*np.exp(-(np.log(2)*(q-q_R)**2)/g_R**2)
    
    voigt_2d_ring = i_R*(a_term_R+b_term_R)**2
    
    # Lobes function
    c_L_L = 0.68188+0.61293*d_L-0.18384*d_L**2-0.11568*d_L**3
    c_G_L = 0.32460-0.61825*d_L+0.17681*d_L**2+0.12109*d_L**3

    a_term_L = c_L_L*(1/np.pi)*(g_L/((q-q_L)**2+g_L**2))
    b_term_L = c_G_L*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_L))*np.exp(-(np.log(2)*(q-q_L)**2)/g_L**2)
    
    if negative_I == True:
        i_L_2nd = (abs(i_L_2)/2*(np.cos(2*theta-2*phi_L)))
        i_L_4th = (abs(i_L_4)/2*(np.cos(4*theta-4*phi_L)))
    else:
        i_L_2nd = (abs(i_L_2)/2*(np.cos(2*theta-2*phi_L)+1))
        i_L_4th = (abs(i_L_4)/2*(np.cos(4*theta-4*phi_L)+1))
    
    i_term_L = (i_L_2nd + i_L_4th)
    voigt_2d_lobe = i_term_L*(a_term_L+b_term_L)**2
    
    return (b + voigt_2d_charge + voigt_2d_ring + voigt_2d_lobe)

def Voigt_full_2D_charge_plot(qxx, qyy, *args):
    u = qxx
    v = qyy
    
    # 'B'  'q0_L'  'Gamma_L'   'I_v_2'  'I_v_4'  'd_L'   'phi'   'X0'  'Y0'
    
    # Use this to allow the lobe oscilation to go below zero in I
    negative_I = False
    
    # Background
    b = args[0]
    
    # Charge scattering
    q_C = 0
    g_C = args[1]
    i_C = args[2]
    d_C = args[3]
    
    # Ring scattering
    q_R = args[4]
    g_R = args[5]
    i_R = args[6]
    d_R = args[7]
    
    # Lobe scattering
    q_L = args[8]
    g_L = args[9]
    i_L_2 = args[10]
    i_L_4 = args[11]
    d_L = args[12]
    phi_L = args[13]
    q_x_cent = args[14]
    q_y_cent = args[15]
    
    # Defining theta and q
    theta = np.arctan2(u-q_x_cent,v-q_y_cent)
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    # Charge scattering function
    c_L_C = 0.68188+0.61293*d_C-0.18384*d_C**2-0.11568*d_C**3
    c_G_C = 0.32460-0.61825*d_C+0.17681*d_C**2+0.12109*d_C**3

    a_term_C = c_L_C*(1/np.pi)*(g_C/((q-q_C)**2+g_C**2))
    b_term_C = c_G_C*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_C))*np.exp(-(np.log(2)*(q-q_C)**2)/g_C**2)
    
    voigt_2d_charge = i_C*(a_term_C+b_term_C)**2
    
    # Ring function
    c_L_R = 0.68188+0.61293*d_R-0.18384*d_R**2-0.11568*d_R**3
    c_G_R = 0.32460-0.61825*d_R+0.17681*d_R**2+0.12109*d_R**3

    a_term_R = c_L_R*(1/np.pi)*(g_R/((q-q_R)**2+g_R**2))
    b_term_R = c_G_R*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_R))*np.exp(-(np.log(2)*(q-q_R)**2)/g_R**2)
    
    voigt_2d_ring = i_R*(a_term_R+b_term_R)**2
    
    # Lobes function
    c_L_L = 0.68188+0.61293*d_L-0.18384*d_L**2-0.11568*d_L**3
    c_G_L = 0.32460-0.61825*d_L+0.17681*d_L**2+0.12109*d_L**3

    a_term_L = c_L_L*(1/np.pi)*(g_L/((q-q_L)**2+g_L**2))
    b_term_L = c_G_L*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_L))*np.exp(-(np.log(2)*(q-q_L)**2)/g_L**2)
    
    if negative_I == True:
        i_L_2nd = (abs(i_L_2)/2*(np.cos(2*theta-2*phi_L)))
        i_L_4th = (abs(i_L_4)/2*(np.cos(4*theta-4*phi_L)))
    else:
        i_L_2nd = (abs(i_L_2)/2*(np.cos(2*theta-2*phi_L)+1))
        i_L_4th = (abs(i_L_4)/2*(np.cos(4*theta-4*phi_L)+1))
    
    i_term_L = (i_L_2nd + i_L_4th)
    voigt_2d_lobe = i_term_L*(a_term_L+b_term_L)**2
    
    return (voigt_2d_charge)

def Voigt_full_2D_ring_plot(qxx, qyy, *args):
    u = qxx
    v = qyy
    
    # 'B'  'q0_L'  'Gamma_L'   'I_v_2'  'I_v_4'  'd_L'   'phi'   'X0'  'Y0'
    
    # Use this to allow the lobe oscilation to go below zero in I
    negative_I = False
    
    # Background
    b = args[0]
    
    # Charge scattering
    q_C = 0
    g_C = args[1]
    i_C = args[2]
    d_C = args[3]
    
    # Ring scattering
    q_R = args[4]
    g_R = args[5]
    i_R = args[6]
    d_R = args[7]
    
    # Lobe scattering
    q_L = args[8]
    g_L = args[9]
    i_L_2 = args[10]
    i_L_4 = args[11]
    d_L = args[12]
    phi_L = args[13]
    q_x_cent = args[14]
    q_y_cent = args[15]
    
    # Defining theta and q
    theta = np.arctan2(u-q_x_cent,v-q_y_cent)
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    # Charge scattering function
    c_L_C = 0.68188+0.61293*d_C-0.18384*d_C**2-0.11568*d_C**3
    c_G_C = 0.32460-0.61825*d_C+0.17681*d_C**2+0.12109*d_C**3

    a_term_C = c_L_C*(1/np.pi)*(g_C/((q-q_C)**2+g_C**2))
    b_term_C = c_G_C*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_C))*np.exp(-(np.log(2)*(q-q_C)**2)/g_C**2)
    
    voigt_2d_charge = i_C*(a_term_C+b_term_C)**2
    
    # Ring function
    c_L_R = 0.68188+0.61293*d_R-0.18384*d_R**2-0.11568*d_R**3
    c_G_R = 0.32460-0.61825*d_R+0.17681*d_R**2+0.12109*d_R**3

    a_term_R = c_L_R*(1/np.pi)*(g_R/((q-q_R)**2+g_R**2))
    b_term_R = c_G_R*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_R))*np.exp(-(np.log(2)*(q-q_R)**2)/g_R**2)
    
    voigt_2d_ring = i_R*(a_term_R+b_term_R)**2
    
    # Lobes function
    c_L_L = 0.68188+0.61293*d_L-0.18384*d_L**2-0.11568*d_L**3
    c_G_L = 0.32460-0.61825*d_L+0.17681*d_L**2+0.12109*d_L**3

    a_term_L = c_L_L*(1/np.pi)*(g_L/((q-q_L)**2+g_L**2))
    b_term_L = c_G_L*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_L))*np.exp(-(np.log(2)*(q-q_L)**2)/g_L**2)
    
    if negative_I == True:
        i_L_2nd = (abs(i_L_2)/2*(np.cos(2*theta-2*phi_L)))
        i_L_4th = (abs(i_L_4)/2*(np.cos(4*theta-4*phi_L)))
    else:
        i_L_2nd = (abs(i_L_2)/2*(np.cos(2*theta-2*phi_L)+1))
        i_L_4th = (abs(i_L_4)/2*(np.cos(4*theta-4*phi_L)+1))
    
    i_term_L = (i_L_2nd + i_L_4th)
    voigt_2d_lobe = i_term_L*(a_term_L+b_term_L)**2
    
    return (voigt_2d_ring)

def Voigt_full_2D_lobe_plot(qxx, qyy, *args):
    u = qxx
    v = qyy
    
    # 'B'  'q0_L'  'Gamma_L'   'I_v_2'  'I_v_4'  'd_L'   'phi'   'X0'  'Y0'
    
    # Use this to allow the lobe oscilation to go below zero in I
    negative_I = False
    
    # Background
    b = args[0]
    
    # Charge scattering
    q_C = 0
    g_C = args[1]
    i_C = args[2]
    d_C = args[3]
    
    # Ring scattering
    q_R = args[4]
    g_R = args[5]
    i_R = args[6]
    d_R = args[7]
    
    # Lobe scattering
    q_L = args[8]
    g_L = args[9]
    i_L_2 = args[10]
    i_L_4 = args[11]
    d_L = args[12]
    phi_L = args[13]
    q_x_cent = args[14]
    q_y_cent = args[15]
    
    # Defining theta and q
    theta = np.arctan2(u-q_x_cent,v-q_y_cent)
    q = np.sqrt((u-q_x_cent)**2+(v-q_y_cent)**2)
    
    # Charge scattering function
    c_L_C = 0.68188+0.61293*d_C-0.18384*d_C**2-0.11568*d_C**3
    c_G_C = 0.32460-0.61825*d_C+0.17681*d_C**2+0.12109*d_C**3

    a_term_C = c_L_C*(1/np.pi)*(g_C/((q-q_C)**2+g_C**2))
    b_term_C = c_G_C*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_C))*np.exp(-(np.log(2)*(q-q_C)**2)/g_C**2)
    
    voigt_2d_charge = i_C*(a_term_C+b_term_C)**2
    
    # Ring function
    c_L_R = 0.68188+0.61293*d_R-0.18384*d_R**2-0.11568*d_R**3
    c_G_R = 0.32460-0.61825*d_R+0.17681*d_R**2+0.12109*d_R**3

    a_term_R = c_L_R*(1/np.pi)*(g_R/((q-q_R)**2+g_R**2))
    b_term_R = c_G_R*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_R))*np.exp(-(np.log(2)*(q-q_R)**2)/g_R**2)
    
    voigt_2d_ring = i_R*(a_term_R+b_term_R)**2
    
    # Lobes function
    c_L_L = 0.68188+0.61293*d_L-0.18384*d_L**2-0.11568*d_L**3
    c_G_L = 0.32460-0.61825*d_L+0.17681*d_L**2+0.12109*d_L**3

    a_term_L = c_L_L*(1/np.pi)*(g_L/((q-q_L)**2+g_L**2))
    b_term_L = c_G_L*(np.sqrt(np.log(2))/(np.sqrt(np.pi)*g_L))*np.exp(-(np.log(2)*(q-q_L)**2)/g_L**2)
    
    if negative_I == True:
        i_L_2nd = (abs(i_L_2)/2*(np.cos(2*theta-2*phi_L)))
        i_L_4th = (abs(i_L_4)/2*(np.cos(4*theta-4*phi_L)))
    else:
        i_L_2nd = (abs(i_L_2)/2*(np.cos(2*theta-2*phi_L)+1))
        i_L_4th = (abs(i_L_4)/2*(np.cos(4*theta-4*phi_L)+1))
    
    i_term_L = (i_L_2nd + i_L_4th)
    voigt_2d_lobe = i_term_L*(a_term_L+b_term_L)**2
    
    return (voigt_2d_lobe)

