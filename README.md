# 2D and Time Constant Fit Demonstration

This repository showcases a demonstration of 2D fitting of a phenomenological model on small-angle X-ray scattering (SAXS) data, capturing coexisting labyrinthine and stripe domain morphologies. The corresponding publication has been uploaded to arXiv and can be accessed using the following DOI: [https://doi.org/10.48550/arXiv.2303.16131](https://doi.org/10.48550/arXiv.2303.16131).

## Structure of Raw Data in the "Raw_data" Directory

The "Raw_data" directory contains the raw data from multiple scans, obtained at a pump fluence ranging from 0.8 to 13.4 mJ/cm<sup>2</sup> and 1 mJ/cm<sup>2</sup> EUV (Ni M<sub>3</sub> edge) fluence. Inside the directory, you will find four subfolders labeled as follows:

1. WX#_Y#_Scan0#/rawdata: Contains the raw detector images from the scan.
2. WX#_Y#_Scan0#_BG/rawdata: Contains detector background images.
3. WX#_Y#_Scan0#_OF/rawdata: Contains detector images with only X-rays on the samples.
4. WX#_Y#_Scan0#_OL/rawdata: Contains detector images with only the pump laser on the samples.

Below are the pump fluences for each scan in the "Raw_data" directory. These scans correspond to the data presented in the manuscript.

1. WX3_Y17_Scan009: 0.8 mJ/cm<sup>2</sup>
2. WX3_Y17_Scan011: 2.1 mJ/cm<sup>2</sup>
3. WX15_Y16_Scan031: 5.0 mJ/cm<sup>2</sup>
4. WX15_Y16_Scan032: 5.9 mJ/cm<sup>2</sup>
5. WX15_Y16_Scan033: 6.7 mJ/cm<sup>2</sup>
6. WX15_Y16_Scan034: 7.5 mJ/cm<sup>2</sup>
7. WX15_Y16_Scan035: 8.4 mJ/cm<sup>2</sup>
8. WX15_Y16_Scan036: 9.2 mJ/cm<sup>2</sup>
9. WX15_Y16_Scan037: 10.0 mJ/cm<sup>2</sup>
10. WX15_Y16_Scan038: 10.9 mJ/cm<sup>2</sup>
11. WX15_Y16_Scan039: 11.7 mJ/cm<sup>2</sup>
12. WX15_Y16_Scan040: 12.6 mJ/cm<sup>2</sup>
13. WX15_Y16_Scan041: 13.4 mJ/cm<sup>2</sup>

## Instructions for Running 2D Fits Using Code_for_2D_fitting.ipynb

To run the 2D fitting, please ensure that the following Python libraries are installed: numpy, matplotlib, pandas, scipy, h5py, h5glance, imageio, pillow, and labellines. Once installed, you can execute the 2D fitting by running the Code_for_2D_fitting.ipynb file.

The setup parameters for the 2D fit are defined in cells labeled "File path and analysis options". In the last cell, uncomment the lines to save the fit parameter results from the 

## Instructions for Running 2D Fits Using Code_for_2D_fitting.ipynb

To run the time constant fits, please install the same aforementioned Python libraries. To do time constant fits, run Code_for_time_constant_fits.ipynb
