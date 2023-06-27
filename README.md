# 2D_and_time_constant_fit_demo
 A demonstration of 2D fitting of a phenomenological model on small angle x-ray scattering (SAXS) from coexisting labyrinthine and stripe domain morphologies for publication uploaded to arXiv at this DOI:https://doi.org/10.48550/arXiv.2303.16131.

## Structure of raw data contained in Raw_data directory
The Raw_data directory contains data from one scan measured at pump fluence of 13.4 mJ/cm<sup>2</sup> and 1 mJ/cm<sup>2</sup> EUV (Ni M<sub>3</sub> edge) fluence. The directory contains four subfolders labeled as follows:-
1. WX15_Y16_Scan041/rawdata:- contains the raw detector images from the scan.
2. WX15_Y16_Scan041_BG/rawdata:- contains detector background images.
3. WX15_Y16_Scan041_OF/rawdata:- contains detector images with just X-rays on the samples.
4. WX15_Y16_Scan041_OL/rawdata:- contains detector images with just the pump laser on the samples.

## Instructions for running 2D fits using Code_for_2D_fitting.ipynb
One can run the 2D fitting using Code_for_2D_fitting.ipynb file after installing the following python libraries:-
- numpy, matplotlib, pandas, scipy, h5py, h5glance, imageio, pillow, labellines.
The setup parameters for the 2D fit are defined in cells labeled "File path and analysis options". Uncomment the lines in the last cell to save the fit parameter results from 2D fitting as .npy files.
