import re
import os

from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np

# Read in full table from Eilers et al.

eilers_table_filename = '/Users/tdm/Downloads/spectrophotometric_parallax_HER2018.fits'

table = fits.getdata(eilers_table_filename)
rgbs = Table(table)


# Export selected columns to pandas DataFrame
columns = ['J', 'J_ERR', 'H', 'H_ERR', 'K', 'K_ERR', 'source_id2', 'ref_epoch', 'ra_1', 'ra_error_1', 'dec_1', 'dec_error_1', 
           'parallax', 'parallax_error', 'duplicated_source', 'phot_g_n_obs', 'phot_g_mean_flux', 'phot_g_mean_flux_error',
           'phot_g_mean_flux_over_error', 'phot_g_mean_mag', 'phot_bp_n_obs', 'phot_bp_mean_flux', 'phot_bp_mean_flux_error',
           'phot_bp_mean_flux_over_error', 'phot_bp_mean_mag', 'phot_rp_n_obs', 'phot_rp_mean_flux', 'phot_rp_mean_flux_error',
           'phot_rp_mean_flux_over_error', 'phot_rp_mean_mag', 'IRAC_3_6', 'IRAC_3_6_ERR', 'IRAC_4_5', 'IRAC_4_5_ERR',
           'IRAC_5_8', 'IRAC_5_8_ERR', 'IRAC_8_0', 'IRAC_8_0_ERR', 'w1mpro', 'w1mpro_error', 'w2mpro', 'w2mpro_error',
           'w3mpro', 'w3mpro_error', 'w4mpro', 'w4mpro_error', 'tmass_key', 'spec_parallax', 'spec_parallax_err']

rgb_df = rgbs[columns].to_pandas()

# Compute Gaia magnitudes from fluxes according to corrected ZPs

ZP_G = 25.6914
ZP_BP = 25.3488
ZP_RP = 24.7627

G_flux = rgb_df['phot_g_mean_flux']
G_flux_unc = rgb_df['phot_g_mean_flux_error']

BP_flux = rgb_df['phot_bp_mean_flux']
BP_flux_unc = rgb_df['phot_bp_mean_flux_error']

RP_flux = rgb_df['phot_rp_mean_flux']
RP_flux_unc = rgb_df['phot_rp_mean_flux_error']

rgb_df['G'] = -2.5 * np.log10(G_flux) + ZP_G
rgb_df['BP'] = -2.5 * np.log10(BP_flux) + ZP_BP
rgb_df['RP'] = -2.5 * np.log10(RP_flux) + ZP_RP

rgb_df['G_unc'] = (2.5 / np.log(10)) * G_flux_unc/G_flux
rgb_df['BP_unc'] = (2.5 / np.log(10)) * BP_flux_unc/BP_flux
rgb_df['RP_unc'] = (2.5 / np.log(10)) * RP_flux_unc/RP_flux

band_map = dict(J='J', H='H', K='K',
                w1mpro='W1', w2mpro='W2', w3mpro='W3', w4mpro='W4',
                IRAC_3_6='IRAC_3.6', IRAC_4_5='IRAC_4.5', IRAC_5_8='IRAC_5.8', IRAC_8_0='IRAC_8.0')

for c in rgb_df.columns:
    m = re.search('(\w+)_[Ee][Rr][Rr]', c)
    if m:
        if m.group(1) in band_map:
            band_map.update({c: '{}_unc'.format(band_map[m.group(1)])})

rgb_df.rename(columns=band_map, inplace=True)

rgb_df.to_hdf('data/rgb.hdf', 'df')
