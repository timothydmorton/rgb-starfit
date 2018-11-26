import os

import pandas as pd
from configobj import ConfigObj

ROOT = os.path.join(os.getenv('PROJECT_DATA'), 'rgb-starfit')
STARMODEL_ROOT = os.path.join(ROOT, 'starmodels')

os.makedirs(STARMODEL_ROOT, exist_ok=True)

rgb_df = pd.read_hdf(os.path.join(ROOT, 'rgb.hdf'), 'df')

bands = ['J', 'H', 'K', 'G','BP', 'RP', 
         'W1', 'W2', 'W3', 'W4', 
         'IRAC_3.6', 'IRAC_4.5', 'IRAC_5.8', 'IRAC_8.0']

for _, star in rgb_df.iloc.iterrows():
    name = star['tmass_key']
    os.makedirs(os.path.join(STARMODEL_ROOT, str(name)), exist_ok=True)
    filename = os.path.join(STARMODEL_ROOT, str(name), 'star.ini')
    c = ConfigObj(filename, create_empty=True)
    c['ra'] = star['ra_1']
    c['dec'] = star['dec_1']
    c['parallax'] = (star['parallax'], star['parallax_error'])
    
    for band in bands:
        val = star[band]
        unc = star['{}_unc'.format(band)]
        if val > 0 and unc > 0:
            c[band] = (val, unc)
    
    c.write()