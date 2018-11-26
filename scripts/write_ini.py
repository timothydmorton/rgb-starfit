import os

import pandas as pd
from configobj import ConfigObj

ROOT = os.path.join(os.getenv('PROJECT_DATA'), 'rgb-starfit')
STARMODEL_ROOT = os.path.join(ROOT, 'starmodels')

os.makedirs(STARMODEL_ROOT, exist_ok=True)

rgb_df = pd.read_hdf(os.path.join(ROOT, 'rgbs.hdf'), 'df')

for _, star in rgb_df.iloc[:10].iterrows():
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