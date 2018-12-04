import os

import pandas as pd
import numpy as np
import schwimmbad

from isochrones.starmodel import BasicStarModel
from isochrones.priors import FehPrior, FlatPrior, GaussianPrior
from isochrones.extinction import get_AV_infinity

from isochrones import get_ichrone

PROJECT_DIR = os.path.join(os.getenv('PROJECT_DATA', '/Users/tdm/dbufl/projects'), 'rgb-starfit')

def mod_from_row(iso, row, props, name_col='tmass_key', 
                 rootdir=os.path.join(PROJECT_DIR, 'fit_results'),
                 halo_fraction=0.5, tag=None, **kwargs):
    
    kwargs.update({p: (np.float64(row[p]), np.float64(row['{}_unc'.format(p)])) for p in props 
                   if row[p] > 0 and row['{}_unc'.format(p)] > 0})
    if 'feh' in props: 
        if np.isfinite(row['feh']):
            kwargs.update({'feh': (np.float64(row['feh']), np.float64(row['feh_unc']))})
    name = row[name_col]
    if tag is not None:
        name = '{}_{}'.format(name, tag)
    kwargs.update({'name': name, 'directory': os.path.join(rootdir, name)})
    
    max_distance = min((1000./(row['parallax'] - row['parallax_unc']))*2, 30000)
    kwargs['max_distance'] = max_distance
    kwargs['halo_fraction'] = halo_fraction
    mod = BasicStarModel(iso, **kwargs)
    mod.set_prior('distance', FlatPrior((0, max_distance)))
    AV_inf = get_AV_infinity(row['ra_1'], row['dec_1'])
    mod.set_prior('feh', FehPrior(halo_fraction=halo_fraction, local=False))
    mod.set_prior('AV', GaussianPrior(AV_inf, AV_inf/2, bounds=(0, 5)))
    mod.set_bounds(mass=(0.1, 20), feh=iso.model_grid.get_limits('feh'))
    print(mod.mnest_basename)
    return mod

class Worker(object):
    def __init__(self, bands, name_col='tmass_key', 
                 results_basename='results', 
                 columns=['mass', 'radius', 'age', 'Teff', 'logg', 'feh', 'distance', 'AV'],
                 quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],
                 fit_kwargs=None):
        self.name_col = name_col
        self.results_basename = results_basename
        self.columns = columns
        self.bands = bands
        self.quantiles = quantiles
        self.fit_kwargs = {} if fit_kwargs is None else fit_kwargs

        # Initialize column headers
        for filename in [self.spec_summary_file, self.phot_summary_file]:
            with open(filename, 'w') as fout:
                s = self.name_col + ' '
                for c in columns + bands:
                    for q in quantiles:
                        s += '{}_{:02.0f} '.format(c, q*100)

                s += 'posterior_predictive'
                fout.write(s + '\n')
        
    @property
    def spec_summary_file(self):
        return os.path.join(PROJECT_DIR, '{}_spec.txt'.format(self.results_basename))

    @property
    def phot_summary_file(self):
        return os.path.join(PROJECT_DIR, '{}_phot.txt'.format(self.results_basename))
        
    def work(self, row):    

        track = get_ichrone('mist', tracks=True, bands=self.bands)
        track.initialize()
        
        mod_spec = mod_from_row(track, row, name_col=self.name_col, 
                                tag='spec', props=['Teff', 'logg', 'feh', 'parallax'])
        mod_phot = mod_from_row(track, row, name_col=self.name_col,
                                tag='phot', props=self.bands + ['parallax'])
        
        name = str(row[self.name_col])
        print('fitting spec model for {}'.format(name))
        mod_spec.fit(**self.fit_kwargs)
        mod_spec.write_results()
        print('spec model for {} complete; results written.'.format(name))
        
        print('fitting phot model for {}'.format(name))
        mod_phot.fit(**self.fit_kwargs)
        mod_phot.write_results()
        print('phot model for {} complete; results written.'.format(name))

        cols = self.columns + ['{}_mag'.format(b) for b in self.bands]
        results_spec = mod_spec.derived_samples[cols].quantile(self.quantiles)
        results_phot = mod_phot.derived_samples[cols].quantile(self.quantiles)    
        
        spec = '{} '.format(row[self.name_col])
        phot = '{} '.format(row[self.name_col])        
        for c in cols:
            for q in self.quantiles:
                spec += '{:.3f} '.format(results_spec.loc[q, c])
                phot += '{:.3f} '.format(results_phot.loc[q, c])
        spec += '{:.3f}'.format(mod_spec.posterior_predictive)
        phot += '{:.3f}'.format(mod_phot.posterior_predictive)
        
        return spec, phot
    
    def write_results(self, result):
        spec, phot = result
        with open(self.spec_summary_file, 'a') as fout:
            fout.write(spec + '\n')
        with open(self.phot_summary_file, 'a') as fout:
            fout.write(phot + '\n')
            
    def __call__(self, row):
        return self.work(row)
        
def main(pool, filename=None, n_rows=None):

    if filename is None:
        filename = os.path.join(PROJECT_DIR, 'rgb.hdf')

    rgbs = pd.read_hdf(filename, 'df')
    if n_rows is not None:
        rgbs = rgbs.iloc[:n_rows]
    
    bands = ['G','BP', 'RP', 'J', 'H', 'K'] #, 'W1', 'W2', 'W3', 'W4']#, 'IRAC_3.6', 'IRAC_4.5', 'IRAC_5.8', 'IRAC_8.0']
    
    fit_kwargs = dict(force_no_MPI=True, verbose=False)
    worker = Worker(bands, fit_kwargs=fit_kwargs)
    
    for r in pool.map(worker, [row for _, row in rgbs.iterrows()],
                      callback=worker.write_results):
        pass
    
    pool.close()

if __name__ == '__main__':
    from argparse import ArgumentParser
        
    from isochrones import get_ichrone
    
    parser = ArgumentParser(description="Run stellar model fits.")
    parser.add_argument('--nrows', dest='n_rows', type=int, default=None)
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args = parser.parse_args()    

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    main(pool, n_rows=args.n_rows)
    
    
    
    
