#!/usr/bin/env python
"""
Example application using ICA
"""
import logging
import resource
from pathlib import Path

import sd_access
from cml_data_tools.config import Config
from cml_data_tools.curves import *
from cml_data_tools.experiment import Experiment
from cml_data_tools.pickle_cache import PickleCache
from cml_data_tools.source_ehr import DataSource, MetaSource
from cml_data_tools.standardizers import *


configs = [
    Config(
        mode='Measurements',
        data=DataSource('cml_test_measurement', sd_access.connect),
        meta=MetaSource('cml_test_measurement_meta', sd_access.connect),
        curve_cls=RegressionCurveBuilder,
        std_cls=LogGelmanStandardizerWithFallbacks,
        std_kws={'log_default': False},
    ),
    Config(
        mode='Conditions',
        data=DataSource('cml_test_condition', sd_access.connect),
        meta=MetaSource('cml_test_condition_meta', sd_access.connect),
        curve_cls=IntensityCurveBuilder,
        std_cls=LogGelmanStandardizerWithFallbacks,
    ),
    Config(
        mode='Medications',
        data=DataSource('cml_test_medication', sd_access.connect),
        meta=MetaSource('cml_test_medication_meta', sd_access.connect),
        curve_cls=BinaryCurveBuilder,
        std_cls=LinearStandardizer,
        std_kws={'scale': 1},
    ),
    Config(
        mode='Race',
        data=DataSource('cml_test_race', sd_access.connect),
        meta=(
            ('W', 'White Race', 0),
            ('U', 'Unknown Race', 0),
            ('N', 'Native American Race', 0),
            ('B', 'Black Race', 0),
            ('M', 'Multiple Race', 0),
            ('A', 'Asian Race', 0),
            ('D', 'Declined Race', 0),
            ('I', 'Pacific Islander Race', 0),
            ('H', 'Hispanic Race', 0)
        ),
        curve_cls=ConstantCurveBuilder,
        std_cls=LinearStandardizer,
        std_kws={'scale': 1},
    ),
    Config(
        mode='Sex',
        data=DataSource('cml_test_sex', sd_access.connect),
        meta=(
            ('M', 'Male Sex', 0),
            ('F', 'Female Sex', 0),
            ('U', 'Unknown Sex', 0)
        ),
        curve_cls=ConstantCurveBuilder,
        std_cls=LinearStandardizer,
        std_kws={'scale': 1},
    ),
    Config(
        mode='Age',
        data=DataSource('cml_test_age', sd_access.connect),
        meta=(('Age', 'Age', 0), ),
        curve_cls=AgeCurveBuilder,
        std_cls=GelmanStandardizer,
    ),
    Config(
        mode='Bmi',
        data=DataSource('cml_test_bmi', sd_access.connect),
        meta=(('BMI', 'Clean BMI', 26.12), ),
        curve_cls=RegressionCurveBuilder,
        std_cls=GelmanStandardizer,
    ),
    Config(
        mode='ANA',
        data=DataSource('cml_test_ana', sd_access.connect),
        meta=(('ANA titer', 'ANA titer', 8.0), ),
        curve_cls=RegressionCurveBuilder,
        std_cls=GelmanStandardizer,
    ),
]

if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')

    N_MODEL = 5
    N_PHENT = 500
    project_name = 'E_05'
    loc = f'./{project_name}'

    logging.basicConfig(filename=f'ICA_{project_name}.log',
                        format='%(asctime)s %(name)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)
    logging.info(f'Setting up Experiment with cache loc {loc}')

    cache = PickleCache(loc=loc)
    experiment = e = Experiment(configs, cache)

    logging.info('Saving configuration objects')
    cache.set('configs', configs)

    logging.info('Fetching data & channel metadata')
    experiment.fetch_data()
    experiment.fetch_meta()

    logging.info('Computing Curves')
    experiment.compute_curves(max_workers=48)

    logging.info('Constructing standardizer')
    experiment.make_standardizer()

    logging.info('Beginning to train submodels')
    for i in range(N_MODEL):
        logging.info(f'Training submodel {i:03}')
        path = Path(cache.loc/f'segment_{i:03}')

        # Complex cache skipping not yet implemented, so to avoid redoing all
        # those standardized matrices / etc we do this for now
        if path.exists():
            continue

        with cache.relocate(path):
            logging.info(f'Computing cross sections for model {i:03}')
            experiment.compute_cross_sections(curves_key='../curves')
            # create and standardize data matrix in one step
            logging.info(f'Computing data matrix for model {i:03}')
            experiment.build_standardized_data_matrix(key='std_matrix',
                                                      meta_key='../meta',
                                                      std_key='../standardizer',
                                                      save_dense=False)
            experiment.learn_model(max_phenotypes=N_PHENT)
            # Remove the std matrix (~40G) for storage considerations
            cache.remove('std_matrix')

            # Try to track memory usage so we can see if we run out of memory
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            logging.info(f'RSS usage: {mem}Kb')

    logging.info('Submodel training complete')
    model_keys = sorted(cache.loc.glob('segment_*/model.pkl'))

#    # Generate PDFs if they don't already exist
#    pdf_dir = cache.loc/'pdfs'
#    try:
#        pdf_dir.mkdir()
#    except FileExistsError:
#        pass
#    else:
#        for i, model_key in enumerate(model_keys):
#            experiment.plot_model(pdf_path=pdf_dir/f'phenotypes_{i:03}.pdf',
#                                  model_key=model_key)

    # Collect phenotypes
    logging.info('Collecting submodel phenotypes together')
    experiment.collect_phenotypes(model_keys)

    logging.info('Creating affinity matrix from collected phenotypes')
    experiment.create_affinity_matrix()

    logging.info('Clustering phenotypes from submodel')
    experiment.cluster_affinities(damping=0.82, threshold=0.6)

    logging.info('Extracting cluster exemplar expressions')
    experiment.extract_exemplar_expressions(model_keys)
