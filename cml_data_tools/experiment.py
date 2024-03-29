import collections
import contextlib
import functools
import inspect
import itertools
import logging
import operator
import pathlib
import pickle
from concurrent.futures import as_completed, ProcessPoolExecutor, TimeoutError

import numpy as np
import pandas as pd

from cml_data_tools.curves import build_patient_curves
from cml_data_tools.clustering import (make_affinity_matrix, iter_clusters,
                                       AffinityPropagationClusterer)
from cml_data_tools.expand_and_fill import expand_and_fill_cross_sections
from cml_data_tools.models import IcaPhenotypeModel
from cml_data_tools.parallelize import parallelize_func
from cml_data_tools.pickle_cache import PickleCache
from cml_data_tools.sampling import binomial_sample_curves, sample_latest
from cml_data_tools.source_ehr import (make_data_df, make_meta_df,
                                       aggregate_data, aggregate_meta)
from cml_data_tools.standardizers import CurveStats, Standardizer


def _curve_gen_worker(df, spec, resolution, calc_stats):
    curves = build_patient_curves(df, spec, resolution)
    if calc_stats:
        stats = CurveStats.from_curves(curves)
    else:
        stats = None
    return curves, stats


def _channel_count_worker(df):
    return {k: len(v) for k, v in df.groupby(['mode', 'channel'])}


def _omit_insufficient_dataframes(dataframe_iter):
    for df in dataframe_iter:
        if len(df.date.dropna().unique()) > 1:
            yield df


def _trunc_to_date(dates, stream):
    """`dates` is a mapping from patient ID's to dates. For each unique
    (ptid, date) pair yields a dataframe with that patient's data truncated to
    the given date. Each yielded dataframe uses the patient ID concatenated to
    the controlling date as patient ID.

    Arguments
    ---------
    dates : Mapping
        Maps patient ids to lists of dates.
    stream : Iterator[pd.DataFrame]
        An iterator yielding pandas dataframes
    """
    for df in stream:
        ptid = df.ptid[0]
        #for dt in dates.get(ptid, [df.date.max()]):
        try:
            ptdates = dates[ptid]
        except KeyError:
            # Skip data if not present in `dates` mapping
            continue
        else:
            nats = df[np.isnat(df.date)]
            for dt in ptdates:
                sub_df = pd.concat([nats, df[df.date <= dt]])
                sub_df.ptid = f'{ptid}_{str(dt.date())}'
                yield sub_df


def cached_operation(func):
    default_key = inspect.signature(func).parameters['key'].default
    @functools.wraps(func)
    def wrapper(self, *args, force=False, **kwargs):
        key = kwargs.get('key', default_key)
        if key not in self.cache or force:
            return func(self, *args, **kwargs)
    return wrapper


class Experiment:
    """
    An Experiment instance is a state machine which orchestrates the various
    phases and tasks involved in running an experiment. The experiment must be
    configured with a list (or other sortable container) of
    ``cml_data_tools.Config`` objects specifying how a data mode is transformed
    into floats and standardized prior to training any model. The experiment
    also requires a cache, which provides the working memory for the
    experiment. The default cache is a PickleCache instance pointed at the
    current working directory.

    Parameters
    ----------
    configs : Config[]
        A list or other sortable container of Config (or compatible) objects
    cache : PickleCache
        An instance of a PickleCache (or compatible) class

    Notes
    -----
    The methods which represent experiment operations follow some common
    patterns. Each returns None; these operations manipulate state (i.e. the
    cache) rather than return a value.

    Each has a keyword argument "key" which is used to set the results of the
    operation in the cache. Thus, if we wanted to compute two cross sections
    from the same set of curves, for example, we could call generate cross
    sections (via `compute_cross_sections`) at two different densities and save
    the results of both operations. I.e.

    >>> experiment.compute_cross_sections(key='xs_yearly', density=1/365)
    >>> experiment.compute_cross_sections(key='xs_monthly', density=1/30)

    Our state-machine memory now contains two entries of cross sections,
    `xs_yearly` and `xs_monthly`, taken from the curves once yearly and once
    monthly respectively.

    Many operations rely on previous computations. These are frequently
    accessed from the cache by appropriately named keys. For example, if we
    wished to compute data matrices for our previously computed sampling rates,
    we would execute:

    >>> experiment.build_data_matrix(key='dm_yearly', xs_key='xs_yearly')
    >>> experiment.build_data_matrix(key='dm_monthly', xs_key='xs_monthly')

    Check each operation's documentation for what other keys it needs in order
    to function.

    The state-machine's memory is truly called a cache. Cached operations check
    the cache before executing, and if their key exists they return.  Each
    cached operation accepts a keyword argument "force", which will force
    re-execution of the operation. For example, in the following sequence

    >>> experiment.fetch_data()
    >>> experiment.fetch_data()
    >>> experiment.fetch_data(force=True)

    The first line establishes a DB connection and downloads the data, storing
    it in the cache at the default key "data." The second line is then a noop.
    The third line re-runs `fetch_data` and overwrites "data" in the cache.
    """
    def __init__(self, configs, cache=None):
        self.configs = sorted(configs, key=operator.attrgetter('mode'))
        # Cache must have an interface compatible with PickleCache, which is
        # assumed to be the default implementation. Experiment state-machine
        # methods will not work without a proper cache to get/set results to
        if cache is None:
            self.cache = PickleCache()
        self.cache = cache

    @cached_operation
    def fetch_data(self, key='data', min_inst=2, min_span=182,
                   counts_key='channel_counts'):
        """Download patient data

        Aggregates by patient ID and converts to pandas.DataFrame before
        caching as a stream of DataFrames. Cf. cml_data_tools.source_ehr

        Keyword Arguments
        -----------------
        key : str
            Default 'data'. Specifies cache key for the fetched data.
        min_inst : int
            Default 2. Minimum number of distinct dates a patient record must
            have in order to be used for further processing.
        min_span : int
            Default 182. Minimum distance between the earliest and latest date
            in the patient record for which there is data. Default is 6 months.
        """
        data_iter = map(make_data_df, aggregate_data(self.configs))
        # Filter out persons who don't have at least min_inst distinct
        # dates of data
        # TODO: min_span
        data_iter = (
            df for df in data_iter
            if len(df['date'][~np.isnan(df['date'])].unique()) >= min_inst
        )
        def intercept_counts(stream):
            grp_key = ['mode', 'channel']
            counts = []
            for df in stream:
                counts.append(_channel_count_worker(df))
                yield df
            self.cache.set(counts_key, counts)
        self.cache.set_stream(key, intercept_counts(data_iter))

    @cached_operation
    def count_channels(self, key='channel_counts',
                       data_key='data',
                       max_workers=0):
        """Count the channels per patient in the dataset.

        Keyword Arguments
        -----------------
        key : str
            Default 'channel_counts'. Specifies cache key for the counts.
        data_key : str
            Default 'data'. Specifies the cache key for the patient dataframes.
        max_workers : int
            Default 0. If above zero, specifies the number of subprocesses to
            use in generating curves in parallel. At default of zero, no
            parallelization is used.
        """
        counts = []
        data = self.cache.get_stream(data_key)
        if max_workers > 0:
            counts = list(parallelize_func(_channel_count_worker, data,
                                           max_workers=max_workers))
        else:
            counts = [_channel_count_worker(df) for df in data]
        self.cache.set(key, counts)

    @cached_operation
    def fetch_meta(self, key='meta'):
        """Download modal data metadata. Cf. cml_data_tools.source_ehr

        Keyword Arguments
        -----------------
        key : str
            Default 'meta'. Specifies cache key for the fetched metadata.
        """
        meta = make_meta_df(aggregate_meta(self.configs))
        self.cache.set(key, meta)

    def filter_data(self, dates, key='filtered_data', data_key='data'):
        """`dates` is a mapping from patient ID's to dates. For each unique
        (ptid, date) pair a patient data dataframe from `data_key` with that
        patient's data truncated to the given date is stored at `key` Each
        generated dataframe uses the patient ID concatenated to the controlling
        date as patient ID.

        `key` and `data_key` must be different.

        Arguments
        ---------
        dates : Mapping
            A mapping from patient ID's to lists of dates.

        Keyword Arguments
        -----------------
        key : str
            Default 'filtered_data'. Specifies cache key for the filtered data.
        data_key : str
            Default 'data'. Specifies cache key for the underlying dataset.
        """
        data = self.cache.get_stream(data_key)
        self.cache.set_stream(key, _trunc_to_date(dates, data))

    @cached_operation
    def compute_curves(self, key='curves',
                       data_key='data',
                       curve_stats_key='curve_stats',
                       n_instances_key='n_instances',
                       resolution='D',
                       extra_curve_kws=None,
                       max_workers=0,
                       calc_stats=True):
        """Compute patient curves. Transforms patient data from heterogeneous
        data types into curves of floats.

        Keyword Arguments
        -----------------
        key : str
            Default 'curves'. The key for the generated curves.
        data_key : str
            Default 'data'. The key of input patient dataframes.
        curve_stats_key : str
            Default 'curve_stats'. The key for curve statistics. Cf.
            `calc_stats`.
        resolution : str
            Default 'D', i.e. daily. The resolution of the generated curves.
        extra_curve_kws : Mapping, optional
            Extra curve gen configuration.
        max_workers : int
            Default 0. If above zero, specifies the number of subprocesses to
            use in generating curves in parallel. At default of zero, no
            parallelization is used.
        calc_stats : bool
            Default True. Flag indicating whether or not to calculate curve
            statistics alongside curve generation. If True, then the curve
            stats are stored at `curve_stats_key`. This allows us to
            parallelize collection of curve statistics, which otherwise must be
            run in serial.
        """
        xtra = extra_curve_kws or {}
        spec = {}
        for config in self.configs:
            extra_kws = xtra.get(config.mode, {})
            func = config.curve_builder(**extra_kws)
            spec[config.mode] = func

        data = self.cache.get_stream(data_key)
        # Wrap the data iterator in a filter to remove entries without enough
        # data points to generate curves over (i.e. patient datasets without at
        # least two data points located in time are filtered)
        data = _omit_insufficient_dataframes(data)

        if max_workers > 0:
            curves_iter = parallelize_func(
                _curve_gen_worker, data,
                spec, resolution, calc_stats,
                max_workers=max_workers,
            )
        # Single core execution
        else:
            curves_iter = (
                _curve_gen_worker(df, spec, resolution, calc_stats)
                for df in data
            )

        if calc_stats:
            def intercept_stats(curves_iter):
                curves_iter = iter(curves_iter)
                curves, stats = next(curves_iter)
                yield curves
                for curves, new_stats in curves_iter:
                    stats = stats.merge(new_stats)
                    yield curves
                self.cache.set(curve_stats_key, stats)
            # Wrap the generator in another generator;
            # e.g. no work happens at the time of this function call.
            curves_iter = intercept_stats(curves_iter)

        def count_instances(curves_iter):
            n_instance = []
            for curves in curves_iter:
                n_instance.append(len(curves))
                yield curves
            self.cache.set(n_instances_key, np.array(n_instance))

        # Wrap the generator in another generator;
        # e.g. no work happens at the time of this function call.
        curves_iter = count_instances(curves_iter)

        # Drives the iterators - i.e. this is when the work happens
        self.cache.set_stream(key, curves_iter)

    @cached_operation
    def compute_cross_sections(self, key='curve_xs',
                               curves_key='curves',
                               density=1/365,
                               keyset=None,
                               parallel=False):
        """Subsamples a set of curves at `density`.

        Keyword Arguments
        -----------------
        key : str
            Default 'curve_xs'. Key for subsampled cross sections'
        curves_key : str
            Default 'curves'. Key for input stream of curves.
        density : float
            Default 1/365 (i.e. roughly yearly). Specifies how frequently to
            subsample each patient dataframe.
        keyset : list[str]
            A list of keys (only used with parallel option)
        parallel : bool
            If True, `keyset` must be a list of keys (`key` is not used), and
            samples are taken from the curves iterator for each key given,
            iterating over the set of curves dataframes only once (this can be
            much more efficient for file-backed caching).  Does not change from
            single process to multiprocess.
        """
        if not parallel:
            def stream():
                for df in self.cache.get_stream(curves_key):
                    samples = binomial_sample_curves(df, density)
                    if samples is not None:
                        yield samples
            self.cache.set_stream(key, stream())
            return

        with contextlib.ExitStack() as stack:
            stream_setters = [
                stack.enter_context(self.cache.stream_setter(k))
                for k in keyset
            ]
            for df in self.cache.get_stream(curves_key):
                for setter in stream_setters:
                    samples = binomial_sample_curves(df, density)
                    if samples is not None:
                        setter(samples)

    @cached_operation
    def make_standardizer(self, key='standardizer', stats_key='curve_stats'):
        """Instantiate and fit a Standardizer.

        Keyword Arguments
        -----------------
        key : str
            Default 'standardizer'. Key for the Standardizer instance.
        stats_key : str
            Default 'curve_stats'. Key for already computed curve stats.
        """
        mode_params = {}
        for cfg in self.configs:
            mode_params[cfg.mode] = {k:v for k, v in cfg.std_params.items()}

        curve_stats = self.cache.get(stats_key)
        n_instances = self.cache.get('n_instances')

        std = Standardizer(mode_params, curve_stats, n_instances)
        self.cache.set(key, std)

    @cached_operation
    def build_standardized_data_matrix(self, key='std_matrix',
                                       xs_key='curve_xs',
                                       meta_key='meta',
                                       std_key='standardizer',
                                       prune_nan=False):
        """Merges a set of dataframes with heterogeneous column labels into a
        single dataframe. Prunes the column labels by what actually exists in
        the curve data (taken from the fit standardizer), and then finally
        standardizes the dataframe.

        Keyword Arguments
        -----------------
        key : str
            Default 'std_matrix'. Key for the final merged, standardized, and
            filled dataframe.
        meta_key : str
            Default 'meta'. Key for the column label metadata.
        xs_key : str
            Default 'curve_xs'. Key for the stream of dataframes to merge.
        std_key : str
            Default 'standardizer'. Key for a fitted standardizer instance.
        prune_nan : bool
            Default False. Cf. `expand_and_fill_cross_sections`.
        """
        meta = self.cache.get(meta_key)
        std = self.cache.get(std_key)
        cross_sections = self.cache.get_stream(xs_key)
        matrix = expand_and_fill_cross_sections(meta, std, cross_sections,
                                                prune_nan=prune_nan)
        self.cache.set(key, matrix)

    @cached_operation
    def sample_last_curve_points(self, key='final_samples',
                                 curves_key='curves'):
        """Subsamples by selecting the final point per set of patient curves.

        Keyword Arguments
        -----------------
        key : str
            Default 'final_samples'. Key for subsampled cross sections'
        curves_key : str
            Default 'curves'. Key for input stream of curves.
        """
        curves_iter = self.cache.get_stream(curves_key)
        self.cache.set_stream(key, sample_latest(curves_iter))

    @cached_operation
    def learn_model(self, key='model',
                    train_data_key='std_matrix',
                    drop_cols=None,
                    **model_kws):
        """Instantiate and fit an IcaPhenotypeModel from preprocessed data.

        Keyword Arguments
        -----------------
        key : str
            Default 'model'. Key for the fit model.
        train_data_key : str
            Default 'std_matrix'. Key for a processed, standardized dataframe.
        drop_cols : list
            Default None. If given, a list of channels in the data to exclude
            from the phenotype model training.
        **kwargs
            All other kwargs are forwarded to IcaPhenotypeModel at
            instantiation.
        """
        data = self.cache.get(train_data_key)
        if drop_cols is not None:
            data = data.drop(drop_cols, axis=1)
        model = IcaPhenotypeModel(**model_kws)
        model.fit(data)
        self.cache.set(key, model)

    @cached_operation
    def collect_phenotypes(self, model_keys, key='phenotypes'):
        """Collect phenotypes from models specified in model_keys into a
        single file for further analysis or hand inspection.

        Arguments
        ---------
        model_keys : Iterator
            An input iterator of keys (or locations) for models to aggregate

        Keyword Arguments
        -----------------
        key : str
            Default 'phenotypes'. Key for the dict of phenotypes.
        """
        phen = {}
        for k in model_keys:
            model = self.cache.get(k)
            phen[k] = model.phenotypes_
        self.cache.set(key, phen)

    @cached_operation
    def create_affinity_matrix(self, key='affinity_matrix',
                               phenotypes_key='phenotypes'):
        """Transform the collected phenotypes from trained submodels into an
        affinity matrix.

        Keyword Arguments
        -----------------
        key : str
            Default 'affinity_matrix'. Key for the 2d ndarray of similarities.
        phenotypes_key : str
            Default 'phenotypes'. Key for the dict of phenotypes.
        """
        phenotypes = self.cache.get(phenotypes_key)
        values = [v for (_, v) in sorted(phenotypes.items())]
        aff_matrix = make_affinity_matrix(values)
        self.cache.set(key, aff_matrix)

    @cached_operation
    def cluster_affinities(self, key='clustering',
                           clusters_key='clusters',
                           affinity_key='affinity_matrix',
                           preference=None,
                           threshold=None,
                           **kwargs):
        """Run affinity propagation on precomputed similarities (affinities).

        Keyword Arguments
        -----------------
        key : str
            Default 'clustering'. Key for a clustering.Clustering object.
        clusters_key : str
            Default 'clusters'. Key for secondary product, triplets of
            (submatrix, indices, center) for each cluster, where the submatrix
            is the original, unmasked values of the affinity matrix for the
            cluster centered at `center`.
        affinities_key : str
            Default 'affinity_matrix'. Key for the precomputed affinities.
        preference : Number
            Default None. The preference is passed through to the Affinity
            Propagation algorithm. If None (the default), the median of the
            unmasked affinity matrix is used.
        threshold : Number
            Default None. Used to construct a sparse affinity matrix. If None,
            the cached affinity matrix is used without any masking.
        **kwargs
            All other kwargs are forwarded to the affinity propagation
            algorithm.
        """
        S = self.cache.get(affinity_key).copy()
        # Use median as default preference for Affinity Propagation
        # median should be calculated prior to threshold masking
        if preference is None:
            preference = np.median(S)
        kwargs['preference'] = preference
        # Generate clustering on copy of affinity matrix
        S0 = S.copy()
        if threshold is not None:
            S0[S0 < threshold] = 0.0
        clusterer = AffinityPropagationClusterer(**kwargs)
        clusterer.fit(S0)
        self.cache.set(key, clusterer)
        # Pull clusters from original affinity matrix
        clusters = list(iter_clusters(S, clusterer.labels_))
        self.cache.set(clusters_key, clusters)

    @cached_operation
    def extract_exemplar_expressions(self, model_keys,
                                     key='exemplar_expressions',
                                     clustering_key='clustering',
                                     n_phenotypes=500):
        """Extract the expressions for exemplars from a given clustering from a
        set of model located at model_keys.
        """
        clustering = self.cache.get(clustering_key)
        expressions = {}
        def gkey(n):
            return divmod(n, n_phenotypes)[0]
        centers = sorted(clustering.centers_)
        for model_id, grp in itertools.groupby(centers, key=gkey):
            model = self.cache.get(model_keys[model_id])
            for n in grp:
                mid, pid = divmod(n, n_phenotypes)
                assert mid == model_id
                logging.debug('Loading expr for center %d: %d %d',
                              n, mid, pid)
                expressions[n] = model.expressions_[f'ICA-{pid:03}'].copy()
            del model
        self.cache.set(key, expressions)

    def combine_models(self):
        # TODO
        pass
