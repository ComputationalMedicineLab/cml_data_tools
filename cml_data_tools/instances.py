"""Instances generation.

An instance is a tuple of (pid, t_obs, t_end, t_evt, x), where
 * pid : str - a patient ID
 * t_obs : np.datetime64 - the datetime of the instance observation
 * t_end : np.datetime64 - the datetime of the final obs. in the patient record
 * t_evt : np.datetime64 or NaT - the date of the event or Not a Time
 * x : np.ndarray[np.float, ndim=1] - the observation data

Instances are generated in the following steps:
 + Downsample a dataframe of patient curves by observation date
 + Fill in missing channel values
 + Annotate each observation with t_end and t_evt

Instance labels are generated by choosing a horizon datetime between t_obs and
t_end, and then recording if the horizon_time occurs before or after t_evt
(before is a False, after is a True) - in other words, given an arbitrary
datetime after the observation and before the end of the record, has the event
occurred by that time?

Instances are saved with these datetimes rather than the actual horizon and
label so that the method of choosing a horizon (and hence generating a label)
may be a model hyperparameter.
"""
import pandas as pd
import numpy as np

__all__ = [
    'EVENT',
    'ONE_YEAR',
    'random_expand_horizons',
    'channel_mask',
    'collate',
    'gen_instances',
    'downsample',
    'fill_channels',
]


EVENT = ('Event', 'event')

# The value of pd.Timedelta(1, 'Y').to_timedelta64()
ONE_YEAR = np.timedelta64(31556952000000000, 'ns')


def random_expand_horizons(t_obs, t_end, t_evt, X):
    """For each instance represented in X, we calculate a horizon time
    randomly sampled from the instance observation date to the end of the
    patient's record and label the instance with whether or not the event has
    occurred (0 if not, 1 if so).

    Arguments
    ---------
    t_obs : np.ndarray[np.datetime64, ndim=1]
    t_end : np.ndarray[np.datetime64, ndim=1]
    t_evt : np.ndarray[np.datetime64, ndim=1]
    X : np.ndarray[np.float64, ndim=2]

    Returns
    -------
    X, y : np.ndarray, np.ndarray
        X is an ndarray of shape (n instances, n features), where the last
        feature is the horizon time expressed as a float64 denoting a number of
        years between the observation date and the end of the patient record

        y is an ndarray of shape (n instances, 1) of flags 0 or 1
        indicating whether or not the event in question has occurred by the
        time of the horizon time
    """
    n = t_obs.shape[0]
    h_spans = np.random.random_sample(n) * (t_end - t_obs)
    h_dates = h_spans + t_obs
    horizons = h_spans / ONE_YEAR
    horizons = horizons.reshape(-1, 1)
    X = np.hstack((X, horizons))
    y = h_dates > t_evt
    y = y.astype(np.float64).reshape(-1, 1)
    return X, y


def channel_mask(channels, dropcols):
    """Produces a mask of len(channels) that will filter out dropcols"""
    return np.array([ch not in dropcols for (_, ch) in channels])


def collate(instances):
    """Converts a list of instances into a tuple of ndarrays.

    Arguments
    ---------
    instances : list of instances
        Cf. gen_instances

    Returns
    -------
    (t_obs, t_end, t_evt, X) : (ndarray, ndarray, ndarray, ndarray)
        The values of t_obs, t_end, t_evt, and x per instance stacked together
        into a single numpy ndarray. Note that the first three are one-dim
        arrays of a length equal to the first dimension of X.
    """
    t_obs, t_end, t_evt, Xs = [], [], [], []
    # Discard the ID; it isn't needed
    for (_, obs, end, evt, x) in instances:
        t_obs.append(obs)
        t_end.append(end)
        t_evt.append(evt)
        Xs.append(x)
    return np.array(t_obs), np.array(t_end), np.array(t_evt), np.array(Xs)


def gen_instances(curves, channels, fillvec, *,
                  rate=None, density=None, period=None, id_dates=None):
    """Runs the full instance pipeline for generating training instances.

    Arguments
    ---------
    For `curves` and each of `rate`, `density`, `period`, and `id_dates`
    cf. `downsample`.
    For `channels` and `fillvec` cf. `fill_channels`.

    Returns
    -------
    A list of instances. Instances are tuples of (pid, t_obs, t_end, t_evt, x)
    pid : str
        A patient ID
    t_obs : np.datetime64
        The datetime of the instance observation
    t_end : np.datetime64
        The datetime of the final obs. in the patient record
    t_evt : np.datetime64 or np.datetime64('NaT')
        The date of the event, if any
    x : np.ndarray[np.float, ndim=1]
        The observation data

    Notes
    -----
    Some of the routines called by this function delegate work to numpy or
    pandas threads that are not hindered by the GIL; therefore, the CPU time of
    this function is approximately 7-8x its wall clock time (according to some
    limited testing). For example, using Ipython's %time magic we have:
    """
    samples = downsample(curves, rate=rate, density=density, period=period, id_dates=id_dates)

    # Calculate t_end, t_evt
    dates = curves.index.get_level_values('date')
    t_end = dates[-1].to_datetime64()
    try:
        evt_dates = dates[curves[EVENT] == 1]
        t_evt = evt_dates[0].to_datetime64()
    except IndexError:
        t_evt = np.datetime64('NaT')

    # Calculate x
    samples = samples.sort_index(axis=1)
    cols = samples.columns.values
    vals = samples.values
    X_i = samples.index.values
    X = fill_channels(cols, vals, channels, fillvec)

    instances = [(i, t_obs.to_datetime64(), t_end, t_evt, x)
                 for (i, t_obs), x in zip(X_i, X)]

    return instances


def downsample(curves, *, rate=None, density=None, period=None, id_dates=None):
    """Generate cross sections from a dataframe of curves.

    Arguments
    ---------
    curves : pandas.DataFrame
        A dataframe of curves. The columns are a multi-index with levels (mode,
        channel). The index is a multi-index with levels ('id', 'date').  The
        values are float64s giving the value of each channel's curve at the
        given date for this patient.

    Keyword Only Arguments
    ----------------------
    rate : [float, pandas.Timedelta]
        If a rate is provided, then it must be a two-element iterable, the
        first of which is the rate, and the second determines the frequency. So
        a rate of (50, pd.Timedelta(1, 'Y')) would indicate that you want 50
        samples per year for which the patient record exists.

    density : float
        The random sampling density of cross sections as a fraction of rows in
        the source dataframe. Sampling is guaranteed to produce at least one
        sample, regardless of how small density is. Cannot be used together
        with period.

    period : str or pandas.DateOffset
        A pandas DateOffset object giving the regular sampling period of the
        cross sections. Setting more frequently than daily produces undefined
        results. Cannot be used together with density.

    id_dates: dict of str,pandas.MultiIndex
        A dict containing patient id's as keys and multi-index objects as
        elements. The multi-index objects contain the desired patient id's and
        dates in a format to be located in curve dataframes. Cannot be used
        together with rate, density, or period

    Returns
    -------
    A pandas.DataFrame, the rows of which are a subset of `curves`

    Raises
    ------
    ValueError if more than one of `rate`, `density`, `period`, or
    `id_dates` is not `None`.

    Notes
    -----
    The time this function takes depends heavily upon the input arguments. As
    an example, timing performance across 1000 random patient curves with
    Ipython's %timeit magic gives:

        %timeit [downsample(x, density=0.05) for x in curves]
        845 ms ± 17 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

        rate = (50, pd.Timedelta(1, 'Y'))
        %timeit [pf.downsample(x, rate=rate) for x in curves]
        2.46 s ± 294 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

        %timeit [downsample(x, density=1) for x in curves]
        5.92 s ± 531 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

        %timeit [downsample(x, period='MS') for x in curves]
        24.7 s ± 1.62 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

    The expected case is for smaller values of density (<0.2) or lower rates
    when generating training instances.  The ability to specify downsampling by
    period exists to support generating prediction instances, which can be done
    on a per-patient or subsample basis.
    """
    if sum((rate is not None, density is not None,
            period is not None, id_dates is not None)) != 1:
        raise ValueError('Must specify exactly one of rate, density, period, '
                         'or id_dates')

    if id_dates:
        patient_id = curves.index.get_level_values('id')[0]
        dates = id_dates[patient_id]
        samples = curves.loc[dates]
    elif period:
        samples = curves.reset_index('id')
        samples = samples.resample(period).first()
        samples = samples.set_index('id', append=True)
        samples = samples.reorder_levels(['id', 'date'])
    else:
        if density:
            n_samples = density * len(curves)
        else:
            dates = curves.index.get_level_values('date')
            interval = dates[-1] - dates[0]
            n_samples = rate[0] * (interval / rate[1])
        n_samples = max(1, round(n_samples))
        samples = curves.sample(n=n_samples)

    return samples


def fill_channels(cols, vals, channels, fillvec):
    """Fast filling in of missing feature values.

    Arguments
    ---------
    cols : numpy.ndarray
        A 1d ndarray of tuples (as if from pd.MultiIndex.to_numpy())
        representing the channels available in vals

    vals : numpy.ndarray
        A 2d ndarray with observations as rows and columns corresponding to the
        channels outlined in cols

    channels : numpy.ndarray
        A 1d ndarray of tuples (as if from pd.MultiIndex.to_numpy())
        representing the full set of channel names

    fillvec : numpy.ndarray
        A 1d ndarray of floats to use if a channel is not present in `X0`;
        corresponds to the names in `channels`.

    Returns
    -------
    An ndarray of shape (len(vals), len(channels)) that has missing channels
    filled in from `fillvec`.
    """
    _, copy_idx, _ = np.intersect1d(channels, cols,
                                    assume_unique=True,
                                    return_indices=True)
    X = np.tile(fillvec, (vals.shape[0], 1))
    X[:, copy_idx] = vals
    return X