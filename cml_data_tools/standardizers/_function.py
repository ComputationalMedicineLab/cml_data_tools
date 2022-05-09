import collections

import numpy as np
import pandas as pd


def nansum(*xs):
    return np.nansum(np.stack(xs), axis=0)

def nanprod(*xs):
    return np.nanprod(np.stack(xs), axis=0)

def concat(*xs):
    return np.concatenate(xs)


_fields = ['channels', 'n_neg', 'base_total', 'base_mean', 'base_var',
           'log_total', 'log_mean', 'log_var', 'curve_min', 'curve_max']


class CurveStats(collections.namedtuple('CurveStats', _fields)):
    """
    A container class for statistics observed in parallel over multiple
    sources. E.g. given a DataFrame with rows of observations and columns of
    sources (or *channels*), statistics over these observations on a per
    channel basis can be contained and operated on via this class.

    This class also, as a subclass of `collections.namedtuple` which contains
    almost entirely ndarrays, is pickleable. This means that curves (or other
    dataframes of observations) can be generated in parallel via child
    processes of a main process and then this class used to generate
    statistics, which can then be sent via pickle to the main process for
    aggregation.

    All the following attributes are same-length 1-dimensional ndarrays of
    various data types, containing various statistics about the underlying
    data's channels. Many of the statistics are calculated over the data
    directly, however the number of finite elements, mean, and variance of the
    log transform of this base data is also tracked.

    Attributes
    ----------
    channels : np.ndarray[object]
        An ndarray of objects containing identifiers for each channel.
    n_neg : np.ndarray[int]
        Count of negative observations per channel.
    base_total : np.ndarray[int]
        Count of finite values per channel of the base data.
    base_mean : np.ndarray[float]
        Per-channel mean of the base data.
    base_var : np.ndarray[float]
        Per-channel variance of the base data.
    log_total : np.ndarray[int]
        Count of finite values per channel of the log transform of the base data.
    log_mean : np.ndarray[float]
        Per-channel mean of the log transform of the base data.
    log_var : np.ndarray[float]
        Per-channel variance of the log transform of the base data.
    curve_min : np.ndarray[float]
        The base data minimum value per channel.
    curve_max : np.ndarray[float]
        The base data maximum value per channel.
    """
    @classmethod
    def from_curves(cls, curves, eps=1e-6):
        """
        Classmethod to construct a new CurveStats object from a DataFrame of
        curve points.

        The argument `eps` gives a slight jitter to avoid Inf in the statistics
        of the log transformed data.
        """
        # prune any channels which are identically zero
        curves = curves.dropna(axis=1, how='all')
        channels = curves.columns.values
        X = curves.values

        # get number negatives & channel curve min / max values
        n_neg = (X < 0.0).sum(axis=0)
        mins = np.nanmin(X, axis=0)
        maxs = np.nanmax(X, axis=0)

        # Collect the basic n, mean, variance per channel curve
        N, M, V = cls.collect_meanvar(X)

        # collect n, mean, variance for log transformed channel curves
        log_X = np.log10(X + eps)
        log_X[np.isinf(log_X)] = np.nan
        log_N, log_M, log_V = cls.collect_meanvar(log_X)

        return cls(channels, n_neg, N, M, V, log_N, log_M, log_V, mins, maxs)

    @staticmethod
    def collect_meanvar(X):
        """Gets (n, m, v) over X.

        Arguments
        ---------
        X : np.ndarray
            A two-dimensional ndarray over which statistics are computed

        Returns
        -------
        A triplet (n, m, v) of 1-dim ndarrays containing, per column, the
        number `n` of finite entries, the nanmean `m`, and nanvar `v`.
        """
        n = np.isfinite(X).sum(axis=0).astype(np.int64)
        m = np.nanmean(X, axis=0)
        v = np.nanvar(X, axis=0)
        return n, m, v

    @staticmethod
    def knuth_update(pn, pm, pv, cn, cm, cv):
        """Applies the Knuth update step for online statistics generation.

        Arguments
        ---------
        pn, pm, pv
            Three 1-dim ndarays containing the number of elements `pn`, mean
            `pm`, and variance `pv` of the data seen so far, one per channel.
            E.g. if the data seen so far has 200 channels, each of these
            vectors is of length `(200,)`, with one entry per channel.
        cn, cm, cv
            Three 1-dim ndarays containing the number of elements `cn`, mean
            `cm`, and variance `cv` of the new data to incorporate into the
            existing statistics via the Knuth update algorithm.

        Returns
        -------
        A triplet of 1-dim ndarrays (N, M, V), which is the updated number of
        elements, mean, and variance of the data.
        """
        # NaN safe implementation, cf. helper functions nansum / nanprod
        N = nansum(pn, cn)
        cf = nanprod(cn, 1/N)
        pf = nanprod(pn, 1/N)
        # dx = cm - pm
        dx = nansum(cm, -pm)
        # M = pm + (cf * dx)
        M = nansum(pm, nanprod(cf, dx))
        # V = (pv * pf) + (cv * cf) + (pf * cf * dx * dx)
        V = nansum(nanprod(pv, pf), nanprod(cv, cf), nanprod(pf, cf, dx, dx))
        return N, M, V

    @staticmethod
    def agg_stats(total, mean, var):
        """Applies the Knuth update across all members of the input vectors.

        Arguments
        ---------
        total, mean, var
            Three 1-dim ndarays containing the number of elements, mean, and
            variance of the data, one per channel. E.g. if the data seen so far
            has 200 channels, each of these vectors is of length `(200,)`, with
            one entry per channel.

        Returns
        -------
        A triplet of 1-dim ndarrays (N, M, V), which are the total number of
        elements, mean, and variance across all channels represented by the
        input vectors.
        """
        N = total[0]
        M = mean[0]
        V = var[0]
        for n, m, v in zip(total[1:], mean[1:], var[1:]):
            N, M, V = CurveStats.knuth_update(N, M, V, n, m, v)
        return N, M, V

    def merge(self, other):
        """Produces a new CurveStats instance merging the statistics in this
        instance with those in another, as though the merged statistics had
        been observed over the conjoint underlying data sets.
        """
        # Find masks for the shared channels between this and other
        C, self_idx, other_idx = np.intersect1d(self.channels,
                                                other.channels,
                                                assume_unique=True,
                                                return_indices=True)

        # Update number of negative elements per curve
        P = self.n_neg[self_idx] + other.n_neg[other_idx]

        # Update base curve statistics
        N, M, V = self.knuth_update(self.base_total[self_idx],
                                    self.base_mean[self_idx],
                                    self.base_var[self_idx],
                                    other.base_total[other_idx],
                                    other.base_mean[other_idx],
                                    other.base_var[other_idx])

        # Update log transformed curve statistics
        log_N, log_M, log_V = self.knuth_update(self.log_total[self_idx],
                                                self.log_mean[self_idx],
                                                self.log_var[self_idx],
                                                other.log_total[other_idx],
                                                other.log_mean[other_idx],
                                                other.log_var[other_idx])

        # Update curve max & min values
        c_max = np.maximum(self.curve_max[self_idx],
                           other.curve_max[other_idx])
        c_min = np.minimum(self.curve_min[self_idx],
                           other.curve_min[other_idx])

        # Mask for prev values not in current (i.e. unchanged vals)
        p_mask = np.ones(self.channels.shape, dtype=np.bool)
        p_mask[self_idx] = False

        # Mask for curr values not in previous (i.e. unchanged vals)
        c_mask = np.ones(other.channels.shape, dtype=np.bool)
        c_mask[other_idx] = False

        # Recombine the updated, unchanged, and new values
        C = concat(C, self.channels[p_mask], other.channels[c_mask])
        P = concat(P, self.n_neg[p_mask], other.n_neg[c_mask])
        N = concat(N, self.base_total[p_mask], other.base_total[c_mask])
        M = concat(M, self.base_mean[p_mask], other.base_mean[c_mask])
        V = concat(V, self.base_var[p_mask], other.base_var[c_mask])
        log_N = concat(log_N, self.log_total[p_mask], other.log_total[c_mask])
        log_M = concat(log_M, self.log_mean[p_mask], other.log_mean[c_mask])
        log_V = concat(log_V, self.log_var[p_mask], other.log_var[c_mask])
        c_min = concat(c_min, self.curve_min[p_mask], other.curve_min[c_mask])
        c_max = concat(c_max, self.curve_max[p_mask], other.curve_max[c_mask])

        # Sort results by channel
        sort = np.argsort(C)
        C = C[sort]
        P = P[sort]
        N = N[sort]
        M = M[sort]
        V = V[sort]
        log_N = log_N[sort]
        log_M = log_M[sort]
        log_V = log_V[sort]
        c_min = c_min[sort]
        c_max = c_max[sort]

        # Produce new CurveStats (or subclass, if in a subclass) object
        cls = self.__class__
        return cls(C, P, N, M, V, log_N, log_M, log_V, c_min, c_max)

    def calculate_postfill_stats(self, mask, n_instances, fill=0.0):
        """
        Updates the statistics for a given set of channels selected by the
        `mask` as though `n_instances` total datapoints had been seen, with any
        missing data points filled by the float value `fill`. E.g., updates the
        selected channels' statistics as though they had already been expanded
        and filled.

        Arguments
        ---------
        mask : np.ndarray
            A boolean mask selecting statistics for given channels.
        n_instances : int
            A number of instances by which to extend the statistics.
        fill : float
            Default 0.0, a number with which to extend the statistics.

        Returns
        -------
        A new copy of this object with the attributes `base_total`,
        `base_mean`, `base_var`, `log_total`, `log_mean`, and `log_var` updated
        as described.
        """
        N = np.copy(self.base_total)
        M = np.copy(self.base_mean)
        V = np.copy(self.base_var)
        log_N = np.copy(self.log_total)
        log_M = np.copy(self.log_mean)
        log_V = np.copy(self.log_var)

        # Calculate postfill stats over the base curves
        pn = N[mask]
        pm = M[mask]
        pv = V[mask]
        cn = n_instances - pn
        cm = np.full_like(cn, fill, dtype=float)
        cv = np.zeros_like(cn, dtype=float)
        N[mask], M[mask], V[mask] = self.knuth_update(pn, pm, pv, cn, cm, cv)

        # Calculate postfill stats over the log transformed curves
        pn = log_N[mask]
        pm = log_M[mask]
        pv = log_V[mask]
        cn = n_instances - pn
        cm = np.full_like(cn, np.log10(fill or 1e-6), dtype=float)
        cv = np.zeros_like(cn, dtype=float)
        log_N[mask], log_M[mask], log_V[mask] = self.knuth_update(pn, pm, pv,
                                                                  cn, cm, cv)

        # Return a *new* instance
        return self._replace(base_total=N, base_mean=M, base_var=V,
                             log_total=log_N, log_mean=log_M, log_var=log_V)

    def calculate_modelevel_stats(self, mask, mean_of_var=True):
        """
        Updates the statistics for a given set of channels selected by the
        `mask` as though the statistics had been calculated over all those
        channels together. E.g. if 10 channels of 100 data points have been
        observed, each channels' statistics are computed as though one channel
        of 1000 data points had been observed (hence all selected channels will
        have the same values for number of elements, mean, and variance).

        Arguments
        ---------
        mask : np.ndarray
            A boolean mask selecting statistics for given channels.
        mean_of_var : bool
            Default True. If True, then the variance per selected channel is
            set to the mean of the variances of all the selected channels
            rather than the variance of the concatenation of the selected
            channels.

        Returns
        -------
        A new copy of this object with the attributes `base_total`,
        `base_mean`, `base_var`, `log_total`, `log_mean`, and `log_var` updated
        as described.
        """
        N = np.copy(self.base_total)
        M = np.copy(self.base_mean)
        V = np.copy(self.base_var)
        log_N = np.copy(self.log_total)
        log_M = np.copy(self.log_mean)
        log_V = np.copy(self.log_var)

        # Aggregate base curve stats over the whole mode
        n_agg, m_agg, v_agg = self.agg_stats(N[mask], M[mask], V[mask])
        N[mask] = n_agg
        M[mask] = m_agg
        if mean_of_var:
            V[mask] = V[mask].mean()
        else:
            V[mask] = v_agg

        # Aggregate log-transformed curve stats over the whole mode
        log_n_agg, log_m_agg, log_v_agg = self.agg_stats(log_N[mask],
                                                         log_M[mask],
                                                         log_V[mask])
        log_N[mask] = log_n_agg
        log_M[mask] = log_m_agg
        if mean_of_var:
            log_V[mask] = log_V[mask].mean()
        else:
            log_V[mask] = log_v_agg

        # Return a *new* instance
        return self._replace(base_total=N, base_mean=M, base_var=V,
                             log_total=log_N, log_mean=log_M, log_var=log_V)

    def as_frame(self):
        """Returns the underlying data as a DataFrame"""
        data = self._asdict()
        index = data.pop('channels')
        if isinstance(index[0], tuple):
            index = pd.MultiIndex.from_tuples(index)
        return pd.DataFrame(data=data, index=index).T


class AffineTransform:
    """A simple transformation bundled with its inverse transformation.

    The forward transformation is `(x - shift) / scale`, and the inverse
    transformation is `(scale * x) + shift`. If log transform is requested,
    then the forward transformation applies `np.log10(x + eps)` prior to the
    shifting & scaling, and the backward transformation applies
    `np.power(10, x) - eps` after undoing the shifting and scaling.

    Parameters
    ----------
    scale, shift : float
        Numbers by which to, respectively, scale and shift the data.
    log : bool
        If True, apply `np.log10` to the input data before scaling & shifting.
    eps : float
        An amount of jitter by which to shift the input data prior to log
        transformation (if log transformation is True) to avoid Inf values.
    """
    def __init__(self, scale=1.0, shift=0.0, log=False, eps=10):
        self.scale = scale
        self.shift = shift
        self.log = log
        self.eps = eps

    def transform(self, x):
        """Cf. class docstring"""
        if self.log:
            x = np.log10(x + self.eps)
        return (x - self.shift) / self.scale

    def inverse_transform(self, x):
        """Cf. class docstring"""
        y = (self.scale * x) + self.shift
        if self.log:
            return np.power(10, y) - self.eps
        return y


class Standardizer:
    """A class which manages mapping data channels to AffineTransform functions.

    Each channel's function is fitted from parameters specified at the mode
    leve and at the channel level. Channels are grouped into modes, and general
    parameters (function type, etc) are specified per mode. But then each
    AffineTransform is fitted with channel-specific statistics.

    The mode-level fitting parameters and their data types are as follows:

    * `"kind"`: one of `{"identity", "standard", "gelman", "gelman_with_fallbacks"}`
    * `"noshift"`: boolean
    * `"log"`: boolean
    * `"postfill"`: boolean
    * `"agg_mode"`: boolean
    * `"fill"`: float
    * `"eps"`: float

    For each mode, if `postfill` is True then the curve statistics for the
    channels in that mode are adjusted by the value given in `fill` and the
    total number of instances in the data (cf. Parameters of this class) using
    the `CurveStats.calculate_postfill_stats` algorithm.

    For each mode, if `agg_mode` is True then the curve statistics for the
    channels in that mode are adjusted by the algorithm given in
    `CurveStats.calculate_modelevel_stats`.

    Then for each mode, the `kind` parameter specifies the basic type of
    function to use in standardizing. The `"identity"` function is simply
    `y = x`, the `"standard"` function is `y = (x - mean) / stdev`, the
    `"gelman"` function is `y = (x - mean) / (2 * stdev)`. The
    `gelman_with_fallbacks` option computes which of these functions to use by
    the following algorithm on the channel statistics:

    * if the channel is empty (mean is NaN or var is 0), then the function is
      the "identity" function
    * if the channel is nearly constant, then the function is "linear", i.e. no
      scale is applied but values are shifted by the channel min
    * if the channel has over 1% negative values, the "log" param is ignored
      but the function is "gelman" (i.e. the scale factor is `2 * stdev`)
    * otherwise, a "gelman" function is given, and the "log" parameter is
      respected.

    For all these function types, if "log" is True, then the function
    constructed will apply a log transformation before shifting & scaling the
    data (cf. AffineTransform) and the shift & scale are the mean and stdev of
    the log transformed channel data.

    For all these function types, if "noshift" is True then the shift is set to
    zero unless overridden by `"gelman_with_fallbacks"`.

    Parameters
    ----------
    mode_params : dict
        A mapping from mode identifiers to fitting parameters.
    curve_stats : CurveStats
        A container of statistics over the input data used for fitting
        functions on a per-channel basis.
    n_instances : np.ndarray[int]
        A 1-dim ndarray giving the number of observations per patient. E.g.,
        `len(n_instances` gives the number of patients, and `sum(n_instances)`
        gives the total number of observations in the data.

    Use of the following attributes is useful for interactive understanding and
    introspection but they are not considered part of the class's public API.

    Attributes
    ----------
    _functions : dict
        A mapping from (mode, channel) keys to AffineTransform instances
    _parameters : dict
        A mapping from (mode, channel) to the final parameters used to
        construct each corresponding AffineTransform instance
    """
    def __init__(self, mode_params, curve_stats, n_instances):
        self.mode_params = mode_params
        self.curve_stats = curve_stats
        self.n_instances = n_instances

        # Adjust statistics by adding postfill values and/or aggregating the
        # statistics over the entire mode's channels
        stats = curve_stats
        total = sum(n_instances)
        for mode, params in self.mode_params.items():
            if params.get('postfill'):
                mask = np.array([m == mode for m, _ in stats.channels])
                fill = params.get('fill', 0.0)
                stats = stats.calculate_postfill_stats(mask, total, fill)
            if params.get('agg_mode'):
                mask = np.array([m == mode for m, _ in stats.channels])
                stats = stats.calculate_modelevel_stats(mask)
        self.final_stats = stats

        # Mappings of (mode, channel) keys to AffineTransform objects and sets
        # of transform parameters (the mode level and channel leve parameters)
        self._functions = {}
        self._parameters = {}

        # Populate the _functions map with concrete functions generated using
        # the mode level and channel level parameters
        stats_df = self.final_stats.as_frame()
        for (mode, channel) in stats_df:
            st = stats_df[(mode, channel)]
            params = {k: v for k, v in self.mode_params[mode].items()}

            log = params.get('log')
            eps = params.get('eps', 1e-6)
            if log:
                shift = st.log_mean
                scale = np.sqrt(st.log_var)
            else:
                shift = st.base_mean
                scale = np.sqrt(st.base_var)

            if params.get('noshift'):
                shift = 0.0

            params['computed_shift'] = shift
            params['computed_scale'] = scale

            kind = params.get('kind')

            if kind == 'identity':
                func = AffineTransform()

            elif kind == 'standard':
                func = AffineTransform(scale, shift, log, eps)

            elif kind == 'gelman':
                scale *= 2
                params['scale'] = scale
                func = AffineTransform(scale, shift, log, eps)

            elif kind == 'gelman_with_fallbacks':
                if np.isnan(st.base_mean) or st.base_var == 0:
                    params['computed_kind'] = 'identity'
                    func = AffineTransform()
                elif ((st.curve_max - st.curve_min) <
                      1e-6 * (st.curve_max + st.curve_min)):
                    params['computed_kind'] = 'linear'
                    func = AffineTransform(shift=st.curve_min)
                elif st.n_neg > 0.01 * st.base_total:
                    scale = np.sqrt(st.base_var) * 2
                    shift = st.base_mean
                    params['computed_kind'] = 'gelman'
                    params['computed_scale'] = scale
                    params['computed_shift'] = shift
                    params['log'] = False
                    func = AffineTransform(scale, shift)
                else:
                    scale *= 2
                    params['scale'] = scale
                    params['computed_kind'] = 'gelman'
                    func = AffineTransform(scale, shift, log=log, eps=eps)

            self._functions[(mode, channel)] = func
            self._parameters[(mode, channel)] = params

    def transform(self, X):
        """In-place transform of each column of X"""
        for col in X:
            X.loc[:, col] = self._functions[col].transform(X[col])
        return X

    def inverse_transform(self, X):
        """In-place inverse transform of each column of X"""
        for col in X:
            X.loc[:, col] = self._functions[col].inverse_transform(X[col])
        return X

    def inverse_transform_label(self, name, delta, anchor=1, spec=None):
        """Provides the meaning of `delta` in the original space of the channel
        specified by `name`, with format `spec` and reference to `anchor`.

        This is for use in phenotype plots. For most transforms, an additive
        change in the amount `delta` corresponds a scaled but still additive
        amount in the original space. In this case, `delta` in the transformed
        space is simply scaled to the original space and given an additive
        label. If the transform includes a logarithm operation, then an
        additive `delta` in the original space corresponds to a multiplicative
        change in the original space. In this case, `delta` is appropriately
        scaled and given a multiplicative label.

        Arguments
        ---------
        name : (str, str)
            A 2-tuple of strings identifying a mode and channel
        delta : float
            A value to map into the original space of the given channel
        anchor : float
            Default 1.
        """
        if spec is None:
            spec = '+.2f'

        params = self._parameters[name]
        X = pd.Series([anchor, anchor+delta])
        X_inv = self._functions[name].inverse_transform(X)

        if params.get('log'):
            spec = spec.lstrip('+')
            impact = X_inv[1] / X_inv[0]
            if impact > 1.0:
                prefix = 'x'
            else:
                prefix = '/'
                impact = 1.0 / impact
        else:
            prefix = ''
            impact = X_inv[1] - X_inv[0]
        return f'{prefix}{impact:{spec}}'
