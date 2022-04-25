import collections

import numpy as np
import pandas as pd

from cml_data_tools.standardizers import (DataframeStandardizer,
                                          LinearStandardizer,
                                          GelmanStandardizer,
                                          LogGelmanStandardizerWithFallbacks)


CurveStats = collections.namedtuple('CurveStats', [
    'channels', 'n_neg', 'base_total', 'base_mean', 'base_var', 'log_total',
    'log_mean', 'log_var', 'curve_min', 'curve_max'
])


def collect_curve_stats(curves, eps=1e-6):
    # We want to track, for each curve, the total number of observations
    # and the number of non-negative values, the mean and variance of the
    # curve, and the mean and variance of the curve to log10
    curves = curves.dropna(axis=1, how='all')
    C = curves.columns.values
    X = curves.values
    n_neg = (X < 0.0).sum(axis=0)

    base_M = np.nanmean(X, axis=0)
    base_V = np.nanvar(X, axis=0)
    base_total = np.isfinite(X).sum(axis=0).astype(np.int64)

    log_X = np.log10(X + eps)
    log_X[np.isinf(log_X)] = np.nan
    log_M = np.nanmean(log_X, axis=0)
    log_V = np.nanvar(log_X, axis=0)
    log_total = np.isfinite(log_X).sum(axis=0).astype(np.int64)

    curve_min = np.nanmin(X, axis=0)
    curve_max = np.nanmax(X, axis=0)

    return CurveStats(C, n_neg, base_total, base_M, base_V, log_total, log_M,
                      log_V, curve_min, curve_max)


def update_curve_stats(prev, curr):
    if prev is None:
        return curr

    prev_chan, prev_pos, prev_n, prev_m, prev_v, prev_log_n, prev_log_m,\
        prev_log_v, prev_min, prev_max = prev

    curr_chan, curr_pos, curr_n, curr_m, curr_v, curr_log_n, curr_log_m,\
        curr_log_v, curr_min, curr_max = curr

    C, prev_idx, curr_idx = np.intersect1d(prev_chan, curr_chan,
                                           assume_unique=True,
                                           return_indices=True)

    # Calculate the updated values for basic mean / var
    cn = curr_n[curr_idx]
    pn = prev_n[prev_idx]
    N = pn + cn
    cf = cn / N
    pf = pn / N

    cm = curr_m[curr_idx]
    cv = curr_v[curr_idx]
    pm = prev_m[prev_idx]
    pv = prev_v[prev_idx]
    dx = cm - pm
    M = pm + (cf * dx)
    V = (pv * pf) + (cv * cf) + (pf * cf * dx * dx)

    # Updated mean / var of log transformed
    log_cn = curr_log_n[curr_idx]
    log_pn = prev_log_n[prev_idx]
    log_N = log_pn + log_cn
    log_cf = log_cn / log_N
    log_pf = log_pn / log_N

    log_cm = curr_log_m[curr_idx]
    log_cv = curr_log_v[curr_idx]
    log_pm = prev_log_m[prev_idx]
    log_pv = prev_log_v[prev_idx]

    log_dx = np.nansum(np.stack((log_cm, -log_pm)), axis=0)
    log_M = np.nansum(np.stack((log_pm, log_cf*log_dx)), axis=0)

    log_V = (np.nanprod(np.stack((log_pv, log_pf)), axis=0)
             + np.nanprod(np.stack((log_cv, log_cf)), axis=0)
             + np.nanprod(np.stack((log_pf, log_cf, log_dx, log_dx)), axis=0))

    # Update num positive, max & min values
    n_neg = prev_pos[prev_idx] + curr_pos[curr_idx]
    c_max = np.maximum(prev_max[prev_idx], curr_max[curr_idx])
    c_min = np.minimum(prev_min[prev_idx], curr_min[curr_idx])

    # Mask for prev values not in current (i.e. unchanged vals)
    p_mask = np.ones(prev_chan.shape, dtype=np.bool)
    p_mask[prev_idx] = False

    # Mask for curr values not in previous (i.e. unchanged vals)
    c_mask = np.ones(curr_chan.shape, dtype=np.bool)
    c_mask[curr_idx] = False

    # Recombine the unchanged, updated, and new values
    C = np.concatenate((C, prev_chan[p_mask], curr_chan[c_mask]))
    P = np.concatenate((n_neg, prev_pos[p_mask], curr_pos[c_mask]))
    N = np.concatenate((N, prev_n[p_mask], curr_n[c_mask]))
    M = np.concatenate((M, prev_m[p_mask], curr_m[c_mask]))
    V = np.concatenate((V, prev_v[p_mask], curr_v[c_mask]))
    log_N = np.concatenate((log_N, prev_log_n[p_mask], curr_log_n[c_mask]))
    log_M = np.concatenate((log_M, prev_log_m[p_mask], curr_log_m[c_mask]))
    log_V = np.concatenate((log_V, prev_log_v[p_mask], curr_log_v[c_mask]))
    c_min = np.concatenate((c_min, prev_min[p_mask], curr_min[c_mask]))
    c_max = np.concatenate((c_max, prev_max[p_mask], curr_max[c_mask]))

    # Stay sorted by channel
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

    return CurveStats(C, P, N, M, V, log_N, log_M, log_V, c_min, c_max)


class OnlineCurveStandardizer(DataframeStandardizer):
    def __init__(self, curve_stats=None, **kwargs):
        super().__init__(**kwargs)
        self.curve_stats = curve_stats

    def update_from_stats(self, stats):
        self.curve_stats = update_curve_stats(self.curve_stats, stats)

    def update_from_curves(self, curves):
        stats = collect_curve_stats(curves)
        self.curve_stats = update_curve_stats(self.curve_stats, stats)

    def stats_as_df(self):
        if self.curve_stats is not None:
            data = self.curve_stats._asdict()
            index = data.pop('channels')
            if isinstance(index[0], tuple):
                index = pd.MultiIndex.from_tuples(index)
            df = pd.DataFrame(data=data, index=index)
            return df

    def fit(self, curves=None, y=None):
        # Hook to fit from curves given at last minute
        if curves is not None:
            for df in curves:
                self.update_from_curves(df)

        stats_df = self.stats_as_df().T
        if stats_df is None:
            raise RuntimeError(f'{self.__class__.__name__} is misconfigured')

        for channel_name in stats_df:
            stats = stats_df[channel_name]
            mode = channel_name[0]
            standardizer, kwargs = self._standardizer_info[mode]
            instance = standardizer(**kwargs)
            # Manually set the standardizers' params
            if standardizer is LinearStandardizer:
                pass
            elif standardizer is GelmanStandardizer:
                if instance.log_transform:
                    mean = stats.log_mean
                    stdev = np.sqrt(stats.log_var)
                else:
                    mean = stats.base_mean
                    stdev = np.sqrt(stats.base_var)
                instance._mean = mean
                instance._stdev = stdev
            elif standardizer is LogGelmanStandardizerWithFallbacks:
                if np.isnan(stats.base_mean) or stats.base_var == 0:
                    transformer = LinearStandardizer()
                elif ((stats.curve_max - stats.curve_min) <
                      (1e-6 * (stats.curve_max + stats.curve_min))):
                    transformer = LinearStandardizer(offset=-stats.curve_min)
                elif stats.n_neg > 0.01 * stats.base_total:
                    transformer = GelmanStandardizer(log_transform=False)
                    transformer._mean = stats.base_mean
                    transformer._stdev = np.sqrt(stats.base_var)
                elif instance.log_default:
                    transformer = GelmanStandardizer(instance.log_default,
                                                     eps=instance.eps)
                    transformer._mean = stats.log_mean
                    transformer._stdev = np.sqrt(stats.log_var)
                else:
                    transformer = GelmanStandardizer(log_transform=False)
                    transformer._mean = stats.base_mean
                    transformer._stdev = np.sqrt(stats.base_var)
                instance._transformer = transformer
            self._transformer[channel_name] = instance
        return self
