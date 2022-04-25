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
    """Aggregates stats calculated over patient curves"""
    @classmethod
    def from_curves(cls, curves, eps=1e-6):
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
        n = np.isfinite(X).sum(axis=0).astype(np.int64)
        m = np.nanmean(X, axis=0)
        v = np.nanvar(X, axis=0)
        return n, m, v

    @staticmethod
    def knuth_update(pn, pm, pv, cn, cm, cv):
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
        """Applies the Knuth update across all members of the input vectors"""
        N = total[0]
        M = mean[0]
        V = var[0]
        for n, m, v in zip(total[1:], mean[1:], var[1:]):
            N, M, V = CurveStats.knuth_update(N, M, V, n, m, v)
        return N, M, V

    def merge(self, other):
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
        data = self._asdict()
        index = data.pop('channels')
        if isinstance(index[0], tuple):
            index = pd.MultiIndex.from_tuples(index)
        return pd.DataFrame(data=data, index=index).T


class AffineTransform:
    def __init__(self, scale=1.0, shift=0.0, log=False, eps=10):
        self.scale = scale
        self.shift = shift
        self.log = log
        self.eps = eps

    def transform(self, x):
        if self.log:
            x = np.log10(x + self.eps)
        return (x - self.shift) / self.scale

    def inverse_transform(self, x):
        y = (self.scale * x) + self.shift
        if self.log:
            return np.power(10, y) - self.eps
        return y


class Standardizer:
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
