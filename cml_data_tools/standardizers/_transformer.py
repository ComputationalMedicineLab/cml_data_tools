import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataframeStandardizer(BaseEstimator, TransformerMixin):
    """A dataframe standardizer with a custom transformer for each data mode.

    This is intended to be used on the dataframe of cross sections destined for
    phenotype model fitting. It allows for transforming all curves into roughly
    the same range of values, which is necessary for getting an interpretable
    model. Curves of different modes or even different channels may otherwise
    cover vastly different scales, which would obscure interpretation of the
    model.

    Usage
    -----
    >>> dfs = DataframeStandardizer()
    >>> # Add some standardizers...
    >>> dfs.add_standardizer('Medications', lambda x: 4 * x)
    >>> # Fit and transform
    >>> dfs.fit(X)
    >>> transformed = dfs.transform(X)
    >>> new_transformed = dfs.transform(new_X)
    """
    def __init__(self, configs=None):
        self._transformer = {}
        self._standardizer_info = {}
        if configs is not None:
            self.configure(configs)

    def fit(self, X, y=None):
        """Fit a transformer for each column of X. Returns self.

        Arguments
        ---------
        X : pandas.DataFrame
            A pandas dataframe with one column per channel of data, typically a
            set of cross sections intended as input to a phenotype model.
        """
        for channel_name in X:
            channel = X[channel_name]
            mode = channel_name[0]
            # There's a different standardizer for each channel, but the
            # standardizer class and kwargs are specified per mode:
            standardizer, kwargs = self._standardizer_info[mode]
            self._transformer[channel_name] = standardizer(**kwargs)
            self._transformer[channel_name].fit(channel)
        return self

    def transform(self, X, y=None):
        """Transform each column of X.

        Arguments
        ---------
        X : pandas.DataFrame
            A pandas dataframe, with columns corresponding to channels of data.
            Typically a set of cross sections intended as input to a phenotype
            model.

        Returns
        -------
        pandas.DataFrame
            The transformed dataframe, same size as X.
        """
        for col_name in X:
            result = self._transformer[col_name].transform(X[col_name])
            X.loc[:, col_name] = result
        return X

    def inverse_transform_label(self, name, delta, spec=None):
        return self._transformer[name].inverse_transform_label(delta=delta,
                                                               spec=spec)

    def add_standardizer(self, mode, standardizer, **kwargs):
        """Add a standardizer for a given mode of data.

        At fitting time, one of these standardizers per column of data in the
        given mode will be created using `kwargs`.

        Arguments
        ---------
        mode : str
            Specifies the mode for which the standardizer will operate.
        standardizer : a Standardizer class
        kwargs : the set of kwargs to be passed to standardizer.__init__().
        """
        self._standardizer_info[mode] = (standardizer, kwargs)

    def get(self, mode):
        return self._standardizer_info.get(mode)

    def get_mode_scales(self):
        scales = {}
        # Lots of repeats, but probably not that slow.
        for channel_name in self._transformer:
            mode = channel_name[0]
            scales[mode] = self._transformer[channel_name].scale
        return scales

    def set_mode_scales(self, scales):
        for channel_name in self._transformer:
            mode = channel_name[0]
            self._transformer[channel_name].scale = scales[mode]

    def transformer_properties(self):
        """Yields the mean and stdev of all fit transformers in this
        DataframeStandardizer.  Requires fitting first.

        Yields
        ------
        Tuple : [obj, float, float]
            3-tuples of (channel key, mean, stdev)
        """
        for key, transformer in self._transformer.items():
            yield (key, transformer.mean_, transformer.stdev_)

    def configure(self, configs, extra_std_kws=None):
        """Convenience method for adding standardizers from conf objects"""
        xtra = extra_std_kws or {}
        for config in configs:
            kws = config.std_kws.copy()
            kws.update(xtra.get(config.mode, {}))
            self.add_standardizer(config.mode, config.std_cls, **kws)


class SeriesStandardizer(BaseEstimator, TransformerMixin):
    def fit(self, series):
        """Fits a standardization function.

        The fitted function transforms an input data series into a specified
        output space. A simple example would be a statistical standardizer that
        shifts and scales `series` so that its mean is zero and standard
        deviation is one, and fixes that transformation so that new data can be
        scaled and shifted by the same amount.

        The fitted function may be learned from the data in `series`, but
        simpler functions that are not learned from data are allowed.
        (Subclassed methods must retain the `series` argument, even if they
        ignore it.)

        Arguments
        ---------
        series : pandas.Series
            Contains data from which to learn the transforming function.
        """
        raise NotImplementedError

    def transform(self, series):
        """Transforms the series by the previously - fit function.

        Arguments
        ---------
        series : pandas.Series
            Contains data to be transformed.
        """
        raise NotImplementedError

    def inverse_transform(self, series):
        """Reverses the transform of the fitted function.

        Usage
        -----
        >>> std = SeriesStandardizer()
        >>> std.fit(my_series)
        >>> transformed_series = std.transform(my_series)
        >>> recovered_series = std.inverse_transform(transformed_series)
        >>> all(recovered_series == my_series)
        True

        Arguments
        ---------
        series : pandas.Series
            Contains data to be inverse transformed.
        """
        raise NotImplementedError

    def inverse_transform_label(self, delta, spec='+'):
        """Provides the meaning of `delta` in the original space, with format
        `spec`.

        This is for use in phenotype plots. For most transforms, an additive
        change in the amount `delta` corresponds an scaled but still additive
        amount in the original space. In this case, `delta` in the transformed
        space is simply scaled to the original space and given an additive
        label. If the transform includes a logarithm operation, then an
        additive `delta` in the original space corresponds to a multiplicative
        change in the original space. In this case, `delta` is appropriately
        scaled and given a multiplicative label.

        Examples
        --------
        # If `x` is the original data, 'y' is the scaled data,
        # if `s` transforms by scaling 0.5 * x + c, then
        # x2 - x1 = 2.0 * (y2 - y1), regardless of the value of `c`:
        >>> s.inverse_transform_label(1.0)
        "+2.0"
        # and likewise:
        >>> s.inverse_transform_label(-1.0)
        "-2.0"

        # If `s` transforms by y = 3 * log10(x) - c, then
        # x2 / x1 = pow(10, (y2 - y1) / 3), regardless of the value of `c`:
        >>> s.inverse_transform_label(1.0)
        "*2.15"
        # and likewise:
        >>> s.inverse_transform_label(-1.0)
        "/2.15"

        """
        raise NotImplementedError

    @property
    def mean_(self):
        raise NotImplementedError

    @property
    def stdev_(self):
        raise NotImplementedError

    def _inverse_transform_difference(self, delta, anchor=0):
        orig = self.inverse_transform(pd.Series([anchor, anchor + delta]))
        return orig[1] - orig[0]

    def _inverse_transform_ratio(self, delta, anchor=1):
        orig = self.inverse_transform(pd.Series([anchor, anchor + delta]))
        return orig[1] / orig[0]


class LinearStandardizer(SeriesStandardizer):
    """A simple transformer that scales and offsets the input. This
    standardizer is non-data dependent.

    Parameters
    ----------
    scale : float (default 1.0)
        Constant scale factor by which to multiply each element
    offset : float (double 0.0)
        Constant offset to add to each element (default 0).
    """
    def __init__(self, scale=1.0, offset=0):
        self.offset = offset
        self.scale = scale

    @property
    def mean_(self):
        return self.offset

    @property
    def stdev_(self):
        return self.scale

    def fit(self, series):
        return self

    def transform(self, series):
        return self.scale * series + self.offset

    def inverse_transform(self, series):
        return (series - self.offset) / self.scale

    def inverse_transform_label(self, delta, spec='+.2f'):
        orig_delta = self._inverse_transform_difference(delta)
        label = f'{orig_delta:{spec}}'
        return label


class GelmanStandardizer(SeriesStandardizer):
    """Subtracts the mean and divides by *2* stdevs. Optionally, takes the log
    first.

    As suggested in Gelman2008: Gelman, A. Scaling regression inputs by
    dividing by two standard deviations Stat Med, Wiley Online Library, 2008,
    27 , 2865-2873.
    """
    def __init__(self, log_transform=False, eps=0, shift=True):
        self.eps = eps
        self.log_transform = log_transform
        self.shift = shift

    @property
    def mean_(self):
        return self._mean

    @property
    def stdev_(self):
        return self._stdev

    def fit(self, series):
        x = np.log10(series + self.eps) if self.log_transform else series
        self._mean = x.mean()
        self._stdev = x.std(ddof=0)
        return self

    def transform(self, series):
        x = np.log10(series + self.eps) if self.log_transform else series
        if self.shift:
            x -= self.mean_
        x /= 2 * self.stdev_
        return x

    def inverse_transform(self, series):
        x = series * (2 * self.stdev_)
        if self.shift:
            x += self.mean_
        if self.log_transform:
            x = np.power(10, x) - self.eps
        return x

    def inverse_transform_label(self, delta, spec=None):
        if self.log_transform:
            orig_frac = self._inverse_transform_ratio(delta)
            if spec.startswith('+'):
                spec = spec[1:]
            elif spec is None:
                spec = '.2f'
            label = f'x{orig_frac:{spec}}' if orig_frac > 1.0 else f'/{1.0/orig_frac:{spec}}'
        else:
            orig_delta = self._inverse_transform_difference(delta)
            if spec is None:
                spec = '+.2f'
            label = f'{orig_delta:{spec}}'
        return label


class LogGelmanStandardizerWithFallbacks(SeriesStandardizer):
    """Creates a standardized log transformer, falling back to:

    * LinearStandardizer(), i.e. identity, if all NaN
    * LinearStandardizer(offset=-series.min), i.e. offset to zero, if all are
      constant or nearly constant
    * LogGelmanStandardizer(log_transform=False) if more than 1% negative
    * LogGelmanStandardizer(log_transform=True) otherwise
    """
    def __init__(self, eps=0, log_default=True, shift=True):
        self.eps = eps
        self.log_default = log_default
        self.shift = shift
        self.log = logging.getLogger(self.__class__.__name__)

    @property
    def mean_(self):
        return self._transformer.mean_

    @property
    def stdev_(self):
        return self._transformer.stdev_

    def fit(self, series):
        # Use LinearStandardizer as placeholder if all NaN
        if (series.isna().all()):
            self.log.info(f'Series {series.name} is all NaN. '
                          f'Using identity standardizer instead.')
            self._transformer = LinearStandardizer()

        # Use LinearStandardizer if all values are basically the same.
        elif ((series.max() - series.min()) < 1e-6 *
              (series.max() + series.min())):
            self.log.info(f'Series {series.name} is nearly constant '
                          f'(value: {(series.min() + series.max()) / 2:.2g}, '
                          f'delta: {series.max() - series.min():.2g}). '
                          f'Using offset standardizer instead.')
            self._transformer = LinearStandardizer(offset=-series.min())

        # Use Gaussian 2sd standardizer if there are more than a few negatives
        elif (series < 0).sum() > 0.01 * series.size:
            num_negative = (series < 0).sum()
            num_total = series.size
            self.log.info(f'Series {series.name} has {num_negative} '
                          f'({num_negative / num_total:.1%}) negative values. '
                          f'Using Gelman standardizer instead.')
            self._transformer = GelmanStandardizer(log_transform=False,
                                                   shift=self.shift)

        else:
            self._transformer = GelmanStandardizer(self.log_default, self.eps,
                                                   shift=self.shift)

        self._transformer.fit(series)
        return self

    def transform(self, series):
        return self._transformer.transform(series)

    def inverse_transform(self, series):
        return self._transformer.inverse_transform(series)

    def inverse_transform_label(self, delta, spec=None):
        if spec is None:
            return self._transformer.inverse_transform_label(delta)
        return self._transformer.inverse_transform_label(delta, spec=spec)


# Compat
GelmanStandardizerWithFallbacks = LogGelmanStandardizerWithFallbacks
