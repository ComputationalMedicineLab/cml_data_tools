"""
Mode configuration for experiments.
"""
import collections


class Config:
    """Generic namespace for carrying around configuration data on a mode.

    Parameters
    ----------
    mode : str
        The string name for the mode this instance configures.
    data : Iterator
    meta : Iterator
    curve_cls : class
    curve_kws : dict
    std_cls : class
    std_kws : dict
    std_params : dict
        All default to None, so that instances may be instantiated but then
        passed around different for configuration from other sources
    """
    def __init__(self, mode,
                 data=None, meta=None,
                 curve_cls=None, std_cls=None,
                 curve_kws=None, std_kws=None,
                 std_params=None):
        self.mode = mode
        self.data = data
        self.meta = meta
        self.curve_cls = curve_cls
        self.curve_kws = {} if curve_kws is None else curve_kws
        self.std_cls = std_cls
        self.std_kws = {} if std_kws is None else std_kws
        self.std_params = {} if std_params is None else std_params

    def __str__(self):
        return f'Config(mode={self.mode})'

    def __repr__(self):
        return f'<{str(self)}>'

    def valid_source(self):
        """Check if the instance is configured to be a data source"""
        return None not in (self.data, self.meta)

    def curve_builder(self, **kws):
        """Instantiate a curve builder class"""
        kwargs = collections.ChainMap(kws, self.curve_kws)
        return self.curve_cls(**kwargs)

    def standardizer(self, **kws):
        """Instantiate a standardizer class"""
        kwargs = collections.ChainMap(kws, self.std_kws)
        return self.std_cls(**kwargs)
