import collections.abc
import logging

import numpy as np
import pandas as pd
from fast_intensity import infer_intensity, regression

__all__ = [
    'build_patient_curves', 'build_all_patient_curves', 'CurveSetBuilder',
    'CurveBuilder', 'IntensityCurveBuilder', 'RegressionCurveBuilder',
    'BinaryCurveBuilder', 'FuzzedBinaryCurveBuilder', 'CumulativeTimeCurveBuilder',
    'AgeCurveBuilder', 'ConstantCurveBuilder', 'CumulativeCurveBuilder',
    'LogCumulativeCurveBuilder', 'EverNeverCurveBuilder', 'EventCurveBuilder',
    'BmiCurveBuilder', 'SmoothExpandingMax', 'ExpandingMean', 'Smoothed',
    'RollingIntensity', 'RollingRegression',
]


def build_patient_curves(df, spec, resolution='D'):
    """Builds curves for a patient.

    Arguments
    ---------
    df : pd.DataFrame[ptid, date, mode, channel, value]
        A dataframe of all the patient data for a given patient with columns
        for patient id (ptid), date, mode, channel, and value.

    spec : dict[str, callable]
        A mapping from modes to curve builder callables (cf. CurveBuilder). The
        callables need not subclass CurveBuilder so long as they have the same
        signature

    resolution : pandas.DateOffset
        A pandas DateOffset object giving the time resolution of the computed
        curves. See pandas Timeseries Offset Aliases for convenient ways to
        specify this. (Default 'D', which gives one - day resolution.)
    """
    ptid = df.ptid.values[0]
    grid = pd.date_range(df.date.min().floor(resolution),
                         df.date.max().ceil(resolution),
                         freq=resolution)

    curves = {}
    for mode, func in spec.items():
        data = df.loc[df['mode'] == mode]
        data = data.set_index('date')
        data = data.sort_index()
        result = func(data, grid)
        curves[mode] = result
    out = pd.concat(curves, axis=1, names=['mode', 'channel'])
    out = pd.concat([out], keys=[ptid], names=['id'], copy=False)
    return out


def build_all_patient_curves(patients, spec, resolution='D'):
    """Lazily applies build_curves to all patient dataframes in `patients`"""
    yield from (build_patient_curves(df, spec, resolution) for df in patients)


def _flatten(pipe):
    """(f0, (f1, (f2, (f3,)), f4)) => (f0, f1, f2, f3, f4)"""
    results = []
    for f in pipe:
        if callable(f):
            results.append(f)
        else:
            results.extend(flatten(f))
    return results


class CurveSetBuilder:
    """
    A CurveSetBuilder accepts a curve specification in the form of a dictionary
    mapping column names to curve processing pipelines (tuples of functions).
    Each pipeline may contain nested iterables of functions; the pipeline is
    flattened before use.

    A simple example:

    >>> builder = CurveSetBuilder({
    ...   'Labs': (get_labs, run_regression),
    ...   'LabsInt': (get_labs, run_regression, integrate)
    ... })
    >>> curves = builder(patient_record)

    The above code could have equivalently been written:

    >>> shared = (get_labs, run_regression)
    >>> builder = CurveSetBuilder({
    ...   'Labs': shared,
    ...   'LabsInt': (shared, integrate)
    ... })
    >>> curves = builder(patient_record)

    In either case, when the builder is called with a patient record, the
    following occurs:

    1) A `grid` of timestamps is generated from the patient record according to
       the specified `resolution` (default is 'D').
    2) The functions in each pipeline are run left to right. Each function in
       the pipe needs to accept the output of the prior function as its first
       positional argument, and accept a keyword argument `grid` (equivalently,
       a second positional argument).  The first function is called with the
       patient record: `func(record, grid=grid)`.
    3) Functions in the pipe are cached by prefix as an alternative to caching
       by input, since our inputs are often expected to be DataFrames or other
       objects that are difficult / expensive to hash.  In other words, if a
       function is shared in several pipes, *and* every function before it is
       also shared, then we infer that the results will also be shared and
       reuse them. In the example given above, `get_labs` and `run_regression`
       are both executed _once_ and then reused in the pipe for "LabsInt."
    3) If any function in a pipeline returns None, the pipe is aborted and the
       key excluded from the final dataset.
    4) After running each pipe, the results are concatenated by key into a
       single DataFrame, which then gains a column for the patient ID and is
       returned.

    Pipelines may theoretically contain inner CurveSetBuilders, but these will
    not cache in tandem.  Implementing caching optimizations across nested
    curve set generation specifications would require significantly more graph
    theoretic machinery than is here implemented.

    Args:
        steps (dict): cf. above

        resolution (pandas.DateOffset):
            A pandas DateOffset object giving the time resolution of the
            computed curves. See pandas Timeseries Offset Aliases for
            convenient ways to specify this. (Default 'D', which gives one -
            day resolution.)
    """
    def __init__(self, steps, resolution='D'):
        step_pairs = dict(steps).items()
        self.steps = {name: _flatten(pipe) for name, pipe in step_pairs}
        self.resolution = resolution

    def __call__(self, record):
        """Build curves for each component in a patient's record.

        Args:
            record (PatientRecord): specifies a patient's EHR

        Returns:
            A dataframe indexed by timestamps between record.start_date and
            record.end_date as specified by `resolution`.  Columns are
            multi-indexed on mode and channel.  All cell values are floats.
        """
        grid = pd.date_range(record.start_date.floor('D'),
                             record.end_date.ceil('D'),
                             freq=self.resolution)
        cache = {}
        curves = {}
        for name, pipe in self.steps.items():
            result = record  # initial value for all pipes
            path = tuple()  # lists have append, but are not hashable
            for func in pipe:
                # Cache by the entire path through the pipe so that results are
                # only shared if they share inputs.  Caching per-func will lead
                # to incorrect results, and trying to decorate each stage of a
                # processing pipeline with something like functools.lru_cache
                # is difficult b/c our inputs are often not hashable.
                path = path + (func, )
                if path not in cache:
                    cache[path] = func(result, grid=grid)
                result = cache[path]
                # If any stage of a pipe returns None, truncate execution and
                # exclude this key from the result dataframe (this is the case
                # if, for example, the patient record doesn't contain any of a
                # given EHR mode - no Surgeries, etc)
                if result is None:
                    break
            else:
                curves[name] = result
        df = pd.concat(curves, axis=1, names=['mode', 'channel'])
        df = pd.concat([df],
                       keys=[record.patient_id],
                       names=['id'],
                       copy=False)
        return df


class CurveBuilder(collections.abc.Callable):
    """Converts data to a curve"""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__qualname__)

    def _build_single_curve(self, data, grid, **kwargs):
        """Computes curve values at times given by grid.

        Args:
            data: a pandas dataframe with DatetimeIndex and columns appropriate
                for the builder.
            grid : a pandas series containing the times at which to compute the
                curve values.
            kwargs: a dictionary containing additional arguments as needed for
                subclasses.

        Returns: A pandas series indexed by the times in `grid`.
        """
        raise NotImplementedError

    def __call__(self, data, grid, **kwargs):
        """Build curves for all channels in `data`.

        Overriding this method allows for analysis of all information in `data`
        before building single curves. If a subclass can build curves using
        only information in a single channel, then it needs to override only
        _build_single_curve() and not this method. If a curve builder needs
        information from other channels in the same mode, then the subclass
        will need to override this method as well.

        This is also the place to modify `data` as a whole set, if needed,
        before building individual curves.

        Args:
            data: a pandas dataframe containing all data for a given mode,
                with a DatetimeIndex. Usually a component from a
                MergedDataRecord.
            grid: A pandas series containing the timestamps at which to compute
                curve values.
            kwargs: Additional information to be passed to _build_single_curve()

        Returns:
            a pandas dataframe indexed by the Timestamps in `grid` as
            'date', with columns corresponding to the variable names in `data`.

        Usage:
            A simple subclass usage may be as follows:

            def __call__(self, data, grid, **kwargs):
                my_info = self._extract_my_info(data)
                kwargs['info'] = my_info
                transformed_data = self._transform(data)
                return super().__call__(filtered_data, grid, **kwargs)
        """
        curveset = {'date': grid}
        for channel, df in data.groupby(by='channel'):
            curveset[channel] = self._build_single_curve(df, grid, **kwargs)

        curve_df = pd.DataFrame.from_dict(curveset)
        curve_df.set_index('date', inplace=True)
        return curve_df


class IntensityCurveBuilder(CurveBuilder):
    """Builds a curve giving the approximate instantaneous time - intensity of
    events.

    Args:
        iterations(int): The number of iterations of the curve building
            algorithm. (Default 100). Larger values produce smoother curves,
            but take longer to compute.
    """
    def __init__(self, iterations=100):
        super().__init__()
        self.iterations = iterations

    def _build_single_curve(self, data, grid, smooth=10, **kwargs):
        """Build a single curve from `data` at time points given by `grid`.

        Args:
            data: a pandas dataframe with a DatetimeIndex giving the event times.
            grid: a pandas DatetimeIndex giving the equally-spaced times at
                which to compute event intensity.

            smooth: an integer > 1 (default 10) containing the smallest number
                of events to smooth over. Larger numbers give smoother curves.

            kwargs: Unused.

        Returns:
            An ndarray of event intensities the same length as `grid`.

        """
        # create bin boundaries in units of days such that grid values are the
        # bin midpoints
        delta = grid[1] - grid[0]
        boundaries = np.empty(len(grid) + 1, dtype=float)
        boundaries[0:-1] = (
            (grid - grid[0] - 0.5 * delta).values) / np.timedelta64(1, 'D')
        boundaries[-1] = boundaries[-2] + (delta / np.timedelta64(1, 'D'))

        days = (data.index - grid[0]).values / np.timedelta64(1, 'D')

        return infer_intensity(events=days,
                               grid=boundaries,
                               iterations=self.iterations,
                               min_count=max(smooth, 1))


class RegressionCurveBuilder(CurveBuilder):
    def _build_single_curve(self, data, grid, **kwargs):
        """Build a single curve from `data` at time points given by `grid`.

        Args:
            data: a pandas dataframe with measurement values in the `value`
                column and a DatetimeIndex.
            grid: a pandas DatetimeIndex giving the times at which to
                estimate the measurement value.
            kwargs: unused.

        Returns:
            An ndarray of event intensities the same length as `grid`.
        """
        ref_date = grid[0]
        grid_deltas = grid - ref_date
        grid_days = grid_deltas.total_seconds().values / (3600 * 24)

        deltas = data.index - ref_date
        days = deltas.total_seconds().values / (3600 * 24)
        measurements = data.value.values

        return regression(days, measurements, grid_days)


class BinaryCurveBuilder(CurveBuilder):
    """This builder estimates a binary signal for each channel.

    The canonical example for this builder is a curve representing whether a
    medication was present or absent. The inference is complicated by the fact
    that medications are only noted when present.  We must infer the absence of
    medication X by finding that medication Y was noted on a given date but
    medication X was not.

    Furthermore, medications are only noted on particular dates, and we must
    infer what happened between those dates. This class assumes by default that
    if a medication was present on two adjacent observed dates, then it was
    also present between those dates, and the same for absence. If a medication
    was present on one observed date and absent on an adjacent obseved date,
    then we must infer when the transition was made. This builder allows the
    user to choose a transition made just after the first observation, just
    before the second observation, or at their midpoint(the default). More
    sophisticated transitions could be constructed, but this class does not
    implement them.

    The default behavior can be changed so that medications are inferred as
    absent except on dates specifically observed as present. This mode can be
    useful for status curves such as inpatient/outpatient, where the imputation
    between observations is unwanted.

    Args:
        imputation_method: {'bfill', 'ffill', 'nearest', None} the interpolation
            method to be used to fill dates in the intervals between observed
            dates in `all_dates`. 'bfill' fills with the next observed value,
            causing any transition to be made just after the first observed
            date. 'ffill' fills with the previous observed value, causing any
            transition to be made just before the second observed
            date. 'nearest' (default) fills with the nearest observed value,
            causing any transition to be made at the midpoint. `None` (the
            keyword, not the string) provides no interpolation between dates -
            any date not specifically observed as present is computed as
            absent.
    """
    def __init__(self, imputation_method='nearest'):
        super().__init__()
        self.imputation_method = imputation_method

    def _build_single_curve(self, data, grid, **kwargs):
        """Build a single curve from `data` at time points given by `grid`.

        Args:
            data: a pandas dataframe with a DatetimeIndex containing the times
                of events for this channel. All other columns are ignored.
            grid: a pandas Index of timestamps giving the times at which to
                estimate the binary value.
        kwargs:
            all_dates: a pandas Index of timestamps giving the times at
                which the events in `data` *could* have occurred.
        Returns:
            An ndarray of floats the same length as `grid`, where the array
                value is 1.0 on dates when the medication is inferred as
                present, 0.0 when absent.
        """
        freqstr = grid.freqstr
        all_dates = kwargs.pop('all_dates').round(freqstr).unique()

        dates = data.index.round(freqstr).unique()

        # Dates with an entry are assigned present (curve = 1)
        curve = pd.Series(1.0, index=dates)

        # Dates for other entries, but not this one, are assigned absent (curve = 0)
        curve = curve.reindex(index=all_dates, fill_value=0.0, copy=False)

        # Fill intervals between observations.
        curve = curve.reindex(index=grid, method=self.imputation_method)

        # If there are any unfilled times at this point (which will only occur
        # if method=None), then they are assigned absent:
        curve.fillna(0.0, inplace=True)

        return curve.values

    def __call__(self, data, grid, **kwargs):
        """Build a single curve from `data` at time points given by `grid`.

        This builder estimates whether something was present(curve=1) or
        absent(curve=0) at the points in `grid`, given the times in `data`
        where it was observed to be present. Times present in `all_dates`, but
        not in `data` are considered observations of absence. Dates between
        `all_dates` are interpolated using `self.imputation_method`.

        This method was originally designed to build curves representing
        presence or absence of medications, which generally occur in the record
        only when the patient is taking them, hence the need to infer absence
        in this way.

        All input times are rounded to 'D' resolution.

        Args:
            data: a pandas dataframe containing a column 'channel' giving the
                channel name observed on that date, and a DatetimeIndex.
            grid: a pandas DatetimeIndex giving the times at which to
                estimate the binary value.
        """
        all_dates = data.index.unique()
        kwargs['all_dates'] = all_dates
        return super().__call__(data, grid, **kwargs)


class FuzzedBinaryCurveBuilder(BinaryCurveBuilder):
    def __init__(self, *args, fuzz_length=10, **kwargs):
        self.fuzz_length = fuzz_length
        super().__init__(*args, **kwargs)

    def _build_single_curve(self, data, grid, **kwargs):
        freqstr = grid.freqstr
        all_dates = kwargs.pop('all_dates')
        all_dates = all_dates.round(freqstr).unique().sort_values()
        dates = data.index.round(freqstr).unique().sort_values()

        # Basic curve imputation - uses fill between
        curve = pd.Series(1.0, index=dates)
        curve = curve.reindex(index=all_dates, fill_value=0.0, copy=False)
        curve = curve.reindex(index=grid, method=self.imputation_method)
        curve.fillna(0.0, inplace=True)

        # Add the fuzz tail for existing dates
        delta = pd.Timedelta(days=self.fuzz_length)
        stopdate = grid.max()
        tail_dates = dates.copy()
        for x in dates:
            dts = pd.date_range(x, min(x+delta, stopdate), freq=freqstr)
            tail_dates = tail_dates.union(dts)
        curve[tail_dates] = 1.0

        return curve.values


class CumulativeTimeCurveBuilder(BinaryCurveBuilder):
    def _build_single_curve(self, data, grid, **kwargs):
        curve = super()._build_single_curve(data, grid, **kwargs)
        return np.cumsum(curve)


class AgeCurveBuilder(CurveBuilder):
    def _build_single_curve(self, data, grid, **kwargs):
        """Build a single curve from `data` at time points given by `grid`.

        Args:
            data: a pandas dataframe with the date of birth in the `value`
                column. Only the first element of this column is used.
            grid: a pandas DatetimeIndex giving the times at which to
                estimate the age.
            kwargs: unused.

        Returns:
            An ndarray of age in years for each date in `grid`.
        """
        dob = pd.to_datetime(data.value.iloc[0])
        if not dob:
            self.logger.info('No DOB for %s', data.ptid.iloc[0])
            dob = grid[0]
        curve = grid.to_series().sub(dob).div(pd.Timedelta(days=365.25))
        return curve.values


class ConstantCurveBuilder(CurveBuilder):
    def _build_single_curve(self, data, grid, **kwargs):
        """Build a single curve from `data` at time points given by `grid`.

        Args:
            data: a pandas dataframe with measurement values in the `value`
                column and a DatetimeIndex.
            grid: a pandas DatetimeIndex giving the times at which to
                estimate the measurement value.
            kwargs: unused.

        Returns:
            An ndarray of event intensities the same length as `grid`.
        """
        curve = np.ones_like(grid.values, dtype=float)
        return curve


class CumulativeCurveBuilder(CurveBuilder):
    """Constructs curves with cumulative counts.

    Intended to be used for procedures, where procedures happen rarely and the
    important property is whether it has happened or not by time t.
    """
    def _build_single_curve(self, data, grid, **kwargs):
        """Build a single curve from `data` at time points given by `grid`.

        Args:
            data: a pandas dataframe with measurement values in the `value`
                column and a DatetimeIndex.
            grid: a pandas DatetimeIndex giving the times at which to
                estimate the measurement value.
            kwargs: unused.

        Returns:
            An ndarray of event intensities the same length as `grid`.
        """
        freqstr = grid.freqstr
        dates = data.index.round(freqstr).unique()

        # Dates with an entry are assigned present (curve = 1)
        curve = pd.Series(1.0, index=dates)

        # Dates for other entries, but not this one, are assigned absent (curve = 0)
        curve = curve.reindex(index=grid, fill_value=0.0, copy=False)
        return np.cumsum(curve).astype(int)


class LogCumulativeCurveBuilder(CumulativeCurveBuilder):
    """Constructs curves with log10(1 + (cumulative counts)).

    Intended to be used for procedures, where procedures happen rarely and the
    important property is whether it has happened or not by time t.
    """
    def _build_single_curve(self, data, grid, **kwargs):
        """Build a single curve from `data` at time points given by `grid`.

        Args:
            data: a pandas dataframe with measurement values in the `value`
                column and a DatetimeIndex.
            grid: a pandas DatetimeIndex giving the times at which to
                estimate the measurement value.
            kwargs: unused.

        Returns:
            An ndarray of event intensities the same length as `grid`.
        """
        curve = super()._build_single_curve(data, grid, **kwargs)
        return np.log10(1 + np.cumsum(curve))


class EverNeverCurveBuilder(CurveBuilder):
    """Constructs curves with ever/never indication.

    Intended to be used for procedures, where procedures happen rarely and the
    important property is whether it has happened (curve = 1) or not
    (curve = 0) by time t.
    """
    def _build_single_curve(self, data, grid, **kwargs):
        """Build a single curve from `data` at time points given by `grid`.

        Args:
            data: a pandas dataframe with measurement values in the `value`
                column and a DatetimeIndex.
            grid: a pandas DatetimeIndex giving the times at which to
                estimate the measurement value.
            kwargs: unused.

        Returns:
            An ndarray of event intensities the same length as `grid`.
        """
        freqstr = grid.freqstr
        dates = data.index.round(freqstr).unique()

        # Dates with an entry are assigned present (curve = 1)
        curve = pd.Series(1.0, index=dates)

        # Dates for other entries, but not this one, are assigned absent (curve = 0)
        curve = curve.reindex(index=grid, fill_value=0.0, copy=False)

        return np.maximum.accumulate(curve)


class EventCurveBuilder(CurveBuilder):
    """Constructs curves with ever/never indication of event value == 1.

    This differs from the EverNeverCurveBuilder in that the event may be
    positive (value == 1) or negative (value == 0). The EverNeverCurveBuilder
    calls any event with a date positive, but this one calls it positive only
    when the value==1. This is useful for handling censored data, where the
    event for a censored record is the last day of the record, but its value is
    0.
    """
    def _build_single_curve(self, data, grid, **kwargs):
        """Build a single curve from `data` at time points given by `grid`.

        Args:
            data: a pandas dataframe with measurement values in the `value`
                column and a DatetimeIndex.
            grid: a pandas DatetimeIndex giving the times at which to
                estimate the measurement value.
            kwargs: unused.

        Returns:
            An ndarray of event intensities the same length as `grid`.
        """
        freqstr = grid.freqstr
        first_event_date = data.index.round(freqstr)[0]
        val = data.value[0]
        curve = pd.Series(val, index=[first_event_date])

        # Dates for other entries, but not this one, are assigned absent (curve = 0)
        curve = curve.reindex(index=grid, fill_value=0.0, copy=False)

        return np.maximum.accumulate(curve)


class BmiCurveBuilder(RegressionCurveBuilder):
    """This builder estimates the BMI curve from height and weight curves.
    """
    def __call__(self, data, grid, **kwargs):
        """Construct curves of height, weight, and bmi.
        All input times are rounded to 'D' resolution.

        Args:
            data: a pandas dataframe containing columns 'channel' (string),
                'value' (numeric), and 'date' (DateTimeIndex). Height must be
                named 'Height' with values in cm. Weight must be named 'Weight'
                with values in kg.
            grid: a pandas DatetimeIndex giving the times at which to
                estimate the values.

        Returns:
            a pandas dataframe containing curves for the columns in 'channel' with
                an additional column named 'BMI'. All columns are calculated
                using a RegressionCurveBuilder, with the exception of the `BMI`
                column, which is computed directly from 'Height' and 'Weight'.
        """
        curves = super().__call__(data, grid, **kwargs)
        if 'Height' in curves.columns and 'Weight' in curves.columns:
            curves.loc[:, 'BMI'] = (10000 * (curves.Weight /
                                             (curves.Height * curves.Height)))
        return curves


def ExpandingMean():
    """A higher-order builder that computes the expanding mean of curves.

    This builder operates on previously-computed curves, rather than a data
    record.  An output curve at time t contains the mean of the input curve
    between time 0 and t.  It approximates the continuous, integrated mean
    using a Riemann sum.

    Returns:
        A function f(curves, **kwargs) that expects a pandas dataframe of
        curves as input and returns a dataframe of the same shape, consisting
        of new curves described above. kwargs are ignored.

    Usage:
        builder = CurveSetBuilder({
            'Integrated Top Labs': (get('Top Lab Values'),
                                    RegressionCurveBuilder(),
                                    ExpandingMean())
        }

    """
    def func(curves, **kwargs):
        return curves.expanding(min_periods=1).mean()

    return func


def SmoothExpandingMax(window='14D'):
    """A higher-order builder that computes the smoothed expanding max of curves.

    This builder operates on previously-computed curves, rather than a data
    record.  An output curve at time t contains the max of the smoothed input
    curve between time 0 and t.

    Smoothing is done with a rolling median of size `window`. For some reason,
    the underlying pandas comaplains if this is expressed in units larger than
    'D'. So '30D' works, but '4W' does not.

    Args:
        window: A pandas.DateOffset or alias string specifying the smoothing
            window size. (default '30D')

    Returns:
        A function f(curves, **kwargs) that expects a pandas dataframe of
        curves as input and returns a dataframe of the same shape, consisting
        of new curves described above. kwargs are ignored.

    Usage:

        builder = CurveSetBuilder({
            'Max Phecodes': (get('Phecodes'),
                             IntensityCurveBuilder(),
                             SmoothExpandingMax('14D')),

    """
    def func(curves, **kwargs):
        return curves.rolling(window).median().expanding(min_periods=1).max()

    return func


def Smoothed(window='14D'):
    """A higher-order builder that computes median-smoothed curves.

    This builder operates on previously-computed curves, rather than a data
    record.  An output curve at time t contains the median of the input curve
    in the window around t. For some reason, the underlying pandas comaplains
    if this window is expressed in units larger than 'D'. So '30D' works, but '4W'
    does not.

    Args:
        window: A pandas.DateOffset or alias string specifying the smoothing
            window size. (default '30D')

    Returns:
        A function f(curves, **kwargs) that expects a pandas dataframe of
        curves as input and returns a dataframe of the same shape, consisting
        of new curves described above. kwargs are ignored.

    Usage:

        builder = CurveSetBuilder({
            'Smooth Phecodes': (get('Phecodes'),
                                IntensityCurveBuilder(),
                                Smoothed('14D')),
            }
    """
    def func(curves, **kwargs):
        return curves.rolling(window).median()

    return func


class RollingIntensity:
    """Wraps the IntensityCurveBuilder with a windowed mean"""
    def __init__(self, window, *args, **kwargs):
        self.window = window
        self.inst = IntensityCurveBuilder(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        curves = self.inst(*args, **kwargs)
        return curves.rolling(self.window, min_periods=1).mean()


class RollingRegression:
    """Wraps the RegressionCurveBuilder with a windowed mean"""
    def __init__(self, window, *args, **kwargs):
        self.window = window
        self.inst = RegressionCurveBuilder(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        curves = self.inst(*args, **kwargs)
        return curves.rolling(self.window, min_periods=1).mean()
