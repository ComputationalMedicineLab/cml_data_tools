import numpy as np
import pandas as pd


def expand_and_fill_cross_sections(meta, standardizer, cross_sections):
    """Merges the cross sections present in `cross_sections` using the metadata
    and standardization functions in `meta` and `standardizer`.

    Arguments
    ---------
    meta : pandas.DataFrames
        A dataframe with columns 'mode', 'channel', 'description', and 'fill',
        giving the fill values for each channel.
    standardizer : cml_data_tools.standardizers.Standardizer
        A Standardizer or compatible object with a `transform` function and
        `curve_stats` attribute containing channel names and statistics.
    cross_sections : Iterator[pandas.DataFrames]
        A finite iterator of dataframes indexed by patient ID and date, with
        (mode, channel) columns. No two need have any channels in common.

    Returns
    -------
    pandas.DataFrame
        A single dataframe containing rows for all patient id, date pairs in
        the input cross sections and columns for all channels in the input
        cross sections, as if produced by pandas `merge`. Anywhere a channel is
        not present in a patient's data the missing values have been filled
        using `meta`.  All values are standardized using the `standardizer`.
    """
    # XXX: This work is bundled into subroutines to (a) keep clean namespaces,
    # but also (b) so that we can easily jettison namespaces (and hence
    # references) and not keep the entire chain of partial products in memory
    channels = standardizer.curve_stats.channels
    matrix, index = construct_matrix(cross_sections, channels)
    matrix, channels = prune_nan_channels(matrix, channels)
    matrix = assemble_dataframe(matrix, index, channels)

    # Q: Would a numpy based transform that circumvents pandas be faster?
    matrix = standardizer.transform(matrix)
    fill = extract_fill_values(meta, standardizer, channels)
    # Sanity check that fill and matrix vals are aligned via channel
    assert np.all(fill.index == matrix.columns)

    # Fill NaNs *inplace!* in data matrix with standardized fill values
    fill_values = fill.values
    for ix in range(len(matrix)):
        row = matrix.values[ix]
        matrix.values[ix] = np.where(np.isnan(row), fill_values, row)

    return matrix


def construct_matrix(cross_sections, channels):
    # Construct the expanded matrix parts filled, for now, with NaN
    n_channels = len(channels)
    dense_parts = []
    index_parts = []
    for df in cross_sections:
        dense = np.full((len(df), n_channels), np.nan, dtype=np.float64)
        _, _, idx = np.intersect1d(df.columns, channels,
                                   assume_unique=True,
                                   return_indices=True)
        dense[:, idx] = df.values
        dense_parts.append(dense)
        index_parts.append(df.index.values)
    dense = np.concatenate(dense_parts)
    index = np.concatenate(index_parts)
    return dense, index


def prune_nan_channels(matrix, channels):
    # Prune all-NaN channels from data matrix and channel list
    not_all_nan = ~np.all(np.isnan(matrix), axis=0)
    matrix = matrix[:, not_all_nan]
    channels = channels[not_all_nan]
    return matrix, channels


def assemble_dataframe(matrix, index, channels):
    # Assemble expanded DataFrame with appropriate index & columns
    index = pd.MultiIndex.from_tuples(index, names=['id', 'date'])
    columns = pd.MultiIndex.from_tuples(channels, names=['mode', 'channel'])
    return pd.DataFrame(matrix, index=index, columns=columns)


def extract_fill_values(meta, standardizer, channels):
    # Get fill Series from `meta`, w. dtype set and pruned by channels
    meta = meta.set_index(['mode', 'channel'])
    meta.fill = meta.fill.astype(np.float64)
    meta = meta.loc[channels]
    # Then cast Series to DataFrame for standardizing, then back to Series for
    # filling in the data matrix
    fill = pd.DataFrame(meta.fill).transpose()
    fill = standardizer.transform(fill)
    fill = fill.loc['fill']
    return fill
