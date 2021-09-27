"""
Tools for plotting objects
"""
import collections
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backends.backend_pdf import PdfPages

from cml_data_tools.clustering import (intra_cluster_mean_std,
                                       PerfectClusterScorer)


MODE_COLOR = {'Measurements': 'lightcoral',
              'Conditions': 'lightskyblue',
              'Medications': 'palegreen',
              'Procedures': 'mediumseagreen',
              'Sex': 'mediumvioletred',
              'Race': 'slateblue',
              'Age': 'teal',
              'Bmi': 'indianred',
              'ANA': 'cadetblue'}




class PhenotypePlotter:
    def __init__(self, meta, standardizer):
        self.meta = meta
        self.standardizer = standardizer

        self.name_descriptions = {(t.mode, t.channel): t.description
                                  for t in meta.itertuples()}

    def get_stripped_label_description(self, name, width):
        flags = re.IGNORECASE
        desc = self.name_descriptions.get(name, 'No Description')
        # Strip out units in square brackets
        desc = re.sub(r'\[.*\]', '', desc, flags=flags)

        # Strip out "in Blood" etc. NB: ORDER is important!
        replacements = {'in Serum, Plasma, or Blood': 'in SPB',
                        'in Serum or Plasma': 'in SP',
                        'in Serum': 'in S',
                        'in Plasma': 'in P',
                        'in Blood': 'in B',
                        'by automated count': 'by AC'}
        for phrase, replacement in replacements.items():
            desc = re.sub(phrase, replacement, desc, flags=flags)

        # Strip any doubled or extra whitespace generated by previous
        desc = re.sub(r'\s+', ' ', desc)
        # Replace any extraneous underscores (usually a programming artifact)
        desc = desc.replace('_', ' ')
        desc = textwrap.shorten(desc, width=width, placeholder='...')
        return desc

    def generate_labels(self, phenotype, trunc_at=100):
        """Generate labels for the given phenotype series"""
        labels = []
        for c, val in phenotype.iteritems():
            ex = self.standardizer.inverse_transform_label(c, val, spec='+.4g')
            impact = f'{ex}'
            desc = self.get_stripped_label_description(c, trunc_at)
            labels.append(f'{desc} ({impact})')
        return labels

    def top_channels(self, phenotype, thresh=None, nmax=None, nmin=None):
        """
        Selects a least nmin and at most nmax channels from a given phenotype
        series, preferring channels with values above the given threshold if
        thresh is given.

        Arguments
        ---------
        phenotype : pd.Series or pd.DataFrame

        Keyword Arguments
        -----------------
        thresh : float
            Include only channels with an absolute value above `thresh`. If
            None is given, `max(nmax, len(phenotype))` channels are returned.
        nmax : int
            Include at most `nmax` channels. Defaults to 60 if None is given.
        nmin : int
            Include at least `nmin` channels. Defaults to 10 if None is given.
        """
        # Using the following idiom instead of putting these directly as
        # keyword arguments allows callers and callers of callers to invoke
        # this function without needing to specify and re-specify the defaults.
        if nmax is None: nmax = 60
        if nmin is None: nmin = 10
        ph = phenotype.sort_values(key=np.absolute)[-nmax:]
        if thresh is not None:
            x = ph[ph.abs() > thresh]
            if len(x) < nmin:
                return ph[-nmin:]
            return x
        return ph

    def _plot_inset_expressions_hist(self, ax, expr):
        # Plot the expressions inset
        # https://matplotlib.org/3.3.3/gallery/axes_grid1/inset_locator_demo.html
        inset = inset_axes(ax, width=1.5, height=1.0, loc=3, borderpad=1.5)
        inset.hist(x=(expr[expr > 0], expr[expr < 0]),
                   bins=100, histtype='stepfilled', log=True,
                   color=('blue', 'red'), alpha=0.4, label=('pos', 'neg'))
        inset.set_title('Expressions', fontsize='small')
        inset.tick_params(direction='in', labelsize='x-small', pad=1.2)
        inset.set_ylim(bottom=0.8, top=None)
        xlim = max(map(abs, inset.get_xlim()))
        inset.set_xlim([-xlim, xlim])
        inset.patch.set_alpha(0.6)
        return inset

    def _get_figure_and_axes(self, n_labels):
        """Produces a figure and the major axes for a plot of n_labels"""
        width = 13
        # 1 inch for title, 0.5 for margin on either side, and 1 per 4 channels
        height = 1.5 + (n_labels*0.25)
        # add_axes(rect); rect=[left, bottom, width, height], as fraction of
        # whole (0..1) 1 inch as fraction of whole is 1/height
        margin = 0.5 / height
        ax_ht = 1 - (3 * margin)

        fig = plt.figure(figsize=(width, height))
        bars = fig.add_axes(rect=(0.65, margin, 0.34, ax_ht))
        dots = fig.add_axes(rect=(0.625, margin, 0.025, ax_ht))
        return fig, bars, dots

    def _fill_dots(self, dots, xs, ys, xmax):
        dot_colors = [MODE_COLOR.get(m, 'white')
                      for m in xs.index.get_level_values(0)]
        dots.scatter(np.full_like(ys, (-xmax*0.8)), ys,
                     color=dot_colors,
                     marker='o', s=100, edgecolors='black', linewidths=0.5)

    def _set_limits(self, bars, dots, ys, labels, xmax):
        ylim = [-1, len(labels)]
        # Remove borders
        bars.spines['left'].set_visible(False)
        dots.spines['right'].set_visible(False)
        # Set correct x/y limits/ticks/labels
        bars.tick_params(axis="x", bottom=True, top=True,
                         labelbottom=True, labeltop=True)
        bars.set_yticks([])
        bars.set_yticklabels([])
        dots.set_yticks(ys)
        dots.set_yticklabels(labels)
        dots.set_xticks([])
        bars.set_ylim(ylim)
        dots.set_ylim(ylim)
        bars.set_xlim([-xmax, +xmax])

    def plot_single_barplot(self, phenotype, expressions, title='',
                            xmax=None, thresh=None):
        xs = self.top_channels(phenotype, thresh=thresh)
        labels = self.generate_labels(xs)
        fig, bars, dots = self._get_figure_and_axes(len(labels))

        bars.axvline(0, color='black', linestyle='solid')
        # Plot the bars as the values from the phenotype
        ys = range(len(labels))
        bars.barh(ys, xs, color='cornflowerblue')

        if xmax is None:
            xmax = max(map(abs, bars.get_xlim())) * 1.05

        self._fill_dots(dots, xs, ys, xmax)
        self._set_limits(bars, dots, ys, labels, xmax)
        self._plot_inset_expressions_hist(bars, expressions)

        title = f'{title or phenotype.name} (thresh={thresh})'
        fig.suptitle(title, fontsize=16)
        return fig, bars

    def plot_multi_barplots(self, phenotypes, expressions, fname,
                            project_name='', **kwargs):
        with PdfPages(fname) as pdf:
            for name in phenotypes.columns:
                title = f'{project_name} {name}'
                self.plot_single_barplot(phenotypes[name],
                                         expressions[name],
                                         title=title,
                                         **kwargs)
                pdf.savefig()
                plt.close()

    def plot_single_model(self, model, fname, project_name='', **kwargs):
        self.plot_multi_barplots(model.phenotypes_,
                                 model.expressions_,
                                 fname,
                                 project_name,
                                 **kwargs)


ClusterData = collections.namedtuple('ClusterData', 'center indices mean std')
ClusterData.__doc__ = """Auxiliary class for structuring model clusters"""


def structure_clustering(clusters, clustering):
    """
    Process the file-based data structures for the clustering into a more user
    friendly data format, an array of ClusterData objects. Also annotate these
    structures with some useful derivative statistics.

    Arguments
    ---------
    clusters : list
        A list of pairs (matrix, [row_idx, col_idx]) indicating the cluster's
        submatrix of the general affinity matrix used for the clustering, and
        the row / col indices into the affinity matrix which produced it.
    clustering : cml_data_tools.clustering.AffinityPropagationClusterer
        A trained AffinityPropagationClusterer object.
    is_perfect : callable
        A function or other callable that accepts a cluster and returns a bool.
    """
    structures = []
    for x in clusters:
        mean, std = intra_cluster_mean_std(x[0])
        indices = x[1][1].flatten()
        # Find the center of this cluster
        for c in indices:
            if np.isin(c, clustering.centers_):
                center = c
                break
        structures.append(ClusterData(center, indices, mean, std))
    return structures


def cluster_score_sort_key(datum):
    m = datum.mean
    if np.isnan(m):
        m = -np.inf
    return -(len(datum.indices) + (100 * m))


def cluster_lexical_sort_key(datum):
    return datum.center


class ClusterPlotter(PhenotypePlotter):
    """Generates horizontal bar charts for clusters of phenotypes."""

    def generate_labels(self, exemplar, members, trunc_at=100):
        """Generate labels for the cluster

        Includes impact expressions for 0.025 and 0.975 percentiles of the
        cluster members' values.
        """
        impact_func = self.standardizer.inverse_transform_label
        labels = []
        vlo = members.quantile(0.025)
        vhi = members.quantile(0.975)
        for c in members.columns:
            ex = impact_func(c, exemplar[c], spec='+.4g')
            lo = impact_func(c, vlo[c], spec='+.4g')
            hi = impact_func(c, vhi[c], spec='+.4g')
            impact = f'{ex} [{lo}, {hi}]'
            desc = self.get_stripped_label_description(c, trunc_at)
            labels.append(f'{desc} ({impact})')
        return labels

    def plot_single_cluster(self, exemplar, members, expressions,
                            title=None, xmax=None,
                            thresh=None, nmax=None, nmin=None):
        """Produces a chart of the given cluster

        Arguments
        ---------
        exemplar : pd.Series
            The cluster's exemplar phenotype's data.
        members : pd.DataFrame
            A dataframe containing the data for the rest of the cluster
            members.
        expressions : pd.DataFrame
            The exemplar's expression data (used for the inset histogram).

        Keyword Arguments
        -----------------
        title : str, default None
            A title for this cluster. If not given, defaults to the exemplar's
            name (probably as determined by `extract_series`
        xmax : float, default None
            A value to set as the horizontal bar's xmax. If not given, will
            default to `max(xs.max(), members.quantile(0.975).max()) * 1.05`

        For `thresh`, `nmax` and `nmin` cf. `PhenotypePlotter.top_channels`

        Returns
        -------
        A tuple, (figure, axes) of the figure object itself and the main axes
        containing the horizontal bars.
        """
        # Very similar to super().plot_single_barplot except the need to use
        # `members` in generating some interior values
        xs = self.top_channels(exemplar, thresh=thresh, nmax=nmax, nmin=nmin)
        labels = self.generate_labels(exemplar, members)
        fig, bars, dots = self._get_figure_and_axes(len(labels))

        bars.axvline(0, color='black', linestyle='solid')
        # Plot the bars as the values from the *exemplar*
        ys = range(len(labels))
        bars.barh(ys, xs, color='cornflowerblue')

        # Plot cluster
        bars.plot([members.quantile(0.025), members.quantile(0.975)],
                  [ys, ys],
                  color='black',
                  solid_capstyle='butt')

        if xmax is None:
            xmax = max(xs.max(), members.quantile(0.975).max()) * 1.05

        self._fill_dots(dots, xs, ys, xmax)
        self._set_limits(bars, dots, ys, labels, xmax)
        self._plot_inset_expressions_hist(bars, expressions)

        fig.suptitle(title, fontsize=16)
        return fig, bars

    def extract_series(self, phenotypes, n, n_phenotypes):
        """Gets a phenotype as pd.Series from the whole list of model
        phenotypes available.

        Arguments
        ---------
        phenotypes : list
            A sorted list of model phenotype dataframes.
        n : int
            The flat-list index of this phenotype to find
        n_phenotypes : int
            The number of phenotypes requested of the models during training.

        Returns
        -------
        The phenotype pd.Series, sorted by absolute value and renamed with
        model and component id's.
        """
        model_id, component_id = divmod(n, n_phenotypes)
        series = phenotypes[model_id].iloc[:, component_id]
        series = series.sort_values(key=np.abs)
        series.name = f'M{model_id:04}-P{component_id:04}'
        return series

    def extract_cluster(self, phenotypes, cluster_data, n,
                        thresh=None, nmax=None, nmin=None):
        """Pulls the exemplar as a Series and corresponding data from the other
        cluster members out of the total list of `phenotypes`.

        Arguments
        ---------
        phenotypes : list
            A sorted list of model phenotype dataframes.
        cluster_data : ClusterData
            A ClusterData object representing the cluster exemplar.
        n : int
            The number of phenotypes requested during model training.

        Keyword Arguments
        -----------------
        For `thresh`, `nmax` and `nmin` cf. `PhenotypePlotter.top_channels`

        Returns
        -------
        A tuple of (pd.Series, pd.DataFrame) containing some number of top
        channels from the exemplar and the corresponding values from all
        members of the cluster.
        """
        exemplar = self.extract_series(phenotypes, cluster_data.center, n)
        channels = self.top_channels(exemplar, thresh=thresh,
                                     nmax=nmax, nmin=nmin)
        members = [self.extract_series(phenotypes, ix, n)
                   for ix in cluster_data.indices]
        columns = [m.loc[channels.index] for m in members]
        return exemplar, pd.concat(columns, axis=1).T

    def plot_clusters(self, clustering, clusters,
                      phenotypes,
                      expressions,
                      *,
                      sort_key='score',
                      n_models=10,
                      n_phents=2000,
                      project_name='',
                      file_name='clusters.pdf',
                      stop_after_n=None,
                      thresh=None,
                      nmax=None,
                      nmin=None):
        """Plots a clustering of models

        Arguments
        ---------
        clustering, clusters :
            cf. function `structure_clustering` in this module.
        phenotypes : list
            A sorted list of phenotype dataframes.
        expressions : dict
            A dict mapping cluster centers (exemplars) to expression
            dataframes.

        Keyword Arguments
        -----------------
        sort_key : str|callable, default 'score'
            Either the string 'score' or 'lexical', or a callable. If a
            callable, must be a function of a ClusterData object to be used as
            a sort key. The 'score' sort uses a function of the cluster's size
            and mean, and the 'lexical' sort orders the clusters by model and
            component ids.
        n_models : int, default 10
            The number of models clustered.
        n_phents : int, default 2000
            The number of phenotypes requested from the clustered models.
        project_name : str, default ''
            An optional project name to prefix each pdf's title with.
        file_name : str, default 'clusters.pdf'
            A file path to save the generated pdf's at.
        stop_after_n : int, default None
            Defaults None. If an integer is given, only the first
            `stop_after_n` clusters are plotted.

        For `thresh`, `nmax` and `nmin` cf. `PhenotypePlotter.top_channels`
        """
        if sort_key == 'score':
            sort_key = cluster_score_sort_key
        elif sort_key == 'lexical':
            sort_key = cluster_lexical_sort_key
        elif callable(sort_key):
            pass
        else:
            raise ValueError('invalid value for keyword arg sort_key')

        if isinstance(phenotypes, dict):
            phenotypes = [v for k, v in sorted(phenotypes.items())]

        cluster_structures = structure_clustering(clusters, clustering)
        cluster_structures = sorted(cluster_structures, key=sort_key)

        if stop_after_n is not None:
            cluster_structures = cluster_structures[:stop_after_n]

        with PdfPages(file_name) as pdf:
            for datum in cluster_structures:
                exemplar, members = self.extract_cluster(phenotypes,
                                                         datum,
                                                         n_phents,
                                                         thresh=thresh,
                                                         nmax=nmax,
                                                         nmin=nmin)
                # Align polarities of members if necessary
                feature = exemplar.idxmax()
                feat_sign = np.sign(exemplar.max())
                mask = (np.sign(members[feature]) == feat_sign).astype(int)
                mask[mask == 0] = -1
                members = (members.T * mask).T

                # Include selection criteria with stats in title
                inner = f'm={len(members.index)}, s={datum.mean:.4f}'
                if thresh is not None:
                    inner = f'{inner}, thresh={thresh}'
                if nmin is not None:
                    inner = f'{inner}, nmin={nmin}'
                if nmax is not None:
                    inner = f'{inner}, nmax={nmax}'
                title = f'{project_name} Source {exemplar.name} ({inner})'

                self.plot_single_cluster(exemplar, members,
                                         expressions[datum.center], title,
                                         thresh=thresh, nmin=nmin, nmax=nmax)
                pdf.savefig()
                plt.close()

    def plot_cluster_statistics(self, clusters,
                                *,
                                n_models=10,
                                n_phents=2000,
                                project_name='',
                                file_name='cluster_stats.pdf'):
        """Generates a histogram of the overall clustering's intra-cluster mean
        values.

        Arguments
        ---------
        cf. `self.plot_clusters`
        """
        scorer = PerfectClusterScorer(n_models, n_phents)

        stats = np.array([intra_cluster_mean_std(x[0]) for x in clusters])
        perf = np.array([scorer.is_perfect(x[1][0]) for x in clusters])
        singles = np.array([len(x[1][0]) == 1 for x in clusters])

        n_perf = np.sum(perf)
        n_single = np.sum(singles)
        n_mixed = len(clusters) - (n_perf + n_single)

        m = stats[:, 0]
        data = np.array([m[perf], m[~perf]], dtype=object)

        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(16, 8))
        all_counts, *_ = ax.hist(data, bins=100,
                                 label=['Perfect', 'Imperfect'],
                                 color=[cmap(0), cmap(1)])
        ax.set_yscale('log')
        ax.set_xticks(np.arange(0.45, 1.05, 0.05))
        ax.set_title(f'{project_name} Distribution of Intra-Cluster Means '
                     f'(Cluster Size > 1)\n'
                     f'Perf={n_perf}, Singles={n_single}, Mixed={n_mixed}')
        ax.set_xlabel('Intra-Cluster Mean')
        ax.set_ylabel('N Clusters')

        for x in (0.5, 0.7, 0.8, 0.9, 0.95):
            ax.axvline(x, color='black', alpha=0.5, linestyle='dashed')
            ax.text(x, all_counts.max(),
                    f' {np.sum(m > x)}')
            ax.text(x, all_counts.max()-200,
                    f' {np.sum(m[perf] > x)}', color=cmap(0))
            ax.text(x, all_counts.max()-350,
                    f' {np.sum(m[~perf] > x)}', color=cmap(1))

        plt.savefig(file_name, dpi=300)
        plt.close()
