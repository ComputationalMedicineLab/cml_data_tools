"""
Tools for plotting objects
"""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backends.backend_pdf import PdfPages


MODE_COLOR = {'Measurements': 'lightcoral',
              'Conditions': 'lightskyblue',
              'Medications': 'palegreen',
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

    def generate_labels(self, phenotype, trunc_at=46):
        """Generate labels for the given phenotype series"""
        labels = []
        for c, val in phenotype.iteritems():
            ex = self.standardizer.inverse_transform_label(c, val, spec='.4g')
            impact = f'{ex}'
            desc = self.name_descriptions.get(c, 'No Description')

            # Strip out units in square brackets
            desc = re.sub(r'\[.*\]', '', desc)

            # Strip out phrases "in blood", "in serum", "in plasma"
            # Should this be (in|of) ?
            desc = re.sub(r'(in|of|or|) (blood|serum|plasma)', '',
                          desc, flags=re.IGNORECASE)

            # Strip out "by automated count"
            desc = re.sub(r'by automated count', 'AC',
                          desc, flags=re.IGNORECASE)

            # Strip any doubled or extra whitespace generated by previous
            desc = re.sub(r'\s+', ' ', desc)

            # Trunc to trunc_at
            if len(desc) >= trunc_at:
                desc = f'{desc[:trunc_at-3]}...'

            label = f'{desc} ({impact})'
            labels.append(label)
        return labels

    def top_channels(self, phenotype, thresh=None, nmax=60, nmin=10):
        """
        Selects a least nmin and at most nmax channels from a given phenotype
        series, preferring channels with values above the given threshold if
        thresh is given.
        """
        ph = phenotype.sort_values(key=np.absolute)[-nmax:]
        if thresh is not None:
            x = ph[ph.abs() > thresh]
            if len(x) < nmin:
                return ph[-nmin:]
            return x
        return ph

    def plot_single_barplot(self, phenotype, expressions, title='',
                            xmax=None, xscale=None, thresh=None):
        ch = self.top_channels(phenotype, thresh=thresh)
        labels = self.generate_labels(ch)

        width = 11
        # 1 inch for title, 0.5 for margin on either side, and 1 per 4 channels
        height = 1.5 + (len(labels)*0.25)

        fig = plt.figure(figsize=(width, height))

        # add_axes(rect); rect=[left, bottom, width, height], as fraction of
        # whole (0..1) 1 inch as fraction of whole is 1/height
        margin = 0.5 / height
        ax_ht = 1 - (3*margin)

        bars = fig.add_axes(rect=(0.5, margin, 0.49, ax_ht))
        dots = fig.add_axes(rect=(0.475, margin, 0.025, ax_ht))

        ys = range(len(labels))
        ylim = [-1, len(labels)]

        bars.axvline(0, color='black', linestyle='solid')

        # Plot the bars as the values from the phenotype
        bars.barh(ys, ch)

        # Draw mode category dots and legend for relevant modes
        dot_colors = [MODE_COLOR.get(m, 'white') for
                      m in ch.index.get_level_values(0)]

        if xmax is None:
            xmax = max(map(abs, bars.get_xlim())) * 1.05

        dots.scatter(np.full_like(ys, -(xmax*0.8)), ys,
                     color=dot_colors,
                     marker='o',
                     s=100,
                     edgecolors='black',
                     linewidths=0.5)

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

        if xscale == 'symlog':
            bars.set_xscale('symlog', linthresh=0.1)

        # Plot the expressions inset
        # https://matplotlib.org/3.3.3/gallery/axes_grid1/inset_locator_demo.html
        expr_ax = inset_axes(bars, width=1.0, height=1.0, loc=4, borderpad=2)
        expr = (expressions[expressions > 0], -expressions[expressions < 0])
        expr_ax.hist(x=expr,
                     bins=100,
                     histtype='stepfilled',
                     log=True,
                     color=('blue', 'red'),
                     alpha=0.4,
                     label=('pos', 'neg'))
        expr_ax.legend(loc='upper right', fontsize='x-small', frameon=False)
        expr_ax.set_title('Expressions')
        expr_ax.set_ylim(bottom=0.8, top=None)
        expr_ax.set_alpha(0.6)
        #expr_ax.set_xticks([])

        fig.suptitle(f'{title or phenotype.name} (thresh={thresh})',
                     fontsize=16)
        return fig, bars

    def plot_single_model(self, model, fname, **kwargs):
        with PdfPages(fname) as pdf:
            for name in model.phenotypes_.columns:
                phen = model.phenotypes_[name]
                expr = model.expressions_[name]
                self.plot_single_barplot(phen, expr, title=name, **kwargs)
                pdf.savefig()
                plt.close()


def plot_phenotypes_to_file(phenotypes, expressions, filepath, channel_data,
                            standardizer):
    emax = np.max(expressions.values) * 1.05
    # ptmax = phenotypes.max().max() * 1.05
    ptmax = None
    # sorted_index = expressions[expressions.abs() >
    #                            0.3].count().sort_values(ascending=False).index
    sorted_index = phenotypes.columns
    name_dict = {(t.mode, t.channel): t.description
                 for t in channel_data.itertuples()}

    with PdfPages(filepath) as pdf:
        for pt in sorted_index:
            plot_phenotype(phenotypes[pt],
                           20,
                           expressions[pt],
                           phenotype_max=ptmax,
                           expression_max=emax,
                           name_descriptions=name_dict,
                           standardizer=standardizer)

            pdf.savefig()
            plt.close()


def plot_phenotype(pt,
                   n,
                   expressions,
                   plot_title=None,
                   phenotype_max=None,
                   expression_max=None,
                   name_descriptions=None,
                   standardizer=None):
    """Plot the top n values of phenotype Series pt.
    """
    colors = [
        'lightcoral', 'lightskyblue', 'palegreen', 'mediumvioletred',
        'slateblue', 'teal', 'indianred', 'cadetblue', 'mediumseagreen',
        'rosybrown', 'darkcyan', 'forestgreen'
    ]

    # modes are colored based on their order in the pt index, which in most
    # cases is the order that they are listed in the configs list of the
    # Experiment object.
    modes = pt.index.get_level_values(0).drop_duplicates()

    # only uses as many colors as needed.
    mode_color = dict(zip(modes, colors))

    plt.clf()
    f = plt.gcf()
    f.set_size_inches(11, 6)

    df = pd.DataFrame({
        'pt': pt,
        'ab': pt.abs()
    }).sort_values('ab', ascending=False)
    v = df.pt[:n]

    y = range(n - 1, -1, -1)
    plt.barh(y, v)
    if standardizer is None:
        label_func = lambda *unused: ''
    else:
        label_func = standardizer.inverse_transform_label

    if name_descriptions is None:
        names = v.index.get_level_values(1)
        modes = [None] * len(names)
    else:
        names = [
            '[{impact}] {desc}'.format(
                impact=f'{label_func(index_elt, v[index_elt])}',
                desc=name_descriptions.get(index_elt, 'No Description'))
            for index_elt in v.index
        ]
        modes = [index_elt[0] for index_elt in v.index]
    y_colors = [mode_color.get(m, 'white') for m in modes]
    if phenotype_max is None:
        xmin, xmax = plt.xlim()
        phenotype_max = max(abs(xmin), xmax)

    labels = [_truncate_string(name, 75) for name in names]
    plt.yticks(y, labels)
    plt.scatter([-0.95 * phenotype_max] * len(y),
                y,
                c=y_colors,
                marker='o',
                s=100,
                edgecolors='black',
                linewidths=0.5)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are on
        labelbottom=False,  # labels along the bottom edge are on
        labeltop=True,  # labels along the top edge are on
        direction='in')

    plt.xlim((-phenotype_max, phenotype_max))
    plt.axvline(x=0)

    if not plot_title:
        plot_title = 'Phenotype ' + pt.name

    plt.title(plot_title, pad=10)
    _plot_inset(f, expressions, expression_max)

    plt.tight_layout()
    f.subplots_adjust(left=0.5, top=0.9)


def _plot_inset(fig, expressions, expression_max):
    left, bottom, width, height = [0.85, 0.11, 0.12, 0.20]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.set_alpha(0.4)
    ax2.hist(x=(expressions[expressions > 0], -expressions[expressions < 0]),
             bins=100,
             histtype='stepfilled',
             log=True,
             color=('blue', 'red'),
             alpha=0.4,
             label=('pos', 'neg'))
    ax2.legend(loc='upper right', fontsize='x-small', frameon=False)

    plt.xlabel('Expression')
    plt.ylabel('Count')
    ax2.patch.set_alpha(0.5)
    ax2.set_ylim(bottom=0.8, top=None)


def _truncate_string(s, n):
    if len(s) > n:
        return s[:n - 3] + '..'
    else:
        return s
