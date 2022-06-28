"""Tools for training ElasticNet models
"""
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_Xy_groups(expressions, labels):
    """Returns `X`, `y`, and `groups` for use in hyperparam_search given
    properly formatted arguments.

    Arguments
    ---------
    expressions : pd.DataFrame
        A pandas DataFrame of expression values indexed by (identifier string,
        actual date) tuples. The identifier string is of format '{PTID}_YYYY-MM-DD'.
    labels : dict
        A mapping of (ptid, target date) tuples to labels.
    """
    df = expressions.reset_index()
    df['label'] = [labels[tuple(x.split('_'))] for x in df['id']]
    X = df.drop(['id', 'date', 'label'], axis=1).to_numpy()
    y = df['label'].to_numpy().astype(int)
    groups = np.array([x.split('_')[0] for x in df['id']])
    return X, y, groups


def hyperparam_search(X, y, groups=None):
    model = LogisticRegression(penalty='elasticnet',
                               solver='saga',
                               max_iter=1000,
                               warm_start=True,
                               tol=0.001)
    cv = GroupKFold(n_splits=10)

    space = {'C': np.logspace(-3, -2, 20),
             'l1_ratio': np.arange(0, 1, 0.01)}

    search = RandomizedSearchCV(model,
                                space,
                                n_iter=2000,
                                scoring='roc_auc',
                                cv=cv,
                                n_jobs=60,
                                verbose=True)

    results = search.fit(X, y, groups=groups)

    return model, cv, space, search, results


def save_results(fname, model, cv, space, search, results):
    """Saves the results of a hyperparam search"""
    with open(fname, 'wb') as file:
        pickle.dump({
            'cv': cv,
            'model': model,
            'results': results,
            'search': search,
            'space': space,
        }, file, protocol=pickle.HIGHEST_PROTOCOL)


def make_results_df(results):
    df = pd.DataFrame(results['results'].cv_results_)
    df.sort_values('rank_test_score', inplace=True)
    df.reset_index(inplace=True)
    return df


def make_heatmap_df(df, values='mean_test_score',
                    columns='param_l1_ratio', index='param_C'):
    heatmap = pd.pivot_table(df, values=values, columns=columns, index=index)
    heatmap.index = np.round(heatmap.index.values, decimals=4)
    heatmap.columns = np.round(heatmap.columns.values, decimals=4)
    heatmap.index.rename('C', inplace=True)
    heatmap.columns.rename('l1_ratio', inplace=True)
    return heatmap


def plot_heatmap(heatmap, ax=None, cbar_label='ROC-AUC Score',
                 highlight=True, hi_only_best=False, hi_perc=99):
    """Uses seaborn to draw a heatmap specified by `heatmap` onto the axis
    specified by `ax`; optionally draws edges on select heatmap cells to
    highlight them.
    """
    if ax is None:
        ax = plt.gca()

    sns.heatmap(heatmap, ax=ax, annot=False, fmt='.3g', square=False,
                cbar_kws={'label': cbar_label})

    # Draws edges to highlight the max values in the heatmap and the values
    # above a certain percentile (given by hi_perc)
    if highlight:
        if not hi_only_best:
            V = heatmap.values.ravel()
            p = np.percentile(V, hi_perc)
            rs, cs, = np.unravel_index(np.argwhere(V > p), heatmap.shape)
            for c, r, v in zip(cs, rs, V > p):
                ax.add_patch(mpl.patches.Rectangle((c, r), 1, 1,
                                                   edgecolor='gray',
                                                   facecolor='none',
                                                   linewidth=1))
        r, c = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        ax.add_patch(mpl.patches.Rectangle((c, r), 1, 1,
                                           edgecolor='black',
                                           facecolor='none',
                                           linewidth=1))
    return ax


def plot_contours(heatmap, model, ax=None):
    if ax is None:
        ax = plt.gca()

    Z = heatmap.values
    Cs = heatmap.index.values
    Rs = heatmap.columns.values

    R = model.l1_ratio
    C = model.C
    k = R / C

    zs = np.sort(Z.ravel())
    levels = np.percentile(zs, [0, 10, 25, 50, 75, 85, 90, 95, 99, 100])

    contours = ax.contour(Rs, Cs, Z, levels, cmap=mpl.cm.PuBu, alpha=0.25)
    ax.clabel(contours, inline=True)

    ax.set_xlabel('R (L1 Ratio)')
    ax.set_ylabel('C')

    # Plot the model constant line
    ax.plot(Rs, Rs / k)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.001, 0.01])
    ax.invert_yaxis()

    return ax


def plot_score_by_nonzero_betas():
    pass
