import itertools
import logging
import operator
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FastICA
from sklearn.exceptions import NotFittedError


def clusters_to_components(clusters, n=2000):
    """Converts the data format produced by `clustering.iter_clusters` into a
    format usable for extracting the cluster centers from the IcaPhenotypeModel
    models contained in a given clustering.

    Arguments
    ---------
    clusters : Iterable
        An iterable of 3-tuples such as those yielded by `iter_clusters` in
        `clustering.py`. Only the exemplar from each 3-tuple is used.
    n : int, default=2000
        The number of phenotypes requested per IcaPhenotypeModel in the
        clustering. This is used to decompose each exemplar extracted from
        `clusters` into a submodel number and component number. I.e., if n is
        2000 and the exemplar is 5500, then the exemplar is the 500th phenotype
        in the third model.

    Returns
    -------
    A list of arrays of component indices. Each array in the list contains all
    exemplars present in its submodel from the clustering; i.e. if the first
    element is `np.array([0, 4, 99])` then `model_0` contains cluster centers
    (exemplars) at those indices (`model.phenotypes_.iloc[:, indices]`).
    """
    # clusters is an iterable of 3-tuples as from `iter_clusters` in
    # cml_data_tools/clustering.py
    exemplars = np.sort(np.array([x[-1] for x in clusters]))
    components = []
    for _, grp in itertools.groupby(zip(*divmod(exemplars, n)),
                                    key=operator.itemgetter(0)):
        components.append(np.array([x[-1] for x in grp]))
    return components


class AggregateIcaModel:
    """Transforms a list of models and a list of clusters (cf. `iter_clusters`
    in clustering.py) into a model capable of transforming using the exemplars
    of each component cluster across the various submodels.

    Parameters
    ----------
    models
        A list of IcaPhenotypeModel instances in order
    clusters
        A list of 3-tuples, the third of which is the exemplar ordinal
        I.e. if the cluster exemplar is the 300th phenotype of the 4th model
        then the ordinal of that exemplar is (3 * n) + 299, where `n` is the
        number of phenotypes that was requested across the submodels
    n : Int
        The number of phenotypes requested per submodel
    """
    def __init__(self, models, clusters, n=2000):
        self.models = [self.ModelCore(m) for m in models]
        self.components = clusters_to_components(clusters, n=n)
        # Sanity check
        assert len(self.models) == len(self.components)

        # Get the Phenotypes of the exemplars and stitch them together
        # We do this on init to avoid keeping a reference to models anywhere
        parts = []
        for i, (model, ids) in enumerate(zip(models, self.components)):
            cols = np.array(model.phenotype_names)[ids]
            ph = model.phenotypes_[cols]
            ph.columns = self.ids_to_names(i, ids)
            parts.append(ph)
        self.phenotypes = pd.concat(parts, axis=1)

    def ids_to_names(self, m, ids):
        """
        Produces an array of appropriate identifiers for the phenotypes
        identified by model_id `m` and column indices `ids`

        Arguments
        ---------
        m : str
            The model number (zero-indexed).
        ids : np.array[int]
            A list of indices

        Returns
        -------
        An ndarray containing strings of the format `fM{m:03}-P{c:04}` for `c`
        in `ids`.
        """
        return np.array([f'M{m:03}-P{c:04}' for c in ids])

    def transform(self, X):
        """Project the data in X onto the aggregated phenotype cluster
        exemplars.

        Arguments
        ---------
        X : pandas.DataFrame
            Has one row per instance, one column per variable

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe with rows corresponding to (and indexed the same
            as) rows in X, and columns corresponding to the aggregated
            phenotypes. Since the exemplar names may not be unique across the
            submodels, the columns are *renamed* from the original. The
            renaming function is given in `AggregateIcaModel.ids_to_names`.
            Each cell contains the amount of the given phenotype expressed by
            the row of X.
        """
        parts = []
        for i, (model, ids) in enumerate(zip(self.models, self.components)):
            columns = np.array(model.phenotype_names)[ids]
            x_hat = model.transform(X)[columns]
            x_hat.columns = self.ids_to_names(i, ids)
            parts.append(x_hat)
        return pd.concat(parts, axis=1)

    class ModelCore:
        """Extracts core functionality from IcaPhenotypeModel"""
        def __init__(self, model):
            self.ica = model.ica
            self.phenotype_names = model.phenotype_names
            self.scale_factors = model.scale_factors_

        def transform(self, X):
            # Tracks the functionality of IcaPhenotypeModel.transform
            expressions = pd.DataFrame(self.ica.transform(X.values),
                                       index=X.index,
                                       columns=self.phenotype_names)
            return expressions * self.scale_factors


class IcaPhenotypeModel(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    name_stem : str
        Prepended to the phenotype number to create the phenotype names. For
        example, setting name_stem='HF' will produce a phenotype matrix with an
        index like ('HF-000', 'HF-001', ...). This helps to keep track of
        models downstream. (Default: 'ICA')
    max_phenotypes : int
        The number of phenotypes to attempt to infer. If there is not enough
        information in the training data, fewer than this number may be
        inferred. (Default: 500)
    max_iter : int
        The maximum number of ICA iterations to attempt (Default: 1000)
    """
    def __init__(self, name_stem='ICA', max_phenotypes=500, max_iter=1000):
        self.name_stem = name_stem
        self.max_phenotypes = max_phenotypes
        self.max_iter = max_iter
        self.logger = logging.getLogger(self.__class__.__qualname__)

    def fit(self, X, y=None):
        """Learns phenotypes from data. Returns self.

        Arguments
        ---------
        X : pandas.DataFrame
            Has one row per instance, one column per variable
        """
        # In this formulation of ICA, X' = AS', where (untransposed) X is the
        # passed parameter `X`.

        # Columns of A are the patterns of channel values for the source
        # signals (phenotypes).

        # Columns of S are the original X expressed in terms of the source
        # phenotype strengths.

        # So a column in A and a column in S are for a single phenotype.
        self.phenotype_names = [
            self.get_nth_name(n) for n in range(self.max_phenotypes)
        ]
        self.channel_names = X.columns

        self.logger.info('Fitting ICA to a {} by {} matrix'.format(*X.shape))
        self._ica = FastICA(n_components=self.max_phenotypes,
                            algorithm='parallel',
                            max_iter=self.max_iter)
        self.ica.fit(X.values)

        self.logger.info('Computing S Matrix.')
        raw_expressions = pd.DataFrame(self.ica.transform(X.values),
                                       index=X.index,
                                       columns=self.phenotype_names)

        self.logger.info('Computing A Matrix.')
        raw_phenotypes = pd.DataFrame(self.ica.mixing_,
                                      index=self.channel_names,
                                      columns=self.phenotype_names)

        self.logger.info('Computing scale factors')
        self.scale_factors_ = self.compute_scale_factors(raw_phenotypes,
                                                         raw_expressions)

        self.phenotypes_ = raw_phenotypes / self.scale_factors_
        self.expressions_ = raw_expressions * self.scale_factors_

        return self

    def fit_transform(self, X, y=None):
        """Learns phenotypes from data and returns the computed expressions.

        Arguments
        ---------
        X : pandas.DataFrame
            Has one row per instance, one column per variable

        Returns
        -------
        pandas.DataFrame
            The expressions of the learned phenotypes present in `X`.
        """
        self.fit(X)
        return self.expressions_

    def transform(self, X):
        """Project the data in X onto the previously-learned phenotypes.

        Arguments
        ---------
        X : pandas.DataFrame
            Has one row per instance, one column per variable

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe with rows corresponding to (and indexed the same
            as) rows in X, and columns corresponding to learned phenotypes.
            Each cell contains the amount of the given phenotype expressed by
            the row of X.
        """
        raw_expressions = pd.DataFrame(self.ica.transform(X.values),
                                       index=X.index,
                                       columns=self.phenotype_names)
        return raw_expressions * self.scale_factors_

    @property
    def ica(self):
        # Basic guard to try to notify user if they use a method relying on a
        # fit FastICA instance when this model has not been fitted
        try:
            return self._ica
        except AttributeError as err:
            raise NotFittedError() from err

    @property
    def means_(self):
        return pd.Series(self.ica.mean_, index=self.channel_names)

    @property
    def raw_components(self):
        """Return the W matrix as DataFrame"""
        return pd.DataFrame(self.ica.components_,
                            index=self.phenotype_names,
                            columns=self.channel_names)

    @property
    def raw_phenotypes(self):
        """Return the A matrix as DataFrame"""
        return pd.DataFrame(self.ica.mixing_,
                            index=self.channel_names,
                            columns=self.phenotype_names)

    @property
    def raw_expressions(self):
        """
        Unscales the `expressions_` attribute using the `scale_factors_`i to
        return the S matrix as DataFrame.
        """
        return self.expressions_ / self.scale_factors_

    def get_nth_name(self, n):
        """Constructs the name of the n-th phenotype"""
        return '{}-{:03d}'.format(self.name_stem, n)

    @staticmethod
    def compute_scale_factors(phenotypes, expressions):
        """Computes the scale factors for each learned phenotype.

        ICA results have an arbitrary sign and scale factor. Here, we assign a
        scale factor such that the expressions have a 1.0 standard deviation,
        and assign the sign such that the largest component is always positive.

        This function does not apply the scale factors, only computes them.
        """
        stdev = expressions.std(axis=0).fillna(1)
        factors = 1 / (2 * stdev)
        # Flip polarity such that max(abs) is always positive
        max_locs = phenotypes.abs().idxmax()
        vals = phenotypes.lookup(max_locs, phenotypes.columns)
        factors[vals < 0] *= -1
        # In case we get a phenotype with all zeros, we arbitrarily set the
        # scale factor to 1.0 to avoid a divide by zero error
        factors.replace(to_replace=0, value=1, inplace=True)
        return factors
