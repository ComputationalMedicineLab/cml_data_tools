"""Get some clustering data for analysis"""
import collections
import datetime
import itertools
import time

import numpy as np
# This must be our fork of pySAP
from pysapc import SAP
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity


def intra_cluster_mean_std(cluster):
    """Returns (mean, std) of a given cluster from an affinity matrix.
    Undefined (np.nan, np.nan) for clusters which contain at most a single
    member.

    Arguments
    ---------
    cluster : ndarray[N, N]
        A square submatrix taken from a given affinity matrix representing the
        affinities of a cluster from the total rows/cols of the affinity
        matrix.

    Returns
    -------
    (float, float) :
        The mean and stdev (ddof=0) of the cluster members with each other.
    """
    if len(cluster) <= 1:
        # Undefined for clusters which contain at most one member
        return np.nan, np.nan
    idx = np.triu_indices_from(cluster, k=1)
    vec = cluster[idx]
    return vec.mean(), vec.std(ddof=0)


def edit_dist_dpmat(x, y):
    """Use matrix dynamic programming to calculate the edit distance between ndarrays x and y"""
    D = np.zeros((len(x)+1, len(y)+1), dtype=int)
    range_y = np.arange(1, len(y)+1)
    range_x = np.arange(1, len(x)+1)
    D[0, 1:] = range_y
    D[1:, 0] = range_x
    for i, j in itertools.product(range_x, range_y):
        delt = 1 if x[i-1] != y[j-1] else 0
        D[i, j] = min(D[i-1, j-1] + delt,
                      D[i-1, j]+1,
                      D[i, j-1]+1)
    return D[len(x), len(y)]


def make_affinity_matrix(phenotypes):
    """
    Generates an affinity matrix from a list of phenotypes as the squared
    cosine similarity of all common channels.
    """
    first, *rest = phenotypes
    idx = first.index
    for df in rest:
        idx = idx.intersection(df.index)

    vals = [df.loc[idx].values for df in phenotypes]
    data = np.concatenate(vals, axis=1).T

    C = cosine_similarity(data)
    C = np.square(C)
    return C


def iter_clusters(matrix, labels):
    """Iterates tuples of (cluster submatrix, cluster indices)"""
    labelset = np.unique(labels)
    #assert (labelset == np.arange(len(labelset))).all(), labelset
    for n in labelset:
        mask = labels == n
        index = np.ix_(mask, mask)
        submat = matrix[index]
        yield submat, index


class PerfectClusterScorer:
    def __init__(self, n_model, n_phent):
        self.n_model = n_model
        self.n_phent = n_phent
        self.lo = np.arange(0, n_model*n_phent, n_phent, dtype=np.int)
        self.hi = self.lo + n_phent

    def is_perfect(self, cluster_indices):
        idx = np.sort(cluster_indices.ravel())
        if len(idx) != self.n_model:
            return False
        return np.all((self.lo <= idx) & (idx < self.hi))

    def __call__(self, matrix, labels, X=None, y=None):
        # Failure of clusterer to converge
        if np.all(labels == -1):
            return -1

        score = 0
        for (_, (_, idx)) in iter_clusters(matrix, labels):
            if self.is_perfect(idx):
                score += 1
        return score


class AffinityPropagationClusterer:
    def __init__(self, *, preference=None, convergence_iter=15,
                 max_iter=200, damping=0.5, copy=True, random_state=0):
        self.preference = preference
        self.convergence_iter = convergence_iter
        self.max_iter = max_iter
        self.damping = damping
        self.copy = copy
        self.random_state = random_state

    def fit(self, S, thresh=0.5):
        S = coo_matrix(S, copy=self.copy, dtype=np.float)

        # Allowing thresh to be None allows this whole conversion to happen
        # elsewhere
        if thresh is not None:
            S[S < thresh] = 0.0

        if self.random_state is not None:
            np.random.seed(self.random_state)

        sap = SAP(preference=self.preference,
                  convergence_iter=self.convergence_iter,
                  max_iter=self.max_iter,
                  damping=self.damping,
                  verboseIter=None,
                  parallel=True)

        tstart = time.time()
        centers, labels, n_iter = sap.fit_predict(S)
        wall_time = time.time() - tstart

        self.centers_ = centers
        self.labels_ = labels
        self.n_iter_ = n_iter
        self.wall_time_ = wall_time
