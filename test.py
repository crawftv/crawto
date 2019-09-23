import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_array, check_random_state
from ..utils.extmath import randomized_svd, safe_sparse_dot, svd_flip
from ..utils.sparsefuncs import mean_variance_axis

__all__ = ["TruncatedSVD"]


class TruncatedSVD(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, algorithm="randomized", n_iter=5,
                 random_state=None, tol=0.):
        self.algorithm = algorithm
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        X = check_array(X, accept_sparse=['csr', 'csc'],
                        ensure_min_features=2)
        random_state = check_random_state(self.random_state)

        if self.algorithm == "arpack":
            U, Sigma, VT = svds(X, k=self.n_components, tol=self.tol)
            # svds doesn't abide by scipy.linalg.svd/randomized_svd
            # conventions, so reverse its outputs.
            Sigma = Sigma[::-1]
            U, VT = svd_flip(U[:, ::-1], VT[::-1])

        elif self.algorithm == "randomized":
            k = self.n_components
            n_features = X.shape[1]
            if k >= n_features:
                raise ValueError("n_components must be < n_features;"
                                 " got %d >= %d" % (k, n_features))
            U, Sigma, VT = randomized_svd(X, self.n_components,
                                          n_iter=self.n_iter,
                                          random_state=random_state)
        else:
            raise ValueError("unknown algorithm %r" % self.algorithm)

        self.components_ = VT

        # Calculate explained variance & explained variance ratio
        X_transformed = U * Sigma
        self.explained_variance_ = exp_var = np.var(X_transformed, axis=0)
        if sp.issparse(X):
            _, full_var = mean_variance_axis(X, axis=0)
            full_var = full_var.sum()
        else:
            full_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var
        self.singular_values_ = Sigma  # Store the singular values.

        return X_transformed

    def transform(self, X):
        X = check_array(X, accept_sparse='csr')
        return safe_sparse_dot(X, self.components_.T)

    def inverse_transform(self, X):

        X = check_array(X)
        return np.dot(X, self.components_)
