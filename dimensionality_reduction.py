import copy

import datetime
import os
import pickle
import random

import numpy as np
from sklearn import random_projection
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.metrics import explained_variance_score
from sklearn.random_projection import johnson_lindenstrauss_min_dim

from datamodel import *


# todo: for some reason, the n_components can be set for NO_METHOD, and the transformer can be set for NO_METHOD

def reduce(method: ReductionMethod, data: ExperimentDataset,
           goal_variance: float = 0.9) -> ExperimentDataset:
    data_new = copy.deepcopy(data)

    def load_if_exists(data, method, goal_variance, dm_fn):
        # load paths from data name folder
        if os.path.exists(f'artifacts/{data.name.name}/dmr_{method.name}.pickle'):
            # load the data
            print(f'Loading {method.name} from artifacts/{data.name.name}/dmr_{method.name}.pickle')
            with open(f'artifacts/{data.name.name}/dmr_{method.name}.pickle', 'rb') as f:
                return pickle.load(f)
        else:
            return dm_fn(data, goal_variance)

    if method == ReductionMethod.PCA:
        """
        Fortunately, the sklearn implementation of PCA has a built-in method for determining the optimal number of 
        components.
        """
        # if pca has already been run, then we can just return the data
        data_new = load_if_exists(data, method, goal_variance, find_optimal_pca)
    elif method == ReductionMethod.ICA:
        data_new = load_if_exists(data, method, goal_variance, find_optimal_fastica)
    elif method == ReductionMethod.RANDOM_PROJECTIONS:
        """
        Random projections are a way to reduce the dimensionality of a dataset by projecting it to a lower-dimensional space.
        The lower-dimensional representation is often much more efficient to work with, and can be used to train a model.
        Unfortunately in the case of the APS Failure dataset, the random projections method does not work well.
        
        [source](https://stackabuse.com/random-projection-theory-and-implementation-in-python-with-scikit-learn/)
        
        The Johnson-Lindenstrauss lemma states that if a dataset is projected to a lower-dimensional space using a random
        projection matrix, then the pairwise distances between the points in the lower-dimensional space are approximately
        preserved. This means that the distances between the points in the original dataset are approximately preserved.
        That distance is called the EPS (epsilon) distance. This EPS can be used to determine the number of components
        through a formula. The formula is: n_components >= 4 * log(n_samples) / (EPS ** 2)
        
        Increasing EPS will decrease the number of components. Because the APS Failure dataset with default EPS generates
        many more dimensions, the eps must clearly be increased. This EPS is the same as the min_dist parameter in UMAP.
        In the case of UMAP, it can be thought of as a parameter that controls how tightly the points are allowed to be
        clustered together. Therefore, it is important to choose a value for EPS that is not too small or too large.
        
        We can determine the desirable EPS value by first determining the number of components that are required to
        preserve some suitable percentage of the variance in the dataset. This is a very similar process to PCA, where
        we can determine the number of components that are required to preserve some percentage of the variance in the
        dataset. The greater the EPS, the fewer components are required to preserve the variance, and the more the points
        will be clustered together, at the expense of losing some information. Ultimately, dimensionality reduction is
        a trade-off between information loss and computational efficiency, much like compression.
        
        How do we measure the variance in the dataset? We can use the pairwise distances between the points in the
        dataset. Sklearn makes that easy for us with the pairwise_distances function. We can then use the variance
        """

        # measure of variance in the dataset to preserve 90% of the variance
        data_new = load_if_exists(data, method, goal_variance, find_optimal_random_projections)
    elif method == ReductionMethod.TRUNCATED_SVD:
        data_new = load_if_exists(data, method, goal_variance, find_optimal_truncated_svd)
    elif method == ReductionMethod.NO_METHOD:
        return data_new
    else:
        raise Exception(f'Unknown methodname {method}')
    return data_new


def find_optimal_random_projections(data: ExperimentDataset, goal_variance: float) -> ExperimentDataset:
    explained_variance_history = []
    # get num dimensions in data
    n_samples, n_features = data.X_train.shape

    # get list of 10 random numbers between 2 and n_features
    n_components_list = random.sample(range(2, n_features), 10)
    X_new = data.X_train

    dmr_params = None

    for n_components in n_components_list:
        """
        In this case, there is a minimum number of samples required to preserve the variance. If the number of samples
        is less than the minimum number of samples, then the variance cannot be preserved.
        
        n_components >= 4 log(n_samples) / (eps^2 / 2 - eps^3 / 3)
        
        while we can solve for eps given n_components analytically, it's easier to just iterate over a range of eps.
        
        The APS Failure dataset is not large enough to require this minimum number of samples, so we can ignore it.
        
        Because Random Projections is never able to preserve the variance, we can just use the number of components
        that is estimated to decrease the number of dimensions by 30% as a reasonable number of dimensions to use.
        
        johnson_lindenstrauss_min_dim(n_samples=n_samples, eps-eps) can be used to determine the minimum number of
        components required to preserve the variance. But, as stated above, the none of the datasets are large enough
        to require this minimum number of samples, so we can ignore it.
        
        Out of curiosity, I tried random projections on many more datasets. Some datasets included millions of samples,
        and some datasets included millions of features. In all cases, the Random Projections method was unable to
        preserve the variance.
        
        In random projections, you're only preserving the pairwise distances between the points in the dataset. This is
        not the same as preserving the global structure of the dataset. The reconstruction variance is a measure of the 
        global structure of the dataset. Therefore, it is not possible to preserve the variance in the dataset upon
         reconstruction with the method impelmented in sklearn. Some research has been done on this topic, and it is
        possible to preserve the variance in the dataset upon reconstruction with random projections.
        That is covered in the following paper: http://people.ece.umn.edu/~jdhaupt/publications/it06_noisy_random_proj.pdf
             
        """
        start_time = datetime.datetime.now()
        reducer = random_projection.GaussianRandomProjection(n_components=n_components).fit(data.X_train)
        train_wall_time = (datetime.datetime.now() - start_time).total_seconds()

        start_time = datetime.datetime.now()
        X_new = reducer.transform(data.X_train)
        inference_wall_time = (datetime.datetime.now() - start_time).total_seconds()

        explained_variance = explained_variance_score(data.X_train, reducer.inverse_transform(X_new))

        print(f'Random Projections: n_components={n_components}, explained_variance={explained_variance}')

        explained_variance_history.append(ExplainedVarianceResult(
            explained_variance=explained_variance,
            n_components=n_components,
        ))
        if explained_variance >= goal_variance:
            data.X_train = X_new
            data.X_test = reducer.transform(data.X_test)
            data.reducer.method = ReductionMethod.RANDOM_PROJECTIONS
            data.reducer.explained_variance_history = explained_variance_history
            data.reducer.training_wall_time = train_wall_time
            data.reducer.inference_wall_time = inference_wall_time
            data.reducer.reducer = reducer
            data.reducer.n_components = n_components
            return data

    data.X_train = X_new
    data.X_test = reducer.transform(data.X_test)
    data.reducer.explained_variance_history = explained_variance_history
    data.reducer.training_wall_time = train_wall_time
    data.reducer.inference_wall_time = inference_wall_time
    data.reducer.reducer = reducer
    data.reducer.method = ReductionMethod.RANDOM_PROJECTIONS
    data.reducer.n_components = n_components
    return data


def find_optimal_truncated_svd(data: ExperimentDataset, goal_variance: float = 0.9) -> ExperimentDataset:
    # if the dataset has already been reduced, then return throw an exception
    data.reducer.method = ReductionMethod.TRUNCATED_SVD

    explained_variance_history = []
    for n_components in range(2, data.X_train.shape[1]):
        start_time = datetime.datetime.now()
        reducer = TruncatedSVD(n_components=n_components).fit(data.X_train)
        train_wall_time = (datetime.datetime.now() - start_time).total_seconds()

        start_time = datetime.datetime.now()
        X_new = reducer.transform(data.X_train)
        inference_wall_time = (datetime.datetime.now() - start_time).total_seconds()

        explained_variance = explained_variance_score(data.X_train, reducer.inverse_transform(X_new))
        print(f'Factor Analysis: n_components={n_components}, explained_variance={explained_variance}')

        explained_variance_history.append(ExplainedVarianceResult(
            explained_variance=explained_variance,
            n_components=n_components,
        ))

        # if the explained variance is greater than the goal variance, then we have found the optimal eps value
        if explained_variance >= goal_variance:
            data.X_train = X_new
            data.X_test = reducer.transform(data.X_test)
            data.reducer.method = ReductionMethod.TRUNCATED_SVD
            data.reducer.explained_variance_history = explained_variance_history
            data.reducer.training_wall_time = train_wall_time
            data.reducer.inference_wall_time = inference_wall_time
            data.reducer.reducer = reducer
            data.reducer.n_components = n_components
            return data


def find_optimal_fastica(data: ExperimentDataset, goal_variance: float = 0.9) -> ExperimentDataset:
    explained_variance_history = []
    for n_components in range(2, data.X_train.shape[1]):
        start_time = datetime.datetime.now()
        reducer = FastICA(n_components=n_components).fit(data.X_train)
        train_wall_time = (datetime.datetime.now() - start_time).total_seconds()

        start_time = datetime.datetime.now()
        X_new = reducer.transform(data.X_train)
        inference_wall_time = (datetime.datetime.now() - start_time).total_seconds()

        explained_variance = explained_variance_score(data.X_train, reducer.inverse_transform(X_new))
        print(f'Factor Analysis: n_components={n_components}, explained_variance={explained_variance}')

        explained_variance_history.append(ExplainedVarianceResult(
            explained_variance=explained_variance,
            n_components=n_components,
        ))

        # if the explained variance is greater than the goal variance, then we have found the optimal eps value
        if explained_variance >= goal_variance:
            data.X_train = X_new
            data.X_test = reducer.transform(data.X_test)
            data.reducer.method = ReductionMethod.ICA
            data.reducer.explained_variance_history = explained_variance_history
            data.reducer.training_wall_time = train_wall_time
            data.reducer.inference_wall_time = inference_wall_time
            data.reducer.reducer = reducer
            data.reducer.n_components = n_components
            return data


def find_optimal_pca(data: ExperimentDataset, goal_variance: float = 0.9) -> ExperimentDataset:
    explained_variance_history = []
    for n_components in range(2, data.X_train.shape[1]):
        start_time = datetime.datetime.now()
        reducer = PCA(n_components=n_components).fit(data.X_train)
        train_wall_time = (datetime.datetime.now() - start_time).total_seconds()

        start_time = datetime.datetime.now()
        X_new = reducer.transform(data.X_train)
        inference_wall_time = (datetime.datetime.now() - start_time).total_seconds()

        explained_variance = explained_variance_score(data.X_train, reducer.inverse_transform(X_new))
        print(f'Factor Analysis: n_components={n_components}, explained_variance={explained_variance}')

        explained_variance_history.append(ExplainedVarianceResult(
            explained_variance=explained_variance,
            n_components=n_components,
        ))

        # if the explained variance is greater than the goal variance, then we have found the optimal eps value
        if explained_variance >= goal_variance:
            data.X_train = X_new
            data.X_test = reducer.transform(data.X_test)
            data.reducer.method = ReductionMethod.PCA
            data.reducer.explained_variance_history = explained_variance_history
            data.reducer.training_wall_time = train_wall_time
            data.reducer.inference_wall_time = inference_wall_time
            data.reducer.reducer = reducer
            data.reducer.n_components = n_components
            return data
