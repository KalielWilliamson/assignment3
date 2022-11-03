import pickle

import click as click
import numpy as np
import os
import pandas as pd
import warnings
from itertools import product
from kneed import KneeLocator
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from typing import List

from datamodel import ExperimentDataset, ClusterMethod, DatasetName, OptimalParameters
from dimensionality_reduction import ReductionMethod


### INSTRUCTIONS ###
# 1. Within a notebook, run the `aps_system_failure_run` function
# 2. Visualize the results of the clustering algorithms and choose the best k and n_components
# 3. Run the `aps_system_failure_optimal` function with the optimal k and n_components
# 2. The code will run on the APS System Failure dataset
# 3. The code will run on the HepMass dataset
# 4. The code will run on the APS System Failure dataset with dimensionality reduction
# 5. The code will run on the HepMass dataset with dimensionality reduction
# 6. The code will run on the APS System Failure dataset with dimensionality reduction and clustering
# 7. The code will run on the HepMass dataset with dimensionality reduction and clustering
from neural_network import train_neural_networks


def get_data(name: DatasetName, test_size: float = 0.2, limit: int = None) -> ExperimentDataset:
    """
    It loads the data from the csv file, and returns an object of type `ExperimentDataset` which contains the data in a
    format that is easy to use for training and testing

    :param name: The name of the dataset
    :type name: DatasetName
    :param test_size: the proportion of the dataset to use for testing
    :type test_size: float
    :param limit: This is the number of rows to use from the dataset. This is useful for testing the code on a smaller
    dataset
    :type limit: int
    :return: an ExperimentDataset object.
    """
    if name == DatasetName.HEPMASS:
        # get the hepmass data
        df = pd.read_csv("artifacts/hepmass.csv")

        X = df.drop(columns=['# label'], axis=1)
        y = df['# label'].values

        return get_experiment_dataset(name, test_size, X, y, limit)
    elif name == DatasetName.APS_SYSTEM_FAILURE:
        X, y = fetch_openml(data_id='41138', cache=True, return_X_y=True)  # APSFailure dataset
        y = np.where(y == 'neg', 0, 1)
        return get_experiment_dataset(DatasetName.APS_SYSTEM_FAILURE, test_size, X, y, limit)
    elif name == DatasetName.HIGGS:
        X, y = fetch_openml(data_id='23512', cache=True, return_X_y=True)  # HIGGS dataset

        # convert y to numeric
        y = pd.to_numeric(y)

        return get_experiment_dataset(DatasetName.HIGGS, test_size, X, y, limit)
    elif name == DatasetName.MNIST:
        X, y = fetch_openml(data_id='40996', cache=True, return_X_y=True)
        # convert y to numeric
        y = pd.to_numeric(y)
        return get_experiment_dataset(DatasetName.MNIST, test_size, X, y, limit)
    elif name == DatasetName.BNG_LABOR:
        X, y = fetch_openml(data_id='246', cache=True, return_X_y=True)

        # convert each feature to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col])

        # convert y to numeric
        y = pd.to_numeric(y)
        return get_experiment_dataset(DatasetName.BNG_LABOR, test_size, X, y, limit)
    elif name == DatasetName.BNG_CREDIT_G:
        X, y = fetch_openml(data_id='40514', cache=True, return_X_y=True)

        y = np.where(y == 'good', 0, 1)

        le = preprocessing.LabelEncoder()
        for cat_col in X.select_dtypes(['category']).columns:
            X[[cat_col]] = le.fit_transform(X[[cat_col]])

        # convert y to numeric
        y = pd.to_numeric(y)
        return get_experiment_dataset(DatasetName.BNG_CREDIT_G, test_size, X, y, limit)
    elif name == DatasetName.KDD98:
        X, y = fetch_openml(data_id='41435', cache=True, return_X_y=True)

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        le = preprocessing.LabelEncoder()
        for cat_col in X.select_dtypes(['category']).columns:
            X[[cat_col]] = le.fit_transform(X[[cat_col]])

        # convert y to numeric
        y = pd.to_numeric(y)
        return get_experiment_dataset(DatasetName.KDD98, test_size, X, y, limit)


def get_experiment_dataset(name, test_size, X, y, limit) -> ExperimentDataset:
    """
    It takes in a dataset name, a test size, a feature matrix, a target vector, and a limit, and returns an
    ExperimentDataset object

    :param name: the name of the dataset
    :param test_size: the proportion of the dataset to use for testing
    :param X: the dataframe of features
    :param y: the target variable
    :param limit: the number of rows to use for training. This is useful for testing the code on a subset of the data
    :return: An ExperimentDataset object
    """
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # min-max scale the data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # fill in missing values with the mean
    X_train = pd.DataFrame(X_train)
    X_train.fillna(X_train.mean(), inplace=True)
    X_test = pd.DataFrame(X_test)
    X_test.fillna(X_test.mean(), inplace=True)
    # apply a limit to the data
    if limit is not None:
        X_train = X_train.iloc[:limit]
        y_train = y_train[:limit]
    dataset = ExperimentDataset(
        name=name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    with open(f"artifacts/{name.name}/raw_{name.name}.pkl", "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def run_experiment(data: ExperimentDataset, goal_variance: float = 0.9):
    """
    > This function runs each dimensinoality reduction method on the dataset, finds the optimal n_components for each
    method, then runs each clustering method on each optimally reduced dataset, and returns the results

    :param data: ExperimentDataset
    :type data: ExperimentDataset
    :param goal_variance: The goal variance to reduce the dataset to
    :type goal_variance: float
    """
    # run dimensionality reduction without clustering
    data_addresses = get_dimensionality_reduction(data, goal_variance)
    cluster_n_components_list = np.arange(2, 50, 1)
    address_components = list(set(product(cluster_n_components_list, data_addresses)))

    # verify that each dimensionality reduction method has a corresponding dataset

    # run clustering on each dimensionally reduced dataset and n_components
    # zip the data addresses and n_components together
    # tqdim loads a progress bar
    print('Running clustering on dimensionally reduced data')
    for n_components, address in tqdm(address_components):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # cluster with dimensionality reduction
            reduced_dataset: ExperimentDataset = ExperimentDataset.load(address)

            # cluster with dimensionality reduction
            reduced_dataset.cluster.n_components = n_components

            dataset = reduced_dataset.copy()
            dataset.set_cluster(method=ClusterMethod.KMEANS)

            dataset = reduced_dataset.copy()
            dataset.set_cluster(method=ClusterMethod.GMM)

    # verify that each clustering method has a corresponding dataset
    data_index: DataFrame = ExperimentDataset.get_index(data.name)
    for reduction_method in ReductionMethod:
        for cluster_method in ClusterMethod:
            assert len(data_index[(data_index['reduction_method'] == reduction_method.name) & (data_index['cluster_method'] == cluster_method.name)]) > 0


def run(name: DatasetName, test_size: float = 0.2, goal_variance: float = 0.9, limit: int = None):
    """
    It loads the data, runs the experiment, and prints the results

    :param name: The name of the dataset to run the experiment on
    :type name: DatasetName
    :param test_size: the proportion of the dataset to use for testing
    :type test_size: float
    :param goal_variance: The variance of the target variable that we want to achieve
    :type goal_variance: float
    :param limit: The number of rows to use from the dataset. This is useful for testing the code on a small subset of the
    data
    :type limit: int
    """
    print(('=' * 20) + str(name.name) + ('=' * 20))
    data: ExperimentDataset = get_data(name, test_size, limit)
    data.save()

    return run_experiment(data, goal_variance)


def find_optimal_n_components(data_index: DataFrame, cluster_method: ClusterMethod, reduction_method: ReductionMethod) -> OptimalParameters:
    """
    [source](https://stackoverflow.com/questions/51762514/find-the-elbow-point-on-an-optimization-curve-with-python)
    We can automatically find the optimal k by automatically finding the elbow point on the optimization curve.
    The elbow point is the point where the optimization curve changes from linear to exponential.
    THe library `kneed` does this for us.
    """

    # for each kmeans artifact, load
    # get files in artifacts directory that has 'aps_system_failure_knn' and in the name
    x, y = [], []
    dataset_paths = data_index[(data_index['reduction_method'] == reduction_method.name) & (data_index['cluster_method'] == cluster_method.name)].path.values

    for path in dataset_paths:
        data = ExperimentDataset.load(path)
        x.append(data.cluster.n_components)
        y.append(data.cluster.loss)

    # find the elbow point with kneed
    optimal_cluster_components = KneeLocator(x, y, direction='decreasing').knee

    # default to the last value if the elbow point is not found
    if optimal_cluster_components is None:
        optimal_cluster_components = x[-1]

    # index of the optimal n_components
    optimal_path = dataset_paths[x.index(optimal_cluster_components)]
    optimal_dataset = ExperimentDataset.load(optimal_path)
    supervised_score = optimal_dataset.cluster.supervised_score
    unsupervised_score = optimal_dataset.cluster.unsupervised_score

    return OptimalParameters(
        supervised_score=supervised_score,
        unsupervised_score=unsupervised_score,
        path=optimal_path,
        cluster_method=cluster_method,
        reduction_method=reduction_method,
    )


def find_optimal(dataset_name: DatasetName, data_index: DataFrame) -> DataFrame:
    """
    For each dimensionality reduction method, find the optimal n_components for each clustering method

    :param dataset_name: The name of the dataset to use
    :type dataset_name: DatasetName
    """
    # for each dimensionality reduction method, find the optimal n_components
    optimal_values = []

    # combine the dimensionality reduction methods and clustering methods
    methods = list(product(ReductionMethod, ClusterMethod))

    for reduction_method, cluster_method in methods:
        if cluster_method == ClusterMethod.NO_CLUSTERING:
            continue
        optimal_params = find_optimal_n_components(data_index, cluster_method, reduction_method)
        optimal_values.append(optimal_params)

    # save output to file as csv
    # convert to dataframe
    df = pd.DataFrame([x.__dict__ for x in optimal_values])
    # save to csv
    df.to_csv(f'./artifacts/{dataset_name.name}/optimal.csv', index=False)


def get_dimensionality_reduction(raw_data: ExperimentDataset, goal_variance: float = 0.9) -> List[str]:
    """
    > For each dimensionality reduction method, if the pickle file doesn't exist, run the dimensionality reduction and save
    the result

    :param raw_data: ExperimentDataset
    :type raw_data: ExperimentDataset
    :param goal_variance: The amount of variance you want to keep in the data
    :type goal_variance: float
    :return: A list of strings that are the paths to the pickle files that contain the data for each dimensionality
    reduction method.
    """
    data_addresses = []
    for method in ReductionMethod:
        data = raw_data.copy()
        data.reducer.method = method
        data.reducer.goal_variance = goal_variance
        data.reducer.n_components = None

        print(f'\tCalculating {method.name} Dimensionality Reduction Datasets')
        data.set_optimal_reduction(method, goal_variance)
        data.save()
        data_addresses.append(data.get_path())

    # dedup the list
    data_addresses = list(set(data_addresses))

    for method in ReductionMethod:
        paths = ExperimentDataset.find(name=raw_data.name, reduction_method=method)
        assert len(paths) > 0, f'No paths found for {method.name}'

    return data_addresses


def clear_artifacts():
    """
    It removes all files with the extension `.pkl` from the `artifacts` directory
    """
    # remove all artifacts recursively with pkl extension
    for root, dirs, files in os.walk('artifacts', topdown=False):
        for name in files:
            if name.endswith('.pkl'):
                os.remove(os.path.join(root, name))


def create_experiment_directory_structure():
    """
    > Create a directory structure for the experiment if it doesn't already exist
    """
    # create directory structure if it doesn't exist
    for dataset_name in DatasetName:
        path = f'artifacts/{dataset_name.name}'
        if not os.path.exists(path):
            os.makedirs(path)


@click.command()
@click.option('--dataset', default='APS_SYSTEM_FAILURE', help='The name of the dataset to use')
@click.option('--clear', default=False, help='Clear all artifacts')
@click.option('--limit', default=None, help='Limit the number of samples to use')
@click.option('--goal_variance', default=0.9, help='The amount of variance you want to keep in the data')
def entry(dataset: str = 'APS_SYSTEM_FAILURE', clear: bool = True, limit: int = 100, goal_variance: float = 0.9):
    if limit is None:
        limit = None
    else:
        limit = int(limit)

    # get dataset name from input string
    dataset_name = DatasetName(dataset)

    # clear artifacts if specified
    if clear:
        clear_artifacts()

    # create directory structure
    create_experiment_directory_structure()

    # run experiment
    run(dataset_name, limit=limit, goal_variance=goal_variance)
    data_index = pd.read_csv(f'./artifacts/{dataset_name.name}/data_index.csv')
    # save data index to csv
    find_optimal(dataset_name, data_index)

    # run neural network experiment on the optimal parameters
    # get optimal parameters
    optimal_index = pd.read_csv(f'./artifacts/{dataset_name.name}/optimal.csv')
    # for each row, run the experiment with neural network
    train_neural_networks(name=dataset_name, optimal_index=optimal_index)


if __name__ == '__main__':
    entry()
