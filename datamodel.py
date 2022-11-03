import copy
import dataclasses
import datetime
import glob
import keras
import os
import pandas as pd
import pickle
import random
import time
from enum import Enum
from keras.callbacks import History
from numpy import ndarray
from pandas import DataFrame
from sklearn import decomposition
from sklearn import random_projection
from sklearn.cluster import KMeans
from sklearn.metrics import explained_variance_score, silhouette_score, adjusted_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from typing import Any, Dict, List


class DatasetName(Enum):
    APS_SYSTEM_FAILURE = 'APS_SYSTEM_FAILURE'
    HEPMASS = 'HEPMASS'


class ClusterMethod(Enum):
    KMEANS = 'kmeans'
    GMM = 'gmm'  # this is an expectation maximization algorithm
    NO_CLUSTERING = 'no_clustering'


class ReductionMethod(Enum):
    PCA = decomposition.PCA
    ICA = decomposition.FastICA
    RANDOM_PROJECTIONS = random_projection.GaussianRandomProjection
    TRUNCATED_SVD = decomposition.TruncatedSVD
    NO_METHOD = 'no_method'


@dataclasses.dataclass
class OptimalParameters:
    reduction_method: ReductionMethod
    cluster_method: ClusterMethod
    path: str
    supervised_score: float = None
    unsupervised_score: float = None


@dataclasses.dataclass
class ClusterParams:
    labels: ndarray = None
    loss: float = None
    n_components: int = None
    method: ClusterMethod = ClusterMethod.NO_CLUSTERING
    model: Any = None
    unsupervised_score: float = None  # silhouette score
    supervised_score: float = None
    supervised_evaluation_method: str = None
    wall_time: float = None  # seconds


@dataclasses.dataclass
class NeuralNetworkOutput:
    inference_wall_time: float = None  # seconds
    training_wall_time: float = None  # seconds
    fitness_curve: Dict = None
    is_cluster: bool = None
    test_loss: float = None
    test_accuracy: float = None


@dataclasses.dataclass
class ExplainedVarianceResult:
    explained_variance: float
    n_components: int
    eps: float = None


@dataclasses.dataclass
class DimensionalityReducer:
    inference_wall_time: float = None
    training_wall_time: float = None
    reducer: Any = None
    explained_variance_history: List[ExplainedVarianceResult] = dataclasses.field(default_factory=list)
    method = ReductionMethod.NO_METHOD
    n_components: int = None


class ExperimentDataset:
    name: DatasetName
    X_train: DataFrame
    y_train: ndarray
    X_test: DataFrame
    y_test: ndarray
    # unique id for the dataset
    id: str = str(random.randint(0, 1000000000))
    test_error: float = None
    reducer: DimensionalityReducer = DimensionalityReducer()
    cluster: ClusterParams = ClusterParams()
    neural_network_output: NeuralNetworkOutput = NeuralNetworkOutput()
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    explained_variance: float = None

    def __init__(self, name: DatasetName, X_train: DataFrame, y_train: ndarray, X_test: DataFrame, y_test: ndarray):
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        id = str(random.randint(0, 1000000000))
        # verify no other dataset with the same id exists
        while os.path.exists(f'./artifacts/{self.name.value}/{id}.pkl'):
            id = str(random.randint(0, 1000000000))
        self.id = id

    def get_path(self):
        # get string with nanoseconds
        return f'./artifacts/{self.name.value}/{self.id}.pkl'

    @staticmethod
    def get_path_from_fields(name: DatasetName,
                             cluster_method: ClusterMethod,
                             dimensionality_reduction_method: ReductionMethod,
                             cluster_components: int = None,
                             reduction_components: int = None) -> str:
        """
        It takes in the name of the dataset, the clustering method, the dimensionality reduction method, and the number of
        components, and returns the path to the file that contains the data for the experiment

        :param cluster_components:
        :param reduction_components:
        :param name: The name of the dataset
        :type name: DatasetName
        :param cluster_method: The clustering method to use
        :type cluster_method: ClusterMethod
        :param dimensionality_reduction_method: The method used to reduce the dimensionality of the data
        :type dimensionality_reduction_method: ReductionMethod
        :return: A string that is the path to the file that contains the data for the experiment.
        """
        # Returning a string that is the path to the file that contains the data for the experiment.
        dm_n_components: str = str(reduction_components) if reduction_components else 'None'
        cluster_n_components: str = str(cluster_components) if cluster_components else 'None'
        return f'artifacts/{name.name}/cluster_method={cluster_method.name}__cluster_components={cluster_n_components}__reduction_method={dimensionality_reduction_method.name}__reduction_components={dm_n_components}__dataset.pkl'

    def save(self):
        # verify that the dimensionality reduction method is not None
        if self.reducer is None:
            raise ValueError('The dimensionality reduction method cannot be None')

        if self.cluster.method != ClusterMethod.NO_CLUSTERING and self.cluster.n_components is None:
            raise ValueError('Cannot save a dataset without a cluster number of components')

        if self.id in os.listdir(f'./artifacts/{self.name.value}'):
            raise ValueError('Cannot save a dataset with an id that already exists')

        path = self.get_path()
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def copy(self):
        new_dataset = ExperimentDataset(self.name, self.X_train, self.y_train, self.X_test, self.y_test)

        new_dataset.reducer = self.reducer

        new_dataset.cluster = self.cluster

        return new_dataset

    def set_optimal_reduction(self, method: ReductionMethod, goal_variance: float = 0.9) -> None:
        """
        Sets the optimal dimensionality reduction method and number of components for the dataset.

        :param method:
        :param goal_variance:
        :return:
        """
        self.reducer = DimensionalityReducer()
        self.reducer.method = method

        if method == ReductionMethod.NO_METHOD:
            return

        self.reducer.explained_variance_history = []
        n_samples, n_features = self.X_train.shape
        n_components_list = random.sample(range(2, n_features), n_features // 10)

        for n_components in n_components_list:
            reduction_fn = method.value(n_components=n_components)

            start_time = time.time()
            reduction_fn.fit(self.X_train)
            training_wall_time = time.time() - start_time

            start_time = time.time()
            X_train = reduction_fn.transform(self.X_train)
            inference_wall_time = time.time() - start_time

            X_test = reduction_fn.transform(self.X_test)

            explained_variance = explained_variance_score(self.X_train, reduction_fn.inverse_transform(X_train))
            self.reducer.explained_variance_history.append(ExplainedVarianceResult(explained_variance, n_components))

            if explained_variance >= goal_variance:
                self.reducer.training_wall_time = training_wall_time
                self.reducer.inference_wall_time = inference_wall_time
                self.reducer.n_components = n_components
                self.X_train = X_train
                self.X_test = X_test
                break

            # if last iteration and no components found
            if n_components == n_components_list[-1] and self.reducer.n_components is None:
                self.reducer.training_wall_time = training_wall_time
                self.reducer.inference_wall_time = inference_wall_time
                self.reducer.n_components = n_components
                self.X_train = X_train
                self.X_test = X_test

    def set_cluster(self, method: ClusterMethod) -> None:
        """
        It takes in an ExperimentDataset object, and it runs a clustering algorithm on the data

        :param raw_data: ExperimentDataset
        :type raw_data: ExperimentDataset
        :param method: ClusterTypes
        :type method: ClusterMethod
        """

        if self.cluster.n_components is None:
            raise ValueError('n_components must be set')

        # if the path to the file where the results of the experiment are stored exists, then we don't need to run the
        # experiment again
        # if file with cluster method, reduction method, and cluster n_components exists, then we don't need to run the
        # get files in the directory
        files = os.listdir(f'artifacts/{self.name.name}')
        # if any file has the cluster method, reduction method, and cluster n_components, then we don't need to run the
        if any([method.name in f and self.reducer.method.name in f and f'cluster_components={self.cluster.n_components}'
                for f in files]):
            return

        kmeans = KMeans(n_clusters=self.cluster.n_components)
        gmm = GaussianMixture(n_components=self.cluster.n_components)

        X: DataFrame = self.X_train
        y: ndarray = self.y_train

        start = datetime.datetime.now()
        if method == ClusterMethod.KMEANS:
            model = kmeans.fit(X)
            labels = kmeans.labels_
            loss = kmeans.inertia_
        elif method == ClusterMethod.GMM:
            model = gmm.fit(X)
            labels = gmm.predict(X)
            loss = gmm.bic(X)
        else:
            raise ValueError(f'Invalid cluster type: {method}')
        wall_time = (datetime.datetime.now() - start).total_seconds()

        # score the clusters
        unsupervised_score, supervised_score, evaluation_method = self.evaluate_cluster_quality(labels)

        self.cluster.model = model
        self.cluster.labels = labels
        self.cluster.loss = loss
        self.cluster.wall_time = wall_time
        self.cluster.method = method
        self.cluster.unsupervised_score = unsupervised_score
        self.cluster.supervised_score = supervised_score
        self.cluster.supervised_evaluation_method = evaluation_method

        # save the output dataclass to pickle file
        self.save()

    def evaluate_cluster_quality(self, labels: ndarray) -> (float, float, str):
        """
        It returns the silhouette score, the adjusted mutual information score, and the name of the score

        :param data: ExperimentDataset
        :type data: ExperimentDataset
        :param labels: ndarray
        :type labels: ndarray
        :return: The silhouette score, the adjusted mutual information score, and the string 'AMI' or 'ARI'
        """
        s_score: float = silhouette_score(self.X_train, labels, metric='euclidean')
        y_is_balanced = abs(self.y_train.sum() / len(self.y_train) - 0.5) < 0.1
        if y_is_balanced:
            return s_score, adjusted_mutual_info_score(self.y_train, labels), 'AMI'
        else:
            return s_score, adjusted_rand_score(self.y_train, labels), 'ARI'

    @staticmethod
    def find(name: DatasetName, cluster_method: ClusterMethod = None, cluster_components: int = None, reduction_method: ReductionMethod = None, reduction_components: int = None) -> List[str]:
        # load files from directory
        results = []
        files = os.listdir(f'artifacts/{name.name}')
        for file in files:
            path = f'artifacts/{name.name}/{file}'
            with open(path, 'rb') as f:
                experiment: ExperimentDataset = pickle.load(f)
                is_cluster_method = cluster_method is None or experiment.cluster.method == cluster_method
                is_cluster_components = cluster_components is None or experiment.cluster.n_components == cluster_components
                is_reduction_method = reduction_method is None or experiment.reducer.method == reduction_method
                is_reduction_components = reduction_components is None or experiment.reducer.n_components == reduction_components
                if is_cluster_method and is_cluster_components and is_reduction_method and is_reduction_components:
                    results.append(path)
        return results

    @staticmethod
    def get_index(name: DatasetName) -> DataFrame:
        results = []
        files = os.listdir(f'artifacts/{name.name}')
        for file in files:
            path = f'artifacts/{name.name}/{file}'
            with open(path, 'rb') as f:
                try:
                    experiment: ExperimentDataset = pickle.load(f)
                    # create dataframe
                    results.append({
                        'dataset': name.name,
                        'cluster_method': experiment.cluster.method.name,
                        'cluster_components': experiment.cluster.n_components,
                        'reduction_method': experiment.reducer.method.name,
                        'reduction_components': experiment.reducer.n_components,
                        'path': path
                    })
                except Exception as e:
                    print(f'Error loading {path}: {e}')
        # save the dataframe to a csv file
        df = DataFrame(results)
        df.to_csv(f'artifacts/{name.name}/data_index.csv', index=False)
        return df

