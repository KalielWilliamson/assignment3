from datamodel import *


def cluster(raw_data: ExperimentDataset, method: ClusterMethod) -> None:
    """
    It takes in an ExperimentDataset object, and it runs a clustering algorithm on the data

    :param raw_data: ExperimentDataset
    :type raw_data: ExperimentDataset
    :param method: ClusterTypes
    :type method: ClusterMethod
    """
    data = raw_data.copy()

    if data.cluster.n_components is None:
        raise ValueError('n_components must be set')

    # if the path to the file where the results of the experiment are stored exists, then we don't need to run the
    # experiment again
    # if file with cluster method, reduction method, and cluster n_components exists, then we don't need to run the
    # get files in the directory
    files = os.listdir(f'artifacts/{data.name.name}')
    # if any file has the cluster method, reduction method, and cluster n_components, then we don't need to run the
    if any([method.name in f and data.reducer.method.name in f and f'cluster_components={data.cluster.n_components}' for f in files]):
        return

    kmeans = KMeans(n_clusters=data.cluster.n_components)
    gmm = GaussianMixture(n_components=data.cluster.n_components)

    X: DataFrame = data.X_train
    y: ndarray = data.y_train

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
    unsupervised_score, supervised_score, evaluation_method = evaluate_cluster_quality(data, labels)

    data.cluster.model = model
    data.cluster.labels = labels
    data.cluster.loss = loss
    data.cluster.wall_time = wall_time
    data.cluster.unsupervised_score = unsupervised_score
    data.cluster.supervised_score = supervised_score
    data.cluster.supervised_evaluation_method = evaluation_method
    data.cluster.method = method

    # save the output dataclass to pickle file
    data.save()


def evaluate_cluster_quality(data: ExperimentDataset, labels: ndarray) -> (float, float, str):
    """
    It returns the silhouette score, the adjusted mutual information score, and the name of the score

    :param data: ExperimentDataset
    :type data: ExperimentDataset
    :param labels: ndarray
    :type labels: ndarray
    :return: The silhouette score, the adjusted mutual information score, and the string 'AMI' or 'ARI'
    """
    # silhouette score
    # basic info: https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
    # pitfalls: https://stats.stackexchange.com/questions/223574/silhouette-score-significance-what-is-a-significant-increase-in-silhouette-scor
    # info on convex clusters:https://math.stackexchange.com/questions/2751592/what-defines-a-convex-cluster-and-how-it-differentiates-from-other-types
    s_score: float = silhouette_score(data.X_train, labels, metric='euclidean')

    # adjusted mutual information score
    # basic info: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html
    # https://en.wikipedia.org/wiki/Adjusted_mutual_information
    # ARI vs AMI: https://stats.stackexchange.com/questions/260487/adjusted-rand-index-vs-adjusted-mutual-information
    # - AMI is better for non-overlapping clusters & large equal sized ground truth clusters
    # - ARI is better for overlapping clusters & unbalanced ground truth clusters
    # - Therefore we should test both and measure the balance of the clusters

    # measure the balance of the clusters from the ground truth
    # balance of y
    y_is_balanced = abs(data.y_train.sum() / len(data.y_train) - 0.5) < 0.1
    if y_is_balanced:
        return s_score, adjusted_mutual_info_score(data.y_train, labels), 'AMI'
    else:
        return s_score, adjusted_rand_score(data.y_train, labels), 'ARI'
