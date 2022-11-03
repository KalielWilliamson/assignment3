# import keras
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.engine.input_layer import InputLayer
from keras.layers import Dense
from typing import Tuple

from datamodel import *


def train_neural_networks(name: DatasetName, optimal_index: DataFrame) -> None:
    # all clustering methods and dimensionality reduction methods
    for index, row in optimal_index.iterrows():
        path = row['path']
        # load the data
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # run the experiment
        train_nn_model(data, is_cluster=True)
        train_nn_model(data, is_cluster=False)

    # train nn model without clustering or dimensionality reduction

    # find data without clustering or dimensionality reduction
    data = ExperimentDataset.load(f'./artifacts/{name.name}/raw_{name.name}.pkl')
    train_nn_model(data, is_cluster=True)
    train_nn_model(data, is_cluster=False)


def train_nn_model(data: ExperimentDataset, is_cluster: bool) -> None:
    data_new = data.copy()
    X_test, X_train, y_test, y_train = get_nn_data(data_new, is_cluster=is_cluster)
    data_new.neural_network_output = train(X_train, X_test, y_train, y_test, is_cluster=is_cluster)
    save(data_new, is_cluster)


def get_model(X_train) -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=X_train.shape[1]))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train(X_test: np.ndarray, X_train: np.ndarray, y_test: np.ndarray, y_train: np.ndarray,
          is_cluster: bool) -> NeuralNetworkOutput:
    model = get_model(X_train)

    # early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # start time
    start = time.time()
    # fit the model
    fitness_curve = model.fit(X_train, y_train, epochs=2000, batch_size=50, validation_data=(X_test, y_test),
                              callbacks=[es]).history
    # wall time
    wall_time_train = (time.time() - start)  # unit: seconds

    # run inference on test set
    start = time.time()
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    wall_time_inference = (time.time() - start)  # unit: seconds

    return NeuralNetworkOutput(
        fitness_curve=fitness_curve,
        training_wall_time=wall_time_train,
        inference_wall_time=wall_time_inference,
        test_loss=loss,
        test_accuracy=accuracy,
        is_cluster=is_cluster
    )


def get_nn_data(data: ExperimentDataset, is_cluster: bool = True) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = data.X_train
    y_train = data.y_train
    X_test = data.X_test
    y_test = data.y_test

    if not is_cluster:
        return X_test, X_train, y_test, y_train

    # apply clustering
    # if gaussian mixture model
    if data.cluster.method == ClusterMethod.GMM:
        # get centroid distances for each point
        X_train = data.cluster.model.predict_proba(X_train)
        X_test = data.cluster.model.predict_proba(X_test)
    elif data.cluster.method == ClusterMethod.KMEANS:
        # get the distance of each point to all centroids
        X_train = np.array([np.linalg.norm(X_train - c, axis=1) for c in data.cluster.model.cluster_centers_]).T
        X_test = np.array([np.linalg.norm(X_test - c, axis=1) for c in data.cluster.model.cluster_centers_]).T
    return X_test, X_train, y_test, y_train


def save(data: ExperimentDataset, is_cluster=True) -> None:
    cluster_method = data.cluster.method
    if is_cluster:
        cluster_method = ClusterMethod.NO_CLUSTERING
    with open(f'artifacts/{data.name.name}/nn_output_{cluster_method.name}_{data.reducer.method.name}.pickle',
              'wb') as f:
        pickle.dump(data, f)
