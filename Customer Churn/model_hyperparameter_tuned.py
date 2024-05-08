import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Iterable, Any

# Importing the dataset
dataset = pd.read_csv('Churn_Modeling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#Create dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

## Concatenate the Data Frames

X=pd.concat([X,geography,gender],axis=1)

## Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)

from sklearn.model_selection import train_test_split
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

## Perform Hyperparameter Optimization

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout, Input
from keras.activations import relu, sigmoid
from keras import regularizers

# model = create_neural_network(inputs=X_train.shape[1:], n_hidden=1, n_neurons=30, learning_rate=3e-3, activation='sigmoid', loss='binary_crossentropy')
def create_neural_network(hidden_layer_sizes):
    model = Sequential()
    model.add(Input(shape=X_train.shape[1:]))
    for layer in hidden_layer_sizes:
        model.add(Dense(layer, activation="relu"))
    model.add(Dense(1, activation='sigmoid'))
    # optimizer = keras.optimizers.get(compile_kwargs["optimizer"])
    # optimizer.learning_rate = compile_kwargs["learning_rate"]
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# def build_fn():
#     return create_neural_network()

keras_clf = KerasClassifier(model=create_neural_network,
    loss="binary_crossentropy",
    optimizer="adam",
    model__hidden_layer_sizes=(100,),
    verbose=False,)
# clf_model = keras_clf.fit(X_train, y_train)
# print(clf_model.score(X_train,y_train))
# mse_test = keras_clf.score(X_test, y_test)
# print(mse_test)
param_grid = {
    'model__hidden_layer_sizes': [(100, ), (50, 50, )]
    }

grid = GridSearchCV(estimator = keras_clf, param_grid = param_grid)
grid.fit(X_train,y_train)

print(grid.best_score_)
print(grid.best_params_)



####################################################################################################################################

# from typing import Iterable, Dict, Any
# from keras.models import Sequential
# from keras.layers import Input, Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# import keras

# def create_neural_network(hidden_layer_sizes: Iterable[int], meta: Dict[str, Any], compile_kwargs: Dict[str, Any]):
#     model = Sequential()
#     model.add(Input(shape=(meta["n_features_in_"],)))
#     for layer in hidden_layer_sizes:
#         model.add(Dense(layer, activation="relu"))
#     model.add(Dense(1, activation='sigmoid'))
#     optimizer = keras.optimizers.get(compile_kwargs.get("optimizer", "adam"))
#     optimizer.learning_rate = compile_kwargs.get("learning_rate", 0.001)
#     model.compile(loss=compile_kwargs.get("loss", "binary_crossentropy"), optimizer=optimizer)
#     return model

# # Initialize the KerasClassifier with the build_fn parameter
# def build_fn():
#     return create_neural_network(hidden_layer_sizes=(7,), meta={"n_features_in_": X_train.shape[1]}, compile_kwargs={"optimizer": "adam", "learning_rate": 0.001, "loss": "binary_crossentropy"})

# keras_clf = KerasClassifier(build_fn=build_fn, epochs=15, verbose=0)
