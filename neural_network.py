import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
time = np.arange(0, 100, 1)
values = 2 * time + 1.5 * np.random.randn(100)

print(values)


def create_dataset(values, time_steps=1):
    X = []
    Y = []
    for i in range(len(values) - time_steps):
        X.append(values[i:(i + time_steps)])
        Y.append(values[i: (i + time_steps)])

    return np.array(X), np.array(Y)


time_steps = 5
X, y = create_dataset(values, time_steps)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10,activation='relu',input_shape=(time_steps,)),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=1)

y_pred = model.predict(X_test)
