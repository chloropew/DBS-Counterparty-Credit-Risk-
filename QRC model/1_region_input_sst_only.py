# SST Forecasting with Linear Neural Network and Linear NN + QRC Comparison

import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

# === Load SST Dataset from Local File ===
file_path = "sst_processed_20160101_20231231.nc"
ds_sst = xr.open_dataset(file_path)

# === Preprocess Dataset ===
ds_sst = ds_sst.rename({'longitude': 'lon', 'latitude': 'lat'})
ds_sst['time'] = pd.to_datetime(ds_sst['time'].values).normalize()

# === Randomly Select a Region Without NaNs ===
valid_region_found = False
for _ in range(1000):
    lat_idx = np.random.randint(0, ds_sst.dims['lat'])
    lon_idx = np.random.randint(0, ds_sst.dims['lon'])
    sst_candidate = ds_sst['sst'][:, lat_idx, lon_idx].values
    if not np.isnan(sst_candidate).any():
        valid_region_found = True
        break

if not valid_region_found:
    raise ValueError("No valid region without NaNs found.")

print(f"Selected region: lat index = {lat_idx}, lon index = {lon_idx}")

# === Extract SST Time Series for Selected Region ===
sst = sst_candidate
sst = np.expand_dims(sst, axis=-1)  # shape: (time, 1)

dates = ds_sst['time'].values

# === Normalize SST ===
sst_mean = np.mean(sst)
sst_std = np.std(sst)
sst_norm = (sst - sst_mean) / sst_std

# === Prepare Input and Output Data ===
input_window = 14
forecast_horizon = 7

X = []
Y = []

for i in range(len(sst_norm) - input_window - forecast_horizon):
    X.append(sst_norm[i:i+input_window])
    Y.append(sst_norm[i+input_window:i+input_window+forecast_horizon])

X = np.array(X)  # shape: (samples, 14, 1)
Y = np.array(Y)  # shape: (samples, 7, 1)

# === Train-Test Split ===
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# === Flatten Output ===
Y_train_flat = Y_train.reshape(len(Y_train), -1)
Y_test_flat = Y_test.reshape(len(Y_test), -1)

# === Linear Neural Network ===
model_linear = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(input_window, 1)),
    tf.keras.layers.Dense(forecast_horizon)
])

model_linear.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                     loss='mse',
                     metrics=['mae'])

checkpoint_cb = ModelCheckpoint('best_linear_model.h5', monitor='val_mae', save_best_only=True, mode='min', verbose=1)

model_linear.fit(X_train, Y_train_flat,
                 epochs=80,
                 batch_size=64,
                 validation_split=0.2,
                 callbacks=[checkpoint_cb],
                 verbose=1)

model_linear.load_weights('best_linear_model.h5')

# === Predict with Linear Model ===
Y_pred_linear = model_linear.predict(X_test)

# === QRC Embedding ===
# Placeholder for QRC embedding function
def qrc_embedding(X):
    # Dummy pass-through for now
    return X.reshape(X.shape[0], -1)  # flatten

X_train_qrc = qrc_embedding(X_train)
X_test_qrc = qrc_embedding(X_test)

# === Linear NN on QRC Embedding ===
model_qrc = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_qrc.shape[1],)),
    tf.keras.layers.Dense(forecast_horizon)
])

model_qrc.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='mse',
                  metrics=['mae'])

checkpoint_cb_qrc = ModelCheckpoint('best_qrc_model.h5', monitor='val_mae', save_best_only=True, mode='min', verbose=1)

model_qrc.fit(X_train_qrc, Y_train_flat,
              epochs=80,
              batch_size=64,
              validation_split=0.2,
              callbacks=[checkpoint_cb_qrc],
              verbose=1)

model_qrc.load_weights('best_qrc_model.h5')

# === Predict with QRC Model ===
Y_pred_qrc = model_qrc.predict(X_test_qrc)

# === Inverse Transform for Plotting ===
def denormalize(data):
    return data * sst_std + sst_mean

true_sst = denormalize(Y_test_flat)
pred_linear_sst = denormalize(Y_pred_linear)
pred_qrc_sst = denormalize(Y_pred_qrc)

# === Plot Comparison ===
plt.figure(figsize=(12, 5))
plt.plot(true_sst[:, 0], label='True SST')
plt.plot(pred_linear_sst[:, 0], label='Linear NN')
plt.plot(pred_qrc_sst[:, 0], label='Linear NN + QRC')
plt.title('SST Forecast Comparison (First Time Step)')
plt.xlabel('Sample')
plt.ylabel('SST (°C)')
plt.legend()
plt.grid(True)
plt.show()
