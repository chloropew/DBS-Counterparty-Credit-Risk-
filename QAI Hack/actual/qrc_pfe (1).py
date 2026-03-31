# ============================================================
# QRC-based Market Price Forecasting with Exposure Diagnostics
# ============================================================

import numpy as np
import pandas as pd
from qutip import basis, expect, sesolve, tensor, qeye, sigmax, sigmaz
from itertools import combinations
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# 1. Load and prepare data
# ------------------------------------------------------------
df = pd.read_csv('marketprice_with_sentiment.csv')
df = df[['Date', 'Close', 'Sentiment_Score']]
df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

# Convert to matrix (features x time)
matrix_2d_subset = df[['Close', 'Sentiment_Score']].T.values
time_index = df['Date'].values

print("2D matrix shape (features, time):", matrix_2d_subset.shape)

# Define train/test split
train_start = pd.to_datetime("2/10/2023", dayfirst=True)
train_end = pd.to_datetime("31/7/2024", dayfirst=True)
test_start = pd.to_datetime("1/8/2024", dayfirst=True)
test_end = pd.to_datetime("30/9/2024", dayfirst=True)

# Get index positions
train_mask = (df['Date'] >= train_start) & (df['Date'] <= train_end)
test_mask = (df['Date'] >= test_start) & (df['Date'] <= test_end)
train_indices = np.where(train_mask)[0]
test_indices = np.where(test_mask)[0]

# Forecasting parameters
input_window = 14
forecast_horizon = 7

def create_multi_step_dataset(data, input_window, forecast_horizon):
    X, Y = [], []
    n_features, n_time = data.shape
    for t in range(n_time - input_window - forecast_horizon + 1):
        x = data[:, t:t + input_window]
        y = data[:, t + input_window:t + input_window + forecast_horizon]
        X.append(x.T)
        Y.append(y.T)
    return np.array(X), np.array(Y)

# Get actual indices
train_start_idx = train_indices[0]
train_end_idx = train_indices[-1] + 1
test_start_idx = test_indices[0]
test_end_idx = test_indices[-1] + 1

# Create datasets
X_train, Y_train = create_multi_step_dataset(
    matrix_2d_subset[:, train_start_idx:train_end_idx],
    input_window,
    forecast_horizon
)

X_test, Y_test = create_multi_step_dataset(
    matrix_2d_subset[:, test_start_idx:test_end_idx],
    input_window,
    forecast_horizon
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ------------------------------------------------------------
# 2. Generate QRC embeddings
# ------------------------------------------------------------
# QRC parameters
def make_qrc_params(dim_pca=9):
    return {
        "atom_number": dim_pca,
        "encoding_scale": 5.0,
        "rabi_frequency": 2 * np.pi,
        "total_time": 2.0,
        "time_steps": 8,
        "readouts": "ZZ"
    }

params = make_qrc_params(dim_pca=6)

# Build Hamiltonian with GLOBAL detuning
def build_hamiltonian_global(x: np.ndarray, params, time_idx: int):
    N = params["atom_number"]
    s = params["encoding_scale"]
    Ω = params["rabi_frequency"]
    
    window_pos = int(time_idx * len(x) / params["time_steps"])
    window_pos = min(window_pos, len(x) - 1)
    Δ_global = -x[window_pos] * s
    
    sx = [tensor([sigmax() if j == i else qeye(2) for j in range(N)]) for i in range(N)]
    sz = [tensor([sigmaz() if j == i else qeye(2) for j in range(N)]) for i in range(N)]
    
    H = sum(Ω * sx[i] / 2 for i in range(N))
    H += Δ_global * sum(sz[i] / 2 for i in range(N))
    
    return H

# Evolve and embed with time-dependent global detuning
def evolve_and_embed_global(x_window, params):
    from qutip import Options
    N = params["atom_number"]
    T = params["total_time"]
    steps = params["time_steps"]
    
    psi0 = tensor([basis(2,0) for _ in range(N)])
    times = np.linspace(0, T, steps)
    
    embedding = []
    current_state = psi0
    opts = Options(nsteps=100000, atol=1e-8, rtol=1e-6)
    
    for step_idx in range(steps):
        H_t = build_hamiltonian_global(x_window, params, step_idx)
        
        if step_idx < steps - 1:
            dt = times[step_idx + 1] - times[step_idx]
            result = sesolve(H_t, current_state, [0, dt], options=opts)
            current_state = result.states[-1]
        
        for i in range(N):
            z_op = tensor([sigmaz() if j == i else qeye(2) for j in range(N)])
            embedding.append(expect(z_op, current_state))
        
        if params["readouts"] == "ZZ":
            for i, j in combinations(range(N), 2):
                zi_op = tensor([sigmaz() if k == i else qeye(2) for k in range(N)])
                zj_op = tensor([sigmaz() if k == j else qeye(2) for k in range(N)])
                zz_op = zi_op * zj_op
                embedding.append(expect(zz_op, current_state))
    
    return np.array(embedding)

def project_to_window(x_feature_vec, window_size):
    x = np.asarray(x_feature_vec).reshape(-1)
    if len(x) < window_size:
        return np.pad(x, (0, window_size - len(x)), mode='edge')
    elif len(x) > window_size:
        indices = np.linspace(0, len(x) - 1, window_size)
        return np.interp(indices, np.arange(len(x)), x)
    else:
        return x

def build_qrc_embeddings_from_windows(X_windows, params, max_samples=None):
    from tqdm import tqdm
    n_samples = len(X_windows) if max_samples is None else min(len(X_windows), max_samples)
    feature_idx = 0  # Use Close price
    
    sample0 = X_windows[0]
    x_timeseries = sample0[:, feature_idx]
    x_proj = project_to_window(x_timeseries, sample0.shape[0])
    test_emb = evolve_and_embed_global(x_proj, params)
    embedding_dim = test_emb.size
    embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)
    
    print(f"Building QRC embeddings: n_samples={n_samples}, embedding_dim={embedding_dim}")
    
    for i in tqdm(range(n_samples), desc="QRC embeddings", unit="sample"):
        window = X_windows[i]
        x_timeseries = window[:, feature_idx]
        x_proj = project_to_window(x_timeseries, window.shape[0])
        emb = evolve_and_embed_global(x_proj, params)
        embeddings[i, :] = emb
        
    return embeddings

QRC_train = build_qrc_embeddings_from_windows(X_train, params)
QRC_test = build_qrc_embeddings_from_windows(X_test, params)
print("QRC embeddings shape:", QRC_train.shape)

# ------------------------------------------------------------
# 3. Normalize target prices
# ------------------------------------------------------------
TARGET_FEATURE_INDEX = 0
Y_train_price = Y_train[:, :, TARGET_FEATURE_INDEX]
Y_test_price = Y_test[:, :, TARGET_FEATURE_INDEX]

scaler_y = MinMaxScaler()
Y_train_scaled = scaler_y.fit_transform(Y_train_price)
Y_test_scaled = scaler_y.transform(Y_test_price)

# ------------------------------------------------------------
# 4. Build neural network model
# ------------------------------------------------------------
def build_qrc_model(input_dim, horizon):
    # Accept either lookback int or (lookback, features)
    if isinstance(input_dim, int):
        timesteps, feature_dim = input_dim, 1
    else:
        timesteps, feature_dim = input_dim

    inputs = layers.Input(shape=(timesteps, feature_dim))
    x = layers.LayerNormalization()(inputs)

    # Local pattern extractor with causal padding
    x = layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    # Long-range context
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)

    # Self-attention over the sequence
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)

    # Sequence compression
    x = layers.Bidirectional(layers.GRU(64))(x)
    x = layers.Dropout(0.2)(x)

    # Projection to horizon
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(horizon)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"],
    )
    return model

model = build_qrc_model(QRC_train.shape[1], Y_train_price.shape[1])
model.summary()

# ------------------------------------------------------------
# 5. Train model with robust callbacks
# ------------------------------------------------------------
ckpt = callbacks.ModelCheckpoint(
    'best_qrc_model.keras', monitor='val_mae', save_best_only=True, verbose=1)
es = callbacks.EarlyStopping(
    monitor='val_mae', patience=30, restore_best_weights=True, verbose=1)

history = model.fit(
    QRC_train, Y_train_scaled,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[ckpt, es],
    verbose=1
)

# ------------------------------------------------------------
# 6. Predict and inverse transform
# ------------------------------------------------------------
Y_pred_scaled = model.predict(QRC_test)
Y_pred = scaler_y.inverse_transform(Y_pred_scaled)

# ------------------------------------------------------------
# 7. Compute evaluation metrics
# ------------------------------------------------------------
mae = mean_absolute_error(Y_test_price, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test_price, Y_pred))
r2 = r2_score(Y_test_price.flatten(), Y_pred.flatten())
print(f"\nMAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

# Show sample predictions
print("\nSample Test Predictions (first 3 samples, day 1 and day 7):")
for i in range(min(3, len(Y_pred))):
    print(f"Sample {i}: Actual Day1={Y_test_price[i,0]:.2f}, Pred Day1={Y_pred[i,0]:.2f} | "
          f"Actual Day7={Y_test_price[i,-1]:.2f}, Pred Day7={Y_pred[i,-1]:.2f}")

# ------------------------------------------------------------
# 8. Exposure and correlation diagnostics
# ------------------------------------------------------------
current_spots = X_test[:, -1, TARGET_FEATURE_INDEX]
exposures = np.maximum(Y_pred - current_spots[:, None], 0.0)

print("\n==== Summary Statistics ====")
print(f"Test set exposure mean: {exposures.mean():.4f}")
print(f"Test set exposure max: {exposures.max():.4f}")
print(f"Percentage of test predictions with positive exposure: "
      f"{(exposures > 0).mean() * 100:.1f}%")

# Correlation diagnostics
corrs = []
for dim in range(QRC_test.shape[1]):
    c = np.corrcoef(QRC_test[:, dim], Y_test_price[:, 0])[0, 1]
    corrs.append(c)
corr_df = pd.DataFrame({
    'dimension': np.arange(QRC_test.shape[1]),
    'correlation': corrs
})
print("\nCorrelation statistics:")
print(corr_df.describe())
strong = np.sum(np.abs(corr_df['correlation']) > 0.2)
print(f"Embedding dims with |corr| > 0.2: {strong}/{QRC_test.shape[1]}")

# ------------------------------------------------------------
# 9. Baseline check (optional)
# ------------------------------------------------------------
from sklearn.linear_model import Ridge
lin = Ridge(alpha=1.0)
lin.fit(QRC_train, Y_train_scaled)
Y_lin_scaled = lin.predict(QRC_test)
Y_lin = scaler_y.inverse_transform(Y_lin_scaled)
mae_lin = mean_absolute_error(Y_test_price, Y_lin)
print(f"\nBaseline Ridge MAE: {mae_lin:.4f}")

# ------------------------------------------------------------
# 10. Future forecast (next 7 days after dataset)
# ------------------------------------------------------------
last_window = matrix_2d_subset[:, -input_window:]
last_window_formatted = last_window.T

x_timeseries = last_window_formatted[:, TARGET_FEATURE_INDEX]
x_proj = project_to_window(x_timeseries, last_window_formatted.shape[0])

final_embedding = evolve_and_embed_global(x_proj, params)
final_embedding = final_embedding.reshape(1, -1)

Y_pred_future_scaled = model.predict(final_embedding)
Y_pred_future = scaler_y.inverse_transform(Y_pred_future_scaled)

last_date = df['Date'].iloc[-1]
current_spot = df['Close'].iloc[-1]

# Compute PFE and EPE from test set distribution
PFE_PERCENTILE = 95
results = []
for day_offset in range(forecast_horizon):
    forecast_date = last_date + pd.Timedelta(days=day_offset + 1)
    
    ee = np.mean(exposures[:, day_offset])
    pfe = np.percentile(exposures[:, day_offset], PFE_PERCENTILE)
    exposure_std = np.std(exposures[:, day_offset])
    
    results.append({
        'date': forecast_date,
        'mean_spot': current_spot,
        'ee': ee,
        'pfe': pfe,
        'exposure_std': exposure_std
    })

results_df = pd.DataFrame(results)

print("\n==== Exposure profile ====")
print(f"Last date: {last_date.strftime('%Y-%m-%d')}, Current spot: {current_spot:.2f}")
print(f"\n{'date':<12} {'mean_spot':>12} {'ee':>12} {'pfe':>12} {'exposure_std':>12}")
for _, row in results_df.iterrows():
    print(f"{row['date'].strftime('%Y-%m-%d'):<12} {row['mean_spot']:>12,.4f} "
          f"{row['ee']:>12,.4f} {row['pfe']:>12,.4f} {row['exposure_std']:>12,.4f}")

print("\nTraining and evaluation complete ✅")
