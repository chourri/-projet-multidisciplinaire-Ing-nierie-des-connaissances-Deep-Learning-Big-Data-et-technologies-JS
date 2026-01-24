# =============================
# 0. Imports
# =============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

# =============================
# 1. Charger le fichier Excel
# =============================
# Assurez-vous d'avoir uploadé "bm(21-25).xlsx" dans Colab
df = pd.read_excel("bm(21-25).xlsx")
print("✅ Fichier chargé :", df.shape)
print(df.head())

# =============================
# 2. Nettoyage date
# =============================
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date').reset_index(drop=True)

# =============================
# 3. Colonnes météo
# =============================
features = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']

# =============================
# 4. Nettoyage robuste
# =============================
for col in features:
    df[col] = (
        df[col].astype(str)
        .str.replace(',', '.', regex=False)
        .str.replace('—', '', regex=False)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=features)

if len(df) < 100:
    raise ValueError("❌ Pas assez de données après nettoyage")

# =============================
# 5. Saisonnalité
# =============================
df['dayofyear'] = df['date'].dt.dayofyear
df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

features_all = features + ['sin_doy', 'cos_doy']

# =============================
# 6. Normalisation
# =============================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features_all])

# =============================
# 7. Séquences temporelles
# =============================
sequence_length = 30
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i][:len(features)])  # uniquement météo

X = np.array(X)
y = np.array(y)

# =============================
# 8. Modèle LSTM
# =============================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, len(features_all))),
    Dropout(0.2),
    LSTM(32),
    Dense(len(features))
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# =============================
# 9. Entraînement
# =============================
early_stop = EarlyStopping(patience=7, restore_best_weights=True)

model.fit(
    X, y,
    epochs=60,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# =============================
# 10. Prédiction 2026
# =============================
last_sequence = scaled_data[-sequence_length:]
predictions_scaled = []

future_dates = pd.date_range("2026-01-01", periods=365)

for date in future_dates:
    day = date.dayofyear
    sin_doy = np.sin(2 * np.pi * day / 365)
    cos_doy = np.cos(2 * np.pi * day / 365)

    next_day = model.predict(
        last_sequence.reshape(1, sequence_length, len(features_all)),
        verbose=0
    )[0]

    predictions_scaled.append(next_day)

    next_full = np.concatenate([next_day, [sin_doy, cos_doy]])
    last_sequence = np.vstack([last_sequence[1:], next_full])

# =============================
# 11. Dénormalisation
# =============================
pad = np.zeros((len(predictions_scaled), 2))
pred_full = np.hstack([predictions_scaled, pad])
predictions = scaler.inverse_transform(pred_full)[:, :len(features)]

# =============================
# 12. Sauvegarde des prédictions en JSON
# =============================
df_2026 = pd.DataFrame(predictions, columns=features)
df_2026.insert(0, 'date', future_dates)

df_2026.to_json("weather_2026_predicted.json", orient='records', date_format='iso', force_ascii=False)
print("✅ Prédictions 2026 générées en JSON")

# =============================
# 13. Évaluation des performances
# =============================
# Test set = dernière année (ou max 365 points)
test_size = min(365, len(X))
X_test = X[-test_size:]
y_test = y[-test_size:]

y_pred_scaled = model.predict(X_test, verbose=0)
pad_test = np.zeros((len(y_pred_scaled), 2))
y_pred_full = np.hstack([y_pred_scaled, pad_test])
y_pred = scaler.inverse_transform(y_pred_full)[:, :len(features)]

# =============================
# 13.1 Erreurs globales corrigées
# =============================
errors_list = []
for i, feat in enumerate(features):
    y_true = y_test[:, i]
    y_pred_feat = y_pred[:, i]

    mse = mean_squared_error(y_true, y_pred_feat)
    mae = mean_absolute_error(y_true, y_pred_feat)
    rmse = np.sqrt(mse)
    # MAPE stable avec np.maximum(y_true, 1)
    mape = np.mean(np.abs((y_true - y_pred_feat) / np.maximum(y_true, 1))) * 100

    errors_list.append([feat, mse, rmse, mae, mape])

df_errors = pd.DataFrame(errors_list, columns=['Feature', 'MSE', 'RMSE', 'MAE', 'MAPE'])
print("\n📊 Erreurs globales corrigées :\n", df_errors)

# -----------------------------
# 13.2 Performances sur extrêmes
# -----------------------------
extreme_thresholds = {'tmax': 35, 'tmin': 0, 'prcp': 20}
extremes_list = []

for feat, thresh in extreme_thresholds.items():
    idx = features.index(feat)
    y_true_extreme = (y_test[:, idx] >= thresh).astype(int)
    y_pred_extreme = (y_pred[:, idx] >= thresh).astype(int)

    # Si pas d'exemple positif, éviter division par 0
    if y_true_extreme.sum() == 0:
        recall = np.nan
        precision = np.nan
        f1 = np.nan
    else:
        recall = recall_score(y_true_extreme, y_pred_extreme, zero_division=0)
        precision = precision_score(y_true_extreme, y_pred_extreme, zero_division=0)
        f1 = f1_score(y_true_extreme, y_pred_extreme, zero_division=0)

    accuracy = accuracy_score(y_true_extreme, y_pred_extreme)
    extremes_list.append([feat, thresh, accuracy, recall, precision, f1])

df_extremes = pd.DataFrame(extremes_list, columns=['Feature', 'Threshold', 'Accuracy', 'Recall', 'Precision', 'F1-score'])
print("\n⚠️ Performances sur extrêmes :\n", df_extremes)
