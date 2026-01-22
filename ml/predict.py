# =============================
# 0. Imports
# =============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =============================
# 1. Charger le fichier Excel
# =============================
df = pd.read_excel("bm(21-25).xlsx")

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
# 5. Saisonnalité (clé)
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
    LSTM(64, return_sequences=True,
         input_shape=(sequence_length, len(features_all))),
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
# 11. Dénormalisation CORRECTE
# =============================
pad = np.zeros((len(predictions_scaled), 2))
pred_full = np.hstack([predictions_scaled, pad])
predictions = scaler.inverse_transform(pred_full)[:, :len(features)]

# =============================
# 12. Sauvegarde
# =============================
df_2026 = pd.DataFrame(predictions, columns=features)
df_2026.insert(0, 'date', future_dates)

df_2026.to_csv("weather_2026_predicted.csv", index=False)
print("✅ Prédictions 2026 réalistes générées")
