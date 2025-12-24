import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error   # ← WAJIB ADA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



# =========================
# SUPPRESS WARNING GIT
# =========================
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# =========================
# KONFIGURASI MLFLOW
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Modelling")

# =========================
# AKTIFKAN MLFLOW AUTOLOG
# =========================
mlflow.sklearn.autolog(
    log_models=True,   # ⬅️ model otomatis tersimpan
    silent=True
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(
    r"datasekolahsiap.csv"
)

# =========================
# VALIDASI DATA
# =========================
if not df.select_dtypes(include="object").empty:
    raise ValueError("Masih ada kolom string. Preprocessing belum selesai!")

# =========================
# TARGET & FEATURES
# =========================
TARGET = "Siswa"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# DAFTAR MODEL
# =========================
models = {
    "LinearRegression": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

# =========================
# TRAINING (AUTOLOG ONLY)
# =========================
# =========================
# TRAINING & LOGGING (FINAL)
# =========================
for model_name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metric
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Log ke MLflow (TANPA start_run)
    mlflow.log_param(f"{model_name}_type", model_name)
    mlflow.log_metric(f"{model_name}_rmse", rmse)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=f"model_{model_name}"
    )

    print(f"Model {model_name} berhasil dicatat | RMSE = {rmse}")




