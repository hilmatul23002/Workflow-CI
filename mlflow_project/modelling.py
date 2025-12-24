import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# =========================
# SUPPRESS WARNING GIT
# =========================
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# =========================
# AMBIL PARAMETER (UNTUK MLFLOW RUN)
# =========================
# default → aman untuk run lokal
if len(sys.argv) > 1:
    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    dataset_path = sys.argv[3]
else:
    n_estimators = 100
    max_depth = 5
    dataset_path = "datasekolahsiap.csv"

# =========================
# KONFIGURASI MLFLOW
# =========================
mlflow.set_experiment("Modelling")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(dataset_path)

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
        n_estimators=n_estimators,
        max_depth=max_depth,
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
# TRAINING & LOGGING
# =========================
# JANGAN pakai start_run
for model_name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metric
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Log ke MLflow
    mlflow.log_param("model_name", model_name)
    mlflow.log_metric(f"{model_name}_rmse", rmse)

    # Log model (INI AMAN)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=model_name
    )
# =========================
# FIX BUG MLFLOW PROJECTS + CI
# =========================
os.environ.pop("MLFLOW_RUN_ID", None)
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
