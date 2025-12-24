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
with mlflow.start_run(run_name="model_comparison"):

    for model_name, model in models.items():

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param(f"{model_name}_n_estimators", n_estimators)
        mlflow.log_param(f"{model_name}_max_depth", max_depth)

        mlflow.log_metric(f"{model_name}_mse", mse)
        mlflow.log_metric(f"{model_name}_rmse", rmse)
        mlflow.log_metric(f"{model_name}_mae", mae)
        mlflow.log_metric(f"{model_name}_r2", r2)

        mlflow.sklearn.log_model(model, f"model_{model_name}")

        print(f"{model_name} selesai")

print("Semua model berhasil ditraining & dicatat")
