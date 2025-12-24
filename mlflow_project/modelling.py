import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# =========================
# SUPPRESS WARNING GIT
# =========================
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# =========================
# AMBIL PARAMETER
# =========================
if len(sys.argv) > 1:
    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    dataset_path = sys.argv[3]
else:
    n_estimators = 100
    max_depth = 5
    dataset_path = "datasekolahsiap.csv"

# =========================
# LOAD & PREPARE DATA
# =========================
df = pd.read_csv(dataset_path)
TARGET = "Siswa"
X = df.drop(columns=[TARGET])
y = df[TARGET]

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
# 🔑 Ambil run_id yang dibuat oleh `mlflow run`
parent_run_id = os.environ.get("MLFLOW_RUN_ID")

if parent_run_id is None:
    raise RuntimeError("MLFLOW_RUN_ID tidak ditemukan. Jalankan via `mlflow run`.")

# =========================
# TRAINING & LOGGING (FIXED)
# =========================
# 🔗 ATTACH ke parent run (TIDAK membuat run baru)
with mlflow.start_run(run_id=parent_run_id):

    for model_name, model in models.items():

        # 🔹 Nested run = sub-run (INI BOLEH)
        with mlflow.start_run(run_name=model_name, nested=True):

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metric
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Logging
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("rmse", rmse)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"model_{model_name}"
            )

            print(f"Model {model_name} berhasil dicatat | RMSE = {rmse}")
