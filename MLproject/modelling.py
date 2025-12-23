import os
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
# KONFIGURASI MLFLOW
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Modelling")

# =========================
# LOAD DATA (HASIL PREPROCESSING)
# =========================
df = pd.read_csv("datasekolahsiap.csv")

# =========================
# VALIDASI DATA (WAJIB)
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
# TRAINING & LOGGING
# =========================
for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name) as run:

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluasi
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameter & metric
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )

        # PRINT RUN ID
        print("MLFLOW_RUN_ID=", run.info.run_id)

        # SIMPAN RUN ID KE FILE (UNTUK CI)
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        # PRINT HASIL
        print(
            f"{model_name} | "
            f"MSE={mse:.2f} | "
            f"RMSE={rmse:.2f} | "
            f"MAE={mae:.2f} | "
            f"R2={r2:.3f}"
        )

print("Semua model berhasil ditraining & tercatat di MLflow")

