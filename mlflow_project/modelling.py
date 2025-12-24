import os
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split
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
    r"E:\MSML\SMSML_Hilmatul-Luthfiyah-Hariroh\Membangun model\datasekolahsiap.csv"
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
for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name) as run:

        model.fit(X_train, y_train)

        # Simpan run_id (untuk CI / CD)
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"{model_name} selesai | RUN_ID = {run.info.run_id}")

print("Semua model berhasil ditraining dengan MLflow Autolog")
