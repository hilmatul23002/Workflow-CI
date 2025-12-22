import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# MLflow Configuration
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Random_Forest_Experiment")

# =========================
# Load Dataset
# =========================
df = pd.read_csv("datasekolahsiap.csv")

# =========================
# Tentukan Target
# =========================
TARGET = "Siswa"   # ⬅️ GANTI sesuai kolom target Anda

X = df.drop(columns=[TARGET])
y = df[TARGET]

# =========================
# Pisahkan Kolom
# =========================
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

print("Categorical:", categorical_cols)
print("Numerical:", numerical_cols)

# =========================
# Preprocessing
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# =========================
# Pipeline
# =========================
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# =========================
# MLflow Autolog
# =========================
mlflow.sklearn.autolog()

# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="RandomForest_Regressor"):
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("rmse_manual", rmse)
    mlflow.log_metric("r2_manual", r2)

    print("RMSE:", rmse)
    print("R2 Score:", r2)