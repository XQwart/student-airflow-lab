import os
import json
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from airflow import DAG
from airflow.operators.python import PythonOperator

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", os.path.expanduser("~/airflow"))
DATA_DIR = os.path.join(AIRFLOW_HOME, "data")
MODEL_DIR = os.path.join(AIRFLOW_HOME, "models")
METRICS_DIR = os.path.join(AIRFLOW_HOME, "metrics")

for d in [DATA_DIR, MODEL_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)


def download_data():
    import urllib.request
    import zipfile
    zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    zip_path = os.path.join(DATA_DIR, "student.zip")
    urllib.request.urlretrieve(zip_url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    df = pd.read_csv(os.path.join(DATA_DIR, "student-mat.csv"), delimiter=";")
    raw_path = os.path.join(DATA_DIR, "student_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"Data downloaded: {raw_path}, shape={df.shape}")
    print(f"Target G3: min={df['G3'].min()}, max={df['G3'].max()}, mean={df['G3'].mean():.2f}")
    return raw_path


def clear_data():
    raw_path = os.path.join(DATA_DIR, "student_raw.csv")
    df = pd.read_csv(raw_path)
    print(f"Original shape: {df.shape}")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {before - len(df)}")

    df = df.dropna()
    print(f"After dropna: {df.shape}")

    cat_columns = df.select_dtypes(include=["object"]).columns.tolist()
    num_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    num_columns.remove("G3")

    print(f"Categorical: {cat_columns}")
    print(f"Numerical: {num_columns}")

    if cat_columns:
        encoder = OrdinalEncoder()
        df[cat_columns] = encoder.fit_transform(df[cat_columns])
        encoder_path = os.path.join(MODEL_DIR, "encoder.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)
        print(f"Encoder saved: {encoder_path}")

    q_low = df["G3"].quantile(0.01)
    q_high = df["G3"].quantile(0.99)
    df = df[(df["G3"] >= q_low) & (df["G3"] <= q_high)]
    df = df.reset_index(drop=True)

    X = df.drop(columns=["G3"])
    y = df["G3"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled["G3"] = y.values

    clean_path = os.path.join(DATA_DIR, "student_clean.csv")
    X_scaled.to_csv(clean_path, index=False)

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Clean data: {clean_path}, shape={X_scaled.shape}")
    return clean_path


def train_model():
    clean_path = os.path.join(DATA_DIR, "student_clean.csv")
    df = pd.read_csv(clean_path)

    X = df.drop(columns=["G3"])
    y = df["G3"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    params = {
        "alpha": [0.0001, 0.001, 0.01, 0.05, 0.1],
        "l1_ratio": [0.001, 0.05, 0.01, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ["squared_error", "huber", "epsilon_insensitive"],
        "fit_intercept": [False, True],
    }

    lr = SGDRegressor(random_state=42, max_iter=1000)
    clf = GridSearchCV(lr, params, cv=3, n_jobs=4, scoring="r2")
    clf.fit(X_train, y_train)

    best = clf.best_estimator_
    print(f"Best params: {clf.best_params_}")
    print(f"Best CV R2: {clf.best_score_:.4f}")

    model_path = os.path.join(MODEL_DIR, "student_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best, f)

    val_path = os.path.join(DATA_DIR, "val_data.pkl")
    with open(val_path, "wb") as f:
        pickle.dump({"X_val": X_val, "y_val": y_val}, f)

    print(f"Model saved: {model_path}")
    return model_path


def validate_model():
    model_path = os.path.join(MODEL_DIR, "student_model.pkl")
    val_path = os.path.join(DATA_DIR, "val_data.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(val_path, "rb") as f:
        val_data = pickle.load(f)

    X_val = val_data["X_val"]
    y_val = val_data["y_val"]

    y_pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    metrics = {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4),
        "timestamp": datetime.now().isoformat(),
        "model_params": {
            "alpha": float(model.alpha),
            "l1_ratio": float(model.l1_ratio),
            "penalty": str(model.penalty),
            "loss": str(model.loss),
            "fit_intercept": bool(model.fit_intercept),
        },
    }

    metrics_path = os.path.join(METRICS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== VALIDATION RESULTS ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    return metrics


def save_artifacts():
    metrics_path = os.path.join(METRICS_DIR, "metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    report_path = os.path.join(AIRFLOW_HOME, "report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("  STUDENT PERFORMANCE - ML PIPELINE REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {metrics['timestamp']}\n")
        f.write(f"Model:     SGDRegressor (GridSearchCV)\n")
        f.write(f"Dataset:   Student Performance (UCI)\n\n")
        f.write(f"--- Best params ---\n")
        for k, v in metrics["model_params"].items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n--- Metrics ---\n")
        f.write(f"  RMSE: {metrics['rmse']}\n")
        f.write(f"  MAE:  {metrics['mae']}\n")
        f.write(f"  R2:   {metrics['r2']}\n")
        f.write("=" * 50 + "\n")

    print(f"Report: {report_path}")
    print(">>> PIPELINE COMPLETED SUCCESSFULLY <<<")


default_args = {
    "owner": "qwart",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    dag_id="student_performance_pipeline",
    default_args=default_args,
    description="ML Pipeline: Student Performance Prediction",
    schedule=timedelta(minutes=5),
    start_date=datetime(2025, 2, 3),
    max_active_runs=1,
    catchup=False,
    tags=["ml", "student", "lab"],
)

t1 = PythonOperator(task_id="download_data", python_callable=download_data, dag=dag)
t2 = PythonOperator(task_id="clear_data", python_callable=clear_data, dag=dag)
t3 = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)
t4 = PythonOperator(task_id="validate_model", python_callable=validate_model, dag=dag)
t5 = PythonOperator(task_id="save_artifacts", python_callable=save_artifacts, dag=dag)

t1 >> t2 >> t3 >> t4 >> t5
