# %% [markdown]
# # Federated Learning Server (Revisi)
#
# Perubahan dibanding versi sebelumnya, sesuai hasil review:
#
# | Poin review | Perubahan |
# |---|---|
# | (1) DP vs epsilon-delta | Deskripsi scenario memakai istilah *update perturbation*. Key `STATIC` / `DP` / `ADP` dipertahankan agar database dan frontend tidak rusak. |
# | (2) Warm-start | **Dihapus.** Global model diinisialisasi dengan bobot acak (seed diatur `--seed`). |
# | (4) Validation vs test | Monitoring per ronde dan pemilihan model terbaik memakai **validation**. Threshold sweep di **validation**. Test dievaluasi **sekali** di akhir run. |
# | (5) Multiple seeds | Argumen `--seed`; hasil tiap run disimpan ke `results/summary_<SCENARIO>_seed<SEED>.json` untuk diagregasi `run_experiments.py`. |
#
# Perbaikan teknis tambahan:
# - Fungsi `export_training_history_to_csv` sebelumnya dipanggil tetapi tidak pernah didefinisikan — sekarang ada.
# - Pemilihan *best model* sebelumnya memakai F1 pada test set (bocor). Sekarang memakai validation.
# - `threading.Lock` pada `/send_update` agar dua client yang mengirim bersamaan tidak memicu agregasi ganda.
# - Model evaluasi dibuat sekali dan dipakai ulang (tidak memanggil `clear_session()` di tengah training).
#
# Cara menjalankan:
# ```
# python federated_server.py --scenario ADP --seed 1 --rounds 50
# ```
# Atau jalankan notebook ini dari atas ke bawah (argumen memakai default).

# %%
# pip install flask flask-cors flask-sqlalchemy psycopg2-binary tensorflow pandas scikit-learn openpyxl

# %%
# ============================================================
# FEDERATED LEARNING SERVER
# Privacy-Preserving Federated Learning with Adaptive Update Perturbation
# Diabetes Prediction Simulation
# ============================================================

import os
import json
import base64
import argparse
import random
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, jsonify, request, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy import text

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

# %%
# ============================================================
# 1. ARGUMENT CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(description="Federated Learning Server")

parser.add_argument("--scenario", type=str, default="STATIC", choices=["STATIC", "DP", "ADP"],
                    help="STATIC (no perturbation), DP (fixed update perturbation), ADP (adaptive update perturbation)")
parser.add_argument("--rounds", type=int, default=50, help="Total number of federated communication rounds")
parser.add_argument("--min_clients", type=int, default=2, help="Minimum number of clients per aggregation round")
parser.add_argument("--seed", type=int, default=42, help="Run seed: weight initialization and server-side randomness")
parser.add_argument("--monitor_threshold", type=float, default=0.5,
                    help="Threshold used for per-round validation monitoring and best-round selection")
parser.add_argument("--fixed_threshold", type=float, default=None,
                    help="If set, skip the validation threshold sweep and use this threshold for the final test "
                         "(use this to apply one common threshold across all scenarios)")
parser.add_argument("--holdout", type=str, default="server_holdout.npz", help="Validation/test archive")
parser.add_argument("--metadata", type=str, default="experiment_metadata.json", help="Experiment metadata")
parser.add_argument("--db_uri", type=str,
                    default=os.environ.get("FL_DB_URI", "postgresql://postgres:YOUR_PASSWORD@localhost:5432/federated_db"),
                    help="SQLAlchemy database URI (prefer setting the FL_DB_URI environment variable)")
parser.add_argument("--save_round_models", action="store_true", help="Save a .keras file for every round")
parser.add_argument("--exit_on_finish", action="store_true",
                    help="Shut the server down shortly after the last round (used by run_experiments.py)")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=5000)

# parse_known_args(): Jupyter/IPython injects extra arguments such as "-f kernel.json"
args, unknown = parser.parse_known_args()
if unknown:
    print(f"Ignored Jupyter/IPython arguments: {unknown}")

# %%
# ============================================================
# 2. REPRODUCIBILITY CONFIGURATION
# ============================================================

SEED = int(args.seed)

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %%
# ============================================================
# 3. GLOBAL EXPERIMENT CONFIGURATION
# ============================================================

SCENARIO_MODE = args.scenario
TOTAL_ROUNDS = int(args.rounds)
MIN_CLIENTS = int(args.min_clients)
MONITOR_THRESHOLD = float(args.monitor_threshold)

RUN_ID = f"{SCENARIO_MODE}_seed{SEED}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

MODEL_DIR = "models"
RESULT_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Scenario keys are kept for database/frontend compatibility.
# Paper names: STATIC = no perturbation, DP = fixed update perturbation, ADP = adaptive update perturbation.
# None of these scenarios carries a formal (epsilon, delta) guarantee; no privacy accounting is performed.
SCENARIO_CONFIG = {
    "STATIC": {
        "noise": 0.0,
        "clip": 100.0,
        "adaptive": False,
        "paper_name": "No perturbation (baseline)",
        "description": "Baseline without update clipping noise",
    },
    "DP": {
        "noise": 1.0,
        "clip": 1.5,
        "adaptive": False,
        "paper_name": "Fixed update perturbation",
        "description": "Fixed-norm clipping of client updates with Gaussian noise (no formal DP accounting)",
    },
    "ADP": {
        "noise": 1.0,
        "clip": 1.0,
        "adaptive": True,
        "paper_name": "Adaptive update perturbation",
        "description": "Quantile-based adaptive clipping of client updates with Gaussian noise (no formal DP accounting)",
    },
}

CURRENT_CONFIG = SCENARIO_CONFIG[SCENARIO_MODE]
NOISE_MULTIPLIER = float(CURRENT_CONFIG["noise"])
current_clip = float(CURRENT_CONFIG["clip"])

# %%
# ============================================================
# 4. FLASK AND DATABASE SETUP
# ============================================================

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "ngrok-skip-browser-warning"],
    }}
)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, ngrok-skip-browser-warning"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response


app.config["SQLALCHEMY_DATABASE_URI"] = args.db_uri
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(100))
    role = db.Column(db.String(20))  # admin | rs
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class LoginSession(db.Model):
    __tablename__ = "login_sessions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    login_time = db.Column(db.DateTime, default=datetime.utcnow)
    logout_time = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)



class TrainingHistory(db.Model):
    __tablename__ = "training_history"

    id = db.Column(db.Integer, primary_key=True)

    run_id = db.Column(db.String(100), nullable=False)
    scenario = db.Column(db.String(20), nullable=False)
    seed = db.Column(db.Integer, nullable=True)

    # entity = new naming, rs_name = old frontend naming
    entity = db.Column(db.String(50), nullable=False)
    rs_name = db.Column(db.String(50), nullable=True)

    round = db.Column(db.Integer, nullable=False)

    # Which data the metrics were computed on:
    # "client_local" (client's own real data), "validation", or "test"
    eval_split = db.Column(db.String(20), nullable=True)
    threshold = db.Column(db.Float, nullable=True)

    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1 = db.Column(db.Float, nullable=True)
    loss = db.Column(db.Float, nullable=True)

    n_samples = db.Column(db.Integer, nullable=True)
    update_norm = db.Column(db.Float, nullable=True)

    clip_threshold = db.Column(db.Float, nullable=True)
    clip = db.Column(db.Float, nullable=True)

    clip_scale = db.Column(db.Float, nullable=True)
    noise_multiplier = db.Column(db.Float, nullable=True)
    noise_std = db.Column(db.Float, nullable=True)

    tn = db.Column(db.Integer, nullable=True)
    fp = db.Column(db.Integer, nullable=True)
    fn = db.Column(db.Integer, nullable=True)
    tp = db.Column(db.Integer, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()
    DB_DIALECT = db.engine.dialect.name

    # db.create_all() does not alter existing tables; add new columns on PostgreSQL.
    if DB_DIALECT == "postgresql":
        migration_sql = [
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS run_id VARCHAR(100);",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS scenario VARCHAR(20);",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS seed INTEGER;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS entity VARCHAR(50);",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS rs_name VARCHAR(50);",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS round INTEGER;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS eval_split VARCHAR(20);",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS threshold DOUBLE PRECISION;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS accuracy DOUBLE PRECISION;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS precision DOUBLE PRECISION;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS recall DOUBLE PRECISION;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS f1 DOUBLE PRECISION;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS loss DOUBLE PRECISION;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS n_samples INTEGER;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS update_norm DOUBLE PRECISION;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS clip_threshold DOUBLE PRECISION;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS clip DOUBLE PRECISION;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS clip_scale DOUBLE PRECISION;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS noise_multiplier DOUBLE PRECISION;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS noise_std DOUBLE PRECISION;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS tn INTEGER;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS fp INTEGER;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS fn INTEGER;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS tp INTEGER;",
            "ALTER TABLE training_history ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;",
        ]
        for sql in migration_sql:
            db.session.execute(text(sql))

        db.session.execute(text(
            "UPDATE training_history SET entity = rs_name WHERE entity IS NULL AND rs_name IS NOT NULL;"))
        db.session.execute(text(
            "UPDATE training_history SET rs_name = entity WHERE rs_name IS NULL AND entity IS NOT NULL;"))
        db.session.execute(text(
            "UPDATE training_history SET clip_threshold = clip WHERE clip_threshold IS NULL AND clip IS NOT NULL;"))
        db.session.execute(text(
            "UPDATE training_history SET clip = clip_threshold WHERE clip IS NULL AND clip_threshold IS NOT NULL;"))
        db.session.commit()

print("=" * 80)
print(f"DATABASE READY ({DB_DIALECT}) | table: training_history")
print("=" * 80)

# %%
# ============================================================
# 5. LOAD EXPERIMENT ARTIFACTS
# ============================================================

for file_path in [args.metadata, args.holdout]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}. Run the data preparation notebook first.")

with open(args.metadata, "r") as f:
    experiment_metadata = json.load(f)

holdout = np.load(args.holdout, allow_pickle=False)

X_val = holdout["X_val"].astype("float32")
y_val = holdout["y_val"].astype("int32")
X_test = holdout["X_test"].astype("float32")
y_test = holdout["y_test"].astype("int32")

INPUT_DIM = int(holdout["input_shape"][0])

if INPUT_DIM != int(experiment_metadata["engineered_feature_count"]):
    raise ValueError("Input dimension mismatch between server_holdout.npz and experiment_metadata.json")

print("=" * 80)
print("FEDERATED SERVER INITIALIZATION")
print("=" * 80)
print(f"Scenario: {SCENARIO_MODE} ({CURRENT_CONFIG['paper_name']})")
print(f"Description: {CURRENT_CONFIG['description']}")
print(f"Run ID: {RUN_ID}")
print(f"Seed: {SEED}")
print(f"Total rounds: {TOTAL_ROUNDS}")
print(f"Minimum clients per round: {MIN_CLIENTS}")
print(f"Initial clip threshold: {current_clip}")
print(f"Noise multiplier: {NOISE_MULTIPLIER}")
print(f"Input dimension: {INPUT_DIM}")
print(f"Validation set: {X_val.shape} | Test set: {X_test.shape}")
print(f"SMOTE policy: {experiment_metadata['smote_policy']}")
print(f"Initialization: {experiment_metadata.get('initialization_policy', 'random')}")
print("=" * 80)

# %%
# ============================================================
# 6. MODEL ARCHITECTURE AND RANDOM INITIALIZATION
# ============================================================

def create_model(input_dim=INPUT_DIM):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


tf.keras.backend.clear_session()
tf.random.set_seed(SEED)

# Random initialization (Glorot uniform for kernels, zeros for biases), controlled by --seed.
# No centralized training on raw data happens before federated training.
global_model = create_model(INPUT_DIM)
global_weights = [w.copy() for w in global_model.get_weights()]

# A second model used only for evaluation, built once and reused.
eval_model = create_model(INPUT_DIM)

initial_model_path = os.path.join(MODEL_DIR, f"initial_random_model_{RUN_ID}.keras")
global_model.save(initial_model_path)
print(f"Global model randomly initialized (seed={SEED}). Saved to: {initial_model_path}")

# %%
# ============================================================
# 7. FEDERATED SERVER STATE
# ============================================================

current_round = 1
client_updates_buffer = {}

best_val_f1 = -1.0
best_round = 0
best_weights = [w.copy() for w in global_weights]
best_model_path = None

final_summary = None
state_lock = threading.Lock()

# %%
# ============================================================
# 8. WEIGHT SERIALIZATION UTILITIES
# ============================================================

def serialize_weights(weights_list):
    serialized = []
    for w in weights_list:
        w = np.ascontiguousarray(w)
        serialized.append({
            "shape": list(w.shape),
            "dtype": str(w.dtype),
            "data": base64.b64encode(w.tobytes()).decode("utf-8"),
        })
    return json.dumps(serialized)


def deserialize_weights(json_str):
    weights_list = []
    for item in json.loads(json_str):
        w = np.frombuffer(base64.b64decode(item["data"]), dtype=np.dtype(item["dtype"]))
        weights_list.append(w.reshape(item["shape"]).copy())
    return weights_list

# %%
# ============================================================
# 9. FEDERATED AVERAGING, EVALUATION, THRESHOLD SELECTION
# ============================================================

def weighted_federated_average(client_payloads):
    total_samples = sum(item["n_samples"] for item in client_payloads)
    if total_samples <= 0:
        raise ValueError("Total client samples must be greater than zero.")

    averaged_weights = []
    for layer_weights in zip(*[item["weights"] for item in client_payloads]):
        weighted_layer = np.zeros_like(layer_weights[0], dtype=np.float32)
        for weights_array, item in zip(layer_weights, client_payloads):
            weighted_layer += weights_array.astype(np.float32) * (item["n_samples"] / total_samples)
        averaged_weights.append(weighted_layer)
    return averaged_weights


def predict_proba(weights, X):
    eval_model.set_weights(weights)
    return eval_model.predict(X, verbose=0, batch_size=4096).reshape(-1)


def metrics_from_proba(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_sweep(y_true, y_prob, start=0.05, stop=0.95, step=0.01):
    rows = []
    for t in np.round(np.arange(start, stop + 1e-9, step), 2):
        m = metrics_from_proba(y_true, y_prob, t)
        rows.append({k: m[k] for k in ["threshold", "accuracy", "precision", "recall", "f1"]})
    return pd.DataFrame(rows)


def select_threshold(sweep_df):
    """Maximum validation F1; ties are broken towards 0.50."""
    best_f1 = sweep_df["f1"].max()
    candidates = sweep_df[np.isclose(sweep_df["f1"], best_f1)]
    idx = (candidates["threshold"] - 0.5).abs().idxmin()
    return float(sweep_df.loc[idx, "threshold"])


def update_adaptive_clip(norm_values, previous_clip):
    if len(norm_values) == 0:
        return previous_clip
    quantile_norm = float(np.quantile(norm_values, 0.80))
    adaptation_rate = 0.05
    updated_clip = previous_clip * np.exp(-adaptation_rate * (previous_clip - quantile_norm))
    return float(np.clip(updated_clip, 0.1, 10.0))


def save_model_file(weights, filename):
    eval_model.set_weights(weights)
    eval_model.save(filename)
    return filename

# %%
# ============================================================
# 10. DATABASE LOGGING AND EXPORT
# ============================================================

def save_history_record(entity, round_num, metrics, eval_split=None, n_samples=None, update_norm=None,
                        clip_threshold=None, clip_scale=None, noise_multiplier=None, noise_std=None):
    entity_value = str(entity)

    record = TrainingHistory.query.filter_by(
        run_id=RUN_ID, scenario=SCENARIO_MODE, entity=entity_value, round=round_num
    ).first()

    if record is None:
        record = TrainingHistory(run_id=RUN_ID, scenario=SCENARIO_MODE, entity=entity_value, round=round_num)
        db.session.add(record)

    record.seed = SEED
    record.rs_name = entity_value
    record.eval_split = eval_split
    record.threshold = metrics.get("threshold")

    record.accuracy = float(metrics.get("accuracy", 0.0))
    record.precision = float(metrics.get("precision", 0.0))
    record.recall = float(metrics.get("recall", 0.0))
    record.f1 = float(metrics.get("f1", 0.0))
    record.loss = float(metrics.get("loss", 0.0))

    record.n_samples = n_samples
    record.update_norm = update_norm
    record.clip_threshold = clip_threshold
    record.clip = clip_threshold
    record.clip_scale = clip_scale
    record.noise_multiplier = noise_multiplier
    record.noise_std = noise_std

    record.tn = metrics.get("tn")
    record.fp = metrics.get("fp")
    record.fn = metrics.get("fn")
    record.tp = metrics.get("tp")

    db.session.commit()


def export_training_history_to_csv(run_id=RUN_ID):
    query = TrainingHistory.query
    if run_id is not None:
        query = query.filter(TrainingHistory.run_id == run_id)

    records = query.order_by(TrainingHistory.round.asc(), TrainingHistory.entity.asc()).all()

    columns = ["id", "run_id", "scenario", "seed", "entity", "round", "eval_split", "threshold",
               "accuracy", "precision", "recall", "f1", "loss", "n_samples", "update_norm",
               "clip_threshold", "clip_scale", "noise_multiplier", "noise_std",
               "tn", "fp", "fn", "tp", "created_at"]

    df = pd.DataFrame([{c: getattr(r, c) for c in columns} for r in records], columns=columns)

    name = f"training_history_{run_id}.csv" if run_id else "training_history_all.csv"
    output_path = os.path.join(RESULT_DIR, name)
    df.to_csv(output_path, index=False)
    return output_path, df


def export_all_scenarios_excel(output_path="training_history_all_scenarios.xlsx"):
    """Optional: one sheet per scenario, all runs and seeds. Requires openpyxl."""
    with app.app_context():
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for s in ["STATIC", "DP", "ADP"]:
                pd.read_sql(
                    text("SELECT * FROM training_history WHERE scenario = :s ORDER BY run_id, round, entity"),
                    db.engine, params={"s": s}
                ).to_excel(writer, sheet_name=s, index=False)
    print(f"Exported: {output_path}")

# %%
# ============================================================
# 11. FINALIZATION: MODEL + THRESHOLD SELECTION ON VALIDATION, SINGLE TEST EVALUATION
# ============================================================

def finalize_run():
    global final_summary

    # 1) Selected model = round with the best validation F1 (at the monitoring threshold).
    p_val = predict_proba(best_weights, X_val)

    # 2) Threshold chosen on validation (or fixed by the user for all scenarios).
    sweep_df = threshold_sweep(y_val, p_val)
    sweep_path = os.path.join(RESULT_DIR, f"threshold_sweep_validation_{SCENARIO_MODE}_seed{SEED}.csv")
    sweep_df.to_csv(sweep_path, index=False)

    if args.fixed_threshold is not None:
        selected_threshold = float(args.fixed_threshold)
        threshold_source = "fixed (--fixed_threshold)"
    else:
        selected_threshold = select_threshold(sweep_df)
        threshold_source = "max F1 on validation"

    # 3) Single evaluation on the test set.
    p_test = predict_proba(best_weights, X_test)
    test_selected = metrics_from_proba(y_test, p_test, selected_threshold)
    test_default = metrics_from_proba(y_test, p_test, 0.5)
    ranking = {
        "average_precision": float(average_precision_score(y_test, p_test)),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
    }

    # Final-round model on test at 0.5, for transparency (not used for any selection).
    p_test_final = predict_proba(global_weights, X_test)
    test_final_round = metrics_from_proba(y_test, p_test_final, 0.5)

    save_history_record("FINAL_TEST", TOTAL_ROUNDS + 1, test_selected, eval_split="test",
                        n_samples=len(X_test), clip_threshold=current_clip, noise_multiplier=NOISE_MULTIPLIER)
    save_history_record("FINAL_TEST_T050", TOTAL_ROUNDS + 1, test_default, eval_split="test",
                        n_samples=len(X_test), clip_threshold=current_clip, noise_multiplier=NOISE_MULTIPLIER)

    final_summary = {
        "run_id": RUN_ID,
        "scenario": SCENARIO_MODE,
        "scenario_paper_name": CURRENT_CONFIG["paper_name"],
        "seed": SEED,
        "total_rounds": TOTAL_ROUNDS,
        "noise_multiplier": NOISE_MULTIPLIER,
        "initial_clip": float(CURRENT_CONFIG["clip"]),
        "final_clip": float(current_clip),
        "selection": {
            "best_round_by_validation_f1": int(best_round),
            "best_validation_f1_at_monitor_threshold": float(best_val_f1),
            "monitor_threshold": MONITOR_THRESHOLD,
            "selected_threshold": selected_threshold,
            "threshold_source": threshold_source,
        },
        "test_at_selected_threshold": test_selected,
        "test_at_0.50": test_default,
        "test_ranking_metrics": ranking,
        "test_final_round_model_at_0.50": test_final_round,
        "files": {"best_model": best_model_path, "threshold_sweep": sweep_path},
    }

    summary_path = os.path.join(RESULT_DIR, f"summary_{SCENARIO_MODE}_seed{SEED}.json")
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=2)

    history_path, _ = export_training_history_to_csv()

    print("=" * 80)
    print("FEDERATED TRAINING COMPLETED")
    print(f"Best round (validation F1): {best_round} | val F1 = {best_val_f1:.4f}")
    print(f"Selected threshold: {selected_threshold:.2f} ({threshold_source})")
    print(
        f"TEST @ {selected_threshold:.2f} | Acc={test_selected['accuracy']:.4f}, "
        f"P={test_selected['precision']:.4f}, R={test_selected['recall']:.4f}, "
        f"F1={test_selected['f1']:.4f}, Loss={test_selected['loss']:.4f}, "
        f"AP={ranking['average_precision']:.4f}"
    )
    print(f"Summary: {summary_path}")
    print(f"History: {history_path}")
    print("=" * 80)

    if args.exit_on_finish:
        # Give clients a few polling cycles to receive the FINISHED status, then exit.
        threading.Timer(10.0, lambda: os._exit(0)).start()

# %%
# ============================================================
# 12. INITIAL (RANDOM) GLOBAL MODEL EVALUATION ON VALIDATION
# ============================================================

initial_metrics = metrics_from_proba(y_val, predict_proba(global_weights, X_val), MONITOR_THRESHOLD)

print(
    f"Round 0 (random init) on validation | Acc={initial_metrics['accuracy']:.4f}, "
    f"F1={initial_metrics['f1']:.4f}, Loss={initial_metrics['loss']:.4f}"
)

with app.app_context():
    save_history_record("INITIAL_GLOBAL", 0, initial_metrics, eval_split="validation",
                        n_samples=len(X_val), clip_threshold=current_clip, noise_multiplier=NOISE_MULTIPLIER)

# %%
# ============================================================
# 13. FEDERATED ROUTES
# ============================================================

def finished_payload():
    return {"status": "FINISHED", "scenario": SCENARIO_MODE, "run_id": RUN_ID, "seed": SEED}


@app.route("/")
def index():
    return jsonify({
        "message": "Federated Learning Server is running.",
        "scenario": SCENARIO_MODE, "run_id": RUN_ID, "seed": SEED,
        "current_round": current_round, "total_rounds": TOTAL_ROUNDS,
        "min_clients": MIN_CLIENTS, "received_clients": list(client_updates_buffer.keys()),
        "clip_threshold": current_clip, "noise_multiplier": NOISE_MULTIPLIER,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "scenario": SCENARIO_MODE, "run_id": RUN_ID, "seed": SEED,
                    "current_round": current_round})


@app.route("/get_model_status")
def get_model_status():
    if current_round > TOTAL_ROUNDS:
        return jsonify(finished_payload())
    return jsonify({
        "status": "TRAINING", "scenario": SCENARIO_MODE, "run_id": RUN_ID, "seed": SEED,
        "round": current_round, "total_rounds": TOTAL_ROUNDS,
        "clip_threshold": float(current_clip), "noise_multiplier": float(NOISE_MULTIPLIER),
        "received_clients": list(client_updates_buffer.keys()), "required_clients": MIN_CLIENTS,
    })


@app.route("/get_model")
def get_model():
    with state_lock:
        if current_round > TOTAL_ROUNDS:
            return jsonify(finished_payload())
        return jsonify({
            "status": "TRAINING", "scenario": SCENARIO_MODE, "run_id": RUN_ID, "seed": SEED,
            "round": current_round, "total_rounds": TOTAL_ROUNDS,
            "weights": serialize_weights(global_weights),
            "clip_threshold": float(current_clip), "noise_multiplier": float(NOISE_MULTIPLIER),
            "required_clients": MIN_CLIENTS,
        })


@app.route("/send_update", methods=["POST"])
def receive_update():
    global current_round, global_weights, current_clip
    global best_val_f1, best_round, best_weights, best_model_path

    with state_lock:
        if current_round > TOTAL_ROUNDS:
            return jsonify(finished_payload())

        payload = request.json or {}
        client_id = payload.get("client_id", "UNKNOWN")
        client_round = int(payload.get("round", -1))
        client_scenario = payload.get("scenario", SCENARIO_MODE)

        if client_round != current_round:
            return jsonify({"status": "rejected", "reason": "round_mismatch",
                            "server_round": current_round, "client_round": client_round}), 409

        if client_scenario != SCENARIO_MODE:
            return jsonify({"status": "rejected", "reason": "scenario_mismatch",
                            "server_scenario": SCENARIO_MODE, "client_scenario": client_scenario}), 409

        if client_id in client_updates_buffer:
            return jsonify({"status": "duplicate",
                            "message": f"Update from {client_id} for round {current_round} already received."}), 200

        try:
            decoded_weights = deserialize_weights(payload["weights"])
            checksum = sum(float(np.sum(w)) for w in decoded_weights)
            if np.isnan(checksum) or np.isinf(checksum):
                return jsonify({"status": "rejected", "reason": "invalid_weight_values"}), 400

            client_metrics = payload.get("metrics", {})
            client_record = {
                "client_id": client_id,
                "n_samples": int(payload.get("n_samples", 1)),
                "weights": decoded_weights,
                "norm": float(payload.get("norm", 0.0)),
                "clip_threshold": float(payload.get("clip_threshold", current_clip)),
                "clip_scale": float(payload.get("clip_scale", 1.0)),
                "noise_multiplier": float(payload.get("noise_multiplier", NOISE_MULTIPLIER)),
                "noise_std": float(payload.get("noise_std", 0.0)),
            }
            client_updates_buffer[client_id] = client_record

            save_history_record(
                client_id, current_round, client_metrics, eval_split="client_local",
                n_samples=client_record["n_samples"], update_norm=client_record["norm"],
                clip_threshold=client_record["clip_threshold"], clip_scale=client_record["clip_scale"],
                noise_multiplier=client_record["noise_multiplier"], noise_std=client_record["noise_std"],
            )

            print(
                f"Received update | Client={client_id}, Round={current_round}, "
                f"Buffer={len(client_updates_buffer)}/{MIN_CLIENTS}, Norm={client_record['norm']:.6f}, "
                f"Clip={client_record['clip_threshold']:.6f}, Noise={client_record['noise_multiplier']:.4f}"
            )

        except Exception as e:
            db.session.rollback()
            print(f"Failed to process client update from {client_id}: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

        if len(client_updates_buffer) >= MIN_CLIENTS:
            print("=" * 80)
            print(f"AGGREGATING ROUND {current_round} | Scenario: {SCENARIO_MODE} | Seed: {SEED}")
            print("=" * 80)

            try:
                round_clip_used = float(current_clip)
                client_payloads = list(client_updates_buffer.values())

                global_weights = weighted_federated_average(client_payloads)
                global_model.set_weights(global_weights)

                val_metrics = metrics_from_proba(y_val, predict_proba(global_weights, X_val), MONITOR_THRESHOLD)

                save_history_record(
                    "AGGREGATED", current_round, val_metrics, eval_split="validation",
                    n_samples=len(X_val), clip_threshold=round_clip_used, noise_multiplier=NOISE_MULTIPLIER,
                )

                print(
                    f"Validation | Acc={val_metrics['accuracy']:.4f}, P={val_metrics['precision']:.4f}, "
                    f"R={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}, Loss={val_metrics['loss']:.4f}"
                )

                if args.save_round_models:
                    save_model_file(global_weights, os.path.join(
                        MODEL_DIR, f"model_round_{current_round:03d}_{RUN_ID}.keras"))

                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    best_round = current_round
                    best_weights = [w.copy() for w in global_weights]
                    best_model_path = save_model_file(best_weights, os.path.join(
                        MODEL_DIR, f"best_model_{RUN_ID}.keras"))
                    print(f"New best validation F1: {best_val_f1:.4f} (round {best_round})")

                if CURRENT_CONFIG["adaptive"]:
                    norm_values = [item["norm"] for item in client_payloads]
                    updated_clip = update_adaptive_clip(norm_values, current_clip)
                    print(f"Adaptive clip | Previous={current_clip:.6f}, Updated={updated_clip:.6f}, "
                          f"Client norms={[round(n, 6) for n in norm_values]}")
                    current_clip = updated_clip

                client_updates_buffer.clear()
                completed_round = current_round
                current_round += 1

                if current_round > TOTAL_ROUNDS:
                    finalize_run()
                else:
                    print(f"Round {completed_round} completed. Next round: {current_round}")

            except Exception as e:
                db.session.rollback()
                print(f"Aggregation failed: {e}")
                return jsonify({"status": "error", "message": f"Aggregation failed: {e}"}), 500

        return jsonify({"status": "ok", "server_round": current_round,
                        "received_clients": list(client_updates_buffer.keys())})


@app.route("/results/summary")
def results_summary_route():
    if final_summary is None:
        return jsonify({"status": "running", "current_round": current_round, "total_rounds": TOTAL_ROUNDS})
    return jsonify({"status": "ok", "summary": final_summary})


@app.route("/export/training-csv")
def export_training_csv_route():
    try:
        output_path, _ = export_training_history_to_csv()
        return send_file(os.path.abspath(output_path), as_attachment=True,
                         download_name=os.path.basename(output_path))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/export/training-json")
def export_training_json_route():
    try:
        output_path, df = export_training_history_to_csv()
        return jsonify({"status": "ok", "csv_path": output_path,
                        "records": json.loads(df.to_json(orient="records", date_format="iso"))})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# %%
# ============================================================
# 13B. FRONTEND COMPATIBILITY ROUTES
# ============================================================

def fmt_time(value):
    return value.strftime("%Y-%m-%d %H:%M:%S") if value else None


def record_to_dict(log):
    return {
        "id": log.id, "run_id": log.run_id, "scenario": log.scenario, "seed": log.seed,
        "entity": log.entity, "rs_name": log.rs_name or log.entity, "round": log.round,
        "eval_split": log.eval_split, "threshold": log.threshold,
        "accuracy": log.accuracy, "precision": log.precision, "recall": log.recall,
        "f1": log.f1, "loss": log.loss, "n_samples": log.n_samples,
        "update_norm": log.update_norm, "clip_threshold": log.clip_threshold,
        "clip": log.clip or log.clip_threshold, "clip_scale": log.clip_scale,
        "noise_multiplier": log.noise_multiplier, "noise_std": log.noise_std,
        "tn": log.tn, "fp": log.fp, "fn": log.fn, "tp": log.tp,
        "created_at": fmt_time(log.created_at),
    }




# ==========================================
# AUTHENTICATION & USER MANAGEMENT
# ==========================================
@app.route("/auth/login", methods=["POST", "OPTIONS"])
def auth_login():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    data = request.json or {}
    username = data.get("username", "")
    password = data.get("password", "")
    user = User.query.filter_by(username=username, password=password).first()
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401

    session = LoginSession(user_id=user.id)
    try:
        db.session.add(session)
        db.session.commit()
    except Exception:
        db.session.rollback()

    return jsonify({
        "id": user.id,
        "user_id": user.id,
        "username": user.username,
        "role": user.role,
        "login_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/auth/logout", methods=["POST", "OPTIONS"])
def auth_logout():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    data = request.json or {}
    user_id = data.get("user_id")
    if user_id:
        session = LoginSession.query.filter_by(user_id=user_id, is_active=True).first()
        if session:
            session.is_active = False
            session.logout_time = datetime.utcnow()
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()
    return jsonify({"message": "Logged out"})

@app.route("/auth/profile", methods=["GET", "OPTIONS"])
def auth_profile():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    user_id = request.args.get("user_id")
    user = User.query.get(user_id) if user_id else None
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({
        "id": user.id,
        "user_id": user.id,
        "username": user.username,
        "role": user.role
    })

@app.route("/admin/users", methods=["GET", "OPTIONS"])
def admin_get_users():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    users = User.query.all()
    return jsonify([{
        "id": u.id,
        "username": u.username,
        "role": u.role
    } for u in users])

@app.route("/heartbeat", methods=["GET", "POST", "OPTIONS"])
def heartbeat():
    return jsonify({
        "status": "ok", "server": "online", "scenario": SCENARIO_MODE, "run_id": RUN_ID, "seed": SEED,
        "current_round": current_round, "total_rounds": TOTAL_ROUNDS,
        "received_clients": list(client_updates_buffer.keys()), "required_clients": MIN_CLIENTS,
        "clip_threshold": float(current_clip), "noise_multiplier": float(NOISE_MULTIPLIER),
    })


@app.route("/dashboard/active-users", methods=["GET", "OPTIONS"])
def dashboard_active_users():
    return jsonify({
        "status": "ok", "active_users": len(client_updates_buffer),
        "received_clients": list(client_updates_buffer.keys()), "required_clients": MIN_CLIENTS,
        "current_round": current_round, "total_rounds": TOTAL_ROUNDS,
        "scenario": SCENARIO_MODE, "run_id": RUN_ID,
    })


@app.route("/training/history", methods=["GET", "OPTIONS"])
def training_history_legacy():
    try:
        query = TrainingHistory.query
        for field, column in [("scenario", TrainingHistory.scenario), ("entity", TrainingHistory.entity),
                              ("run_id", TrainingHistory.run_id)]:
            value = request.args.get(field)
            if value and value != "ALL":
                query = query.filter(column == value)
        seed_filter = request.args.get("seed")
        if seed_filter and seed_filter != "ALL":
            query = query.filter(TrainingHistory.seed == int(seed_filter))

        rows = [record_to_dict(log) for log in
                query.order_by(TrainingHistory.created_at.asc(), TrainingHistory.round.asc()).all()]
        return jsonify({"status": "ok", "count": len(rows), "history": rows, "records": rows})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/training/runs", methods=["GET", "OPTIONS"])
def training_runs():
    try:
        records = (
            db.session.query(
                TrainingHistory.run_id, TrainingHistory.scenario, TrainingHistory.seed,
                db.func.min(TrainingHistory.created_at).label("started_at"),
                db.func.max(TrainingHistory.created_at).label("ended_at"),
                db.func.max(TrainingHistory.round).label("max_round"),
            )
            .group_by(TrainingHistory.run_id, TrainingHistory.scenario, TrainingHistory.seed)
            .order_by(db.func.max(TrainingHistory.created_at).desc())
            .all()
        )
        rows = [{"run_id": r.run_id, "scenario": r.scenario, "seed": r.seed,
                 "started_at": fmt_time(r.started_at), "ended_at": fmt_time(r.ended_at),
                 "max_round": int(r.max_round or 0)} for r in records]
        return jsonify({"status": "ok", "count": len(rows), "runs": rows})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def aggregated_summary_dict(record):
    return {
        "scenario": record.scenario, "seed": record.seed, "round": record.round,
        "eval_split": record.eval_split, "accuracy": record.accuracy, "precision": record.precision,
        "recall": record.recall, "f1": record.f1, "loss": record.loss,
        "clip_threshold": record.clip_threshold, "created_at": fmt_time(record.created_at),
    }


@app.route("/dashboard/summary", methods=["GET", "OPTIONS"])
def dashboard_summary():
    try:
        aggregated_records = (TrainingHistory.query.filter(TrainingHistory.entity == "AGGREGATED")
                              .order_by(TrainingHistory.created_at.desc()).all())
        latest = aggregated_records[0] if aggregated_records else None

        scenario_latest = {}
        for record in aggregated_records:
            if record.scenario not in scenario_latest:
                scenario_latest[record.scenario] = aggregated_summary_dict(record)

        return jsonify({
            "status": "ok", "server": "online", "run_id": RUN_ID, "scenario": SCENARIO_MODE, "seed": SEED,
            "current_round": current_round, "total_rounds": TOTAL_ROUNDS,
            "active_users": len(client_updates_buffer), "received_clients": list(client_updates_buffer.keys()),
            "required_clients": MIN_CLIENTS, "scenarios_run_today": len(scenario_latest),
            "latest": None if latest is None else aggregated_summary_dict(latest),
            "latest_by_scenario": scenario_latest,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/dashboard/scenario-comparison", methods=["GET", "OPTIONS"])
def dashboard_scenario_comparison():
    try:
        records = (TrainingHistory.query.filter(TrainingHistory.entity == "AGGREGATED")
                   .order_by(TrainingHistory.scenario.asc(), TrainingHistory.round.asc(),
                             TrainingHistory.created_at.asc()).all())
        rows = []
        for log in records:
            row = aggregated_summary_dict(log)
            row.update({"entity": log.entity, "rs_name": log.entity, "run_id": log.run_id})
            rows.append(row)
        return jsonify({"status": "ok", "count": len(rows), "records": rows})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/dashboard/latest-aggregated", methods=["GET", "OPTIONS"])
def dashboard_latest_aggregated():
    try:
        log = (TrainingHistory.query.filter(TrainingHistory.entity == "AGGREGATED")
               .order_by(TrainingHistory.created_at.desc()).first())
        if log is None:
            return jsonify({"status": "empty", "message": "No aggregated record found yet."})
        return jsonify({"status": "ok", "record": record_to_dict(log)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# %%
# ============================================================
# 14. START SERVER
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STARTING FLASK FEDERATED SERVER")
    print(f"Server URL: http://{args.host}:{args.port} | Scenario: {SCENARIO_MODE} | Seed: {SEED}")
    print("=" * 80)
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
