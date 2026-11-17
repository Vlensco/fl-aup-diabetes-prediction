import argparse
import base64
import json
import os
import random
import time

import numpy as np
import requests
import tensorflow as tf

from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, log_loss


# ============================================================
# CLIENT CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(description="Federated Learning Hospital Client")
parser.add_argument("--id", type=str, default="RS_01", help="Client ID: RS_01 or RS_02")
parser.add_argument("--url", type=str, default="http://127.0.0.1:5000", help="Federated server URL")
parser.add_argument("--data", type=str, default=None,
                    help="Path to this client's archive (default: client_<ID>.npz)")
parser.add_argument("--epochs", type=int, default=1, help="Local training epochs per round")
parser.add_argument("--batch", type=int, default=64, help="Local batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Local learning rate")
parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for local metrics")
parser.add_argument("--seed", type=int, default=42, help="Run seed (must match the server --seed)")
parser.add_argument("--no_smote", action="store_true", help="Disable local SMOTE (ablation)")
parser.add_argument("--smote_k", type=int, default=5, help="SMOTE k_neighbors")
parser.add_argument("--max_wait", type=int, default=120,
                    help="Stop after this many seconds without a server response once training has started")

args = parser.parse_args()

SERVER_URL = args.url.strip().rstrip("/")
CLIENT_ID = args.id.strip()
FILE_DATA_NPZ = args.data or f"client_{CLIENT_ID}.npz"

LOCAL_EPOCHS = args.epochs
BATCH_SIZE = args.batch
LEARNING_RATE = args.lr
DECISION_THRESHOLD = args.threshold
RUN_SEED = int(args.seed)

VALID_CLIENTS = {"RS_01": 1, "RS_02": 2}

if CLIENT_ID not in VALID_CLIENTS:
    raise ValueError(f"Invalid client ID: {CLIENT_ID}. Use RS_01 or RS_02.")

# Each client gets its own deterministic stream derived from the run seed,
# so the two hospitals do not share random numbers but the whole run is reproducible.
CLIENT_SEED = RUN_SEED * 100 + VALID_CLIENTS[CLIENT_ID]

os.environ["PYTHONHASHSEED"] = str(CLIENT_SEED)
random.seed(CLIENT_SEED)
np.random.seed(CLIENT_SEED)
tf.random.set_seed(CLIENT_SEED)
noise_rng = np.random.default_rng(CLIENT_SEED)


# ============================================================
# DATA LOADING AND LOCAL SMOTE
# ============================================================

def print_distribution(title, y):
    unique, counts = np.unique(y.astype(int), return_counts=True)
    print(title)
    for cls, count in zip(unique, counts):
        label = "Non-Diabetic" if cls == 0 else "Diabetic"
        print(f"- Class {cls} ({label}): {count:,} ({count / len(y) * 100:.2f}%)")


def load_local_data():
    if not os.path.exists(FILE_DATA_NPZ):
        raise FileNotFoundError(f"Data file not found: {FILE_DATA_NPZ}")

    archive = np.load(FILE_DATA_NPZ, allow_pickle=False)
    X_real = archive["X"].astype("float32")
    y_real = archive["y"].astype("int32")
    input_dim = int(archive["input_shape"][0])

    print(f"Data loaded for {CLIENT_ID} from {FILE_DATA_NPZ}")
    print(f"Real local samples: {len(X_real):,} | Input dimension: {input_dim}")
    print_distribution("Local class distribution (before SMOTE):", y_real)

    if args.no_smote:
        print("Local SMOTE disabled (--no_smote).")
        return X_real, y_real, X_real, y_real.astype("float32"), input_dim

    # SMOTE is fitted on this client's own data only.
    # Synthetic samples never leave the client and are never mixed with another client's data.
    smote = SMOTE(random_state=CLIENT_SEED, k_neighbors=args.smote_k)
    X_train, y_train = smote.fit_resample(X_real, y_real)
    print_distribution("Local class distribution (after local SMOTE):", y_train)

    return X_real, y_real, X_train.astype("float32"), y_train.astype("float32"), input_dim


# ============================================================
# WEIGHT SERIALIZATION
# ============================================================

def serialize_weights(weights_list):
    serialized = []
    for w in weights_list:
        w = np.ascontiguousarray(w)
        serialized.append({
            "shape": list(w.shape),
            "dtype": str(w.dtype),
            "data": base64.b64encode(w.tobytes()).decode("utf-8")
        })
    return json.dumps(serialized)


def deserialize_weights(json_str):
    weights_list = []
    for item in json.loads(json_str):
        w = np.frombuffer(base64.b64decode(item["data"]), dtype=np.dtype(item["dtype"]))
        weights_list.append(w.reshape(item["shape"]).copy())
    return weights_list


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

def build_model(input_dim: int):
    tf.keras.backend.clear_session()

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# LOCAL METRICS (computed on the client's REAL data, not on synthetic samples)
# ============================================================

def compute_local_metrics(model, X, y, threshold=0.5):
    y_prob = model.predict(X, verbose=0, batch_size=4096).reshape(-1)
    y_prob_safe = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "loss": float(log_loss(y, y_prob_safe, labels=[0, 1]))
    }


# ============================================================
# UPDATE PERTURBATION (clipping + Gaussian noise)
# ============================================================

def protect_model_update(new_weights, old_weights, clip_threshold, noise_multiplier, n_samples):
    """
    Clips the client model delta to an L2 norm of `clip_threshold` and adds Gaussian noise.

    This is an update perturbation mechanism. It does NOT by itself provide a formal
    (epsilon, delta)-differential privacy guarantee, because no privacy accounting is performed.
    """

    weight_updates = [new_w - old_w for new_w, old_w in zip(new_weights, old_weights)]

    update_norm = float(np.sqrt(sum(np.sum(np.square(u)) for u in weight_updates)))
    clip_scale = min(1.0, clip_threshold / (update_norm + 1e-12))

    if noise_multiplier > 0:
        noise_std = (noise_multiplier * clip_threshold) / max(float(n_samples), 1.0)
    else:
        noise_std = 0.0

    protected_weights = []
    for update, old_w in zip(weight_updates, old_weights):
        clipped_update = update * clip_scale
        if noise_multiplier > 0:
            noise = noise_rng.normal(0.0, noise_std, size=clipped_update.shape).astype(clipped_update.dtype)
            clipped_update = clipped_update + noise
        protected_weights.append(old_w + clipped_update)

    protection_info = {
        "update_norm": float(update_norm),
        "clip_threshold": float(clip_threshold),
        "clip_scale": float(clip_scale),
        "noise_multiplier": float(noise_multiplier),
        "noise_std": float(noise_std)
    }

    return protected_weights, protection_info


# ============================================================
# CLIENT LOOP
# ============================================================

def start_client():
    print("=" * 80)
    print(f"FEDERATED CLIENT STARTED: {CLIENT_ID}")
    print(f"Server URL: {SERVER_URL}")
    print(f"Run seed: {RUN_SEED} | Client seed: {CLIENT_SEED}")
    print(f"Local epochs per round: {LOCAL_EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print("=" * 80)

    X_real, y_real, X_train, y_train, input_dim = load_local_data()
    model = build_model(input_dim)

    last_processed_round = 0
    last_contact = None
    seed_warning_shown = False

    while True:
        try:
            try:
                response = requests.get(f"{SERVER_URL}/get_model", timeout=20)
                response.raise_for_status()
                server_payload = response.json()
                last_contact = time.time()
            except Exception:
                if last_contact is not None and time.time() - last_contact > args.max_wait:
                    print(f"\nNo server response for {args.max_wait}s. Stopping client.")
                    break
                print("\rWaiting for server connection...", end="")
                time.sleep(3)
                continue

            if server_payload.get("status") == "FINISHED":
                print("\nTraining has finished on the server.")
                break

            server_seed = server_payload.get("seed")
            if server_seed is not None and int(server_seed) != RUN_SEED and not seed_warning_shown:
                print(f"\nWARNING: server seed={server_seed} but client --seed={RUN_SEED}. "
                      f"Results will not be reproducible for this seed.")
                seed_warning_shown = True

            server_round = int(server_payload["round"])
            scenario = server_payload.get("scenario", "UNKNOWN")

            if server_round < last_processed_round:
                print("\nServer round reset detected. Restarting local round tracker.")
                last_processed_round = 0

            if server_round <= last_processed_round:
                print(f"\rWaiting for round {last_processed_round + 1}...", end="")
                time.sleep(2)
                continue

            print("\n" + "=" * 80)
            print(f"Starting round {server_round} | Scenario: {scenario}")
            print("=" * 80)

            try:
                global_weights = deserialize_weights(server_payload["weights"])
                model.set_weights(global_weights)
                original_weights = [w.copy() for w in model.get_weights()]
            except Exception as e:
                print(f"Failed to load global weights from server: {e}")
                print("Skipping this round to avoid contaminating aggregation.")
                time.sleep(3)
                continue

            try:
                model.fit(X_train, y_train, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=True)
                local_metrics = compute_local_metrics(model, X_real, y_real, threshold=DECISION_THRESHOLD)

                print(
                    "Local metrics (real local data) | "
                    f"Acc={local_metrics['accuracy']:.4f}, Precision={local_metrics['precision']:.4f}, "
                    f"Recall={local_metrics['recall']:.4f}, F1={local_metrics['f1']:.4f}, "
                    f"Loss={local_metrics['loss']:.4f}"
                )

            except Exception as e:
                print(f"Local training failed: {e}")
                print("Restoring original global weights and sending safe fallback metrics.")
                model.set_weights(original_weights)
                local_metrics = {"threshold": DECISION_THRESHOLD, "accuracy": 0.0, "precision": 0.0,
                                 "recall": 0.0, "f1": 0.0, "loss": 10.0}

            new_weights = model.get_weights()

            clip_threshold = float(server_payload.get("clip_threshold", 10.0))
            noise_multiplier = float(server_payload.get("noise_multiplier", 0.0))

            # n_samples = number of REAL local records (synthetic SMOTE samples are not counted),
            # used both for the noise scale and for the FedAvg weight on the server.
            protected_weights, protection_info = protect_model_update(
                new_weights=new_weights,
                old_weights=original_weights,
                clip_threshold=clip_threshold,
                noise_multiplier=noise_multiplier,
                n_samples=len(X_real)
            )

            print(
                "Update perturbation | "
                f"Norm={protection_info['update_norm']:.6f}, Clip={protection_info['clip_threshold']:.4f}, "
                f"Scale={protection_info['clip_scale']:.6f}, "
                f"Noise multiplier={protection_info['noise_multiplier']:.4f}, "
                f"Noise std={protection_info['noise_std']:.8f}"
            )

            payload = {
                "client_id": CLIENT_ID,
                "round": server_round,
                "scenario": scenario,
                "seed": RUN_SEED,
                "n_samples": int(len(X_real)),
                "n_train_samples_after_smote": int(len(X_train)),
                "weights": serialize_weights(protected_weights),
                "metrics": local_metrics,
                "norm": float(protection_info["update_norm"]),
                "clip_threshold": float(protection_info["clip_threshold"]),
                "clip_scale": float(protection_info["clip_scale"]),
                "noise_multiplier": float(protection_info["noise_multiplier"]),
                "noise_std": float(protection_info["noise_std"])
            }

            try:
                upload_response = requests.post(f"{SERVER_URL}/send_update", json=payload, timeout=120)
                if upload_response.status_code == 200:
                    print("Upload successful. Waiting for the next round.")
                    last_processed_round = server_round
                else:
                    print(f"Server returned an error: {upload_response.text}")
            except Exception as e:
                print(f"Network error during upload: {e}")

        except KeyboardInterrupt:
            print("\nClient stopped manually.")
            break

        except Exception as e:
            print(f"\nUnexpected client error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    start_client()
