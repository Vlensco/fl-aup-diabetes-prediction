"""
Runs every scenario x seed combination (server + RS_01 + RS_02), then builds the mean ± SD tables.

Usage:
    python run_experiments.py                                  # STATIC, DP, ADP x seeds 1..5, 50 rounds
    python run_experiments.py --scenarios ADP --seeds 1 2      # subset
    python run_experiments.py --summarize_only                 # only rebuild tables from results/
    python run_experiments.py --fixed_threshold 0.5            # one common threshold for all scenarios

Requires the outputs of the data preparation notebook in the working directory:
    client_RS_01.npz, client_RS_02.npz, server_holdout.npz, experiment_metadata.json
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import requests

parser = argparse.ArgumentParser(description="Run federated experiments over scenarios and seeds")
parser.add_argument("--scenarios", nargs="+", default=["STATIC", "DP", "ADP"])
parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
parser.add_argument("--rounds", type=int, default=50)
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--fixed_threshold", type=float, default=None)
parser.add_argument("--db_uri", type=str, default=None, help="Passed to the server (default: server's own default)")
parser.add_argument("--server_script", type=str, default="federated_server.py")
parser.add_argument("--client_script", type=str, default="auto_client.py")
parser.add_argument("--timeout", type=int, default=7200, help="Max seconds per run")
parser.add_argument("--skip_existing", action="store_true", help="Skip runs whose summary JSON already exists")
parser.add_argument("--summarize_only", action="store_true")
args = parser.parse_args()

PY = sys.executable

# Console output is kept ASCII-only (see console_text) because Jupyter on Windows decodes
# child-process output as cp1252, which turns "±" into "Â±". The CSV tables keep "±".
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass


def console_text(text):
    return text.replace("±", "+/-").replace("−", "-").replace("Δ", "d_")

CHILD_ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")

RESULT_DIR = "results"
LOG_DIR = "logs"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

METRICS = ["accuracy", "precision", "recall", "f1", "loss"]
PAPER_NAMES = {"STATIC": "No perturbation", "DP": "Fixed update perturbation", "ADP": "Adaptive update perturbation"}


class ServerStartError(Exception):
    pass


def print_log_tail(path, n_lines=40):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        print(f"----- last {min(n_lines, len(lines))} lines of {path} -----")
        print("".join(lines[-n_lines:]).rstrip())
        print("-" * 60)
    except Exception as e:
        print(f"(could not read {path}: {e})")


def wait_for_server(url, process, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        if process.poll() is not None:
            return False
        try:
            if requests.get(f"{url}/health", timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_one(scenario, seed):
    summary_path = os.path.join(RESULT_DIR, f"summary_{scenario}_seed{seed}.json")
    if args.skip_existing and os.path.exists(summary_path):
        print(f"[skip] {scenario} seed {seed} (summary exists)")
        return True

    url = f"http://127.0.0.1:{args.port}"
    server_cmd = [PY, args.server_script, "--scenario", scenario, "--seed", str(seed),
                  "--rounds", str(args.rounds), "--port", str(args.port), "--host", "127.0.0.1",
                  "--exit_on_finish"]
    if args.fixed_threshold is not None:
        server_cmd += ["--fixed_threshold", str(args.fixed_threshold)]
    if args.db_uri:
        server_cmd += ["--db_uri", args.db_uri]

    tag = f"{scenario}_seed{seed}"
    print(f"\n=== {tag} ===")

    server_log_path = os.path.join(LOG_DIR, f"server_{tag}.log")
    server_log = open(server_log_path, "w", encoding="utf-8")
    server = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT, env=CHILD_ENV)

    clients, client_logs = [], []
    try:
        if not wait_for_server(url, server):
            print(f"[error] server for {tag} did not start.")
            server_log.flush()
            print_log_tail(server_log_path)
            raise ServerStartError(tag)

        for cid in ["RS_01", "RS_02"]:
            log = open(os.path.join(LOG_DIR, f"client_{cid}_{tag}.log"), "w", encoding="utf-8")
            client_logs.append(log)
            clients.append(subprocess.Popen(
                [PY, args.client_script, "--id", cid, "--url", url, "--seed", str(seed),
                 "--epochs", str(args.epochs), "--batch", str(args.batch)],
                stdout=log, stderr=subprocess.STDOUT, env=CHILD_ENV))

        start = time.time()
        while server.poll() is None:
            if time.time() - start > args.timeout:
                print(f"[error] {tag} timed out")
                return False
            if all(c.poll() is not None for c in clients) and not os.path.exists(summary_path):
                time.sleep(15)
                if server.poll() is None and not os.path.exists(summary_path):
                    print(f"[error] both clients stopped before the run finished.")
                    for cid in ["RS_01", "RS_02"]:
                        print_log_tail(os.path.join(LOG_DIR, f"client_{cid}_{tag}.log"), 20)
                    return False
            time.sleep(2)

        ok = os.path.exists(summary_path)
        print(f"[{'done' if ok else 'error'}] {tag} in {time.time() - start:.0f}s")
        if not ok:
            server_log.flush()
            print_log_tail(server_log_path)
        return ok

    finally:
        for c in clients:
            if c.poll() is None:
                c.terminate()
        if server.poll() is None:
            server.terminate()
        for c in clients:
            try:
                c.wait(timeout=10)
            except Exception:
                c.kill()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()
        server_log.close()
        for log in client_logs:
            log.close()
        time.sleep(2)  # let the port be released


def load_summaries():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULT_DIR, "summary_*_seed*.json"))):
        with open(path) as f:
            s = json.load(f)
        base = {"scenario": s["scenario"], "seed": s["seed"],
                "best_round": s["selection"]["best_round_by_validation_f1"],
                "threshold": s["selection"]["selected_threshold"],
                "average_precision": s["test_ranking_metrics"]["average_precision"],
                "roc_auc": s["test_ranking_metrics"]["roc_auc"],
                "final_clip": s["final_clip"]}
        for m in METRICS:
            base[m] = s["test_at_selected_threshold"][m]
            base[f"{m}@0.50"] = s["test_at_0.50"][m]
        rows.append(base)
    return pd.DataFrame(rows)


def mean_sd_table(df, columns, decimals=4):
    order = [s for s in ["STATIC", "DP", "ADP"] if s in df["scenario"].unique()]
    out = []
    for scen in order:
        sub = df[df["scenario"] == scen]
        row = {"Scenario": scen, "Paper name": PAPER_NAMES.get(scen, scen), "n_seeds": len(sub)}
        for c in columns:
            sd = sub[c].std(ddof=1) if len(sub) > 1 else float("nan")
            row[c] = f"{sub[c].mean():.{decimals}f} ± {sd:.{decimals}f}"
        out.append(row)
    return pd.DataFrame(out)


def paired_differences(df, reference="STATIC"):
    """Per-seed difference of each scenario vs. the reference, then mean ± SD of those differences."""
    out = []
    ref = df[df["scenario"] == reference].set_index("seed")
    for scen in [s for s in ["DP", "ADP"] if s in df["scenario"].unique()]:
        sub = df[df["scenario"] == scen].set_index("seed")
        common = sub.index.intersection(ref.index)
        if len(common) == 0:
            continue
        row = {"Comparison": f"{scen} − {reference}", "n_paired_seeds": len(common)}
        for m in ["f1", "recall", "precision", "average_precision"]:
            d = sub.loc[common, m] - ref.loc[common, m]
            sd = d.std(ddof=1) if len(d) > 1 else float("nan")
            row[f"Δ{m}"] = f"{d.mean():+.4f} ± {sd:.4f}"
        out.append(row)
    return pd.DataFrame(out)


def summarize():
    df = load_summaries()
    if df.empty:
        print("No summaries found in results/.")
        return

    df.sort_values(["scenario", "seed"]).to_csv(os.path.join(RESULT_DIR, "all_runs_test_metrics.csv"), index=False, encoding="utf-8-sig")

    main = mean_sd_table(df, METRICS + ["average_precision", "threshold"])
    at050 = mean_sd_table(df, [f"{m}@0.50" for m in METRICS])
    diffs = paired_differences(df)

    main.to_csv(os.path.join(RESULT_DIR, "table_test_mean_sd_selected_threshold.csv"), index=False, encoding="utf-8-sig")
    at050.to_csv(os.path.join(RESULT_DIR, "table_test_mean_sd_threshold_050.csv"), index=False, encoding="utf-8-sig")
    if not diffs.empty:
        diffs.to_csv(os.path.join(RESULT_DIR, "table_paired_differences_vs_static.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(console_text("\nTEST SET, threshold selected on validation (mean ± SD over seeds)"))
    print(console_text(main.to_string(index=False)))
    print(console_text("\nTEST SET, threshold 0.50 (mean ± SD over seeds)"))
    print(console_text(at050.to_string(index=False)))
    if not diffs.empty:
        print(console_text("\nPaired differences vs STATIC (same seed), mean ± SD"))
        print(console_text(diffs.to_string(index=False)))
    print(f"\nTables written to {RESULT_DIR}/")


if __name__ == "__main__":
    if not args.summarize_only:
        failed = []
        try:
            for scenario in args.scenarios:
                for seed in args.seeds:
                    if not run_one(scenario, seed):
                        failed.append(f"{scenario}_seed{seed}")
        except ServerStartError as e:
            print(f"\nStopped: the server could not start ({e}). "
                  f"The same error would repeat for every run, so the remaining runs were skipped.")
            print("Fix the error shown above, then run this script again.")
            sys.exit(1)
        if failed:
            print(f"\nFailed runs: {failed}")
    summarize()
