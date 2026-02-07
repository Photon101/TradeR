"""
market_patterns_experiments.py
====================================================

BACKGROUND (for a fresh reader / fresh GPT instance)
----------------------------------------------------
You have minute-by-minute return series for multiple assets. Each file is a
single-column text file with one return per line (often in percent units).

Goal:
  Discover "patterns" that tend to precede an upswing/downswing 60 minutes ahead,
  while avoiding overfitting and measuring whether any discovered edge is
  plausibly tradable after costs.

Key design choices in this project:
  1) What is a "pattern"?
     Instead of assuming a fixed technical shape, we treat a pattern as a
     *bundle of features* extracted from the recent return history (multi-scale).
     This keeps the pattern bounded (lookback window <= max scale) while allowing
     the model/selection process to decide which aspects matter.

  2) Outcome definition (classification labels):
     For a given decision time t, define y = sum of returns over the next H minutes.
     Then define three regimes {down, side, up} by TRAIN tertiles:
       q1 = 1/3 quantile of y on TRAIN only
       q2 = 2/3 quantile of y on TRAIN only
       down if y<=q1, side if q1<y<=q2, up if y>q2
     This forces balanced labels in TRAIN, while VAL/TEST may be imbalanced.

  3) Splitting and generalization:
     The most realistic split found so far is:
       - If an asset has "part1" and "part2": use part1 for train/val and part2 as true test.
       - Otherwise fallback to a contiguous split within the single file.
     This avoids leakage and tests regime-shift robustness.

  4) Scoring for trading:
     Fit a probabilistic 3-class model to predict {down,side,up}.
     Convert its output to a single directional score:
       score = P(up) - P(down)
     Intuition: score is positive when the model sees "up skew" and negative
     when it sees "down skew".

  5) Trading evaluation:
     Evaluate non-overlapping decisions every H minutes (e.g., H=60):
       - Enter positions based on score exceeding a threshold, optionally gated
         by risk filters and/or "species" (clusters).
       - Hold for H minutes and realize y (future sum).
       - Subtract a round-trip cost in bps.

This script:
  - Loads multiple assets (either from Google Drive folder, local folder, or zip uploads in Colab).
  - Runs a suite of experiments:
      * horizon sweep (optional)
      * feature-set ablation (fewer vs more features)
      * optional species (KMeans) gating with K sweep
      * risk-filter search selected on VAL (no test leakage)
      * long-only / short-only / long-short symmetry checks
      * cost sweep
  - Selects best configs on VAL (by t-stat per trade, with min trades) and reports TEST once.

IMPORTANT:
  - Any parameter selection (thresholds, filters, clusters) is done on VAL only.
  - TEST is used only for reporting final out-of-sample results.

----------------------------------------------------
USAGE (Colab-friendly)
----------------------------------------------------
1) Put your .txt files in a Drive folder (recommended) OR upload zips.
   Filenames should ideally include "_part1" and "_part2" to enforce true OOT test:
     EURUSD_part1.txt, EURUSD_part2.txt, etc.

2) In Colab:
   - Set DATA_SOURCE="drive" and DRIVE_FOLDER to your folder.
   - Run this script in a cell:
       !python market_patterns_experiments.py

3) Locally:
   - Set DATA_SOURCE="local_folder" and LOCAL_FOLDER.
   - Run:
       python market_patterns_experiments.py

----------------------------------------------------
NOTES ON "SPECIES"
----------------------------------------------------
"Species" = clusters of feature vectors (not pre-defined). We:
  - Fit KMeans on TRAIN feature vectors.
  - Identify candidate clusters with enough support and positive mean future return.
  - Keep only those that also look positive on VAL.
This yields a cluster gate: allow trades only when the current feature vector falls
in an allowed cluster set. Species selection is *optional* and treated as another
axis in ablation.

----------------------------------------------------
"""

from __future__ import annotations

import glob
import math
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler


# ============================
# CONFIG
# ============================

# ---- Data loading ----
DATA_SOURCE = "local_folder"  # "drive" | "zip_upload" | "local_folder"
ASSUME_INPUT_IS_PERCENT = True

# Google Drive (Colab)
DRIVE_FOLDER = "/content/drive/MyDrive/TradeR"
TXT_GLOB = "*.txt"

# Local folder (non-Colab)
LOCAL_FOLDER = "./data"
DATASETS_ZIP = "datasets.zip"

# Zip upload (Colab) expects user to upload one or many zips; each zip should contain one txt.

# ---- Experiment core ----
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

# Horizons to test (minutes). Set to [60] for the main line.
HORIZONS = [60]  # e.g. [30, 60, 120]

# Non-overlapping decision step = horizon (classic "one trade per horizon")
DECISION_STEP_IS_HORIZON = True

# How densely to sample overlapping windows for model fitting
SAMPLE_STEP = 10

# Caps to keep runtime/memory reasonable
MAX_TRAIN_SAMPLES_PER_ASSET = 120_000
MAX_VAL_SAMPLES_PER_ASSET = 40_000
MAX_TEST_SAMPLES_PER_ASSET = 60_000

# ---- Splitting ----
# If an asset has part1+part2: part1 -> train/val, part2 -> test (true OOT).
P1_TRAIN_FRAC = 0.90  # of part1
P1_VAL_FRAC = 0.10  # remainder of part1

# Fallback if no part2: contiguous train/val/test
FB_TRAIN_FRAC = 0.60
FB_VAL_FRAC = 0.10  # test is remainder

# ---- Threshold selection / objectives ----
SCORE_PCTS = [95, 97, 99]
MIN_TRADES_VAL = 200  # minimum trades required on VAL for a config to be considered
MIN_POSITIVE_ASSETS_VAL = 4
ASSET_T_CUTOFF = 0.5

# We select configs on VAL by maximizing t-stat per trade.
# (Alternative: maximize mean bps per trade with trade count constraint.)

# ---- Costs to test (bps round-trip) ----
COSTS_BPS = [1.0]

# ---- Modes: long_only, short_only, long_short ----
MODES = ["long_only", "long_short"]

# ---- Species (clusters) ----
K_LIST = [None, 10]  # None means no species gating
MIN_CLUSTER_N_TRAIN = 2000
TOP_M_CLUSTERS = 10
MIN_SPECIES_COVERAGE_VAL = 0.05

# ---- Risk filters (searched on VAL only) ----
# Each risk filter is (feature_name, direction, quantiles).
# direction:
#   "cap"   => keep trades with feature <= cutoff
#   "floor" => keep trades with feature >= cutoff
RISK_FILTER_CANDIDATES = [
    ("logvol_60", "cap", [0.80, 0.85, 0.90, 0.95]),
    ("logvol_30", "cap", [0.80, 0.85, 0.90, 0.95]),
    ("logvol_15", "cap", [0.80, 0.85, 0.90, 0.95]),
    ("logvol_120", "cap", [0.80, 0.85, 0.90, 0.95]),
    ("logvol_240", "cap", [0.80, 0.85, 0.90, 0.95]),
    ("signflip_60", "floor", [0.50, 0.60, 0.70, 0.80]),
    ("signflip_30", "floor", [0.50, 0.60, 0.70, 0.80]),
    ("signflip_15", "floor", [0.50, 0.60, 0.70, 0.80]),
    ("signflip_120", "floor", [0.50, 0.60, 0.70, 0.80]),
    ("signflip_240", "floor", [0.50, 0.60, 0.70, 0.80]),
]
# plus "None" risk filter implicitly.

# ============================
# FEATURE SPECS (ABLATION)
# ============================
# "Pattern" is represented by features from the recent history.
# We intentionally try smaller feature sets to encourage generalization.


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    scales: Tuple[int, ...]  # lookback windows in minutes
    include: Tuple[str, ...]  # feature types per scale


# Core features:
#   drift_norm: sum(r)/std(r)  (trend strength normalized by vol)
#   signflip: fraction of sign changes (chop proxy)
#   logvol: log(std(r)) (vol regime)
#
# Optional extended features (can help but may overfit):
#   mean_abs: mean(|r|)/std(r)
#   max_abs: max(|r|)/std(r)
#   jump_rate: fraction(|r| > 2*std(r))  (tail/jump proxy)
#   ac1: lag-1 autocorr in window

FEATURE_SPECS = [
    FeatureSpec("core_30_60", (30, 60), ("drift_norm", "signflip", "logvol")),
    FeatureSpec("macro_60_240", (60, 120, 240), ("drift_norm", "signflip", "logvol")),
]


# ============================
# UTILITIES
# ============================


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def clean_returns(r: np.ndarray) -> np.ndarray:
    r = r.astype(np.float32, copy=False)
    if ASSUME_INPUT_IS_PERCENT:
        r = r / 100.0
    r = r[~np.isnan(r)]
    return r


def read_returns_from_txt_path(txt_path: str) -> np.ndarray:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().strip().splitlines()
    r = np.array([float(x) for x in lines if x.strip() != ""], dtype=np.float32)
    return clean_returns(r)


def read_returns_from_zip(zip_path: str) -> Tuple[np.ndarray, str]:
    with zipfile.ZipFile(zip_path, "r") as z:
        txts = [n for n in z.namelist() if n.lower().endswith(".txt")]
        if not txts:
            raise ValueError(f"{zip_path}: no .txt in zip.")
        txt = txts[0]
        lines = z.read(txt).decode("utf-8", errors="ignore").strip().splitlines()
        r = np.array([float(x) for x in lines if x.strip() != ""], dtype=np.float32)
    return clean_returns(r), os.path.basename(txt)


def cap_evenly(arr: np.ndarray, cap: int) -> np.ndarray:
    if len(arr) <= cap:
        return arr
    idx = np.linspace(0, len(arr) - 1, cap).astype(int)
    return arr[idx]


def make_prefix(r: np.ndarray) -> np.ndarray:
    p = np.empty(len(r) + 1, dtype=np.float64)
    p[0] = 0.0
    np.cumsum(r.astype(np.float64), out=p[1:])
    return p


def future_sum(prefix: np.ndarray, t_end: np.ndarray, horizon: int) -> np.ndarray:
    # y = sum of returns from t+1 ... t+horizon inclusive (with our indexing convention)
    return prefix[t_end + horizon + 1] - prefix[t_end + 1]


def parse_asset_part(filename: str) -> Tuple[str, Optional[int]]:
    """
    Parse asset and part from names like "EURUSD_part1.txt".
    Returns (asset, part) where part can be None if not found.
    """
    base = os.path.basename(filename).replace(".txt", "")
    match = re.match(r"^(.*)_part(\d+)$", base)
    if match:
        return match.group(1), int(match.group(2))
    return base, None


def trade_stats(pnl_bps: np.ndarray) -> Dict[str, float]:
    pnl_bps = np.asarray(pnl_bps, dtype=np.float32)
    n = len(pnl_bps)
    if n == 0:
        return dict(trades=0, mean=0.0, std=0.0, t=0.0)
    mu = float(pnl_bps.mean())
    sd = float(pnl_bps.std(ddof=1)) if n > 1 else 0.0
    t_stat = float(mu / (sd / math.sqrt(n) + 1e-12)) if n > 1 else 0.0
    return dict(trades=n, mean=mu, std=sd, t=t_stat)


# ============================
# FEATURE EXTRACTION
# ============================


def _autocorr_lag1(w: np.ndarray) -> np.ndarray:
    """
    Compute lag-1 autocorrelation row-wise for a window matrix w (n, L).
    Returns (n,) in float32.
    """
    x0 = w[:, :-1]
    x1 = w[:, 1:]
    x0 = x0 - x0.mean(axis=1, keepdims=True)
    x1 = x1 - x1.mean(axis=1, keepdims=True)
    num = np.sum(x0 * x1, axis=1)
    den = np.sqrt(np.sum(x0 * x0, axis=1) * np.sum(x1 * x1, axis=1)) + 1e-12
    return (num / den).astype(np.float32)


def feats_from_windows(
    win: np.ndarray, include: Tuple[str, ...]
) -> Tuple[np.ndarray, List[str]]:
    """
    win: (n_windows, L) return windows
    include: feature types to compute
    Returns:
      X: (n_windows, n_feats)
      names: list of names (without scale suffix)
    """
    eps = 1e-8
    vol = win.std(axis=1, ddof=1).astype(np.float32) + eps
    out = []
    names = []

    if "drift_norm" in include:
        drift_norm = (win.sum(axis=1).astype(np.float32) / vol).reshape(-1, 1)
        out.append(drift_norm)
        names.append("drift_norm")

    if "signflip" in include:
        sign_flip = (
            (np.diff(np.sign(win), axis=1) != 0)
            .mean(axis=1)
            .astype(np.float32)
            .reshape(-1, 1)
        )
        out.append(sign_flip)
        names.append("signflip")

    if "logvol" in include:
        log_vol = np.log(vol).astype(np.float32).reshape(-1, 1)
        out.append(log_vol)
        names.append("logvol")

    if "mean_abs" in include:
        mean_abs = (
            (np.mean(np.abs(win), axis=1).astype(np.float32) / vol).reshape(-1, 1)
        )
        out.append(mean_abs)
        names.append("mean_abs")

    if "max_abs" in include:
        max_abs = (
            (np.max(np.abs(win), axis=1).astype(np.float32) / vol).reshape(-1, 1)
        )
        out.append(max_abs)
        names.append("max_abs")

    if "jump_rate" in include:
        # "jump" = |r| > 2*std(window)
        jump_rate = (
            (np.abs(win) > (2.0 * vol[:, None]))
            .mean(axis=1)
            .astype(np.float32)
            .reshape(-1, 1)
        )
        out.append(jump_rate)
        names.append("jump_rate")

    if "ac1" in include:
        ac1 = _autocorr_lag1(win).reshape(-1, 1)
        out.append(ac1)
        names.append("ac1")

    X = np.hstack(out).astype(np.float32)
    return X, names


def build_Xy_for_times(
    r: np.ndarray, t_end: np.ndarray, prefix: np.ndarray, horizon: int, spec: FeatureSpec
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build features for endpoints t_end using multi-scale windows defined by spec.
    Returns X, y, feature_names (with scale suffix).
    """
    feats = []
    feat_names = []
    r32 = r.astype(np.float32, copy=False)

    for ww in spec.scales:
        view = sliding_window_view(r32, ww)  # (N-ww+1, ww) view
        starts = (t_end - ww + 1).astype(np.int64)
        win = view[starts]  # (len(t_end), ww) materialized subset
        Xw, names = feats_from_windows(win, spec.include)
        feats.append(Xw)
        feat_names += [f"{n}_{ww}" for n in names]

    X = np.hstack(feats).astype(np.float32)
    y = future_sum(prefix, t_end, horizon).astype(np.float32)
    return X, y, feat_names


# ============================
# SPLITTING
# ============================


@dataclass
class AssetSplit:
    asset: str
    train_name: str
    test_name: str
    # Overlapping samples for ML (train/val/test)
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    # Non-overlapping decision sets for trading (val/test). Train decisions optional.
    Xtr_d: np.ndarray
    ytr_d: np.ndarray
    Xva_d: np.ndarray
    yva_d: np.ndarray
    Xte_d: np.ndarray
    yte_d: np.ndarray
    # Feature name mapping
    feat_names: List[str]


def build_splits_for_asset(
    asset: str,
    r_part1: np.ndarray,
    name_part1: str,
    r_part2: Optional[np.ndarray],
    name_part2: Optional[str],
    horizon: int,
    spec: FeatureSpec,
) -> Optional[AssetSplit]:
    """
    Returns an AssetSplit with:
      - part1 train/val (and possibly part1 test if no part2)
      - part2 test if present
    """
    Wmax = max(spec.scales)
    step = horizon if DECISION_STEP_IS_HORIZON else max(1, horizon // 2)

    def build_overlapping_samples(
        r: np.ndarray, t_end: np.ndarray, prefix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        return build_Xy_for_times(r, t_end, prefix, horizon, spec)

    def build_decisions(
        r: np.ndarray, prefix: np.ndarray, start: int, end: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        t0 = max(Wmax - 1, start)
        t_end = np.arange(t0, min(end, len(r) - horizon), step, dtype=np.int64)
        X, y, fn = build_Xy_for_times(r, t_end, prefix, horizon, spec)
        return X, y, fn

    # Validate length
    if len(r_part1) < (Wmax + horizon + 10_000):
        return None

    prefix1 = make_prefix(r_part1)
    N1 = len(r_part1)

    # If part2 exists: part1 -> train/val, part2 -> test
    if r_part2 is not None and name_part2 is not None:
        if len(r_part2) < (Wmax + horizon + 2_000):
            return None
        prefix2 = make_prefix(r_part2)
        N2 = len(r_part2)

        i_train = int(P1_TRAIN_FRAC * N1)

        # Overlapping samples on part1 (train/val)
        t_all1 = np.arange(Wmax - 1, N1 - horizon, SAMPLE_STEP, dtype=np.int64)
        t_tr = t_all1[t_all1 < (i_train - horizon)]
        t_va = t_all1[t_all1 >= i_train]

        t_tr = cap_evenly(t_tr, MAX_TRAIN_SAMPLES_PER_ASSET)
        t_va = cap_evenly(t_va, MAX_VAL_SAMPLES_PER_ASSET)

        Xtr, ytr, feat_names = build_overlapping_samples(r_part1, t_tr, prefix1)
        Xva, yva, _ = build_overlapping_samples(r_part1, t_va, prefix1)

        # Overlapping samples on part2 (test)
        t_all2 = np.arange(Wmax - 1, N2 - horizon, SAMPLE_STEP, dtype=np.int64)
        t_te = cap_evenly(t_all2, MAX_TEST_SAMPLES_PER_ASSET)
        Xte, yte, _ = build_overlapping_samples(r_part2, t_te, prefix2)

        # Non-overlapping decisions
        Xtr_d, ytr_d, _ = build_decisions(r_part1, prefix1, 0, i_train)
        Xva_d, yva_d, _ = build_decisions(r_part1, prefix1, i_train, N1)
        Xte_d, yte_d, _ = build_decisions(r_part2, prefix2, 0, N2)

        return AssetSplit(
            asset,
            name_part1,
            name_part2,
            Xtr,
            ytr,
            Xva,
            yva,
            Xte,
            yte,
            Xtr_d,
            ytr_d,
            Xva_d,
            yva_d,
            Xte_d,
            yte_d,
            feat_names,
        )

    # Otherwise fallback contiguous split within part1
    i_train = int(FB_TRAIN_FRAC * N1)
    i_val = int((FB_TRAIN_FRAC + FB_VAL_FRAC) * N1)

    t_all = np.arange(Wmax - 1, N1 - horizon, SAMPLE_STEP, dtype=np.int64)
    t_tr = t_all[t_all < (i_train - horizon)]
    t_va = t_all[(t_all >= i_train) & (t_all < i_val)]
    t_te = t_all[t_all >= i_val]

    t_tr = cap_evenly(t_tr, MAX_TRAIN_SAMPLES_PER_ASSET)
    t_va = cap_evenly(t_va, MAX_VAL_SAMPLES_PER_ASSET)
    t_te = cap_evenly(t_te, MAX_TEST_SAMPLES_PER_ASSET)

    Xtr, ytr, feat_names = build_Xy_for_times(r_part1, t_tr, prefix1, horizon, spec)
    Xva, yva, _ = build_Xy_for_times(r_part1, t_va, prefix1, horizon, spec)
    Xte, yte, _ = build_Xy_for_times(r_part1, t_te, prefix1, horizon, spec)

    Xtr_d, ytr_d, _ = build_Xy_for_times(
        r_part1,
        np.arange(
            max(Wmax - 1, 0), min(i_train, N1 - horizon), step, dtype=np.int64
        ),
        prefix1,
        horizon,
        spec,
    )
    Xva_d, yva_d, _ = build_decisions(r_part1, prefix1, i_train, i_val)
    Xte_d, yte_d, _ = build_decisions(r_part1, prefix1, i_val, N1)

    return AssetSplit(
        asset,
        name_part1,
        f"{name_part1} (tail)",
        Xtr,
        ytr,
        Xva,
        yva,
        Xte,
        yte,
        Xtr_d,
        ytr_d,
        Xva_d,
        yva_d,
        Xte_d,
        yte_d,
        feat_names,
    )


# ============================
# MODEL + TRADING
# ============================


def label_from_y(y: np.ndarray, q1: float, q2: float) -> np.ndarray:
    return np.where(y <= q1, 0, np.where(y <= q2, 1, 2)).astype(np.int8)


def score_from_proba(pr: np.ndarray) -> np.ndarray:
    return (pr[:, 2] - pr[:, 0]).astype(np.float32)


def positions_from_score(score: np.ndarray, thr: float, mode: str) -> np.ndarray:
    if mode == "long_only":
        return (score > thr).astype(np.float32)
    if mode == "short_only":
        return -(score < -thr).astype(np.float32)
    if mode == "long_short":
        pos = np.zeros_like(score, dtype=np.float32)
        pos[score > thr] = 1.0
        pos[score < -thr] = -1.0
        return pos
    raise ValueError("Unknown mode")


@dataclass
class SpeciesGate:
    K: int
    km: MiniBatchKMeans
    good_clusters: np.ndarray
    coverage_val: float


def fit_species_gate(
    Xtrz: np.ndarray, ytr: np.ndarray, Xvaz: np.ndarray, yva: np.ndarray, K: int
) -> SpeciesGate:
    """
    Fit KMeans on TRAIN feature vectors (scaled), then choose "good clusters" by:
      - Eligible clusters: train count >= MIN_CLUSTER_N_TRAIN
      - Take top TOP_M_CLUSTERS by train y-mean
      - Keep only those with val y-mean > 0
      - Ensure minimum val coverage by adding more clusters (still positive val mean)
    """
    km = MiniBatchKMeans(
        n_clusters=K, random_state=RANDOM_SEED, batch_size=4096, n_init=10
    )
    ztr = km.fit_predict(Xtrz)
    df_tr = pd.DataFrame({"cluster": ztr, "y": ytr})
    stat_tr = (
        df_tr.groupby("cluster").agg(n=("y", "size"), ymean=("y", "mean")).reset_index()
    )
    eligible = stat_tr[stat_tr["n"] >= MIN_CLUSTER_N_TRAIN].sort_values(
        "ymean", ascending=False
    )
    top = eligible.head(TOP_M_CLUSTERS)["cluster"].to_numpy()

    zva = km.predict(Xvaz)
    df_va = pd.DataFrame({"cluster": zva, "y": yva})
    stat_va = (
        df_va.groupby("cluster").agg(n=("y", "size"), ymean=("y", "mean")).reset_index()
    )
    val_map = dict(zip(stat_va["cluster"].to_numpy(), stat_va["ymean"].to_numpy()))

    good = np.array(
        [c for c in top.tolist() if val_map.get(int(c), -1e9) > 0.0], dtype=int
    )

    def coverage_val(good_clusters: np.ndarray) -> float:
        if len(good_clusters) == 0:
            return 0.0
        return float(np.isin(zva, good_clusters).mean())

    cov = coverage_val(good)
    if cov < MIN_SPECIES_COVERAGE_VAL:
        for cluster in eligible["cluster"].to_numpy():
            cluster = int(cluster)
            if cluster in set(good.tolist()):
                continue
            if val_map.get(cluster, -1e9) <= 0.0:
                continue
            good = np.append(good, cluster)
            cov = coverage_val(good)
            if cov >= MIN_SPECIES_COVERAGE_VAL:
                break

    return SpeciesGate(K=K, km=km, good_clusters=good, coverage_val=cov)


def apply_risk_filter_mask(
    X_raw: np.ndarray, feat_index: Dict[str, int], risk: Optional[Tuple[str, str, float]]
) -> np.ndarray:
    """
    risk = (feature_name, direction, cutoff)
    returns boolean mask (True if passes).
    """
    if risk is None:
        return np.ones(X_raw.shape[0], dtype=bool)
    fname, direction, cut = risk
    idx = feat_index[fname]
    vals = X_raw[:, idx]
    if direction == "cap":
        return vals <= cut
    if direction == "floor":
        return vals >= cut
    raise ValueError("Bad risk direction")


def evaluate_split_trading(
    clf: HistGradientBoostingClassifier,
    scaler: StandardScaler,
    X_raw: np.ndarray,
    y: np.ndarray,
    mode: str,
    thr: float,
    cost_bps: float,
    species_gate: Optional[SpeciesGate],
    risk: Optional[Tuple[str, str, float]],
    feat_index: Dict[str, int],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute pnl (bps) for the split given config.
    """
    Xz = scaler.transform(X_raw)
    pr = clf.predict_proba(Xz)
    score = score_from_proba(pr)
    pos = positions_from_score(score, thr, mode)

    allow = np.ones_like(score, dtype=bool)
    if species_gate is not None:
        z = species_gate.km.predict(Xz)
        allow &= np.isin(z, species_gate.good_clusters)

    allow &= apply_risk_filter_mask(X_raw, feat_index, risk)

    traded = (pos != 0) & allow
    pnl = (pos * y) - (traded.astype(np.float32) * (cost_bps / 10_000.0))
    pnl_bps = pnl[traded] * 10_000.0
    return pnl_bps, trade_stats(pnl_bps)


def per_asset_trade_table(trade_rows: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(trade_rows)
    if df.empty:
        return df
    g = df.groupby("asset")["pnl_bps"]
    out = pd.DataFrame(
        {"trades": g.size(), "mean": g.mean(), "std": g.std(ddof=1)}
    ).reset_index()
    out["t"] = out["mean"] / (out["std"] / np.sqrt(out["trades"]) + 1e-12)
    return out.sort_values("mean", ascending=False)


def count_positive_assets(
    asset_splits: List["AssetSplit"],
    clf: HistGradientBoostingClassifier,
    scaler: StandardScaler,
    mode: str,
    thr: float,
    cost_bps: float,
    species_gate: Optional["SpeciesGate"],
    risk: Optional[Tuple[str, str, float]],
    feat_index: Dict[str, int],
    split_attr: str,
    t_cutoff: float,
) -> int:
    positive = 0
    for split in asset_splits:
        Xd = getattr(split, split_attr)
        yd = getattr(split, split_attr.replace("X", "y"))
        pnl_bps, _ = evaluate_split_trading(
            clf, scaler, Xd, yd, mode, thr, cost_bps, species_gate, risk, feat_index
        )
        stats = trade_stats(pnl_bps)
        if stats["trades"] > 0 and stats["t"] > t_cutoff:
            positive += 1
    return positive


def make_feat_index(feat_names: List[str]) -> Dict[str, int]:
    """
    feat_names are like 'logvol_60' etc. Return mapping -> column index.
    """
    return {n: i for i, n in enumerate(feat_names)}


# ============================
# EXPERIMENT RUNNER
# ============================


def run_one_pipeline(
    horizon: int, spec: FeatureSpec, cost_bps: float, K: Optional[int]
) -> Dict[str, object]:
    """
    Build splits for all assets, train model, (optional) fit species gate,
    then select best trading config on VAL and report TEST.
    Returns a results dict with summary and tables.
    """
    # ---- Build all asset splits ----
    asset_splits: List[AssetSplit] = []
    for asset, parts in DATASETS_BY_ASSET.items():
        r1, nm1 = parts.get(1, (None, None))
        r2, nm2 = parts.get(2, (None, None))
        if r1 is None:
            # fallback: pick any available part
            any_part = next(iter(parts.values()))
            r1, nm1 = any_part
            r2, nm2 = None, None

        sp = build_splits_for_asset(asset, r1, nm1, r2, nm2, horizon, spec)
        if sp is not None:
            asset_splits.append(sp)

    if len(asset_splits) == 0:
        return {"error": f"No assets usable for horizon={horizon} spec={spec.name}"}

    # ---- Pool arrays ----
    Xtr = np.concatenate([p.Xtr for p in asset_splits])
    ytr = np.concatenate([p.ytr for p in asset_splits])
    Xva = np.concatenate([p.Xva for p in asset_splits])
    yva = np.concatenate([p.yva for p in asset_splits])
    Xte = np.concatenate([p.Xte for p in asset_splits])
    yte = np.concatenate([p.yte for p in asset_splits])

    # Train tertile thresholds from TRAIN y only
    q1, q2 = np.quantile(ytr, [1 / 3, 2 / 3])
    lab_tr = label_from_y(ytr, q1, q2)
    lab_va = label_from_y(yva, q1, q2)
    lab_te = label_from_y(yte, q1, q2)

    # ---- Fit scaler + classifier ----
    scaler = StandardScaler()
    Xtrz = scaler.fit_transform(Xtr)
    Xvaz = scaler.transform(Xva)
    Xtez = scaler.transform(Xte)

    clf = HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_depth=3)
    clf.fit(Xtrz, lab_tr)

    pva = clf.predict_proba(Xvaz)
    pte = clf.predict_proba(Xtez)

    model_diag = dict(
        val_logloss=float(log_loss(lab_va, pva)),
        test_logloss=float(log_loss(lab_te, pte)),
        val_acc=float(accuracy_score(lab_va, pva.argmax(1))),
        test_acc=float(accuracy_score(lab_te, pte.argmax(1))),
        q1=float(q1),
        q2=float(q2),
        n_assets=len(asset_splits),
        n_train_samples=int(len(ytr)),
        n_val_samples=int(len(yva)),
        n_test_samples=int(len(yte)),
    )

    # ---- Optional species gate (clusters) ----
    species_gate = None
    species_info = None
    if K is not None:
        species_gate = fit_species_gate(Xtrz, ytr, Xvaz, yva, K)
        species_info = dict(
            K=K,
            good_clusters=species_gate.good_clusters.tolist(),
            val_coverage=float(species_gate.coverage_val),
        )

    # ---- Prepare pooled decision sets (non-overlapping) ----
    Xtr_d = np.concatenate([p.Xtr_d for p in asset_splits])
    ytr_d = np.concatenate([p.ytr_d for p in asset_splits])
    Xva_d = np.concatenate([p.Xva_d for p in asset_splits])
    yva_d = np.concatenate([p.yva_d for p in asset_splits])
    Xte_d = np.concatenate([p.Xte_d for p in asset_splits])
    yte_d = np.concatenate([p.yte_d for p in asset_splits])

    feat_names = asset_splits[0].feat_names
    feat_index = make_feat_index(feat_names)

    # Candidate thresholds are computed from pooled VAL score distribution.
    val_score = score_from_proba(clf.predict_proba(scaler.transform(Xva_d)))
    thr_candidates = [(pct, float(np.percentile(val_score, pct))) for pct in SCORE_PCTS]

    # ---- Selection on VAL: choose (mode, pct/thr, risk-filter) that maximizes val t-stat ----
    best = None  # will store dict with val stats
    for mode in MODES:
        for pct, thr in thr_candidates:
            # Base gating on VAL for conditional risk cutoff computation
            sc = val_score
            pos = positions_from_score(sc, thr, mode)
            base_enter = pos != 0

            # species gate on val (if enabled)
            if species_gate is not None:
                zva = species_gate.km.predict(scaler.transform(Xva_d))
                base_enter &= np.isin(zva, species_gate.good_clusters)

            if base_enter.sum() < MIN_TRADES_VAL:
                continue

            # Option 0: no risk filter
            _, st_val = evaluate_split_trading(
                clf,
                scaler,
                Xva_d,
                yva_d,
                mode,
                thr,
                cost_bps,
                species_gate,
                None,
                feat_index,
            )
            if st_val["trades"] >= MIN_TRADES_VAL:
                positive_assets = count_positive_assets(
                    asset_splits,
                    clf,
                    scaler,
                    mode,
                    thr,
                    cost_bps,
                    species_gate,
                    None,
                    feat_index,
                    "Xva_d",
                    ASSET_T_CUTOFF,
                )
                if positive_assets >= MIN_POSITIVE_ASSETS_VAL:
                    key = st_val["t"]
                    if (best is None) or (key > best["val"]["t"]):
                        best = dict(
                            mode=mode,
                            pct=pct,
                            thr=thr,
                            risk=None,
                            val=st_val,
                            val_positive_assets=positive_assets,
                        )

            # Option 1..N: risk filters (cutoff computed on VAL among base-enter set)
            for fname, direction, qs in RISK_FILTER_CANDIDATES:
                if fname not in feat_index:
                    continue
                vals = Xva_d[base_enter, feat_index[fname]]
                if len(vals) < MIN_TRADES_VAL:
                    continue
                for q in qs:
                    cut = float(np.quantile(vals, q))
                    risk = (fname, direction, cut)
                    _, st_val2 = evaluate_split_trading(
                        clf,
                        scaler,
                        Xva_d,
                        yva_d,
                        mode,
                        thr,
                        cost_bps,
                        species_gate,
                        risk,
                        feat_index,
                    )
                    if st_val2["trades"] < MIN_TRADES_VAL:
                        continue
                    positive_assets = count_positive_assets(
                        asset_splits,
                        clf,
                        scaler,
                        mode,
                        thr,
                        cost_bps,
                        species_gate,
                        (fname, direction, float(cut)),
                        feat_index,
                        "Xva_d",
                        ASSET_T_CUTOFF,
                    )
                    if positive_assets < MIN_POSITIVE_ASSETS_VAL:
                        continue
                    key = st_val2["t"]
                    if (best is None) or (key > best["val"]["t"]):
                        best = dict(
                            mode=mode,
                            pct=pct,
                            thr=thr,
                            risk=(fname, direction, q, cut),
                            val=st_val2,
                            val_positive_assets=positive_assets,
                        )

    if best is None:
        return dict(
            model_diag=model_diag,
            species_info=species_info,
            error="No VAL configuration met MIN_TRADES_VAL (try lowering MIN_TRADES_VAL or pct list).",
        )

    # ---- Evaluate chosen config on TRAIN/VAL/TEST and produce per-asset table (TEST) ----
    def eval_and_log(split_name: str, Xd: np.ndarray, yd: np.ndarray) -> Dict[str, float]:
        risk_tuple = None
        if best["risk"] is not None:
            fname, direction, _, cut = best["risk"]
            risk_tuple = (fname, direction, float(cut))
        _, st = evaluate_split_trading(
            clf,
            scaler,
            Xd,
            yd,
            best["mode"],
            best["thr"],
            cost_bps,
            species_gate,
            risk_tuple,
            feat_index,
        )
        return {f"{split_name}_{k}": float(v) for k, v in st.items()}

    stats = {}
    stats.update(eval_and_log("train", Xtr_d, ytr_d))
    stats.update(eval_and_log("val", Xva_d, yva_d))
    stats.update(eval_and_log("test", Xte_d, yte_d))

    # Per-asset TEST table: need to apply same rule asset-by-asset and collect trades
    risk_tuple = None
    if best["risk"] is not None:
        fname, direction, _, cut = best["risk"]
        risk_tuple = (fname, direction, float(cut))

    trade_rows = []
    for p in asset_splits:
        Xd, yd = p.Xte_d, p.yte_d
        Xz = scaler.transform(Xd)
        sc = score_from_proba(clf.predict_proba(Xz))
        pos = positions_from_score(sc, best["thr"], best["mode"])
        allow = np.ones_like(sc, dtype=bool)

        if species_gate is not None:
            z = species_gate.km.predict(Xz)
            allow &= np.isin(z, species_gate.good_clusters)

        allow &= apply_risk_filter_mask(Xd, feat_index, risk_tuple)
        traded = (pos != 0) & allow
        pnl_bps = (pos[traded] * yd[traded]) * 10_000.0 - cost_bps

        for v in pnl_bps:
            trade_rows.append({"asset": p.asset, "pnl_bps": float(v)})

    per_asset = per_asset_trade_table(trade_rows)

    return dict(
        horizon=horizon,
        feature_spec=spec.name,
        cost_bps=cost_bps,
        K=K,
        model_diag=model_diag,
        species_info=species_info,
        best_on_val=best,
        split_stats=stats,
        per_asset_test=per_asset,
        feat_names=feat_names,
    )


# ============================
# DATA LOADING (standalone)
# ============================


def _maybe_extract_local_zip() -> None:
    if not os.path.exists(DATASETS_ZIP):
        return
    if os.path.isdir(LOCAL_FOLDER):
        existing = glob.glob(os.path.join(LOCAL_FOLDER, TXT_GLOB))
        if existing:
            return
    os.makedirs(LOCAL_FOLDER, exist_ok=True)
    with zipfile.ZipFile(DATASETS_ZIP, "r") as zf:
        zf.extractall(LOCAL_FOLDER)


def load_datasets() -> Tuple[List[np.ndarray], List[str]]:
    """
    Returns (series_list, name_list)
    - series_list: list of np arrays of returns
    - name_list:   list of dataset names (filenames)
    """
    series: List[np.ndarray] = []
    names: List[str] = []

    if DATA_SOURCE == "drive":
        if not is_colab():
            raise RuntimeError(
                "DATA_SOURCE='drive' is intended for Colab. Use local_folder instead."
            )
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)

        pattern = os.path.join(DRIVE_FOLDER, TXT_GLOB)
        txt_files = sorted(glob.glob(pattern))
        if not txt_files:
            if not os.path.exists(DATASETS_ZIP):
                raise ValueError(
                    f"No .txt files found at: {pattern} and {DATASETS_ZIP} is missing."
                )
            raise ValueError(f"No .txt files found at: {pattern}")

        print(f"Found {len(txt_files)} txt files in: {DRIVE_FOLDER}")
        for fp in txt_files:
            r = read_returns_from_txt_path(fp)
            base = os.path.basename(fp)
            print(f"{base}: N={len(r):,}, mean={r.mean():.2e}, std={r.std():.2e}")
            series.append(r)
            names.append(base)

    elif DATA_SOURCE == "local_folder":
        _maybe_extract_local_zip()
        pattern = os.path.join(LOCAL_FOLDER, TXT_GLOB)
        txt_files = sorted(glob.glob(pattern))
        if not txt_files:
            raise ValueError(f"No .txt files found at: {pattern}")

        print(f"Found {len(txt_files)} txt files in: {LOCAL_FOLDER}")
        for fp in txt_files:
            r = read_returns_from_txt_path(fp)
            base = os.path.basename(fp)
            print(f"{base}: N={len(r):,}, mean={r.mean():.2e}, std={r.std():.2e}")
            series.append(r)
            names.append(base)

    elif DATA_SOURCE == "zip_upload":
        if not is_colab():
            raise RuntimeError("DATA_SOURCE='zip_upload' is intended for Colab.")
        from google.colab import files  # type: ignore

        uploaded = files.upload()
        zip_files = list(uploaded.keys())
        print("Uploaded:", zip_files)

        for zf in zip_files:
            r, inner_name = read_returns_from_zip(zf)
            print(f"{zf} ({inner_name}): N={len(r):,}, mean={r.mean():.2e}, std={r.std():.2e}")
            series.append(r)
            # Use inner txt name if present; otherwise zip name
            names.append(inner_name if inner_name else os.path.basename(zf))
    else:
        raise ValueError("DATA_SOURCE must be 'drive', 'local_folder', or 'zip_upload'")

    return series, names


def group_by_asset(
    series: List[np.ndarray], names: List[str]
) -> Dict[str, Dict[Optional[int], Tuple[np.ndarray, str]]]:
    """
    Build mapping:
      DATASETS_BY_ASSET[asset][part] = (returns, name)
    part is int if parsed from _part#, else None.
    """
    out: Dict[str, Dict[Optional[int], Tuple[np.ndarray, str]]] = {}
    for r, nm in zip(series, names):
        asset, part = parse_asset_part(nm)
        out.setdefault(asset, {})[part] = (r, nm)
    return out


# Global container set in main()
DATASETS_BY_ASSET: Dict[str, Dict[Optional[int], Tuple[np.ndarray, str]]] = {}


# ============================
# MAIN
# ============================


def main() -> None:
    global DATASETS_BY_ASSET

    print("====================================================")
    print("Market pattern experiments (standalone)")
    print("====================================================")
    print("DATA_SOURCE:", DATA_SOURCE)
    print("ASSUME_INPUT_IS_PERCENT:", ASSUME_INPUT_IS_PERCENT)
    print("HORIZONS:", HORIZONS)
    print("FEATURE_SPECS:", [fs.name for fs in FEATURE_SPECS])
    print("K_LIST (species):", K_LIST)
    print("COSTS_BPS:", COSTS_BPS)
    print("MODES:", MODES)
    print("MIN_TRADES_VAL:", MIN_TRADES_VAL)
    print("====================================================\n")

    series, names = load_datasets()
    DATASETS_BY_ASSET = group_by_asset(series, names)

    assets = sorted(DATASETS_BY_ASSET.keys())
    print("\nAssets found:", len(assets))
    # Count how many have part2
    n_p2 = sum(1 for a in assets if (2 in DATASETS_BY_ASSET[a]))
    print("Assets with part2 available:", n_p2)

    all_results = []
    for horizon in HORIZONS:
        for spec in FEATURE_SPECS:
            for cost in COSTS_BPS:
                for K in K_LIST:
                    res = run_one_pipeline(
                        horizon=horizon, spec=spec, cost_bps=cost, K=K
                    )
                    if "error" in res:
                        print(
                            f"[SKIP] horizon={horizon} spec={spec.name} cost={cost} K={K} => {res['error']}"
                        )
                        continue

                    md = res["model_diag"]
                    best = res["best_on_val"]
                    st = res["split_stats"]

                    row = dict(
                        horizon=horizon,
                        feature_spec=spec.name,
                        cost_bps=cost,
                        K=("none" if K is None else K),
                        mode=best["mode"],
                        val_pct=best["pct"],
                        thr=best["thr"],
                        risk=(
                            "none"
                            if best["risk"] is None
                            else f"{best['risk'][0]}:{best['risk'][1]}@q{best['risk'][2]}"
                        ),
                        val_positive_assets=best.get("val_positive_assets", 0),
                        val_trades=st["val_trades"],
                        val_mean=st["val_mean"],
                        val_t=st["val_t"],
                        test_trades=st["test_trades"],
                        test_mean=st["test_mean"],
                        test_t=st["test_t"],
                        train_trades=st["train_trades"],
                        train_mean=st["train_mean"],
                        train_t=st["train_t"],
                        val_logloss=md["val_logloss"],
                        test_logloss=md["test_logloss"],
                        val_acc=md["val_acc"],
                        test_acc=md["test_acc"],
                        n_assets=md["n_assets"],
                    )
                    all_results.append(row)

                    print("\n----------------------------------------------------")
                    print(f"h={horizon} | spec={spec.name} | cost={cost} | K={K}")
                    print(
                        "Model:",
                        {
                            k: md[k]
                            for k in [
                                "val_logloss",
                                "test_logloss",
                                "val_acc",
                                "test_acc",
                                "q1",
                                "q2",
                                "n_assets",
                            ]
                        },
                    )
                    if res["species_info"] is not None:
                        print("Species:", res["species_info"])
                    print("Best on VAL:", best)
                    print("Split stats:", st)
                    print("Per-asset TEST (top 5):")
                    print(
                        res["per_asset_test"]
                        .head(5)
                        .to_string(index=False, float_format=lambda x: f"{x:0.4f}")
                    )
                    print("Per-asset TEST (bottom 5):")
                    print(
                        res["per_asset_test"]
                        .tail(5)
                        .to_string(index=False, float_format=lambda x: f"{x:0.4f}")
                    )
                    print("----------------------------------------------------\n")

    if not all_results:
        print("No successful runs produced results. Consider loosening constraints or reducing pct list.")
        return

    df = pd.DataFrame(all_results)

    # Sort by out-of-sample test t-stat, then test mean
    df_sorted = df.sort_values(["test_t", "test_mean"], ascending=False)

    print("\n====================================================")
    print("TOP RESULTS (sorted by TEST t-stat, then TEST mean)")
    print("====================================================")
    cols = [
        "horizon",
        "feature_spec",
        "cost_bps",
        "K",
        "mode",
        "val_pct",
        "thr",
        "risk",
        "val_trades",
        "val_mean",
        "val_t",
        "val_positive_assets",
        "test_trades",
        "test_mean",
        "test_t",
        "train_trades",
        "train_mean",
        "train_t",
        "val_logloss",
        "test_logloss",
        "val_acc",
        "test_acc",
        "n_assets",
    ]
    print(df_sorted[cols].head(25).to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    # Save to CSV for checkpointing
    out_csv = "experiment_summary.csv"
    df_sorted.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
