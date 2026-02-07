"""
market_patterns_robustness.py
====================================================
RESEARCH OBJECTIVE:
1. Address the "Regime Shift" issue (high VAL, weak TEST) by controlling
   threshold selection (train vs. val) and using fixed percentile triggers.
2. Compare raw labels (TRAIN tertiles) vs. volatility-standardized labels
   with fixed thresholds.
3. Test whether macro lookbacks (60/120/240, 30/120/240) yield more stable
   test behavior.

KEY DIFFERENCES FROM market_patterns_experiments.py
- A/B targets: raw quantiles vs. vol-standardized targets.
- Macro feature spec (60/120/240) added.
- Thresholds are sourced from TRAIN or VAL score distributions, then applied
  to TEST to limit leakage.
====================================================
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ============================
# CONFIG
# ============================

DATA_SOURCE = "local_folder"  # "drive" | "local_folder" | "zip_upload"
DRIVE_FOLDER = "/content/drive/MyDrive/TradeR"
LOCAL_FOLDER = "./data"
DATASETS_ZIP = "datasets.zip"
TXT_GLOB = "*.txt"
ASSUME_INPUT_IS_PERCENT = True

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

HORIZON = 60
SAMPLE_STEP = 5
DECISION_STEP = HORIZON

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10

TARGET_TYPES = ["raw_quantile", "vol_std"]

VOL_STD_WINDOW = 60
VOL_STD_THRESH = (-0.5, 0.5)

COST_BPS = 1.0
THRESH_PCTS = [90, 95, 97]
MODES = ["long_only", "short_only", "long_short"]
THRESH_SOURCES = ["train", "val"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    scales: Tuple[int, ...]
    include: Tuple[str, ...]


FEATURE_SPECS = [
    FeatureSpec("core_30_60", (30, 60), ("drift_norm", "signflip", "logvol")),
    FeatureSpec("macro_60_240", (60, 120, 240), ("drift_norm", "signflip", "logvol")),
    FeatureSpec("macro_30_120_240", (30, 120, 240), ("drift_norm", "signflip", "logvol")),
]


# ============================
# DATA UTILS
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
    return r[~np.isnan(r)]


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
        print(f"Loading {len(txt_files)} files...")
        for fp in txt_files:
            r = read_returns_from_txt_path(fp)
            base = os.path.basename(fp)
            series.append(r)
            names.append(base)
    elif DATA_SOURCE == "local_folder":
        _maybe_extract_local_zip()
        pattern = os.path.join(LOCAL_FOLDER, TXT_GLOB)
        txt_files = sorted(glob.glob(pattern))
        if not txt_files:
            raise ValueError(f"No .txt files found at: {pattern}")
        print(f"Loading {len(txt_files)} files...")
        for fp in txt_files:
            r = read_returns_from_txt_path(fp)
            base = os.path.basename(fp)
            series.append(r)
            names.append(base)
    elif DATA_SOURCE == "zip_upload":
        if not is_colab():
            raise RuntimeError("DATA_SOURCE='zip_upload' is intended for Colab.")
        from google.colab import files  # type: ignore

        uploaded = files.upload()
        zip_files = list(uploaded.keys())
        print(f"Loading {len(zip_files)} files...")
        for zf in zip_files:
            r, inner_name = read_returns_from_zip(zf)
            series.append(r)
            names.append(inner_name if inner_name else os.path.basename(zf))
    else:
        raise ValueError("DATA_SOURCE must be 'drive', 'local_folder', or 'zip_upload'")

    return series, names


def parse_asset_part(filename: str) -> Tuple[str, Optional[int]]:
    base = os.path.basename(filename).replace(".txt", "")
    match = re.match(r"^(.*)_part(\d+)$", base)
    if match:
        return match.group(1), int(match.group(2))
    return base, None


def group_by_asset(
    series: List[np.ndarray], names: List[str]
) -> Dict[str, Dict[Optional[int], Tuple[np.ndarray, str]]]:
    out: Dict[str, Dict[Optional[int], Tuple[np.ndarray, str]]] = {}
    for r, nm in zip(series, names):
        asset, part = parse_asset_part(nm)
        out.setdefault(asset, {})[part] = (r, nm)
    return out


def make_prefix(r: np.ndarray) -> np.ndarray:
    p = np.empty(len(r) + 1, dtype=np.float64)
    p[0] = 0.0
    np.cumsum(r.astype(np.float64), out=p[1:])
    return p


def future_sum(prefix: np.ndarray, t_end: np.ndarray, horizon: int) -> np.ndarray:
    return prefix[t_end + horizon + 1] - prefix[t_end + 1]


# ============================
# FEATURES + TARGETS
# ============================


def feats_from_windows(
    win: np.ndarray, include: Tuple[str, ...]
) -> Tuple[np.ndarray, List[str]]:
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

    X = np.hstack(out).astype(np.float32)
    return X, names


def build_Xy_for_times(
    r: np.ndarray, t_end: np.ndarray, prefix: np.ndarray, horizon: int, spec: FeatureSpec
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feats = []
    feat_names: List[str] = []
    r32 = r.astype(np.float32, copy=False)

    for ww in spec.scales:
        view = sliding_window_view(r32, ww)
        starts = (t_end - ww + 1).astype(np.int64)
        win = view[starts]
        Xw, names = feats_from_windows(win, spec.include)
        feats.append(Xw)
        feat_names += [f"{n}_{ww}" for n in names]

    X = np.hstack(feats).astype(np.float32)
    y = future_sum(prefix, t_end, horizon).astype(np.float32)
    return X, y, feat_names


def vol_at_times(r: np.ndarray, t_end: np.ndarray, window: int) -> np.ndarray:
    r32 = r.astype(np.float32, copy=False)
    view = sliding_window_view(r32, window)
    starts = (t_end - window + 1).astype(np.int64)
    win = view[starts]
    return win.std(axis=1, ddof=1).astype(np.float32) + 1e-8


def labels_from_target(
    y: np.ndarray,
    vol: np.ndarray,
    target_type: str,
    train_quantiles: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    if target_type == "raw_quantile":
        if train_quantiles is None:
            q1, q2 = np.quantile(y, [1 / 3, 2 / 3])
        else:
            q1, q2 = train_quantiles
        labels = np.where(y <= q1, 0, np.where(y <= q2, 1, 2)).astype(np.int8)
        return labels, (float(q1), float(q2))

    if target_type == "vol_std":
        low, high = VOL_STD_THRESH
        y_norm = y / (vol * math.sqrt(HORIZON))
        labels = np.where(y_norm <= low, 0, np.where(y_norm <= high, 1, 2)).astype(
            np.int8
        )
        return labels, (float(low), float(high))

    raise ValueError(f"Unknown target_type: {target_type}")


def score_positions(score: np.ndarray, thr: float, mode: str) -> np.ndarray:
    if mode == "long_only":
        return (score > thr).astype(np.float32)
    if mode == "short_only":
        return -(score < -thr).astype(np.float32)
    if mode == "long_short":
        pos = np.zeros_like(score, dtype=np.float32)
        pos[score > thr] = 1.0
        pos[score < -thr] = -1.0
        return pos
    raise ValueError(f"Unknown mode: {mode}")


def pnl_stats(pnl_bps: np.ndarray) -> Tuple[float, float, float, float]:
    if len(pnl_bps) == 0:
        return 0.0, 0.0, 0.0, 0.0
    mu = float(pnl_bps.mean())
    sd = float(pnl_bps.std(ddof=1)) if len(pnl_bps) > 1 else 0.0
    t_stat = float(mu / (sd / math.sqrt(len(pnl_bps)) + 1e-12)) if len(pnl_bps) > 1 else 0.0
    win_rate = float((pnl_bps > 0).mean())
    return mu, sd, t_stat, win_rate


# ============================
# EXPERIMENT ENGINE
# ============================


def run_pipeline() -> None:
    series, names = load_datasets()
    data_by_asset = group_by_asset(series, names)
    assets = sorted(data_by_asset.keys())
    print(f"\nAssets: {len(assets)} | {', '.join(assets[:5])}...")

    results: List[Dict[str, object]] = []

    for spec in FEATURE_SPECS:
        for target_type in TARGET_TYPES:
            print(f"\n>>> Running: {spec.name} | Target: {target_type}")

            X_tr_list: List[np.ndarray] = []
            y_tr_list: List[np.ndarray] = []
            v_tr_list: List[np.ndarray] = []

            X_va_list: List[np.ndarray] = []
            y_va_list: List[np.ndarray] = []
            v_va_list: List[np.ndarray] = []

            X_te_list: List[np.ndarray] = []
            y_te_list: List[np.ndarray] = []
            v_te_list: List[np.ndarray] = []
            test_slices: List[Tuple[str, int, int]] = []

            for asset in assets:
                parts = data_by_asset[asset]
                if 1 not in parts:
                    continue
                r1, _ = parts[1]
                prefix1 = make_prefix(r1)

                Wmax = max(spec.scales + (VOL_STD_WINDOW,))
                if len(r1) < (Wmax + HORIZON + 2000):
                    continue

                train_end = int(len(r1) * TRAIN_FRAC)
                val_end = int(len(r1) * (TRAIN_FRAC + VAL_FRAC))

                t_end_train = np.arange(
                    Wmax - 1, train_end - HORIZON, SAMPLE_STEP, dtype=np.int64
                )
                t_end_val = np.arange(
                    max(Wmax - 1, train_end),
                    val_end - HORIZON,
                    DECISION_STEP,
                    dtype=np.int64,
                )

                if len(t_end_train) == 0 or len(t_end_val) == 0:
                    continue

                Xtr, ytr, _ = build_Xy_for_times(r1, t_end_train, prefix1, HORIZON, spec)
                vtr = vol_at_times(r1, t_end_train, VOL_STD_WINDOW)
                Xva, yva, _ = build_Xy_for_times(r1, t_end_val, prefix1, HORIZON, spec)
                vva = vol_at_times(r1, t_end_val, VOL_STD_WINDOW)

                if len(Xtr) > 100_000:
                    idx = np.linspace(0, len(Xtr) - 1, 100_000).astype(int)
                    Xtr, ytr, vtr = Xtr[idx], ytr[idx], vtr[idx]

                X_tr_list.append(Xtr)
                y_tr_list.append(ytr)
                v_tr_list.append(vtr)

                X_va_list.append(Xva)
                y_va_list.append(yva)
                v_va_list.append(vva)

                if 2 in parts:
                    r2, _ = parts[2]
                    prefix2 = make_prefix(r2)
                    t_end_test = np.arange(
                        Wmax - 1, len(r2) - HORIZON, DECISION_STEP, dtype=np.int64
                    )
                    Xte, yte, _ = build_Xy_for_times(
                        r2, t_end_test, prefix2, HORIZON, spec
                    )
                    vte = vol_at_times(r2, t_end_test, VOL_STD_WINDOW)
                else:
                    t_end_test = np.arange(
                        max(Wmax - 1, val_end),
                        len(r1) - HORIZON,
                        DECISION_STEP,
                        dtype=np.int64,
                    )
                    if len(t_end_test) == 0:
                        continue
                    Xte, yte, _ = build_Xy_for_times(
                        r1, t_end_test, prefix1, HORIZON, spec
                    )
                    vte = vol_at_times(r1, t_end_test, VOL_STD_WINDOW)

                start_loc = sum(len(x) for x in X_te_list)
                X_te_list.append(Xte)
                y_te_list.append(yte)
                v_te_list.append(vte)
                test_slices.append((asset, start_loc, start_loc + len(Xte)))

            if not X_tr_list or not X_va_list or not X_te_list:
                print("  [SKIP] Not enough data after filtering.")
                continue

            X_tr = np.concatenate(X_tr_list)
            y_tr_raw = np.concatenate(y_tr_list)
            v_tr = np.concatenate(v_tr_list)

            X_va = np.concatenate(X_va_list)
            y_va_raw = np.concatenate(y_va_list)
            v_va = np.concatenate(v_va_list)

            X_te = np.concatenate(X_te_list)
            y_te_raw = np.concatenate(y_te_list)
            v_te = np.concatenate(v_te_list)

            labels_tr, thresh = labels_from_target(y_tr_raw, v_tr, target_type)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)
            X_te_s = scaler.transform(X_te)

            clf = HistGradientBoostingClassifier(
                max_depth=3, random_state=RANDOM_SEED, early_stopping=True
            )
            clf.fit(X_tr_s, labels_tr)

            score_tr = clf.predict_proba(X_tr_s)
            score_tr = score_tr[:, 2] - score_tr[:, 0]
            score_va = clf.predict_proba(X_va_s)
            score_va = score_va[:, 2] - score_va[:, 0]
            score_te = clf.predict_proba(X_te_s)
            score_te = score_te[:, 2] - score_te[:, 0]

            for thresh_source in THRESH_SOURCES:
                source_scores = score_tr if thresh_source == "train" else score_va
                for pct in THRESH_PCTS:
                    thr = float(np.percentile(source_scores, pct))
                    for mode in MODES:
                        pos = score_positions(score_te, thr, mode)
                        traded = pos != 0
                        pnl = (pos * y_te_raw) - traded.astype(np.float32) * (
                            COST_BPS / 10_000.0
                        )
                        pnl_bps = pnl[traded] * 10_000.0
                        mu, sd, t_stat, win_rate = pnl_stats(pnl_bps)

                        asset_ts = []
                        for asset, start, end in test_slices:
                            a_scores = score_te[start:end]
                            a_y = y_te_raw[start:end]
                            a_pos = score_positions(a_scores, thr, mode)
                            a_traded = a_pos != 0
                            if a_traded.sum() < 5:
                                continue
                            a_pnl = (a_pos[a_traded] * a_y[a_traded]) * 10_000.0 - COST_BPS
                            if len(a_pnl) > 1:
                                t = a_pnl.mean() / (
                                    a_pnl.std(ddof=1) / math.sqrt(len(a_pnl)) + 1e-12
                                )
                                asset_ts.append(float(t))

                        positive_assets = sum(1 for t in asset_ts if t > 0.5)
                        print(
                            "  "
                            f"Source={thresh_source} | pct={pct} | mode={mode} | "
                            f"Trades={len(pnl_bps)} | Mean={mu:.2f}bps | "
                            f"T-Stat={t_stat:.2f} | Win%={win_rate * 100:.1f} | "
                            f"Assets T>0.5: {positive_assets}/{len(asset_ts)}"
                        )

                        results.append(
                            dict(
                                spec=spec.name,
                                target=target_type,
                                thresh_source=thresh_source,
                                pct=pct,
                                mode=mode,
                                n_trades=len(pnl_bps),
                                mean_bps=mu,
                                t_stat=t_stat,
                                win_rate=win_rate,
                                positive_assets=positive_assets,
                                label_thresh=thresh,
                            )
                        )

    print("\n================ FINAL COMPARISON ================")
    df = pd.DataFrame(results)
    if not df.empty:
        print(
            df.sort_values(["t_stat", "mean_bps"], ascending=False).to_string(index=False)
        )
        df.to_csv("robustness_results.csv", index=False)
        print("==================================================")
        print("Saved: robustness_results.csv")
    else:
        print("No results computed.")
        print("==================================================")


if __name__ == "__main__":
    run_pipeline()
