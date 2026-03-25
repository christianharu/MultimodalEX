#!/usr/bin/env python3
"""
Train a multimodal SVM empathy classifier on:
classifiers/PBC4emp/processed_databases/mex_fusion_lex_llm.csv

Feature variants:
- av: structured multimodal features (without mimicry_video_face)
- mimicry: non-AV structured features + mimicry_video_face
- av_mimicry: structured multimodal features + mimicry_video_face

Target classes:
empathy in {0, 1, 2}
"""

import argparse
import json
import os
import pickle
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC


VIDEO_AV_COLS = [
    "arousal_speaker_video",
    "valence_speaker_video",
    "arousal_listener_video",
    "valence_listener_video",
]
MIMICRY_VIDEO_COL = "mimicry_video_face"
STRUCTURED_BASE_COLS = [
    "s_word_len",
    "l_word_len",
    "arousal_listener_video",
    "valence_listener_video",
    "who",
    "sentiment",
    "emotional_reaction",
    "interpretations",
    "explorations",
    "intent",
    "arousal_speaker_video",
    "valence_speaker_video",
    "speaker_who",
    "speaker_sentiment",
    "dominance_speaker",
    "dominance_listener",
]
MIMICRY_FEATURE_COLS = [
    "s_word_len",
    "l_word_len",
    "who",
    "sentiment",
    "emotional_reaction",
    "interpretations",
    "explorations",
    "intent",
    "speaker_who",
    "speaker_sentiment",
    "dominance_speaker",
    "dominance_listener",
    MIMICRY_VIDEO_COL,
]
STRUCTURED_ALL_COLS = STRUCTURED_BASE_COLS + [MIMICRY_VIDEO_COL]

LABEL_COL = "empathy"
EXPECTED_LABELS = [0, 1, 2]
FEATURE_SET_CHOICES = ["av", "mimicry", "av_mimicry"]
SVM_PARAMS = {
    "kernel": "rbf",
    "gamma": "auto",
    "degree": 2,
    "C": 1000,
}
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def validate_and_prepare_dataframe(csv_path: str, feature_set: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [LABEL_COL]
    if feature_set in {"av", "av_mimicry"}:
        required_cols += STRUCTURED_BASE_COLS
    if feature_set == "mimicry":
        required_cols += MIMICRY_FEATURE_COLS
    if feature_set == "av_mimicry":
        required_cols += [MIMICRY_VIDEO_COL]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: %s" % missing)

    numeric_cols = [
        "s_word_len",
        "l_word_len",
        "arousal_listener_video",
        "valence_listener_video",
        "emotional_reaction",
        "interpretations",
        "explorations",
        "intent",
        "arousal_speaker_video",
        "valence_speaker_video",
        "dominance_speaker",
        "dominance_listener",
        MIMICRY_VIDEO_COL,
    ]
    categorical_cols = ["who", "sentiment", "speaker_who", "speaker_sentiment"]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    for c in VIDEO_AV_COLS:
        if c in df.columns:
            df[c] = df[c].clip(-1.0, 1.0)

    if MIMICRY_VIDEO_COL in df.columns:
        df[MIMICRY_VIDEO_COL] = df[MIMICRY_VIDEO_COL].clip(0.0, 1.0)

    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].fillna("unknown").astype(str)

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    if df[LABEL_COL].isna().any():
        bad_rows = df[df[LABEL_COL].isna()].index.tolist()[:10]
        raise ValueError("Found non-numeric labels in '%s' at rows (sample): %s" % (LABEL_COL, bad_rows))
    df[LABEL_COL] = df[LABEL_COL].astype(np.int64)

    uniq = sorted(df[LABEL_COL].unique().tolist())
    if not set(uniq).issubset(set(EXPECTED_LABELS)):
        raise ValueError("Unexpected labels found: %s. Expected subset of %s" % (uniq, EXPECTED_LABELS))

    return df


def build_context_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "av":
        raw = df[STRUCTURED_BASE_COLS].copy()
    elif feature_set == "mimicry":
        raw = df[MIMICRY_FEATURE_COLS].copy()
    elif feature_set == "av_mimicry":
        raw = df[STRUCTURED_ALL_COLS].copy()
    else:
        raise ValueError("Unknown feature_set: %s. Expected one of %s" % (feature_set, FEATURE_SET_CHOICES))

    categorical_cols = [c for c in ["who", "sentiment", "speaker_who", "speaker_sentiment"] if c in raw.columns]
    encoded = pd.get_dummies(raw, columns=categorical_cols, drop_first=False)
    return encoded


def build_inputs(df: pd.DataFrame, feature_set: str) -> Tuple[np.ndarray, np.ndarray, int]:
    context_df = build_context_features(df, feature_set)
    x = context_df.to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=np.int64)
    return x, y, int(x.shape[1])


def random_split_indices(
    n_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("Split ratios must sum to 1.0, got %.6f" % ratio_sum)
    if n_samples < 3:
        raise ValueError("Need at least 3 samples for train/val/test split, got %d" % n_samples)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            "Invalid split sizes from ratios: train=%d, val=%d, test=%d." % (n_train, n_val, n_test)
        )

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train : n_train + n_val]
    test_idx = shuffled[n_train + n_val :]
    return train_idx, val_idx, test_idx


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    cm = confusion_matrix(y_true, y_pred, labels=EXPECTED_LABELS).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=EXPECTED_LABELS,
        target_names=[str(i) for i in EXPECTED_LABELS],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def default_prediction_column_name(feature_set: str) -> str:
    if feature_set == "av":
        return "multimodal_svm"
    if feature_set == "mimicry":
        return "multimodal_svm_mimicry"
    if feature_set == "av_mimicry":
        return "multimodal_svm_av_mimicry"
    raise ValueError("Unknown feature_set: %s" % feature_set)


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "PBC4emp", "processed_databases", "mex_fusion_lex_llm.csv")
    default_out = os.path.join(script_dir, "PBC4emp", "results", "multimodal_svm_mex")

    parser = argparse.ArgumentParser(description="Train multimodal SVM empathy classifier (0/1/2).")
    parser.add_argument("--csv-path", type=str, default=default_csv)
    parser.add_argument("--output-dir", type=str, default=default_out)
    parser.add_argument(
        "--baseline-csv-out",
        type=str,
        default="",
        help="Optional full path for [original]_multimodal_baselines.csv export.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=FEATURE_SET_CHOICES,
        default="av",
        help="Feature variant: av, mimicry, av_mimicry.",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default="",
        help="Prediction column name for baseline CSV. If empty, inferred from --feature-set.",
    )

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset from: %s" % args.csv_path)
    df = validate_and_prepare_dataframe(args.csv_path, feature_set=args.feature_set)
    print("Dataset shape: %s" % (df.shape,))
    print("Label distribution:")
    print(df[LABEL_COL].value_counts().sort_index())
    print("Feature set: %s" % args.feature_set)
    print("Using structured feature columns from dataset.")
    if args.feature_set == "av":
        print("Columns: %s" % STRUCTURED_BASE_COLS)
    elif args.feature_set == "mimicry":
        print("Columns: %s" % MIMICRY_FEATURE_COLS)
    else:
        print("Columns: %s" % STRUCTURED_ALL_COLS)

    x, y, context_dim = build_inputs(df=df, feature_set=args.feature_set)
    print("Input tensor shape: %s (N, feature_dim=%d)" % (x.shape, context_dim))

    train_idx, val_idx, test_idx = random_split_indices(
        n_samples=len(y),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(
        "Split sizes | train=%d (%.2f%%), val=%d (%.2f%%), test=%d (%.2f%%)"
        % (
            len(train_idx),
            100.0 * len(train_idx) / len(y),
            len(val_idx),
            100.0 * len(val_idx) / len(y),
            len(test_idx),
            100.0 * len(test_idx) / len(y),
        )
    )

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    model = SVC(**SVM_PARAMS)
    model.fit(x_train, y_train)

    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)
    y_all_pred = model.predict(x)

    val_eval = evaluate_predictions(y_val, y_val_pred)
    test_eval = evaluate_predictions(y_test, y_test_pred)

    feature_tag = args.feature_set
    metrics_path = os.path.join(args.output_dir, "metrics_%s.json" % feature_tag)
    model_path = os.path.join(args.output_dir, "best_multimodal_svm_mex_%s.pkl" % feature_tag)
    val_preds_path = os.path.join(args.output_dir, "validation_predictions_%s.csv" % feature_tag)
    test_preds_path = os.path.join(args.output_dir, "test_predictions_%s.csv" % feature_tag)
    prediction_col = args.prediction_column or default_prediction_column_name(args.feature_set)

    if args.baseline_csv_out:
        baseline_csv_path = args.baseline_csv_out
    else:
        original_name = os.path.splitext(os.path.basename(args.csv_path))[0]
        baseline_csv_path = os.path.join(os.path.dirname(args.csv_path), "%s_multimodal_baselines.csv" % original_name)

    meta = {
        "args": vars(args),
        "feature_set": args.feature_set,
        "prediction_column": prediction_col,
        "context_feature_dim": context_dim,
        "input_feature_dim": context_dim,
        "baseline_csv_path": baseline_csv_path,
        "split": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        },
        "model_params": SVM_PARAMS,
        "validation_metrics": val_eval,
        "test_metrics": test_eval,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_set": args.feature_set,
                "context_dim": context_dim,
                "prediction_column": prediction_col,
                "model_params": SVM_PARAMS,
                "args": vars(args),
            },
            f,
        )

    val_df = df.iloc[val_idx].copy()
    val_df["pred_empathy"] = y_val_pred
    val_df["split"] = "val"
    val_df.to_csv(val_preds_path, index=False)

    test_df = df.iloc[test_idx].copy()
    test_df["pred_empathy"] = y_test_pred
    test_df["split"] = "test"
    test_df.to_csv(test_preds_path, index=False)

    split_labels = np.full(len(df), "train", dtype=object)
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    if os.path.exists(baseline_csv_path):
        try:
            existing_baseline_df = pd.read_csv(baseline_csv_path)
        except Exception:  # noqa: BLE001
            existing_baseline_df = None
    else:
        existing_baseline_df = None

    if existing_baseline_df is not None and len(existing_baseline_df) == len(df):
        baseline_df = existing_baseline_df
        for c in df.columns:
            baseline_df[c] = df[c]
    else:
        baseline_df = df.copy()

    baseline_df["split"] = split_labels
    baseline_df[prediction_col] = y_all_pred
    baseline_df.to_csv(baseline_csv_path, index=False)

    print("\nTraining completed.")
    print("Validation weighted F1: %.4f" % val_eval["f1_weighted"])
    print("Test weighted F1: %.4f" % test_eval["f1_weighted"])
    print("Saved model: %s" % model_path)
    print("Saved metrics: %s" % metrics_path)
    print("Saved validation predictions: %s" % val_preds_path)
    print("Saved test predictions: %s" % test_preds_path)
    print("Saved baseline CSV: %s" % baseline_csv_path)
    print("Saved prediction column: %s" % prediction_col)


if __name__ == "__main__":
    main()
