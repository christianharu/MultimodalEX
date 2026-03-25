#!/usr/bin/env python3
"""
Evaluate multimodal baseline predictions on the test split.

Expected input CSV:
- original dataset columns
- split column with values train/val/test
- prediction column (default: multimodal_lstm)
"""

import argparse
import json
import os
import sys
from typing import Dict

import pandas as pd


def load_metrics_library():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_db_dir = os.path.join(script_dir, "PBC4emp", "processed_databases")

    if processed_db_dir not in sys.path:
        sys.path.insert(0, processed_db_dir)

    try:
        import bert_metrics_lib as metrics_lib  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "Could not import processed_databases metrics library "
            "(`classifiers/PBC4emp/processed_databases/bert_metrics_lib.py`)."
        ) from exc
    return metrics_lib


def default_prediction_column_name(model_type: str, feature_set: str) -> str:
    if model_type == "lstm":
        if feature_set == "av":
            return "multimodal_lstm"
        if feature_set == "mimicry":
            return "multimodal_lstm_mimicry"
        if feature_set == "av_mimicry":
            return "multimodal_lstm_av_mimicry"
    elif model_type == "bert":
        if feature_set == "av":
            return "multimodal_bert"
        if feature_set == "mimicry":
            return "multimodal_bert_mimicry"
        if feature_set == "av_mimicry":
            return "multimodal_bert_av_mimicry"
    elif model_type == "rf":
        if feature_set == "av":
            return "multimodal_rf"
        if feature_set == "mimicry":
            return "multimodal_rf_mimicry"
        if feature_set == "av_mimicry":
            return "multimodal_rf_av_mimicry"
    elif model_type == "svm":
        if feature_set == "av":
            return "multimodal_svm"
        if feature_set == "mimicry":
            return "multimodal_svm_mimicry"
        if feature_set == "av_mimicry":
            return "multimodal_svm_av_mimicry"
    elif model_type == "pbc":
        if feature_set == "av":
            return "multimodal_pbc"
        if feature_set == "mimicry":
            return "multimodal_pbc_mimicry"
        if feature_set == "av_mimicry":
            return "multimodal_pbc_av_mimicry"
    elif model_type == "llm":
        if feature_set == "av":
            return "multimodal_llm_av"
        raise ValueError(
            "LLM labels are currently available only for feature-set 'av'. "
            "Use --feature-set av, or pass --pred-col explicitly if you add custom LLM columns."
        )
    raise ValueError(
        "Could not infer prediction column. "
        "Use --pred-col explicitly or valid --model-type/--feature-set combination."
    )


def compute_metric_set(y_true, y_pred, metrics_lib, average: str) -> Dict:
    result = metrics_lib.compute_metric_set(y_true, y_pred, average=average)
    # Keep evaluator field names explicit while preserving notebook-aligned values.
    return {
        "n_test_samples": result["n_samples"],
        "accuracy": result["accuracy"],
        "cem": result["cem"],
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["f1"],
        "average": result["average"],
    }


def compute_metrics(test_df: pd.DataFrame, label_col: str, pred_col: str, average: str) -> Dict:
    y_true = pd.to_numeric(test_df[label_col], errors="coerce")
    y_pred = pd.to_numeric(test_df[pred_col], errors="coerce")

    valid_mask = y_true.notna() & y_pred.notna()
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(f"Dropping {dropped} rows with non-numeric labels/predictions before scoring.")

    y_true = y_true[valid_mask].astype(int).to_numpy()
    y_pred = y_pred[valid_mask].astype(int).to_numpy()

    if len(y_true) == 0:
        raise ValueError("No valid test rows available after cleaning.")

    metrics_lib = load_metrics_library()

    multiclass_metrics = compute_metric_set(y_true, y_pred, metrics_lib, average=average)

    # Binary mapping: 0 -> 0, {1,2} -> 1
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    binary_metrics = compute_metric_set(y_true_bin, y_pred_bin, metrics_lib, average=average)

    return {
        "label_mapping_binary": {"0": 0, "1": 1, "2": 1},
        "metrics_library": "classifiers/PBC4emp/processed_databases/bert_metrics_lib.py",
        "multiclass": multiclass_metrics,
        "binary": binary_metrics,
    }


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(
        script_dir,
        "PBC4emp",
        "processed_databases",
        "mex_fusion_lex_llm_multimodal_baselines.csv",
    )
    default_metrics_out = os.path.join(
        script_dir,
        "PBC4emp",
        "results",
        "multimodal_lstm_mex",
        "test_metrics_from_baseline_csv.json",
    )

    parser = argparse.ArgumentParser(description="Compute test metrics from baseline CSV predictions.")
    parser.add_argument("--csv-path", type=str, default=default_csv)
    parser.add_argument("--label-col", type=str, default="empathy")
    parser.add_argument(
        "--pred-col",
        type=str,
        default="",
        help="Prediction column to evaluate. If omitted, inferred from --model-type and --feature-set.",
    )
    parser.add_argument("--model-type", type=str, choices=["lstm", "bert", "rf", "svm", "pbc", "llm"], default="lstm")
    parser.add_argument("--feature-set", type=str, choices=["av", "mimicry", "av_mimicry"], default="av")
    parser.add_argument("--split-col", type=str, default="split")
    parser.add_argument("--test-value", type=str, default="test")
    parser.add_argument("--average", type=str, choices=["macro", "weighted"], default="macro")
    parser.add_argument("--metrics-out", type=str, default=default_metrics_out)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    pred_col = args.pred_col if args.pred_col else default_prediction_column_name(args.model_type, args.feature_set)
    print(f"Using prediction column: {pred_col}")

    required = [args.label_col, pred_col, args.split_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    test_df = df[df[args.split_col] == args.test_value].copy()
    if test_df.empty:
        raise ValueError(
            f"No rows found for {args.split_col} == '{args.test_value}'. "
            "Make sure the baseline CSV contains split labels."
        )

    metrics = compute_metrics(test_df, args.label_col, pred_col, average=args.average)
    metrics["prediction_column"] = pred_col
    metrics["model_type"] = args.model_type
    metrics["feature_set"] = args.feature_set
    metrics_dir = os.path.dirname(args.metrics_out)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model_name = args.model_type.upper()
    feature_name = {
        "av": "AV",
        "mimicry": "MIMICRY",
        "av_mimicry": "AV + MIMICRY",
    }.get(args.feature_set, args.feature_set.upper())

    print(f"{model_name} {feature_name} 3-level")
    for k, v in metrics["multiclass"].items():
        print(f"{k}: {v}")

    print(f"\n{model_name} {feature_name} 2-level")
    for k, v in metrics["binary"].items():
        print(f"{k}: {v}")
    print(f"Saved metrics JSON to: {args.metrics_out}")


if __name__ == "__main__":
    main()
