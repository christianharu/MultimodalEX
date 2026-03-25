import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from mimicry_face import _load_status_files, compute_mimicry_scores_for_file


REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DATASET_DIR = REPO_ROOT / "video-based_dataset"
PREDICTIONS_FILE = REPO_ROOT / "classifiers" / "PBC4emp" / "mEX_all_predictions.csv"
OUTPUT_DIR = VIDEO_DATASET_DIR / "mimicry_threshold_random10"
RAW_SCORES_CACHE = OUTPUT_DIR / "raw_mimicry_scores.csv"
LABEL_COLUMN = "mimicry_video_joint"
THRESHOLDS = np.round(np.arange(0.0, 0.1001, 0.001), 3)
DISPLAY_TICKS = np.round(np.arange(0.0, 0.1001, 0.01), 2)
RANDOM_SEED = 42
SAMPLE_SIZE = 10


def compute_raw_scores():
    score_frames = []
    for status_file in tqdm(_load_status_files(str(VIDEO_DATASET_DIR)), desc="Raw mimicry scores"):
        conversation_id = status_file.replace("_speaker_status.csv", "")
        with contextlib.redirect_stdout(io.StringIO()):
            scores_df = compute_mimicry_scores_for_file(str(VIDEO_DATASET_DIR), conversation_id)
        scores_df["id"] = conversation_id
        score_frames.append(scores_df)
    raw_scores = pd.concat(score_frames, ignore_index=True)
    return raw_scores[["id", "exchange", "mean_mimicry_face", "mean_mimicry_pose"]]


def load_or_compute_raw_scores():
    if RAW_SCORES_CACHE.exists():
        return pd.read_csv(RAW_SCORES_CACHE)

    raw_scores = compute_raw_scores()
    RAW_SCORES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    raw_scores.to_csv(RAW_SCORES_CACHE, index=False)
    return raw_scores


def load_ground_truth():
    gold = pd.read_csv(PREDICTIONS_FILE).copy()
    gold["exchange"] = gold.groupby("id").cumcount()
    return gold


def sample_rows(gold):
    rng = np.random.default_rng(RANDOM_SEED)
    labels = sorted(gold[LABEL_COLUMN].dropna().astype(int).unique().tolist())
    per_label = SAMPLE_SIZE // max(len(labels), 1)

    sampled_parts = []
    sampled_index = set()

    for label in labels:
        candidates = gold[gold[LABEL_COLUMN] == label]
        take = min(per_label, len(candidates))
        if take == 0:
            continue
        picked = candidates.sample(n=take, random_state=RANDOM_SEED + label)
        sampled_parts.append(picked)
        sampled_index.update(picked.index.tolist())

    remaining = SAMPLE_SIZE - sum(len(part) for part in sampled_parts)
    if remaining > 0:
        leftovers = gold.loc[~gold.index.isin(sampled_index)]
        extra_idx = rng.choice(leftovers.index.to_numpy(), size=remaining, replace=False)
        sampled_parts.append(leftovers.loc[extra_idx])

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return sampled


def compute_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float((y_true == y_pred).mean())


def evaluate_thresholds(sampled, raw_scores):
    merged = sampled.merge(
        raw_scores[["id", "exchange", "mean_mimicry_face", "mean_mimicry_pose"]],
        on=["id", "exchange"],
        how="inner",
    )

    results = []
    for threshold in THRESHOLDS:
        face_predicted = (merged["mean_mimicry_face"] > float(threshold)).astype(int)
        pose_predicted = (merged["mean_mimicry_pose"] > float(threshold)).astype(int)
        face_accuracy = compute_accuracy(merged[LABEL_COLUMN], face_predicted)
        pose_accuracy = compute_accuracy(merged[LABEL_COLUMN], pose_predicted)
        results.append(
            {
                "threshold_face": float(threshold),
                "face_accuracy": face_accuracy,
                "pose_accuracy": pose_accuracy,
                "rows_evaluated": len(merged),
                "face_positive_rate": float(face_predicted.mean()),
                "pose_positive_rate": float(pose_predicted.mean()),
            }
        )

    return merged, pd.DataFrame(results)


def plot_results(results_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        results_df["threshold_face"],
        results_df["face_accuracy"],
        linewidth=2.0,
        marker="o",
        markersize=4,
        markevery=10,
        label="Mimicry Face",
    )
    ax.plot(
        results_df["threshold_face"],
        results_df["pose_accuracy"],
        linewidth=2.0,
        marker="s",
        markersize=4,
        markevery=10,
        label="Mimicry Pose",
    )
    ax.set_xlabel("Mimicry Threshold")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(DISPLAY_TICKS)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    png_path = OUTPUT_DIR / "random10_threshold_accuracy.png"
    pdf_path = OUTPUT_DIR / "random10_threshold_accuracy.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_scores = load_or_compute_raw_scores()
    gold = load_ground_truth()
    sampled = sample_rows(gold)
    sampled.to_csv(OUTPUT_DIR / "random10_sample.csv", index=False)

    merged, results_df = evaluate_thresholds(sampled, raw_scores)
    merged.to_csv(OUTPUT_DIR / "random10_sample_with_scores.csv", index=False)
    results_df.to_csv(OUTPUT_DIR / "random10_threshold_metrics.csv", index=False)

    png_path, pdf_path = plot_results(results_df)

    print("Random seed:", RANDOM_SEED)
    print("Sample size:", SAMPLE_SIZE)
    print("Ground truth label:", LABEL_COLUMN)
    print("Saved sample to:", OUTPUT_DIR / "random10_sample.csv")
    print("Saved metrics to:", OUTPUT_DIR / "random10_threshold_metrics.csv")
    print("Saved plot to:", png_path)
    print("Saved plot to:", pdf_path)


if __name__ == "__main__":
    main()
