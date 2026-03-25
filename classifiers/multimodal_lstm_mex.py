#!/usr/bin/env python3
"""
Train a multimodal LSTM empathy classifier on:
classifiers/PBC4emp/processed_databases/mex_fusion_lex_llm.csv

Feature variants:
- av: text + speaker/listener arousal/valence video cues
- mimicry: text + mimicry cue only
- av_mimicry: text + arousal/valence + mimicry

Target classes:
empathy in {0, 1, 2}
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Compatibility shim for older ONNX versions with newer NumPy.
# ONNX may still reference the removed alias `np.object`.
if not hasattr(np, "object"):
    np.object = object  # type: ignore[attr-defined]

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


TEXT_COLS = ["speaker_utterance", "listener_utterance"]
VIDEO_AV_COLS = [
    "arousal_speaker_video",
    "valence_speaker_video",
    "arousal_listener_video",
    "valence_listener_video",
]
VIDEO_COLS = VIDEO_AV_COLS
MIMICRY_COL = "mimicry_video_face"
LABEL_COL = "empathy"
EXPECTED_LABELS = [0, 1, 2]
FEATURE_SET_CHOICES = ["av", "mimicry", "av_mimicry"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_torch_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_embedding_device(arg_value: str) -> str:
    if arg_value in {"cpu", "cuda"}:
        return arg_value
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_and_prepare_dataframe(csv_path: str, feature_set: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = TEXT_COLS + [LABEL_COL]
    if feature_set in {"av", "av_mimicry"}:
        required_cols += VIDEO_COLS
    if feature_set in {"mimicry", "av_mimicry"}:
        required_cols += [MIMICRY_COL]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in TEXT_COLS:
        df[c] = df[c].fillna("").astype(str)

    if feature_set in {"av", "av_mimicry"}:
        for c in VIDEO_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(-1.0, 1.0).astype(np.float32)

    if feature_set in {"mimicry", "av_mimicry"}:
        df[MIMICRY_COL] = pd.to_numeric(df[MIMICRY_COL], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(np.float32)

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    if df[LABEL_COL].isna().any():
        bad_rows = df[df[LABEL_COL].isna()].index.tolist()[:10]
        raise ValueError(f"Found non-numeric labels in '{LABEL_COL}' at rows (sample): {bad_rows}")

    df[LABEL_COL] = df[LABEL_COL].astype(np.int64)
    uniq = sorted(df[LABEL_COL].unique().tolist())
    if not set(uniq).issubset(set(EXPECTED_LABELS)):
        raise ValueError(f"Unexpected labels found: {uniq}. Expected subset of {EXPECTED_LABELS}")

    return df


def build_context_features(df: pd.DataFrame, feature_set: str) -> Tuple[np.ndarray, np.ndarray]:
    if feature_set == "av":
        speaker_context = df[["arousal_speaker_video", "valence_speaker_video"]].to_numpy(dtype=np.float32)
        listener_context = df[["arousal_listener_video", "valence_listener_video"]].to_numpy(dtype=np.float32)
        return speaker_context, listener_context

    mimicry = df[[MIMICRY_COL]].to_numpy(dtype=np.float32)
    if feature_set == "mimicry":
        return mimicry, mimicry

    if feature_set == "av_mimicry":
        speaker_video = df[["arousal_speaker_video", "valence_speaker_video"]].to_numpy(dtype=np.float32)
        listener_video = df[["arousal_listener_video", "valence_listener_video"]].to_numpy(dtype=np.float32)
        speaker_context = np.concatenate([speaker_video, mimicry], axis=1)
        listener_context = np.concatenate([listener_video, mimicry], axis=1)
        return speaker_context, listener_context

    raise ValueError(f"Unknown feature_set: {feature_set}. Expected one of {FEATURE_SET_CHOICES}")


def build_sequence_inputs(
    df: pd.DataFrame,
    sbert_model_name: str,
    emb_batch_size: int,
    emb_device: str,
    feature_set: str,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    encoder = SentenceTransformer(sbert_model_name, device=emb_device)

    speaker_emb = encoder.encode(
        df["speaker_utterance"].tolist(),
        batch_size=emb_batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    listener_emb = encoder.encode(
        df["listener_utterance"].tolist(),
        batch_size=emb_batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    speaker_context, listener_context = build_context_features(df, feature_set)

    speaker_step = np.concatenate([speaker_emb, speaker_context], axis=1)
    listener_step = np.concatenate([listener_emb, listener_context], axis=1)

    sequence_inputs = np.stack([speaker_step, listener_step], axis=1).astype(np.float32)
    labels = df[LABEL_COL].to_numpy(dtype=np.int64)
    text_dim = speaker_emb.shape[1]
    context_dim = speaker_context.shape[1]

    return sequence_inputs, labels, text_dim, context_dim


def random_split_indices(
    n_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}")

    if n_samples < 3:
        raise ValueError(f"Need at least 3 samples for train/val/test split, got {n_samples}")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Invalid split sizes from ratios: train={n_train}, val={n_val}, test={n_test}. "
            "Adjust ratios or dataset size."
        )

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train : n_train + n_val]
    test_idx = shuffled[n_train + n_val :]
    return train_idx, val_idx, test_idx


def build_balanced_class_weights(labels: np.ndarray, classes: List[int]) -> torch.Tensor:
    counts = np.bincount(labels, minlength=max(classes) + 1)
    n_total = len(labels)
    n_classes = len(classes)

    weights = []
    for c in classes:
        count_c = counts[c] if c < len(counts) else 0
        if count_c > 0:
            weights.append(n_total / (n_classes * count_c))
        else:
            weights.append(0.0)
    return torch.tensor(weights, dtype=torch.float32)


class MultimodalSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)


class MultimodalLSTM(nn.Module):
    """
    Better than plain concatenation:
    - Keep concatenated input format requested by user.
    - Add a light FiLM-style fusion to condition text embeddings with contextual cues.
    """

    def __init__(
        self,
        text_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.context_dim = context_dim
        self.input_dim = text_dim + self.context_dim

        self.context_to_film = nn.Sequential(
            nn.Linear(self.context_dim, text_dim * 2),
            nn.Tanh(),
        )
        self.input_norm = nn.LayerNorm(self.input_dim)

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        text = x[:, :, : self.text_dim]  # [B, 2, D]
        context = x[:, :, self.text_dim :]  # [B, 2, C]

        film = self.context_to_film(context)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        fused_text = text * (1.0 + gamma) + beta

        fused = torch.cat([fused_text, context], dim=-1)
        fused = self.input_norm(fused)

        lstm_out, _ = self.lstm(fused)  # [B, 2, 2H]

        attn_logits = self.attn(lstm_out).squeeze(-1)  # [B, 2]
        attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [B, 2, 1]
        pooled = torch.sum(lstm_out * attn_weights, dim=1)  # [B, 2H]

        return self.classifier(pooled)


@dataclass
class EpochMetrics:
    loss: float
    acc: float
    f1_weighted: float
    f1_macro: float


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses = []
    all_pred = []
    all_true = []

    for seq_x, labels in data_loader:
        seq_x = seq_x.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(seq_x)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        losses.append(loss.detach().item())
        all_pred.extend(preds.detach().cpu().tolist())
        all_true.extend(labels.detach().cpu().tolist())

    return EpochMetrics(
        loss=float(np.mean(losses)) if losses else 0.0,
        acc=accuracy_score(all_true, all_pred),
        f1_weighted=f1_score(all_true, all_pred, average="weighted", zero_division=0),
        f1_macro=f1_score(all_true, all_pred, average="macro", zero_division=0),
    )


def evaluate_with_details(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    all_pred = []
    all_true = []

    with torch.no_grad():
        for seq_x, labels in data_loader:
            seq_x = seq_x.to(device)
            labels = labels.to(device)
            logits = model(seq_x)
            preds = torch.argmax(logits, dim=1)
            all_pred.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    cm = confusion_matrix(all_true, all_pred, labels=EXPECTED_LABELS).tolist()
    report = classification_report(
        all_true,
        all_pred,
        labels=EXPECTED_LABELS,
        target_names=[str(i) for i in EXPECTED_LABELS],
        output_dict=True,
        zero_division=0,
    )
    return {
        "y_true": all_true,
        "y_pred": all_pred,
        "accuracy": accuracy_score(all_true, all_pred),
        "f1_weighted": f1_score(all_true, all_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def predict_labels(model: nn.Module, data_loader: DataLoader, device: torch.device) -> List[int]:
    model.eval()
    all_pred = []
    with torch.no_grad():
        for seq_x, _ in data_loader:
            seq_x = seq_x.to(device)
            logits = model(seq_x)
            preds = torch.argmax(logits, dim=1)
            all_pred.extend(preds.cpu().tolist())
    return all_pred


def default_prediction_column_name(feature_set: str) -> str:
    if feature_set == "av":
        return "multimodal_lstm"
    if feature_set == "mimicry":
        return "multimodal_lstm_mimicry"
    if feature_set == "av_mimicry":
        return "multimodal_lstm_av_mimicry"
    raise ValueError(f"Unknown feature_set: {feature_set}")


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "PBC4emp", "processed_databases", "mex_fusion_lex_llm.csv")
    default_out = os.path.join(script_dir, "PBC4emp", "results", "multimodal_lstm_mex")

    parser = argparse.ArgumentParser(description="Train multimodal LSTM for empathy prediction (0/1/2).")
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
        help="Feature variant: av (text+arousal/valence), mimicry (text+mimicry), av_mimicry (text+both).",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default="",
        help="Prediction column name for baseline CSV. If empty, inferred from --feature-set.",
    )

    parser.add_argument("--sbert-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-device", type=str, choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset from: {args.csv_path}")
    df = validate_and_prepare_dataframe(args.csv_path, feature_set=args.feature_set)
    print(f"Dataset shape: {df.shape}")
    print("Label distribution:")
    print(df[LABEL_COL].value_counts().sort_index())
    print(f"Feature set: {args.feature_set}")
    if args.feature_set in {"av", "av_mimicry"}:
        print(f"Using video AV columns: {VIDEO_AV_COLS}")

    emb_device = resolve_embedding_device(args.embedding_device)
    print(f"Embedding device: {emb_device}")
    x, y, text_dim, context_dim = build_sequence_inputs(
        df=df,
        sbert_model_name=args.sbert_model,
        emb_batch_size=args.embedding_batch_size,
        emb_device=emb_device,
        feature_set=args.feature_set,
    )
    print(
        f"Input tensor shape: {x.shape} "
        f"(N, seq_len=2, feature_dim={x.shape[-1]}, text_dim={text_dim}, context_dim={context_dim})"
    )

    train_idx, val_idx, test_idx = random_split_indices(
        n_samples=len(y),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    print(
        "Split sizes | "
        f"train={len(train_idx)} ({len(train_idx)/len(y):.2%}), "
        f"val={len(val_idx)} ({len(val_idx)/len(y):.2%}), "
        f"test={len(test_idx)} ({len(test_idx)/len(y):.2%})"
    )

    train_dataset = MultimodalSequenceDataset(x_train, y_train)
    val_dataset = MultimodalSequenceDataset(x_val, y_val)
    test_dataset = MultimodalSequenceDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.force_cpu,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.force_cpu,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.force_cpu,
    )
    all_loader = DataLoader(
        MultimodalSequenceDataset(x, y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.force_cpu,
    )

    device = resolve_torch_device(force_cpu=args.force_cpu)
    print(f"Training device: {device}")

    class_weights = build_balanced_class_weights(y_train, EXPECTED_LABELS).to(device)

    model = MultimodalLSTM(
        text_dim=text_dim,
        context_dim=context_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(EXPECTED_LABELS),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, args.patience // 2),
    )

    best_state = None
    best_epoch = -1
    best_val_f1 = -1.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_m = run_epoch(model, val_loader, criterion, device, optimizer=None)

        scheduler.step(val_m.f1_weighted)

        row = {
            "epoch": epoch,
            "train_loss": train_m.loss,
            "train_acc": train_m.acc,
            "train_f1_weighted": train_m.f1_weighted,
            "train_f1_macro": train_m.f1_macro,
            "val_loss": val_m.loss,
            "val_acc": val_m.acc,
            "val_f1_weighted": val_m.f1_weighted,
            "val_f1_macro": val_m.f1_macro,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_m.loss:.4f} train_f1w={train_m.f1_weighted:.4f} | "
            f"val_loss={val_m.loss:.4f} val_f1w={val_m.f1_weighted:.4f}"
        )

        if val_m.f1_weighted > best_val_f1:
            best_val_f1 = val_m.f1_weighted
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    val_eval = evaluate_with_details(model, val_loader, device)
    test_eval = evaluate_with_details(model, test_loader, device)

    feature_tag = args.feature_set
    history_path = os.path.join(args.output_dir, f"training_history_{feature_tag}.csv")
    metrics_path = os.path.join(args.output_dir, f"metrics_{feature_tag}.json")
    model_path = os.path.join(args.output_dir, f"best_multimodal_lstm_mex_{feature_tag}.pt")
    val_preds_path = os.path.join(args.output_dir, f"validation_predictions_{feature_tag}.csv")
    test_preds_path = os.path.join(args.output_dir, f"test_predictions_{feature_tag}.csv")
    prediction_col = args.prediction_column or default_prediction_column_name(args.feature_set)
    if args.baseline_csv_out:
        baseline_csv_path = args.baseline_csv_out
    else:
        original_name = os.path.splitext(os.path.basename(args.csv_path))[0]
        baseline_csv_path = os.path.join(
            os.path.dirname(args.csv_path),
            f"{original_name}_multimodal_baselines.csv",
        )

    pd.DataFrame(history).to_csv(history_path, index=False)

    meta = {
        "args": vars(args),
        "feature_set": args.feature_set,
        "prediction_column": prediction_col,
        "text_embedding_dim": text_dim,
        "context_feature_dim": context_dim,
        "input_dim_per_timestep": text_dim + context_dim,
        "sequence_length": 2,
        "baseline_csv_path": baseline_csv_path,
        "split": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        },
        "best_epoch": best_epoch,
        "best_val_f1_weighted": best_val_f1,
        "validation_metrics": {
            "accuracy": val_eval["accuracy"],
            "f1_weighted": val_eval["f1_weighted"],
            "f1_macro": val_eval["f1_macro"],
            "confusion_matrix": val_eval["confusion_matrix"],
            "classification_report": val_eval["classification_report"],
        },
        "test_metrics": {
            "accuracy": test_eval["accuracy"],
            "f1_weighted": test_eval["f1_weighted"],
            "f1_macro": test_eval["f1_macro"],
            "confusion_matrix": test_eval["confusion_matrix"],
            "classification_report": test_eval["classification_report"],
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    torch.save(
        {
            "model_state_dict": best_state,
            "text_dim": text_dim,
            "context_dim": context_dim,
            "num_classes": len(EXPECTED_LABELS),
            "args": vars(args),
        },
        model_path,
    )

    val_df = df.iloc[val_idx].copy()
    val_df["pred_empathy"] = val_eval["y_pred"]
    val_df["split"] = "val"
    val_df.to_csv(val_preds_path, index=False)

    test_df = df.iloc[test_idx].copy()
    test_df["pred_empathy"] = test_eval["y_pred"]
    test_df["split"] = "test"
    test_df.to_csv(test_preds_path, index=False)

    all_preds = predict_labels(model, all_loader, device)
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
    baseline_df[prediction_col] = all_preds
    baseline_df.to_csv(baseline_csv_path, index=False)

    print("\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation weighted F1: {best_val_f1:.4f}")
    print(f"Final validation weighted F1 (best checkpoint): {val_eval['f1_weighted']:.4f}")
    print(f"Final test weighted F1 (best checkpoint): {test_eval['f1_weighted']:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved validation predictions: {val_preds_path}")
    print(f"Saved test predictions: {test_preds_path}")
    print(f"Saved baseline CSV: {baseline_csv_path}")
    print(f"Saved prediction column: {prediction_col}")


if __name__ == "__main__":
    main()
