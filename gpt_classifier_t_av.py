import argparse
import os
import re
from typing import List, Tuple

import openai
import pandas as pd

from gpt_classifier_prompt import build_behavioral_empathy_prompt


FEATURE_SETS = {"av", "mimicry", "av_mimicry"}
DEFAULT_DATASET = "classifiers/PBC4emp/processed_databases/mex_fusion_lex_llm.csv"
DEFAULT_SPLIT_SOURCE = "classifiers/PBC4emp/processed_databases/mex_fusion_lex_llm_multimodal_baselines.csv"


def build_prompt(speaker_utterance: str, listener_utterance: str, feature_set: str, row: pd.Series) -> str:
    cues = []
    if feature_set in {"av", "av_mimicry"}:
        cues.append(
            """Arousal and Valence values from video:
- Speaker arousal: {sa}
- Speaker valence: {sv}
- Listener arousal: {la}
- Listener valence: {lv}
Arousal (range: -1 to 1) reflects the level of emotional intensity of the speaker and listener, with -1 being very calm and 1 being highly aroused.
Valence (range: -1 to 1) indicates the emotional valence of the speaker and listener, with -1 being very negative and 1 being very positive.""".format(
                sa=row["arousal_speaker_video"],
                sv=row["valence_speaker_video"],
                la=row["arousal_listener_video"],
                lv=row["valence_listener_video"],
            )
        )

    if feature_set in {"mimicry", "av_mimicry"}:
        cues.append(
            "Mimicry cue from video: mimicry_video={mimicry}. This binary value indicates whether the listener mimics the speaker's body language or head movement (1 for yes, 0 for no).".format(
                mimicry=row["mimicry_video"]
            )
        )

    return build_behavioral_empathy_prompt(
        speaker_utterance,
        listener_utterance,
        extra_cues=cues,
    )


def parse_response(text: str) -> Tuple[str, str]:
    label_match = re.search(r"classification_label\s*:\s*([0-9]+)", text, flags=re.IGNORECASE)
    reason_match = re.search(r"reason\s*:\s*(.*)", text, flags=re.IGNORECASE)

    label = label_match.group(1).strip() if label_match else ""
    reason = reason_match.group(1).strip() if reason_match else text.strip()
    return label, reason


def classify_conversation(client: openai.OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in affective computing and empathy classification."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def required_columns(feature_set: str) -> List[str]:
    cols = ["speaker_utterance", "listener_utterance"]
    if feature_set in {"av", "av_mimicry"}:
        cols.extend(
            [
                "arousal_speaker_video",
                "valence_speaker_video",
                "arousal_listener_video",
                "valence_listener_video",
            ]
        )
    if feature_set in {"mimicry", "av_mimicry"}:
        cols.append("mimicry_video")
    return cols


def validate_columns(df: pd.DataFrame, feature_set: str) -> None:
    missing = [c for c in required_columns(feature_set) if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns for '{feature_set}': {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM empathy classifier with AV/mimicry feature flags.")
    parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DATASET,
        help="Input CSV. Defaults to the same mEX file used by LSTM/BERT/RF/PBC.",
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name.")
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=sorted(FEATURE_SETS),
        default="av",
        help="Feature variant: av, mimicry, av_mimicry.",
    )
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="OpenAI API key.")
    parser.add_argument("--split-col", type=str, default="split", help="Column containing split labels.")
    parser.add_argument(
        "--split-source-csv",
        type=str,
        default=DEFAULT_SPLIT_SOURCE,
        help="CSV used to recover split labels when --split-col is missing in --database.",
    )
    parser.add_argument("--test-value", type=str, default="test", help="Value used for test split rows.")
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Run inference on all rows (overrides default test-only behavior).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional explicit output CSV path. If omitted, uses default name derived from input/model/feature-set.",
    )
    return parser.parse_args()


def attach_split_from_source(df: pd.DataFrame, split_source_csv: str, split_col: str) -> pd.DataFrame:
    split_path = split_source_csv
    if not os.path.isabs(split_path):
        split_path = os.path.join(os.getcwd(), split_path)

    if not os.path.exists(split_path):
        raise ValueError(
            f"Split column not found in dataset and split source file does not exist: {split_path}"
        )

    split_df = pd.read_csv(split_path)
    required = ["speaker_utterance", "listener_utterance", split_col]
    missing = [c for c in required if c not in split_df.columns]
    if missing:
        raise ValueError(f"Split source CSV missing required columns: {missing}")

    left = df.copy()
    right = split_df[required].copy()

    for key in ["speaker_utterance", "listener_utterance"]:
        left[key] = left[key].astype(str).str.strip()
        right[key] = right[key].astype(str).str.strip()

    merged = left.merge(right, on=["speaker_utterance", "listener_utterance"], how="left")
    matched = int(merged[split_col].notna().sum())
    print(f"Recovered split labels from source: {matched}/{len(merged)} rows matched.")
    if matched == 0:
        raise ValueError("Could not recover any split labels from split source CSV.")
    return merged


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("OpenAI API key is required. Set --api-key or OPENAI_API_KEY.")

    db_path = args.database
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.getcwd(), db_path)

    print(f"Loading dataset: {db_path}")
    print(f"Feature set: {args.feature_set}")

    df = pd.read_csv(db_path)
    validate_columns(df, args.feature_set)
    only_test = not args.all_rows

    if args.split_col not in df.columns and only_test:
        df = attach_split_from_source(df, args.split_source_csv, args.split_col)

    client = openai.OpenAI(api_key=args.api_key)

    df["classification_label"] = ""
    df["reason"] = ""

    if only_test:
        if args.split_col not in df.columns:
            raise ValueError(f"Split column not found in dataset: {args.split_col}")
        test_mask = df[args.split_col].astype(str) == str(args.test_value)
        target_indices = df.index[test_mask].tolist()
    else:
        target_indices = df.index.tolist()

    if not target_indices:
        raise ValueError("No rows selected for inference. Check --split-col/--test-value.")

    print(f"Rows selected for inference: {len(target_indices)}")

    out_path = args.output_csv
    if not out_path:
        out_path = f"{db_path.rsplit('.csv', 1)[0]}_{args.model}_{args.feature_set}_classified.csv"
    elif not os.path.isabs(out_path):
        out_path = os.path.join(os.getcwd(), out_path)

    for index in target_indices:
        row = df.loc[index]
        speaker = str(row["speaker_utterance"])
        listener = str(row["listener_utterance"])
        prompt = build_prompt(speaker, listener, args.feature_set, row)
        print(f"Index: {index}")

        try:
            result = classify_conversation(client, args.model, prompt)
            label, reason = parse_response(result)
        except openai.OpenAIError as exc:
            label, reason = "", f"Error: {exc}"

        df.at[index, "classification_label"] = label
        df.at[index, "reason"] = reason

        df.to_csv(out_path, index=False)

    print(f"Finished. Classification results saved to: {out_path}")


if __name__ == "__main__":
    main()
