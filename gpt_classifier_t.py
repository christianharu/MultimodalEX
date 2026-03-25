import argparse
import os

import openai
import pandas as pd

from gpt_classifier_prompt import build_behavioral_empathy_prompt


def classify_conversation(client: openai.OpenAI, model: str, speaker_utterance: str, listener_utterance: str) -> str:
    """Send one exchange to OpenAI for empathy classification."""
    prompt = build_behavioral_empathy_prompt(speaker_utterance, listener_utterance)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in affective computing and empathy classification."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    except openai.OpenAIError as exc:
        return f"Error: {exc}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an utterance based on empathy-related factors.")
    parser.add_argument("--database", type=str, help="The name of the database to load", default="mEX_text_only.csv")
    parser.add_argument("--model", type=str, help="The name LLM model to use", default="gpt-4o")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="OpenAI API key.")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("OpenAI API key is required. Set --api-key or OPENAI_API_KEY.")

    current_folder = os.getcwd()
    database_path = os.path.join(current_folder, args.database)
    model = args.model
    client = openai.OpenAI(api_key=args.api_key)

    database = pd.read_csv(database_path)
    database["classification_label"] = ""
    database["reason"] = ""

    for index, row in database.iterrows():
        speaker_utterance = row["speaker_utterance"]
        listener_utterance = row["listener_utterance"]

        print("\nIndex:", index)

        result = classify_conversation(client, model, speaker_utterance, listener_utterance)
        classification_label = result.split("classification_label: ")[1].split("\n")[0].strip()
        reason = result.split("reason: ")[1].split("\n")[0].strip()

        database.at[index, "classification_label"] = classification_label
        database.at[index, "reason"] = reason
        database.to_csv(f"{database_path.split('.csv')[0]}_{model}_classified.csv", index=False)

    print("\n\nFinished! Classification results saved to:", f"{database_path.split('.csv')[0]}_{model}_classified.csv")
