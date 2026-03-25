# Multimodal Empathetic Exchanges

This repository accompanies the article "Behavioral Empathy Prediction in Dyadic Conversations Using Multimodal Cues".

The public release focuses on the materials needed to inspect the mEX dataset, review the multimodal cues used in the paper, and reproduce the main analysis notebooks and multimodal baselines reported in the manuscript.

## Public Release Contents

The public version of this repository includes:

- the README file
- the mEX text dataset
- the public subset of the video-based cue files
- the main multimodal analysis notebooks
- the aggregated mEX results dataframe
- the multimodal baseline scripts used in the paper
- the GPT benchmarking scripts with the centralized behavioral-empathy prompt
- Talking Room labels only

## Text-Based Dataset

The text-based dataset is stored in `text-based_dataset/`.

- `text-based_dataset/text_exchanges.csv`: minimal exchange-level file with IDs, utterances, and empathy labels
- `text-based_dataset/text_exchanges_cues_included.csv`: exchange-level file with the contextual cues described in the paper

The cue file includes exchange identifiers, speaker and listener utterances, sentiment, arousal and valence values from text, text-based mimicry, utterance lengths, EPITOME-related cues, empathetic intent scores, emotion labels, and empathy annotations.

## Video-Based Dataset

The public video-based subset is stored in `video-based_dataset/`.

Included folders:

- `video-based_dataset/av_per_exchange/`: exchange-level arousal and valence values derived from video
- `video-based_dataset/mimicry_per_exchange_0.01/`: exchange-level mimicry labels using the 0.01 threshold described in the paper
- `video-based_dataset/speaker_status/`: speaker-status annotations used to align speaker and listener roles over time

Included cue-generation scripts:

- `video-based_dataset/video_cues.py`
- `video-based_dataset/mimicry_threshold.py`
- `video-based_dataset/mimicry_face.py`

These files are the public video-cue package used by the multimodal analysis and baseline scripts.

## Main Results Dataframe

The main aggregated dataframe with model predictions for mEX is:

- `classifiers/PBC4emp/mEX_all_predictions.csv`

This is the public dataframe with the combined mEX results used throughout the notebook analysis. It is the main table to inspect if you want all reported mEX predictions in one place.

## Analysis Notebooks

### Full mEX Demo

The main notebook for the mEX analysis is:

- `classifiers/PBC4emp/mEX_analysis.ipynb`

This notebook is the most complete walkthrough for the mEX experiments and analysis reported in the paper. It uses the following public companion files already included in `classifiers/PBC4emp/`:

- `mEX_all_predictions.csv`: aggregated mEX predictions and features
- `features_master_file.csv`: precomputed 3-level influential-feature analysis table
- `features_binary_file.csv`: precomputed 2-level influential-feature analysis table

Short explanation:
This notebook is intended as the main demo for the paper. In the public repository, it is best used together with the precomputed CSV files above. Some optional recomputation sections in the notebook were originally tied to additional internal artifacts from development, but the included CSV files provide the public-ready analysis outputs needed to inspect the reported results.

### Talking Room Demo

The notebook for the Talking Room analysis is:

- `classifiers/PBC4emp/tsc_analysis.ipynb`

Short explanation:
This notebook documents the Talking Room analysis used in the paper. Because the Talking Room utterances come from children, the public repository only includes:

- `TalkingRoom_labels.csv`

The raw Talking Room utterances are not distributed in the public version. The notebook is therefore included as a reference demo for the reported analysis, while the public CSV release for this part is restricted to labels only.

## Multimodal Scripts

The repository includes the multimodal baseline scripts reported in the paper:

- `classifiers/multimodal_pbc_mex.py`
- `classifiers/multimodal_rf_mex.py`
- `classifiers/multimodal_svm_mex.py`
- `classifiers/multimodal_lstm_mex.py`
- `classifiers/bert_multimodal_mex.py`
- `classifiers/evaluate_multimodal_baselines.py`

## GPT Benchmarking Scripts

The repository includes the GPT-based benchmarking scripts:

- `gpt_classifier_prompt.py`
- `gpt_classifier_t.py`
- `gpt_classifier_t_av.py`

`gpt_classifier_prompt.py` contains the centralized long behavioral-empathy prompt shared by the text-only and multimodal GPT scripts.

## PBC4emp Folder

The `classifiers/PBC4emp/` folder contains the main analysis materials for the paper, including:

- `mEX_all_predictions.csv`
- `mEX_analysis.ipynb`
- `tsc_analysis.ipynb`
- `features_master_file.csv`
- `features_binary_file.csv`

This folder is the main place to start if you want to inspect the reported results and analysis outputs.

## References

- [1] Montiel-Vázquez, Edwin C., et al. "EmpatheticExchanges: Towards Understanding the Cues for Empathy in Dyadic Conversations." IEEE Access (2024).
- [2] Sharma, Ashish, et al. "A computational approach to understanding empathy expressed in text-based mental health support." arXiv preprint arXiv:2009.08441 (2020).
- [3] Welivita, Anuradha, and Pearl Pu. "A taxonomy of empathetic response intents in human social conversations." arXiv preprint arXiv:2012.04080 (2020).
- [4] Arzate Cruz, Christian, et al. "Data Augmentation for 3DMM-based Arousal-Valence Prediction for HRI." 2024 33rd IEEE International Conference on Robot and Human Interactive Communication (ROMAN). IEEE, 2024.
- [5] Daněček, Radek, Michael J. Black, and Timo Bolkart. "Emoca: Emotion driven monocular face capture and animation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
