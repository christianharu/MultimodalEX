# Multimodal Empathetic Exchanges

This is a repository for the article "Behavioral Empathy Prediction in Dyadic Conversations Using Multimodal Cues" that is under review in the IEEE Access journal.

We provide access to our multimodal empathy dataset **M**ultimodal **E**mpathetic E**x**changes (mEX). mEX presents dyadic conversations. It comprises 86 conversations with 214 exchanges between 12 participants.

# text_exchanges_cues_included.csv

Our empathy cues are based on the work in [1].

<div style="max-width: 100%; overflow-x: auto;">

- *id*: The format is subject\_[num]_conv\_[num] to identify the subject and the conversation number.
- *speaker utterance*: The utterance by the first person involved in the exchange.
- *listener utterance*: The utterance by the second person involved in the exchange.
- *s_sentiment*: Values for negative, neutral, and positive sentiment for the speaker utterance. Given in range [0,1].
- *l_sentiment*: Values for negative, neutral, and positive sentiment for the listener utterance. Given in range [0,1].
- *Emotions*: Arousal and valence both utterances.
- *Text-based Mimicry*: Binary label defining whether emotional mimicry is present in the exchange.
- *s_word_len*: Length of speaker utterance.
- *l_word_len*: Length of listener utterance.
- *EPITOME values*: Values for Emotional Reaction, Interpretation, and Exploration communication mechanisms [2] of the exchange. Given in range [0,2].
- Empathetic intent scores*: Values for possible empathetic intents of the listener response [3]. Given in range [0,1].
- *Emotions*: Emotion labels for both utterances.
- *Empathy labels*: 
</div>

The file **text_eschanges.csv** only includes the id, speaker and listener utterances, and empathy labels.


## Video-based Dataset
All the data is in the folder named **video-based_dataset**. This folder contains the next files with the same naming format we use in the text-based dataset:

<div style="max-width: 100%; overflow-x: auto;">

- *sub\_[num]_conv[num]_av_av_per_frame.csv*: It contains the arousal and valence values per frame using the method in [4] in the range [-1, 1].
- *sub\_[num]_conv[num]_av_face_data.npy*: This is python array containing the EMOCA facial expression coefficients [5] of the participants per frame.
- *sub\_[num]_conv[num]_av_pose_data.npy*: This is python array containing the EMOCA head position coefficients [5] of the participants per frame.
- *sub\_[num]_conv[num]_speaker_status.npy*: This file contains the speaking status for each frame. There are three possible labels: **nothing**, **left**, and **right**. **Nothing** is assigned when no one is speaking. **Right** always refers to the speaker. **Left** refers to the listener.

</div>


## References

- [1] Montiel-Vázquez, Edwin C., et al. "EmpatheticExchanges: Towards Understanding the Cues for Empathy in Dyadic Conversations." IEEE Access (2024).
- [2] Sharma, Ashish, et al. "A computational approach to understanding empathy expressed in text-based mental health support." arXiv preprint arXiv:2009.08441 (2020).
- [3] Welivita, Anuradha, and Pearl Pu. "A taxonomy of empathetic response intents in human social conversations." arXiv preprint arXiv:2012.04080 (2020).
- [4] Arzate Cruz, Christian, et al. "Data Augmentation for 3DMM-based Arousal-Valence Prediction for HRI." 2024 33rd IEEE International Conference on Robot and Human Interactive Communication (ROMAN). IEEE, 2024.
- [5] Daněček, Radek, Michael J. Black, and Timo Bolkart. "Emoca: Emotion driven monocular face capture and animation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.