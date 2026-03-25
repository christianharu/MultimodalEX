from __future__ import annotations

from typing import Sequence


def build_behavioral_empathy_prompt(
    speaker_utterance: str,
    listener_utterance: str,
    extra_cues: Sequence[str] | None = None,
) -> str:
    utterance_classify = (
        f"Rate the exchange: speaker: {speaker_utterance}; "
        f"listener: {listener_utterance}."
    )

    prompt = """
Rate the behavioral empathy displayed by the listener per exchange. Our definition of empathy supports three dimensions in a hierarchical order: Affective, Cognitive, and Behavioral.

Affective empathy means having emotional congruence with another person (same or similar emotion);
Cognitive empathy is to understand the feelings and emotions of another and why they feel that way; Behavioral empathy is to communicate in a manner that demonstrates cognitive and affective empathy.
Note that high empathy includes expressions of interest for the other person's emotional state or their situation.

Give your answer on this scale:
* little to no empathy (Label: 0)
* somewhat empathetic (Label: 1)
* empathetic (Label: 2)

*Empathic Conversation Examples*
* An example of a pair of exchanges with *empathy level 1* - speaker: There was strange noise from the kitchen at night. listener: Did you have your alarm on speaker: Yes, but it didnt ring. I was spooked but it turned out I had left the window open and it was the wind. listener: Oh wow, I would have thought maybe an animal had gotten in.

* An example of a pair of exchanges with *empathy level 2* - speaker: I was babysitting a few days ago and my cousin kept throwing her food on the floor. listener: That must have been frustrating. What did you do? speaker: I kept cleaning it up and put down a bunch of towels around her to protect the floor. I think it comes with the territory, she gets what she wants. listener: That made me laugh. And you are a great babysitter.

* An example of a pair of exchanges with *empathy level 3* - speaker: My dad knew that I have recently hit a rough patch with my finances and gave me 2 grand. I cannot thank him enough. listener: How kind of him! I bet that is going to help you out a lot! speaker: It really is going to help a lot. I feel so blessed! listener: Dads are the best, what would we do without them?
"""

    if extra_cues:
        prompt += "\nAlso consider the additional metrics we extracted from video:\n"
        for cue in extra_cues:
            prompt += f"\n{cue}\n"

    prompt += """

Conversation to classify:
Utterance: {utterance_classify}

Answer format:
reason:_
classification_label:_
""".format(utterance_classify=utterance_classify)

    return prompt
