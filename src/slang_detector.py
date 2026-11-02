"""
STEP 4 - Slang Detector
Context-Aware Multi-Meaning Slang Classification Using Graph Theory

Purpose:
  Given a raw input sentence, detect which known slang word(s) it contains
  and return their BASE form + position in the sentence - even when the
  sentence uses an inflected form (ghosted -> ghost, cooked -> cook, etc.)

Why lemmatization matters here:
  Step 3 found that exact string matching fails on inflected slang
  (e.g. "ghosted" != "ghost"). spaCy's lemmatizer reduces each token to its
  dictionary base form, so "ghosted"/"ghosting" both lemmatize to "ghost",
  "cooked"/"cooking" both lemmatize to "cook", etc. Matching on lemmas
  instead of raw tokens fixes that gap.

This module is a GATE: nothing downstream (Step 5 full NLP context
extraction, Step 6 graph relationships) should run on a sentence unless
this detector confirms at least one known slang word is present.

Usage:
    from slang_detector import SlangDetector

    detector = SlangDetector("../data/processed/slang_dataset_clean.csv")
    result = detector.detect("He totally ghosted me after our third date.")
    print(result)
"""

import re
import pandas as pd
# pyrefly: ignore[missing-import]
import spacy


class SlangDetector:
    def __init__(self, dataset_path: str, spacy_model: str = "en_core_web_sm"):
        """
        dataset_path: path to slang_dataset_clean.csv (Step 3 output).
                      Used to build the known-slang vocabulary and the
                      known meanings per slang word.
        spacy_model:  spaCy pipeline to use for tokenization + lemmatization.
        """
        self.nlp = spacy.load(spacy_model)

        df = pd.read_csv(dataset_path)
        df["Slang"] = df["Slang"].str.strip().str.lower()

        # IMPORTANT: the dataset's "Slang" labels are NOT all dictionary base
        # forms - e.g. the label is "cooked" (not "cook"), "lit" (not "light"),
        # but "ghost" (already the base form, since sentences use "ghosted").
        # spaCy will lemmatize a detected token to ITS dictionary lemma
        # ("cooked" -> "cook", "lit" -> "light", "ghosted" -> "ghost"), so we
        # must index everything by lemma, not by the raw dataset spelling, or
        # lookups silently fail for exactly the words that were already
        # spelled in base form in the CSV.
        #
        # self.slang_vocab:    {lemma, ...}                         e.g. {"cook", "light", "ghost", ...}
        # self.label_by_lemma: {lemma: canonical dataset spelling}  e.g. {"cook": "cooked", "light": "lit"}
        # self.slang_meanings: {lemma: [meanings]}
        self.slang_vocab = set()
        self.label_by_lemma = {}
        self.slang_meanings = {}

        for slang_label, group in df.groupby("Slang"):
            lemma = self._lemmatize_word(slang_label)
            self.slang_vocab.add(lemma)
            self.label_by_lemma[lemma] = slang_label
            self.slang_meanings[lemma] = sorted(group["Meaning"].unique().tolist())

    def _lemmatize_word(self, word: str) -> str:
        """Lemmatize a single word in isolation (used for building vocab)."""
        doc = self.nlp(word.lower())
        return doc[0].lemma_ if len(doc) else word.lower()

    def detect(self, sentence: str) -> dict:
        """
        Detect known slang words in a sentence.

        Returns a dict:
        {
            "sentence": original sentence,
            "detected": [
                {
                    "surface_form": "ghosted",   # word as it appeared
                    "base_form": "ghost",        # lemma / dictionary form
                    "position": 2,                # token index in sentence
                    "possible_meanings": ["to stop responding/disappear",
                                           "literal spirit/apparition"]
                },
                ...
            ],
            "has_slang": True/False
        }
        """
        doc = self.nlp(sentence)
        detected = []

        for i, token in enumerate(doc):
            if token.is_punct or token.is_space:
                continue
            lemma = token.lemma_.lower()
            if lemma in self.slang_vocab:
                detected.append({
                    "surface_form": token.text,
                    "base_form": self.label_by_lemma[lemma],  # canonical dataset spelling
                    "position": i,
                    "possible_meanings": self.slang_meanings.get(lemma, []),
                })

        return {
            "sentence": sentence,
            "detected": detected,
            "has_slang": len(detected) > 0,
        }

    def detect_batch(self, sentences: list) -> list:
        """Run detect() over a list of sentences."""
        return [self.detect(s) for s in sentences]


def _demo():
    """Quick manual test against tricky inflected forms."""
    detector = SlangDetector("../data/processed/slang_dataset_clean.csv")

    test_sentences = [
        "He totally ghosted me after our third date.",       # inflected -> ghost
        "The chicken is cooked and ready to serve.",          # exact match -> cooked
        "I'm cooked after that exam.",                         # exact match -> cooked
        "The candles were lit around the room.",               # exact match -> lit
        "This is a perfectly normal sentence with no slang.",  # none
        "That match was sick, and he's also feeling sick.",    # same word, two spots
        "The fire in the kitchen was so fire it made the news.",  # same slang, two meanings intended
    ]

    for s in test_sentences:
        result = detector.detect(s)
        print(f"\nSentence: {result['sentence']}")
        print(f"Has slang: {result['has_slang']}")
        for d in result["detected"]:
            print(f"  -> '{d['surface_form']}' (base: '{d['base_form']}', "
                  f"pos: {d['position']}) | candidates: {d['possible_meanings']}")


if __name__ == "__main__":
    _demo()
