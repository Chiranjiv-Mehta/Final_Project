"""
STEP 5 - spaCy NLP Processing
Context-Aware Multi-Meaning Slang Classification Using Graph Theory

Purpose:
  A single, reusable NLP pipeline wrapper around spaCy that turns a raw
  sentence into structured linguistic annotations:
    - Tokens (with index)
    - Lemmas (dictionary base form of each token)
    - POS tags (part of speech)
    - Dependency relations (how each token relates to others grammatically)
    - Named entities (people, places, organizations, etc.)

  Step 4 (slang_detector.py) only used spaCy internally for lemma matching.
  This module exposes the FULL annotation, because Step 6 (context +
  relationship extraction) needs dependency parses to build real "X
  describes Y" / "X belongs to Y" relationships - not just nearby words.

Usage:
    from nlp_pipeline import NLPPipeline

    nlp_pipeline = NLPPipeline()
    result = nlp_pipeline.process("He totally ghosted me after our third date.")
    print(result)
"""

import spacy


class NLPPipeline:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)

    def process(self, sentence: str) -> dict:
        """
        Run full spaCy annotation on a sentence.

        Returns:
        {
            "sentence": "...",
            "tokens": [
                {
                    "text": "ghosted",
                    "index": 2,
                    "lemma": "ghost",
                    "pos": "VERB",
                    "tag": "VBD",
                    "dep": "ROOT",
                    "head": "ghosted",        # the token this one is attached to
                    "head_index": 2,
                    "children": ["He", "totally", "me", "after"],  # tokens attached to this one
                    "is_stop": False,
                },
                ...
            ],
            "entities": [
                {"text": "...", "label": "...", "start": 0, "end": 1}
            ]
        }
        """
        doc = self.nlp(sentence)

        tokens = []
        for token in doc:
            children = [child.text for child in token.children]
            tokens.append({
                "text": token.text,
                "index": token.i,
                "lemma": token.lemma_.lower(),
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "head": token.head.text,
                "head_index": token.head.i,
                "children": children,
                "is_stop": token.is_stop,
                "is_punct": token.is_punct,
            })

        entities = [
            {"text": ent.text, "label": ent.label_, "start": ent.start, "end": ent.end}
            for ent in doc.ents
        ]

        return {
            "sentence": sentence,
            "tokens": tokens,
            "entities": entities,
        }

    def get_token_by_index(self, processed: dict, index: int) -> dict:
        """Helper: fetch a specific token dict by its index from a process() result."""
        for t in processed["tokens"]:
            if t["index"] == index:
                return t
        return None


def _demo():
    pipeline = NLPPipeline()

    test_sentences = [
        "He totally ghosted me after our third date.",
        "The chicken is cooked and ready to serve.",
        "That match was sick, best game I've seen all season.",
    ]

    for s in test_sentences:
        result = pipeline.process(s)
        print(f"\nSentence: {result['sentence']}")
        print(f"{'Token':<12}{'Lemma':<12}{'POS':<8}{'Dep':<10}{'Head':<12}")
        print("-" * 54)
        for t in result["tokens"]:
            if t["is_punct"]:
                continue
            print(f"{t['text']:<12}{t['lemma']:<12}{t['pos']:<8}{t['dep']:<10}{t['head']:<12}")
        if result["entities"]:
            print("Entities:", result["entities"])


if __name__ == "__main__":
    _demo()
