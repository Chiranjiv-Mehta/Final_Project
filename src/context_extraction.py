"""
STEP 6 - Extract Context and Relationships
Context-Aware Multi-Meaning Slang Classification Using Graph Theory

Purpose:
  This is the module that turns a detected slang word (Step 4) plus its
  full spaCy annotation (Step 5) into the actual GRAPH INGREDIENTS Payal's
  Context Graph Builder needs: typed relationships like
  (slang)-[:DESCRIBES]->(word), (slang)-[:HAS_CONTEXT]->(word), and
  (slang)-[:HAS_SENTIMENT]->(label) - not just a flat list of nearby words.

  This directly replaces the rough "nearby non-stopword tokens" placeholder
  from Step 2's Context_Words column with real dependency-based context,
  exactly as flagged back in Step 2/3.

How relationships are derived from the dependency parse:
  - If the slang token's dependency relation to its head is "amod"
    (adjectival modifier, e.g. "sick match"), it DESCRIBES its head noun.
  - If the slang token is a VERB (e.g. "ghosted", "cooked" used as verbs),
    its syntactic subject/object children are RELATED_TO it.
  - Every other dependency-connected content word (head + children, minus
    stopwords/punctuation) gets a HAS_CONTEXT edge - this is the general
    "this word is part of the slang word's immediate context" relationship.
  - Sentence-level sentiment (via TextBlob polarity) becomes a
    HAS_SENTIMENT edge to a positive/negative/neutral node.

Note on topic/category: the dataset's Category/Topic field (e.g.
"Music/Social") is a TRAINING LABEL, not something inferable from a single
sentence without a trained classifier. So no BELONGS_TO edges are invented
here at inference time - topic-based edges only exist for dataset-derived
knowledge-graph seeding (Payal's Neo4j side), not for a fresh input sentence.
This is called out explicitly rather than faked.

Usage:
    from context_extraction import ContextExtractor

    extractor = ContextExtractor("../data/processed/slang_dataset_clean.csv")
    result = extractor.extract("He totally ghosted me after our third date.")
    print(result)
"""

from slang_detector import SlangDetector
from nlp_pipeline import NLPPipeline
from textblob import TextBlob


class ContextExtractor:
    def __init__(self, dataset_path: str, spacy_model: str = "en_core_web_sm"):
        self.detector = SlangDetector(dataset_path, spacy_model=spacy_model)
        # Reuse the SAME spaCy pipeline object for the full NLP pass so we
        # don't load the model twice.
        self.nlp_pipeline = NLPPipeline(spacy_model=spacy_model)

    @staticmethod
    def _sentiment_label(sentence: str) -> dict:
        polarity = TextBlob(sentence).sentiment.polarity
        if polarity > 0.15:
            label = "positive"
        elif polarity < -0.15:
            label = "negative"
        else:
            label = "neutral"
        return {"label": label, "polarity": round(polarity, 3)}

    def extract(self, sentence: str) -> dict:
        """
        Full Step 6 output for a sentence:
        {
            "sentence": "...",
            "has_slang": True,
            "sentiment": {"label": "negative", "polarity": -0.3},
            "slang_contexts": [
                {
                    "surface_form": "ghosted",
                    "base_form": "ghost",
                    "position": 2,
                    "possible_meanings": [...],
                    "context_words": ["date", "third"],
                    "relationships": [
                        {"subject": "ghost", "relation": "RELATED_TO", "object": "me"},
                        {"subject": "ghost", "relation": "HAS_CONTEXT", "object": "date"},
                        {"subject": "ghost", "relation": "HAS_SENTIMENT", "object": "negative"}
                    ]
                }
            ]
        }
        """
        detection = self.detector.detect(sentence)
        sentiment = self._sentiment_label(sentence)

        if not detection["has_slang"]:
            return {
                "sentence": sentence,
                "has_slang": False,
                "sentiment": sentiment,
                "slang_contexts": [],
            }

        nlp_result = self.nlp_pipeline.process(sentence)
        tokens_by_index = {t["index"]: t for t in nlp_result["tokens"]}

        slang_contexts = []
        for d in detection["detected"]:
            slang_token = tokens_by_index.get(d["position"])
            if slang_token is None:
                continue  # safety guard, shouldn't happen since both use same tokenizer

            relationships = []
            context_words = []

            head_text = slang_token["head"]
            head_index = slang_token["head_index"]
            is_self_root = head_index == slang_token["index"]

            # --- Relation to syntactic head -----------------------------------
            if not is_self_root:
                head_token = tokens_by_index.get(head_index)
                if head_token and not head_token["is_punct"]:
                    if slang_token["dep"] == "amod":
                        relation = "DESCRIBES"
                    else:
                        relation = "RELATED_TO"
                    relationships.append({
                        "subject": d["base_form"], "relation": relation, "object": head_text
                    })
                    context_words.append(head_text)

            # --- Relations to syntactic children ------------------------------
            # NOTE: a prepositional object (e.g. "cooked AFTER that EXAM") is two
            # hops away from the slang word, not one - "after" is a direct child
            # of "cooked", and "exam" is a child of "after". Since "after" is a
            # stopword, naively filtering stopword children silently drops the
            # most meaning-carrying word in the sentence. So: when a direct
            # child is a preposition ("prep"), we look one hop further into
            # ITS children for the actual object (pobj) and link that instead.
            for child_text in slang_token["children"]:
                child_token = next(
                    (t for t in nlp_result["tokens"] if t["text"] == child_text), None
                )
                if child_token is None or child_token["is_punct"]:
                    continue

                if child_token["dep"] == "prep":
                    # Look past the preposition to its object.
                    for grandchild_text in child_token["children"]:
                        gc_token = next(
                            (t for t in nlp_result["tokens"] if t["text"] == grandchild_text), None
                        )
                        if gc_token is None or gc_token["is_punct"] or gc_token["is_stop"]:
                            continue
                        if gc_token["dep"] == "pobj":
                            relationships.append({
                                "subject": d["base_form"], "relation": "RELATED_TO",
                                "object": grandchild_text
                            })
                            context_words.append(grandchild_text)
                    continue

                if child_token["is_stop"]:
                    continue
                if child_token["dep"] in ("nsubj", "nsubjpass", "dobj", "attr"):
                    relation = "RELATED_TO"
                else:
                    relation = "HAS_CONTEXT"
                relationships.append({
                    "subject": d["base_form"], "relation": relation, "object": child_text
                })
                context_words.append(child_text)

            # --- Sentiment edge --------------------------------------------
            relationships.append({
                "subject": d["base_form"], "relation": "HAS_SENTIMENT", "object": sentiment["label"]
            })

            # de-duplicate context words, preserve order
            seen = set()
            ordered_context = []
            for w in context_words:
                if w.lower() not in seen:
                    seen.add(w.lower())
                    ordered_context.append(w)

            slang_contexts.append({
                "surface_form": d["surface_form"],
                "base_form": d["base_form"],
                "position": d["position"],
                "possible_meanings": d["possible_meanings"],
                "context_words": ordered_context,
                "relationships": relationships,
            })

        return {
            "sentence": sentence,
            "has_slang": True,
            "sentiment": sentiment,
            "slang_contexts": slang_contexts,
        }


def _demo():
    extractor = ContextExtractor("../data/processed/slang_dataset_clean.csv")

    test_sentences = [
        "He totally ghosted me after our third date.",
        "That match was sick, best game I've seen all season.",
        "The chicken is cooked and ready to serve.",
        "I'm cooked after that exam.",
    ]

    for s in test_sentences:
        result = extractor.extract(s)
        print(f"\n{'='*70}")
        print(f"Sentence: {result['sentence']}")
        print(f"Sentiment: {result['sentiment']}")
        for ctx in result["slang_contexts"]:
            print(f"\n  Slang detected: '{ctx['surface_form']}' (base: '{ctx['base_form']}')")
            print(f"  Context words: {ctx['context_words']}")
            print(f"  Candidate meanings: {ctx['possible_meanings']}")
            print("  Relationships (graph edges):")
            for r in ctx["relationships"]:
                print(f"    ({r['subject']}) -[:{r['relation']}]-> ({r['object']})")


if __name__ == "__main__":
    _demo()
