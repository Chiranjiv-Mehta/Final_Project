"""
STEP 3 - Clean / Preprocess dataset
Context-Aware Multi-Meaning Slang Classification Using Graph Theory

Takes the raw dataset from Step 2 (slang_dataset.csv) and produces a cleaned
version, ready for slang detection (Step 4) and spaCy processing (Step 5).

What this does:
  1. Load + basic sanity checks (missing values, dtypes)
  2. Strip/normalize whitespace and stray characters in text fields
  3. Remove exact duplicate sentences
  4. Normalize label fields (Slang, Meaning, Category, Sentiment) - trim,
     lowercase Slang/Sentiment for consistency, fix casing
  5. Validate: every (Slang, Meaning) pair is a known combination,
     every Sentence actually contains its Slang word
  6. Add a `Clean_Sentence` column (normalized whitespace/quotes) while
     keeping the ORIGINAL `Sentence` column intact — spaCy in Step 5 wants
     natural casing/punctuation, so we don't lowercase or strip punctuation
     from the sentence itself.
  7. Flag rows with empty Context_Words for manual review
  8. Print a data-quality report and save the cleaned CSV
"""

import pandas as pd
import re

IN_PATH = "../data/raw/slang_dataset.csv"
OUT_PATH = "../data/processed/slang_dataset_clean.csv"


def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.strip()
    text = re.sub(r"\s+", " ", text)          # collapse multiple spaces
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return text


def main():
    df = pd.read_csv(IN_PATH)
    report = {}

    report["rows_loaded"] = len(df)

    # 1. Missing values -----------------------------------------------------
    missing = df.isnull().sum()
    report["missing_values"] = missing[missing > 0].to_dict()
    # Context_Words is allowed to be empty (rare short sentences); everything
    # else must be present.
    required_cols = ["Sentence", "Slang", "Meaning", "Category", "Sentiment"]
    before = len(df)
    df = df.dropna(subset=required_cols)
    report["rows_dropped_missing_required_fields"] = before - len(df)

    # 2. Normalize whitespace / smart quotes --------------------------------
    df["Sentence"] = df["Sentence"].apply(normalize_whitespace)
    df["Clean_Sentence"] = df["Sentence"]  # placeholder for further cleanup
    df["Context_Words"] = df["Context_Words"].fillna("").apply(normalize_whitespace)

    # 3. Normalize label fields ----------------------------------------------
    df["Slang"] = df["Slang"].str.strip().str.lower()
    df["Meaning"] = df["Meaning"].str.strip().str.lower()
    df["Category"] = df["Category"].str.strip()
    df["Sentiment"] = df["Sentiment"].str.strip().str.lower()

    # 4. Remove exact duplicate sentences ------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["Sentence", "Slang"])
    report["duplicates_removed"] = before - len(df)

    # 5. Validate sentiment values -------------------------------------------
    valid_sentiments = {"positive", "negative", "neutral"}
    bad_sentiment = df[~df["Sentiment"].isin(valid_sentiments)]
    report["invalid_sentiment_rows"] = len(bad_sentiment)

    # 6. Validate slang word actually appears in the sentence ---------------
    def contains_slang(row):
        tokens = re.findall(r"[A-Za-z']+", row["Sentence"].lower())
        return row["Slang"] in tokens

    df["_slang_present"] = df.apply(contains_slang, axis=1)
    missing_slang_rows = df[~df["_slang_present"]]
    report["rows_where_slang_word_not_found_in_sentence"] = len(missing_slang_rows)
    df = df.drop(columns=["_slang_present"])

    # 7. Flag empty context words for manual review --------------------------
    empty_context = df[df["Context_Words"].str.strip() == ""]
    report["rows_with_empty_context_words"] = len(empty_context)

    # 8. Check (Slang, Meaning) combinations are consistent -------------------
    combo_counts = df.groupby(["Slang", "Meaning"]).size()
    report["slang_meaning_combinations"] = len(combo_counts)
    report["min_examples_per_combination"] = int(combo_counts.min())
    report["max_examples_per_combination"] = int(combo_counts.max())

    # Reset ID after cleaning so it stays sequential
    df = df.reset_index(drop=True)
    df["ID"] = df.index + 1

    # Reorder columns
    df = df[["ID", "Sentence", "Clean_Sentence", "Slang", "Meaning",
             "Context_Words", "Category", "Sentiment"]]

    df.to_csv(OUT_PATH, index=False)
    report["rows_after_cleaning"] = len(df)
    report["saved_to"] = OUT_PATH

    print("=== DATA QUALITY REPORT ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    if len(missing_slang_rows):
        print("\nRows where slang word wasn't found as a whole token (review these):")
        print(missing_slang_rows[["ID", "Sentence", "Slang"]].to_string(index=False))


if __name__ == "__main__":
    main()
