# Context-Aware Multi-Meaning Slang Classification Using Graph Theory
## Conversation Log & Project Progress

**Team:** Chiranjiv & Payal

---

## 1. Project Overview

**Title:** Context-Aware Multi-Meaning Slang Classification Using Graph Theory

**Goal:** Classify the correct meaning of slang words that have multiple meanings, depending on the sentence context (e.g. "fire" = amazing vs. literal fire; "sick" = amazing vs. ill; "cooked" = in trouble vs. food prepared).

**Approach:** Hybrid NLP + Graph Theory + Machine Learning.

**High-level pipeline:**
```
Input Sentence → NLP Processing → Slang Detection → Context Extraction →
Context Graph Construction → Graph Theory Analysis → Meaning Classification →
Prediction + Confidence + Explanation
```

**Core idea:** Represent context as a graph — nodes are words/concepts/meanings/topics/sentiment, edges are relationships between them (e.g. `DESCRIBES`, `RELATED_TO`, `BELONGS_TO`, `HAS_CONTEXT`, `HAS_SENTIMENT`, `HAS_MEANING`). Graph analysis produces a score that combines with an ML model's prediction for the final, explainable meaning classification.

---

## 2. Team Division

### Chiranjiv — Person 1: NLP + Dataset + ML Model
Owns dataset creation, spaCy NLP processing, feature extraction, and ML model training/evaluation.

### Payal — Person 2: Graph Theory + Neo4j + Application
Owns the context graph builder, NetworkX algorithms, Neo4j knowledge graph, graph-based scoring, FastAPI backend, and React + Cytoscape.js frontend/visualization.

**Integration point:** Chiranjiv's NLP module outputs structured context features (JSON) → Payal's graph builder consumes them to build the context graph → graph score + ML score combine into the final prediction, served via Payal's FastAPI/React app.

*(Full task breakdown, deliverables, and an 8-week suggested timeline were provided separately as `project-division.md`.)*

---

## 3. Chiranjiv's 10-Step Build Plan

| Step | Task | Status |
|---|---|---|
| 1 | Setup Python + libraries | ✅ Done |
| 2 | Create CSV dataset ⭐ | ✅ Done |
| 3 | Clean/preprocess dataset | ✅ Done |
| 4 | Build slang detector | 🔜 Next |
| 5 | Add spaCy NLP processing | Pending |
| 6 | Extract context and relationships | Pending |
| 7 | Train first model (TF-IDF + Logistic Regression) | Pending |
| 8 | Evaluate the model | Pending |
| 9 | Create structured JSON output | Pending |
| 10 | Give output interface to Payal | Pending |

---

## 4. Step 2 — CSV Dataset

**Script:** `src/build_dataset.py`
**Output:** `data/raw/slang_dataset.csv`

- Built a starter dataset: **8 slang words × 2 meanings × 10 sentences = 160 rows**
- Slang words: Fire, Sick, Cooked, Ghost, Lit, Dead, Salty, Killer
- Columns: `ID, Sentence, Slang, Meaning, Context_Words, Category, Sentiment`
- `Context_Words` in this step is a **rough placeholder** (nearby non-stopword tokens via regex) — real dependency-based context extraction comes in Step 5/6 with spaCy.
- Noted: this is a *starter* dataset. More sentence variety (not just swapped nouns) will help the model generalize better — worth expanding once the pipeline works end-to-end.

---

## 5. Step 3 — Clean / Preprocess Dataset

**Script:** `src/preprocess_dataset.py`
**Input:** `data/raw/slang_dataset.csv`
**Output:** `data/processed/slang_dataset_clean.csv`

What it does:
1. Loads data, checks for missing required fields
2. Normalizes whitespace and smart quotes
3. Removes exact duplicate sentences
4. Normalizes label fields (lowercases `Slang`/`Meaning`/`Sentiment`, trims `Category`)
5. Validates sentiment values against `{positive, negative, neutral}`
6. Checks that each sentence actually contains its labeled slang word (exact token match)
7. Flags rows with empty `Context_Words` for review
8. Adds a `Clean_Sentence` column while preserving the original `Sentence` (spaCy needs real casing/punctuation)
9. Prints a full data-quality report and saves the cleaned CSV

**Data quality report (current run):**
- Rows loaded: 160 → Rows after cleaning: 160
- Missing required fields: 0
- Duplicates removed: 0
- Invalid sentiment labels: 0
- Rows where slang word not found as exact token: **10** (all "ghost" rows — sentences use "ghosted," not bare "ghost")

**Key finding carried forward:** Exact-token matching fails on inflected slang forms (ghosted, cooked, lit, etc.). This isn't a bug — it's the problem **Step 4 (slang detector)** needs to solve via **lemmatization**, so "ghosted" correctly maps back to the base slang word "ghost."

---

## 6. Project Folder Structure

```
slang-classifier/
│
├── data/
│   ├── raw/
│   │   └── slang_dataset.csv          ← Step 2 output (original, untouched)
│   └── processed/
│       └── slang_dataset_clean.csv    ← Step 3 output (cleaned)
│
├── src/
│   ├── build_dataset.py               ← Step 2 script
│   ├── preprocess_dataset.py          ← Step 3 script
│   ├── slang_detector.py              ← Step 4 (next)
│   ├── nlp_pipeline.py                ← Step 5
│   ├── context_extraction.py          ← Step 6
│   ├── train_model.py                 ← Step 7
│   └── evaluate_model.py              ← Step 8
│
├── models/                            ← trained model files (Step 7)
├── notebooks/                         ← optional, for experimentation
├── requirements.txt
└── README.md
```

Both scripts use **relative paths** (`../data/raw/...`, `../data/processed/...`) and must be run from inside `src/`:
```bash
cd slang-classifier/src
python build_dataset.py
python preprocess_dataset.py
```
Run order matters: `build_dataset.py` must run before `preprocess_dataset.py`, since Step 3 depends on Step 2's output.

**Confirmed working** on the user's local machine (Windows, `D:\slang-classifier\src`) — output matched exactly, including the 10 flagged "ghost" rows.

---

## 7. Next Up — Step 4: Slang Detector

Plan:
- Load `slang_dataset_clean.csv`
- Use **spaCy lemmatization** so inflected forms (ghosted, ghosting, cooked, cooking, lit, lighting, etc.) correctly map back to their base slang word
- Scan an input sentence, detect which slang word(s) it contains, and return the base form + position in the sentence
- This detection step gates everything downstream — Step 5's full NLP context extraction only runs if a slang word is detected first

**Prerequisite check pending:** confirm spaCy is installed —
```bash
pip install spacy
python -m spacy download en_core_web_sm
```
