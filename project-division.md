# Context-Aware Multi-Meaning Slang Classification Using Graph Theory
## Project Division: Chiranjiv & Payal

---

## Chiranjiv — Person 1: NLP + Dataset + ML Model

### 1. Dataset Creation
- Pick ~15–20 slang words (start with: Fire, Sick, Cooked, Ghost, Lit, Dead, Salty, Killer)
- For each slang word, list all its common meanings
- Write/collect 15–30 example sentences per meaning (aim for a balanced dataset)
- Structure each row as:

| ID | Sentence | Slang | Meaning | Context Words | Category/Topic | Sentiment |
|----|----------|-------|---------|----------------|-----------------|-----------|

- Clean the data: remove duplicates, fix typos, balance classes across meanings

### 2. NLP Processing (spaCy)
- Tokenization, lemmatization, POS tagging
- Dependency parsing (find what the slang word is attached to — subject, object, modifier)
- Named Entity Recognition (topic/domain hints)
- Slang detection: match tokens against your slang dictionary

### 3. Feature Extraction
- Context words (nearby + dependency-linked words)
- POS pattern around the slang word
- Sentiment score (via spaCy/TextBlob/VADER)
- Topic/domain tag
- These features feed both your ML model AND Payal's graph builder

### 4. Model Training
Pipeline: `Dataset → Preprocessing → Feature Extraction → Train/Test Split → Model Training → Evaluation`

- Baseline: Logistic Regression
- Better: Random Forest or XGBoost
- Optional stretch: fine-tune a small Transformer

### 5. Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion matrix per slang word (to catch which meanings get confused)

### 6. Deliverables
- [ ] Cleaned labeled dataset (CSV)
- [ ] spaCy processing script (extracts context features per sentence)
- [ ] Feature extraction module (outputs features Payal's graph builder + your ML model both consume)
- [ ] Trained ML model + evaluation report
- [ ] Short doc explaining feature choices and model results

---

## Payal — Person 2: Graph Theory + Neo4j + Application

### 1. Context Graph Builder
- Take Chiranjiv's NLP output (context words, POS, dependencies, sentiment, topic)
- Convert each sentence into a graph: nodes = words/concepts/meanings/topics/sentiment, edges = relationships
- Relationship types: `DESCRIBES`, `RELATED_TO`, `BELONGS_TO`, `HAS_CONTEXT`, `HAS_SENTIMENT`, `HAS_MEANING`

Example:
```
(Sick)-[:DESCRIBES]->(Match)
(Match)-[:BELONGS_TO]->(Sports)
(Sick)-[:HAS_CONTEXT]->(Positive)
```

### 2. Graph Algorithms (NetworkX)
- BFS / k-hop neighborhood around the slang word
- Weighted relationship scoring
- Shortest path between slang node and candidate-meaning nodes
- Graph similarity (compare sentence graph to known meaning patterns)
- Neighborhood analysis

### 3. Meaning Scoring
Formula: `Meaning Score = Σ(Context Relationship Weight × Meaning Association Weight)`

- Combine graph-derived score with Chiranjiv's ML model output for final prediction + confidence

### 4. Neo4j Knowledge Graph
- Store: slang words, meanings, contextual concepts, relationships, example patterns
- Write Cypher queries to retrieve nearest-meaning patterns for a given context

### 5. Backend (FastAPI)
- Endpoint: accept sentence → call Chiranjiv's NLP/ML module → build graph → run graph analysis → return prediction + confidence + explanation

### 6. Frontend (React + Cytoscape.js)
- Input box for a sentence
- Display predicted meaning + confidence score
- Visualize the context graph (nodes/edges) so the prediction is explainable
- Show "why" — which relationships/context drove the decision

### 7. Deliverables
- [ ] Context Graph Builder module
- [ ] NetworkX algorithm implementations
- [ ] Neo4j schema + populated knowledge graph
- [ ] Graph-based scoring function
- [ ] FastAPI backend with working endpoint
- [ ] React frontend with graph visualization

---

## Where the Work Connects

```
Chiranjiv's NLP output (context features)
        ↓
Payal's Context Graph Builder
        ↓
Payal's Graph Analysis → Graph Score
        ↓
Combined with Chiranjiv's ML Model Score
        ↓
Final Prediction + Confidence + Explanation (Payal's API/Frontend)
```

**Key integration point:** Agree early on the exact JSON format for "context features" that Chiranjiv's NLP module outputs — this is what Payal's graph builder will consume. Suggest locking this down in week 1.

## Suggested Timeline (adjust to your deadline)
1. **Week 1–2:** Dataset collection + labeling (Chiranjiv), Neo4j schema design (Payal)
2. **Week 3–4:** NLP feature extraction (Chiranjiv), Context graph builder (Payal)
3. **Week 5:** ML model training (Chiranjiv), Graph algorithms + scoring (Payal)
4. **Week 6:** FastAPI integration (Payal), model evaluation (Chiranjiv)
5. **Week 7:** React frontend + visualization (Payal), final testing both sides
6. **Week 8:** Buffer, polish, documentation, demo prep
