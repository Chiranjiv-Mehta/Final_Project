# Context-Aware Multi-Meaning Slang Classification Using Graph Theory

Team: Chiranjiv Mehta, Payal More, Aditya Mehta

## Setup
pip install -r requirements.txt
python -m spacy download en_core_web_sm

## Run order (from inside src/)
cd src
python build_dataset.py        # Step 2 - creates data/raw/slang_dataset.csv
python preprocess_dataset.py   # Step 3 - creates data/processed/slang_dataset_clean.csv
python slang_detector.py       # Step 4 - demo/test
python nlp_pipeline.py         # Step 5 - demo/test
python context_extraction.py   # Step 6 - demo/test

## Status
Steps 1-6 complete (Chiranjiv's side). Steps 7-10 (ML training, evaluation,
JSON output) pending. See PROGRESS_LOG.md for full details.
