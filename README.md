# CPC Evaluator 

## Getting Started
Setup is pretty easy. Just follow the steps below:

### Step 1️⃣: Install Dependencies
Install dependencies by executing:
```bash
poetry install
```
### Step 2️⃣: Start the API
Start the API by executing:
```bash
poetry run python -m uvicorn api:app --port 6000 --host 0.0.0.0
```
### Step 3️⃣: Run the Evaluator
```bash
poetry run python evaluator.py
```
## Evaluation Results
```bash
Evaluation DateTime: 2024-07-28 15:49:51

Num Samples: 3
Model: gpt-4o-mini

ICD-10 Stats:
min: 0.467
max: 0.950
mean: 0.672
median: 0.600
std_dev: 0.204
Quartiles: 0.533, 0.600, 0.775

exact_matches:  4
partial_matches: 1
over_suggestions: 3
under_suggestions: 1
precision:      1.5833333333333333
recall:         2.1666666666666665
f1_score:       1.8

CPT Stats:
min: 0.000
max: 0.000
mean: 0.000
median: 0.000
std_dev: 0.000
Quartiles: 0.000, 0.000, 0.000

exact_matches:  0
over_suggestions: 5
under_suggestions: 3
precision:      0.0
recall:         0.0
f1_score:       0.0
```