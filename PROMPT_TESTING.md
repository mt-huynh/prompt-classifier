# Prompt testing examples

These examples use `sample_moral_foundations.csv` with the schema in
`labels.example.yaml`. The CSV is synthetic and intended only for smoke tests,
not for measuring real-world model quality.

## 1. Render a prompt without calling the API

This is the safest first test. It loads the data and schema and prints the
system prompt plus five examples per class.

```powershell
python run.py --data sample_moral_foundations.csv --schema labels.example.yaml --preview
```

The preview should include the five labels and end with an unlabeled test tweet.

## 2. Run a small prompting experiment

This calls Claude, so set `ANTHROPIC_API_KEY` first. Start with one seed and
one shot per class to keep the test small.

```powershell
$env:ANTHROPIC_API_KEY = "your-api-key"
python run.py --data sample_moral_foundations.csv --schema labels.example.yaml --mode prompt --k 1 --seeds 1 --test-per-class 1 --workers 1 --out sample_prompt_results.csv
```

The command should print a macro-F1 score and write `sample_prompt_results.csv`.
The expected labels for the following texts are shown for manual spot checks:

```text
"The food bank opened its doors during the storm." -> CARE/HARM
"Everyone must receive the same refund under the policy." -> FAIRNESS/CHEATING
"The team stood together after its captain was injured." -> LOYALTY/BETRAYAL
"Residents followed the emergency coordinator's instructions." -> AUTHORITY/SUBVERSION
"The shrine was kept clean and protected from disrespect." -> PURITY/DEGRADATION
```

## 3. Test prompt construction in Python

This checks label normalization without making an API request:

```powershell
python -c "from data import load_schema; schema = load_schema('labels.example.yaml'); print([schema.normalize(x) for x in ['care/harm', 'FAIRNESS/CHEATING', 'not-a-label']])"
```

Expected output ends with:

```text
['CARE/HARM', 'FAIRNESS/CHEATING', None]
```

## Notes

- `--preview` does not require an API key.
- Prompt mode requires `ANTHROPIC_API_KEY` and network access.
- RoBERTa mode may download `roberta-base` and is much heavier than this smoke
  test.
- The synthetic examples are deliberately short and clearly labeled so prompt
  formatting and data plumbing can be checked independently of model quality.
