# Few-shot text classification: prompting vs. finetuned RoBERTa

An adaptation of Roy, Nakshatri & Goldwasser, *Towards Few-Shot Identification of
Morality Frames using In-Context Learning* ([arXiv:2302.02029](https://arxiv.org/abs/2302.02029)),
generalised to an arbitrary label set so you can run it on your own data.

Both arms of the comparison — one-pass prompting and a finetuned RoBERTa
classifier — draw from the same shot samples and the same test set for a given
(k, seed), so the gap you measure belongs to the method rather than the draw.

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

`torch` and `transformers` are only needed for the RoBERTa arm. If you only want
the prompting arm, `pip install anthropic pandas pyyaml scikit-learn` is enough.

## Your data

A CSV with two columns:

```csv
text,label
"some text to classify",CLASS_A
"some other text",CLASS_B
```

Then copy `labels.example.yaml` to `labels.yaml` and replace the five moral
foundations with your own classes. Write the `definition` fields carefully —
at k=1..5 the definitions carry roughly as much of the signal as the examples do,
and a vague definition for one class shows up as that class being over- or
under-predicted across the board.

## Workflow

**1. Look at the prompt before you trust any number.**

```bash
python run.py --data mydata.csv --schema labels.yaml --preview
```

This renders one complete prompt, system message included. A silently malformed
prompt is the most common cause of a disappointing few-shot result, and it is
invisible in the metrics.

**2. Establish the prompting ceiling.** Cheap, no training, minutes.

```bash
python run.py --data mydata.csv --schema labels.yaml --mode prompt --k 0 1 3 5 --seeds 3
```

k=0 is meaningful here because the definitions act as task instruction — the
paper makes the same point about its own setup. If zero-shot is already close to
your requirement, you may not need a training pipeline at all.

**3. Run the baseline.** Needs a GPU to be pleasant, but `roberta-base` is 125M
parameters and will run on CPU for small k.

```bash
python run.py --data mydata.csv --schema labels.yaml --mode roberta --k 1 3 5 --seeds 5
```

Add `--frozen` to reproduce the paper's floor condition. It should land near
chance; if it does not, something in your setup is leaking.

**4. Compare.** `--mode both` runs the arms on identical splits and prints
mean (std) per method per k.

**5. If prompting wins and you need cheap bulk inference, distil.** Label an
unlabelled pool with the prompting arm, then train RoBERTa on the result:

```bash
python run.py --data mydata.csv --schema labels.yaml --label-pool unlabelled.csv
# filter silver_labels.csv on confidence >= 0.8, then:
python run.py --data silver_labels.csv --schema labels.yaml --mode roberta --k 50
```

This is the path worth taking if your end goal is running over a large corpus:
prompting quality at classifier cost. Labelling uses majority voting
automatically so the confidence column is meaningful.

## Reading the results honestly

- **Never report a single seed.** Few-shot finetuning at k=5 is a few dozen
  gradient steps; the paper's RoBERTa standard deviations run to ±9.6, wide
  enough that its 2-point win over prompting at 5 shots is inside the noise.
  Three seeds is a minimum, five is better.
- **Unparseable generations count as errors.** `macro_f1` maps them to a
  sentinel rather than dropping them. Dropping them flatters the prompting arm.
  If the unparsed rate is above a percent or two, tighten your definitions or
  shorten your label names.
- **Check the per-class report, not just macro F1.** The paper's 5-shot run
  averaged 43.56 macro F1 while Fairness/Cheating sat at 66.67 precision against
  10.00 recall — a class the model had essentially stopped predicting. The
  average hid that completely.
- **Class order in the prompt is a real variable.** Shots are interleaved and
  shuffled within each round by default. If you disable that, expect recency
  bias toward whichever class lands last.

## What differs from the paper

| | Paper | Here |
|---|---|---|
| LLM | GPT-J-6B | current Claude models |
| Decoding | top-k=5, temp 0.5, 10 samples, majority vote | greedy single sample, voting optional via `--votes` |
| Definitions | prepended to completion | system prompt |
| Labels | 5 moral foundations | your schema |

The paper's headline finding — prompting loses to finetuned RoBERTa on subtle
concepts — is largely an artifact of the 6B model. The authors flagged this
themselves and listed larger models as future work. On a current model you
should expect the prompting arm to look substantially better than their Table 2
suggests. Their other finding, that prompting wins comfortably on the more
surface-evident entity-role task, is the one more likely to transfer.

## Files

| File | What it does |
|---|---|
| `labels.example.yaml` | label schema — the one file you must edit |
| `data.py` | loading, stratified k-shot sampling, leakage guard |
| `prompt_classifier.py` | one-pass prompting arm |
| `roberta_classifier.py` | frozen and finetuned RoBERTa arms |
| `run.py` | CLI: sweep, compare, preview, distil |
