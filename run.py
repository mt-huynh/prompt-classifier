"""Run the k-shot sweep for either arm, or both, on shared splits.

Usage
-----
  python run.py --data mydata.csv --schema labels.yaml --preview
  python run.py --data mydata.csv --schema labels.yaml --mode prompt  --k 0 1 3 5
  python run.py --data mydata.csv --schema labels.yaml --mode roberta --k 1 3 5
  python run.py --data mydata.csv --schema labels.yaml --mode both --seeds 5
  python run.py --data mydata.csv --schema labels.yaml --label-pool unlabelled.csv

Both arms receive the identical shot draw and the identical test set for a given
(k, seed), so the difference you read off the table is the method.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from data import holdout, load_data, load_schema, sample_shots, stratified_test_set


def macro_f1(gold: list[str], pred: list[str | None], labels: list[str]) -> float:
    # Unparseable generations become a sentinel that matches no gold label, so
    # they are counted as errors rather than dropped. Dropping them inflates
    # the prompting arm and makes the comparison dishonest.
    pred = [p if p is not None else "__UNPARSED__" for p in pred]
    return f1_score(gold, pred, labels=labels, average="macro", zero_division=0) * 100


def report(gold: list[str], pred: list[str | None], labels: list[str]) -> str:
    pred = [p if p is not None else "__UNPARSED__" for p in pred]
    return classification_report(gold, pred, labels=labels, zero_division=0, digits=2)


def run_prompt(schema, shots, test, args) -> tuple[float, list]:
    from prompt_classifier import PromptClassifier

    clf = PromptClassifier(
        schema, shots, model=args.model, n_votes=args.votes,
        max_workers=args.workers,
    )
    preds = clf.predict(test["text"].tolist())
    labels = [p.predicted for p in preds]
    return macro_f1(test["label"].tolist(), labels, schema.names), labels


def run_roberta(schema, shots, test, args, seed) -> tuple[float, list]:
    from roberta_classifier import RobertaClassifier

    clf = RobertaClassifier(
        schema, encoder=args.encoder, freeze_encoder=args.frozen,
        epochs=args.epochs, seed=seed,
    ).fit(shots)
    labels = clf.predict(test["text"].tolist())
    return macro_f1(test["label"].tolist(), labels, schema.names), labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with `text` and `label`")
    ap.add_argument("--schema", required=True, help="YAML label schema")
    ap.add_argument("--mode", default="both",
                    choices=["prompt", "roberta", "both"])
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="Shots per class to sweep")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--test-per-class", type=int, default=20)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--votes", type=int, default=1,
                    help=">1 enables majority voting, as in the paper")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--encoder", default="roberta-base")
    ap.add_argument("--frozen", action="store_true",
                    help="Freeze the encoder, reproducing the paper's ~8 F1 floor")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--preview", action="store_true",
                    help="Print one fully rendered prompt and exit")
    ap.add_argument("--label-pool",
                    help="CSV of unlabelled text (column `text`) to annotate "
                         "with the prompting arm, for distillation")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    schema = load_schema(args.schema)
    df = load_data(args.data, schema)
    print(f"Loaded {len(df)} rows, {len(schema.names)} classes: {schema.names}")
    print(df["label"].value_counts().to_string(), "\n")

    if args.preview:
        from prompt_classifier import preview_prompt
        shots = sample_shots(df, schema, k=max(args.k), seed=0)
        rest = holdout(df, shots)
        print(preview_prompt(schema, shots, rest["text"].iloc[0]))
        return 0

    if args.label_pool:
        from prompt_classifier import PromptClassifier
        pool = pd.read_csv(args.label_pool)
        if "text" not in pool.columns:
            raise ValueError(
                f"Label-pool CSV must contain a 'text' column; found: "
                f"{list(pool.columns)}"
            )
        shots = sample_shots(df, schema, k=max(args.k), seed=0)
        clf = PromptClassifier(schema, shots, model=args.model,
                               n_votes=max(3, args.votes), max_workers=args.workers)
        preds = clf.predict(pool["text"].astype(str).tolist())
        out = pd.DataFrame({
            "text": pool["text"],
            "label": [p.predicted for p in preds],
            "confidence": [p.confidence for p in preds],
        }).dropna(subset=["label"])
        out.to_csv("silver_labels.csv", index=False)
        print(f"\nWrote {len(out)} silver labels to silver_labels.csv")
        print("Filter on confidence before training on these, e.g. >= 0.8, then "
              "feed the result to --data for the RoBERTa arm.")
        return 0

    rows = []
    for k in args.k:
        for seed in range(args.seeds):
            shots = sample_shots(df, schema, k=k, seed=seed)
            pool = holdout(df, shots)
            test = stratified_test_set(pool, schema, args.test_per_class, seed=seed)

            if args.mode in ("prompt", "both"):
                if k == 0 and args.mode == "both":
                    pass  # RoBERTa cannot do 0-shot; prompting still can
                score, preds = run_prompt(schema, shots, test, args)
                rows.append({"method": "prompt", "k": k, "seed": seed, "macro_f1": score})
                print(f"k={k} seed={seed}  prompt   macro-F1 {score:5.2f}")
                if k == max(args.k) and seed == 0:
                    print(report(test["label"].tolist(), preds, schema.names))

            if args.mode in ("roberta", "both") and k > 0:
                score, preds = run_roberta(schema, shots, test, args, seed)
                tag = "roberta-frozen" if args.frozen else "roberta"
                rows.append({"method": tag, "k": k, "seed": seed, "macro_f1": score})
                print(f"k={k} seed={seed}  {tag:<8} macro-F1 {score:5.2f}")
                if k == max(args.k) and seed == 0:
                    print(report(test["label"].tolist(), preds, schema.names))

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)

    print("\n" + "=" * 52)
    print("Macro F1, mean (std) across seeds")
    print("=" * 52)
    summary = res.groupby(["method", "k"])["macro_f1"].agg(["mean", "std", "count"])
    for (method, k), row in summary.iterrows():
        std = 0.0 if np.isnan(row["std"]) else row["std"]
        print(f"{method:<16} k={k}  {row['mean']:5.2f} ({std:4.1f})  "
              f"n={int(row['count'])}")
    print(f"\nPer-run scores written to {args.out}")
    if args.seeds < 3:
        print("\n[warn] Fewer than 3 seeds. The spread between draws at low k is "
              "large enough to flip the ranking; do not conclude anything yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
