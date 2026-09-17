"""Data loading, schema handling, and few-shot sampling.

Both the prompting arm and the RoBERTa arm draw their training examples from
`sample_shots` with the same seed, so any performance gap between them is a
property of the method and not of the draw. This matters more than it sounds:
at k=1..5 per class the variance between draws is large enough to reverse a
ranking, which is why the paper reports standard deviations alongside means.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass
class Label:
    name: str
    definition: str


@dataclass
class Schema:
    task_description: str
    label_field: str
    text_field: str
    labels: list[Label]

    @property
    def names(self) -> list[str]:
        return [lab.name for lab in self.labels]

    def normalize(self, raw: str) -> str | None:
        """Map a model generation onto the label space, or None if it misses.

        Tries exact match, then case-insensitive, then unique prefix, then
        unique substring. Anything still ambiguous returns None and is counted
        as an unparseable generation rather than silently guessed -- the paper
        ran into exactly this problem in the zero-shot role setting, where
        open-ended generations could not be mapped to fixed labels.
        """
        if raw is None:
            return None
        cleaned = raw.strip().strip(".\"'`").strip()
        if not cleaned:
            return None

        for name in self.names:
            if cleaned == name:
                return name

        low = cleaned.lower()
        lowered = {name.lower(): name for name in self.names}
        if low in lowered:
            return lowered[low]

        first_line = low.splitlines()[0].strip()
        if first_line in lowered:
            return lowered[first_line]

        hits = [n for l, n in lowered.items() if first_line.startswith(l)]
        if len(hits) == 1:
            return hits[0]

        hits = [n for l, n in lowered.items() if l in low]
        if len(hits) == 1:
            return hits[0]

        return None


def load_schema(path: str | Path) -> Schema:
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    labels = [
        Label(name=item["name"], definition=" ".join(item["definition"].split()))
        for item in raw["labels"]
    ]
    if len(labels) < 2:
        raise ValueError("Need at least two labels.")
    if len({lab.name for lab in labels}) != len(labels):
        raise ValueError("Label names must be unique.")

    return Schema(
        task_description=" ".join(raw.get("task_description", "").split()),
        label_field=raw.get("label_field", "Label"),
        text_field=raw.get("text_field", "Text"),
        labels=labels,
    )


def load_data(path: str | Path, schema: Schema,
              text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Load a CSV with `text` and `label` columns and validate against schema."""
    df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing column(s): {sorted(missing)}")

    df = df[[text_col, label_col]].rename(
        columns={text_col: "text", label_col: "label"}
    )
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    unknown = set(df["label"]) - set(schema.names)
    if unknown:
        raise ValueError(
            f"Labels in the CSV that are not in the schema: {sorted(unknown)}. "
            f"Schema has: {schema.names}"
        )

    missing_labels = [name for name in schema.names
                      if not (df["label"] == name).any()]
    if missing_labels:
        raise ValueError(
            f"CSV has no examples for schema label(s): {missing_labels}. "
            "Add at least one row for every label."
        )
    return df.reset_index(drop=True)


def sample_shots(df: pd.DataFrame, schema: Schema, k: int, seed: int,
                 interleave: bool = True) -> pd.DataFrame:
    """Draw k examples per class.

    `interleave=True` round-robins across classes rather than grouping them.
    The paper's reading of its own results is that one-pass prompting beats
    one-vs-all because the model sees contrastive examples; grouping all the
    CARE/HARM examples together and then all the FAIRNESS ones weakens that and
    also invites recency bias toward whichever class lands last in the prompt.
    """
    if k == 0:
        return df.iloc[0:0].copy()

    rng = random.Random(seed)
    per_class: dict[str, list[int]] = {}

    for name in schema.names:
        pool = df.index[df["label"] == name].tolist()
        if len(pool) < k:
            raise ValueError(
                f"Class {name!r} has {len(pool)} example(s) but k={k} were "
                f"requested. Lower k or add data for this class."
            )
        rng.shuffle(pool)
        per_class[name] = pool[:k]

    if interleave:
        order: list[int] = []
        for i in range(k):
            round_i = [per_class[name][i] for name in schema.names]
            rng.shuffle(round_i)
            order.extend(round_i)
    else:
        order = [idx for name in schema.names for idx in per_class[name]]

    return df.loc[order].reset_index(drop=True)


def holdout(df: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Everything not used as a shot. Guards against train/test leakage."""
    used = set(zip(shots["text"], shots["label"]))
    mask = [(t, l) not in used for t, l in zip(df["text"], df["label"])]
    return df[mask].reset_index(drop=True)


def stratified_test_set(df: pd.DataFrame, schema: Schema, n_per_class: int,
                        seed: int) -> pd.DataFrame:
    """Balanced test set, so macro F1 is not distorted by class imbalance."""
    rng = random.Random(seed)
    picks: list[int] = []
    for name in schema.names:
        pool = df.index[df["label"] == name].tolist()
        rng.shuffle(pool)
        if len(pool) < n_per_class:
            print(f"  [warn] class {name!r} has only {len(pool)} test examples")
        picks.extend(pool[:n_per_class])
    rng.shuffle(picks)
    return df.loc[picks].reset_index(drop=True)
