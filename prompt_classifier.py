"""One-pass few-shot prompting classifier.

Structure follows Roy et al. (arXiv:2302.02029) section 5.2:

  1. all label definitions in the header, as a guideline
  2. k labeled examples per class, interleaved, separated by ###
  3. the test item last, left incomplete so the continuation is the label

What is deliberately different from the paper: it runs against a current model
rather than GPT-J-6B, the definitions go in a system prompt instead of being
prepended to the completion, and the default is a single greedy sample instead
of a 10-way majority vote. The paper needed the vote because sampling at
temperature 0.5 from a 6B model is noisy; a modern model at temperature 0 is
not, and the vote mostly buys you a 10x bill. Turn it back on via `n_votes` if
you want confidence estimates rather than bare predictions.
"""

from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd

from data import Schema

SEPARATOR = "###"

# Current model IDs. Haiku is the one to reach for when labelling a large
# unlabelled pool for the distillation path in README step 5.
DEFAULT_MODEL = "claude-sonnet-5"
CHEAP_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class Prediction:
    text: str
    predicted: str | None          # None when the generation did not map to a label
    raw: str
    votes: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        if not self.votes:
            return 1.0 if self.predicted else 0.0
        total = sum(self.votes.values())
        return max(self.votes.values()) / total if total else 0.0


def build_system_prompt(schema: Schema) -> str:
    lines = []
    if schema.task_description:
        lines.append(schema.task_description)
        lines.append("")
    lines.append("Label definitions:")
    for lab in schema.labels:
        lines.append(f"{lab.name}: {lab.definition}")
    lines.append("")
    lines.append(
        "Answer with exactly one of these labels and nothing else: "
        + ", ".join(schema.names)
        + ". Do not explain, do not add punctuation, do not hedge. If the text "
        "is ambiguous, pick the single best fit."
    )
    return "\n".join(lines)


def build_user_prompt(schema: Schema, shots: pd.DataFrame, text: str) -> str:
    blocks = []
    for _, row in shots.iterrows():
        blocks.append(
            f"{SEPARATOR}\n"
            f"{schema.text_field}: {row['text']}\n"
            f"{schema.label_field}: {row['label']}"
        )
    blocks.append(f"{SEPARATOR}\n{schema.text_field}: {text}")
    return "\n".join(blocks)


class PromptClassifier:
    def __init__(self, schema: Schema, shots: pd.DataFrame,
                 model: str = DEFAULT_MODEL, n_votes: int = 1,
                 temperature: float | None = None, prefill: bool = True,
                 max_workers: int = 8, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY or pass api_key=")

        self.client = anthropic.Anthropic(api_key=key)
        self.schema = schema
        self.shots = shots
        self.model = model
        self.n_votes = max(1, n_votes)
        # Greedy for a single sample; sampled when voting, or there is nothing
        # for the vote to disagree about.
        self.temperature = temperature if temperature is not None else (
            0.0 if self.n_votes == 1 else 0.7
        )
        self.prefill = prefill
        self.max_workers = max_workers
        self.system = build_system_prompt(schema)

        # Longest label, generously: labels are short, and capping output is
        # the cheapest guard against a chatty model.
        self.max_tokens = max(16, max(len(n) for n in schema.names) // 2 + 12)

    def _one_call(self, text: str) -> str:
        messages = [{"role": "user",
                     "content": build_user_prompt(self.schema, self.shots, text)}]
        if self.prefill:
            # Ends the prompt mid-pattern, the same trick the paper uses when it
            # cuts the final block off after the label field. Remove this if you
            # switch to a model with extended thinking enabled, which rejects
            # a prefilled assistant turn.
            messages.append({"role": "assistant",
                             "content": f"{self.schema.label_field}:"})

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system,
            messages=messages,
            stop_sequences=[SEPARATOR, "\n\n"],
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )

    def predict_one(self, text: str) -> Prediction:
        raws, errors, votes = [], [], Counter()
        for _ in range(self.n_votes):
            try:
                raw = self._one_call(text)
            except Exception as exc:                      # noqa: BLE001
                message = str(exc)
                errors.append(message)
                raws.append(f"<error: {message}>")
                continue
            raws.append(raw)
            mapped = self.schema.normalize(raw)
            if mapped:
                votes[mapped] += 1

        predicted = votes.most_common(1)[0][0] if votes else None
        return Prediction(text=text, predicted=predicted,
                          raw=" | ".join(raws), votes=dict(votes),
                          errors=errors)

    def predict(self, texts: list[str], progress: bool = True) -> list[Prediction]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(self.predict_one, texts))
        failures = sum(bool(result.errors) for result in results)
        if failures:
            raise RuntimeError(
                f"{failures}/{len(results)} predictions failed at the API. "
                "Check the API key, model name, network, and rate limits."
            )
        if progress:
            unparsed = sum(1 for r in results if r.predicted is None)
            if unparsed:
                print(f"  [warn] {unparsed}/{len(results)} generations did not "
                      f"map to a label")
        return results


def preview_prompt(schema: Schema, shots: pd.DataFrame, text: str) -> str:
    """Print exactly what the model will see. Read this before trusting any
    number the pipeline gives you -- a silently malformed prompt is the single
    most common cause of a bad few-shot result."""
    return (
        "----- SYSTEM -----\n" + build_system_prompt(schema)
        + "\n\n----- USER -----\n" + build_user_prompt(schema, shots, text)
        + f"\n\n----- ASSISTANT (prefill) -----\n{schema.label_field}:"
    )
