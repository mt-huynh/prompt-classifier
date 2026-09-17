"""Finetuned RoBERTa baseline.

Mirrors the paper's baseline: encode the text with RoBERTa, take the [CLS]
embedding from the last layer, classify from it. Two modes, as in their Table 2:

  frozen    -- only the classification head trains. The paper gets ~8 macro F1
               this way, barely above chance on 5 classes. Included so you can
               reproduce that floor and confirm your wiring is sane.
  finetuned -- the whole encoder trains. This is the arm that actually competed
               with prompting.

Few-shot finetuning is unstable. With 5 examples per class you are taking a few
dozen gradient steps on a 125M-parameter model, and the seed matters as much as
the method -- hence the +/- 6 to 9 point standard deviations in the paper. Never
report a single run from this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import Schema

DEFAULT_ENCODER = "roberta-base"


class RobertaClassifier:
    def __init__(self, schema: Schema, encoder: str = DEFAULT_ENCODER,
                 freeze_encoder: bool = False, lr: float | None = None,
                 epochs: int = 20, batch_size: int = 8, max_length: int = 128,
                 seed: int = 0, device: str | None = None):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("pip install torch transformers") from exc

        self.torch = torch
        self.schema = schema
        self.freeze_encoder = freeze_encoder
        # A frozen encoder needs a much larger step size: the head is the only
        # thing learning and it starts from scratch.
        self.lr = lr if lr is not None else (1e-3 if freeze_encoder else 2e-5)
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label2id = {name: i for i, name in enumerate(schema.names)}
        self.id2label = {i: name for name, i in self.label2id.items()}

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(encoder)
        self.encoder = AutoModel.from_pretrained(encoder).to(self.device)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden = self.encoder.config.hidden_size
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, len(schema.names)),
        ).to(self.device)

    def _encode(self, texts: list[str]):
        return self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)

    def _cls(self, batch):
        out = self.encoder(**batch)
        return out.last_hidden_state[:, 0, :]   # [CLS] of the last layer

    def fit(self, train: pd.DataFrame, verbose: bool = False) -> "RobertaClassifier":
        torch = self.torch
        texts = train["text"].tolist()
        labels = torch.tensor(
            [self.label2id[l] for l in train["label"]], device=self.device
        )

        params = list(self.head.parameters())
        if not self.freeze_encoder:
            params += list(self.encoder.parameters())
        optim = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        n = len(texts)
        rng = np.random.default_rng(self.seed)

        self.encoder.train() if not self.freeze_encoder else self.encoder.eval()
        self.head.train()

        for epoch in range(self.epochs):
            order = rng.permutation(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                batch = self._encode([texts[i] for i in idx])
                target = labels[list(idx)]

                logits = self.head(self._cls(batch))
                loss = loss_fn(logits, target)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                total += loss.item() * len(idx)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"    epoch {epoch + 1:>3}  loss {total / n:.4f}")

        return self

    @property
    def _no_grad(self):
        return self.torch.no_grad()

    def predict(self, texts: list[str], batch_size: int = 32) -> list[str]:
        self.encoder.eval()
        self.head.eval()
        preds: list[str] = []
        with self._no_grad:
            for start in range(0, len(texts), batch_size):
                chunk = texts[start:start + batch_size]
                logits = self.head(self._cls(self._encode(chunk)))
                preds.extend(
                    self.id2label[int(i)] for i in logits.argmax(dim=-1).cpu()
                )
        return preds
