#!/usr/bin/env python3
"""Train FastText classifier from train.txt and save papra_model.bin."""

import fasttext
import pathlib
import sys

train_file = pathlib.Path(__file__).parent / "train.txt"
model_file = pathlib.Path(__file__).parent / "papra_model.bin"

if not train_file.exists():
    print(f"Error: {train_file} not found", file=sys.stderr)
    sys.exit(1)

print(f"Training on {train_file} ...", flush=True)
model = fasttext.train_supervised(
    str(train_file),
    epoch=50,
    lr=0.5,
    wordNgrams=2,
    verbose=2,
)

result = model.test(str(train_file))
print(f"Samples: {result[0]}  Precision: {result[1]:.3f}  Recall: {result[2]:.3f}")

model.save_model(str(model_file))
print(f"Model saved to {model_file}")
