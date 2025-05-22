#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.3.1")
from npfl138.datasets.uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `window`.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=100, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=2048, type=int, help="Batch size.")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=7, type=int, help="Window size to use.")
parser.add_argument("--hidden_layers", default=[4096], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--label_smoothing", default=0.3, type=float, help="Label smoothing.")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout regularization.")
parser.add_argument("--char_dropout", default=0.01, type=float, help="Character dropout regularization.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")


class BatchGenerator:
    """A simple batch generator, optionally with suffling.

    The functionality of this batch generator is very similar to
        torch.utils.data.DataLoader(
            torch.utils.data.StackDataset(inputs, outputs),
            batch_size=batch_size, shuffle=shuffle,
        )
    but if the data is stored in a single tensor, it is much faster.
    """
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor, batch_size: int, shuffle: bool):
        self._inputs = inputs
        self._outputs = outputs
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __len__(self):
        return (len(self._inputs) + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        indices = torch.randperm(len(self._inputs)) if self._shuffle else torch.arange(len(self._inputs))
        while len(indices):
            batch = indices[:self._batch_size]
            indices = indices[self._batch_size:]
            yield self._inputs[batch], self._outputs[batch]


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args

        # TODO: Implement a suitable model. The inputs are _windows_ of fixed size
        # (`args.window` characters on the left, the character in question, and
        # `args.window` characters on the right), where each character is
        # represented by a `torch.int64` index. To suitably represent the
        # characters, you can:
        # - Convert the character indices into _one-hot encoding_, which you can
        #   achieve by using `torch.nn.functional.one_hot` on the characters,
        #   and then concatenate the one-hot encodings of the window characters.
        # - Alternatively, you can experiment with `torch.nn.Embedding`s (an
        #   efficient implementation of one-hot encoding followed by a Dense layer)
        #   and flattening afterwards, or suitably using `torch.nn.EmbeddingBag`.
        self.embedding = torch.nn.Embedding(args.alphabet_size + 2, 16)  # Additional indices for padding/dropout replacement
        input_size = (2 * args.window + 1) * 16

        layers = []
        for i, hidden_layer in enumerate(args.hidden_layers):
            layers.append(torch.nn.Linear(input_size if i == 0 else args.hidden_layers[i - 1], hidden_layer))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(args.dropout))
        layers.append(torch.nn.Linear(args.hidden_layers[-1], 1))
        layers.append(torch.nn.Sigmoid())

        self.network = torch.nn.Sequential(*layers)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass.
        embedded = self.embedding(windows)
        flattened = embedded.view(embedded.size(0), -1)
        return self.network(flattened)

def train_model(model, train_data, dev_data, args):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        for batch_inputs, batch_labels in train_data:
            optimizer.zero_grad()
            outputs = model(batch_inputs).squeeze()
            loss = criterion(outputs, batch_labels.float())
            loss.backward()
            optimizer.step()

        # Evaluate on the dev set
        model.eval()
        with torch.no_grad():
            dev_loss = 0
            accuracy = torchmetrics.Accuracy(task="binary")
            for batch_inputs, batch_labels in dev_data:
                outputs = model(batch_inputs).squeeze()
                dev_loss += criterion(outputs, batch_labels.float()).item()
                accuracy.update(outputs.round(), batch_labels)
            print(f"Epoch {epoch + 1}/{args.epochs}, Dev Loss: {dev_loss:.4f}, Accuracy: {accuracy.compute().item():.4f}")


def generate_predictions(model, test_data, text, output_file):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_inputs, _ in test_data:
            outputs = model(batch_inputs).squeeze()
            predictions.extend(outputs.round().tolist())

    result = []
    for i, char in enumerate(text):
        if char.isalpha():
            result.append(char.upper() if predictions[i] else char.lower())
        else:
            result.append(char)

    return result


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. The default label dtype of torch.float32 is suitable for binary classification,
    # but you should change it to torch.int64 if you use 2-class classification (CrossEntropyLoss).
    uppercase_data = UppercaseData(args.window, args.alphabet_size, label_dtype=torch.float32)

    # Instead of using
    #   train = torch.utils.data.DataLoader(
    #     torch.utils.data.StackDataset(uppercase_data.train.windows, uppercase_data.train.labels),
    #     batch_size=args.batch_size, shuffle=True)
    # we use the BatchGenerator, which is about an order of magnitude faster.
    train = BatchGenerator(uppercase_data.train.windows, uppercase_data.train.labels, args.batch_size, shuffle=True)
    dev = BatchGenerator(uppercase_data.dev.windows, uppercase_data.dev.labels, args.batch_size, shuffle=False)
    test = BatchGenerator(uppercase_data.test.windows, uppercase_data.test.labels, args.batch_size, shuffle=False)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters, and train the model.
    model = Model(args)
    train_model(model, train, dev, args)

    # TODO: Generate correctly capitalized test set. Use `uppercase_data.test.text`
    # as input, capitalize suitable characters, and write the result to `predictions_file`
    # (which is by default `uppercase_test.txt` in the `args.logdir` directory).
    os.makedirs(args.logdir, exist_ok=True)
    result = generate_predictions(model, test, uppercase_data.test.text, os.path.join(args.logdir, "uppercase_test.txt"))
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        predictions_file.write("".join(result))


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
