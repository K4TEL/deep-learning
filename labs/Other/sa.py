#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchmetrics
import transformers

import npfl138
npfl138.require_version("2425.10")
from npfl138.datasets.text_classification_dataset import TextClassificationDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, eleczech: transformers.PreTrainedModel,
                 dataset: TextClassificationDataset.Dataset) -> None:
        super().__init__()

        # TODO: Define the model. Note that
        # - the dimension of the EleCzech output is `eleczech.config.hidden_size`;
        # - the size of the vocabulary of the output labels is `len(dataset.label_vocab)`.
        self._output_layer = torch.nn.Linear(eleczech.config.hidden_size, len(dataset.label_vocab))
        self._eleczech = eleczech

    # TODO: Implement the model computation.
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Model computation
        hidden = self._eleczech(input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden[:, 0]  # Use the representation of the [CLS] token.
        return self._output_layer(hidden)


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: TextClassificationDataset.Dataset, tokenizer) -> None:
        super().__init__(dataset)
        self.tokenizer = tokenizer

    def transform(self, example):
        # TODO: Process single examples containing `example["document"]` and `example["label"]`.
        tokenized = self.tokenizer(example["document"], truncation=True, padding="max_length", return_tensors="pt")
        label = example["label"]
        return tokenized, label

    def collate(self, batch):
        # TODO: Construct a single batch using a list of examples from the `transform` function.
        tokenized, labels = zip(*batch)
        input_ids = torch.cat([t["input_ids"] for t in tokenized], dim=0)
        attention_mask = torch.cat([t["attention_mask"] for t in tokenized], dim=0)
        # print(labels)
        labels = torch.tensor([self.dataset._label_vocab.index(label) if len(label) > 0 else 1 for label in labels], dtype=torch.long)
        # print(labels)
        return (input_ids, attention_mask), labels


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

    # Load the Electra Czech small lowercased.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.AutoModel.from_pretrained("ufal/eleczech-lc-small")

    # Load the data.
    facebook = TextClassificationDataset("czech_facebook")

    # TODO: Prepare the data for training.
    train_dataset = TrainableDataset(facebook.train, tokenizer)
    dev_dataset = TrainableDataset(facebook.dev, tokenizer)
    test_dataset = TrainableDataset(facebook.test, tokenizer)

    train_loader = train_dataset.dataloader(batch_size=args.batch_size, shuffle=True)
    dev_loader = dev_dataset.dataloader(batch_size=args.batch_size, shuffle=False)
    test_loader = test_dataset.dataloader(batch_size=args.batch_size, shuffle=False)

    # Create the model.
    model = Model(args, eleczech, facebook.train)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(facebook.train.label_vocab))

    # TODO: Configure and train the model
    model.configure(
        optimizer=optimizer,
        loss=loss_fn,
        metrics={"accuracy": accuracy}
    )
    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set.
        predictions = model.predict(test_loader, data_with_labels=True)
        # print(predictions)

        for document_logits in predictions:
            print(facebook.train.label_vocab.string(np.argmax(document_logits)), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
