#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch

import npfl138
npfl138.require_version("2425.11")
from npfl138.datasets.modelnet import ModelNet
from npfl138 import TrainableModule, TransformedDataset
import torchmetrics

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")


# 3D CNN Model inheriting from TrainableModule
class Conv3DNet(TrainableModule):
    def __init__(self, input_dim=20, num_classes=10):
        super(Conv3DNet, self).__init__()  # Initialize TrainableModule
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool3d(2)

        self.conv2 = torch.nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool3d(2)

        self.conv3 = torch.nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool3d(2)

        # Calculate the flattened size dynamically
        with torch.no_grad():
            # Create a dummy input of the expected shape for one item (C, D, H, W)
            # ModelNet.Dataset item['grid'] should be (1, input_dim, input_dim, input_dim)
            dummy_input_item = torch.zeros(1, 1, self.input_dim, self.input_dim, self.input_dim)
            dummy_out = self.pool3(self.relu3(
                self.conv3(self.pool2(self.relu2(self.conv2(self.pool1(self.relu1(self.conv1(dummy_input_item)))))))))
            self._flattened_size = dummy_out.nelement()  # Elements in the output feature map for one item

        self.fc1 = torch.nn.Linear(self._flattened_size, 128)
        self.relu4 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(128, self.num_classes)

    def forward(self, x):
        # x shape: (batch, 1, D, D, D)
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = self.pool3(self.relu3(self.conv3(out)))

        out = out.view(out.size(0), -1)  # Flatten
        if out.size(1) != self._flattened_size:
            raise ValueError(
                f"Flattened size mismatch. Expected {self._flattened_size}, got {out.size(1)}. Input shape was {x.shape}")

        out = self.relu4(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# Custom collate function for ModelNet data
def modelnet_collate_fn(batch_list):
    # Each item in batch_list is a dict {'grid': tensor, 'label': tensor}
    # item['grid'] has shape (C, D, H, W), e.g. (1, 20, 20, 20)
    # item['label'] has shape () (scalar tensor)
    grids = torch.stack([item['grid'].float() for item in batch_list])
    labels = torch.stack([item['label'].long() for item in batch_list])
    return grids, labels


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

    # Load the data.
    modelnet = ModelNet(args.modelnet)

    # TODO: Create the model and train it
    # Create TransformedDataset instances
    train_transformed_dataset = TransformedDataset(dataset=modelnet.train)
    train_transformed_dataset.collate = modelnet_collate_fn

    dev_transformed_dataset = TransformedDataset(dataset=modelnet.dev)
    dev_transformed_dataset.collate = modelnet_collate_fn

    test_transformed_dataset = TransformedDataset(dataset=modelnet.test)
    test_transformed_dataset.collate = modelnet_collate_fn  # Test data also has labels

    # Determine num_workers for DataLoaders
    num_workers = args.threads
    if args.threads == 0:  # Use all available CPU cores
        num_workers = os.cpu_count() or 1  # os.cpu_count() might be None or 0

    # Create DataLoaders using TransformedDataset.dataloader() method
    train_loader = train_transformed_dataset.dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=num_workers, seed=args.seed
    )
    dev_loader = dev_transformed_dataset.dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers, seed=args.seed
    )
    test_loader = test_transformed_dataset.dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers, seed=args.seed
    )

    # Initialize the model
    model = Conv3DNet(input_dim=args.modelnet, num_classes=modelnet.LABELS)

    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Define metrics (using torchmetrics)
    metrics = {
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=modelnet.LABELS)
    }

    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader),
        eta_min=1e-8
    )


    # Configure the TrainableModule
    model.configure(
        optimizer=optimizer,
        scheduler=schedule,
        loss=criterion,
        metrics=metrics,
        logdir=args.logdir,
    )
    print(f"Model configured to run on: {model.device}")
    print(model)
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}") # TrainableModule wraps module

    # Train the model using TrainableModule.fit()
    print("Starting training with TrainableModule...")
    model.fit(
        train_loader,
        epochs=args.epochs,
        dev=dev_loader,
        console=2  # For progress bar and epoch logs
    )

    # Generate test set annotations using TrainableModule.predict()
    print("Generating test predictions...")
    # model.predict returns list of NumPy arrays by default (as_numpy=True)
    # Each array corresponds to one sample's output logits/scores
    # data_with_labels=True because our collate_fn for test_loader returns (grids, labels)
    # and TrainableModule needs to know to extract only the grids.
    # predictions_outputs = model.predict(test_loader, data_with_labels=True, as_numpy=True)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(grid, target)` pairs.
        for prediction in model.predict(test_loader, data_with_labels=True, as_numpy=True):
            print(np.argmax(prediction), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
