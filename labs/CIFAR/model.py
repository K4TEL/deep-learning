#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch

import npfl138
npfl138.require_version("2425.4")
from npfl138.datasets.cifar10 import CIFAR10
from torchvision.transforms import v2
from torch_dataset import ManualDataset, TransformedDataset
import torchmetrics

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--activation", default="relu", type=str, help="Activation type")
parser.add_argument("--augment", default="basic", type=str, help="Augmentation type (basic/autoaugment/randaugment/cutout/cutmix/mixup)")
parser.add_argument("--decay", default="cosine", type=str, help="Decay type (cosine/piecewise)")
parser.add_argument("--momentum", default=None, type=float, help="Nesterov momentum to use in SGD.")
parser.add_argument("--depth", default=56, type=int, help="Model depth")
parser.add_argument("--width", default=3, type=int, help="Model width")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout")
parser.add_argument("--label_smoothing", default=0.2, type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--learning_rate_final", default=1e-5, type=float, help="Final learning rate")
parser.add_argument("--model", default="widenet", type=str, help="Model type (resnet/widenet)")
parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer type (SGD/Adam)")


class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, stochastic_depth_prob=0.0, layer_index=0, total_layers=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.shortcut = torch.nn.Identity()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

        self.stochastic_depth_prob = stochastic_depth_prob * layer_index / total_layers

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply stochastic depth
        if self.training and self.stochastic_depth_prob > 0:
            if torch.rand(1).item() < self.stochastic_depth_prob:
                return identity

        out += identity
        out = torch.nn.functional.relu(out)

        return out


class ResNet(torch.nn.Module):
    def __init__(self, depth=56, num_classes=10, dropout=0.0, stochastic_depth=0.0, activation='relu'):
        super().__init__()
        n = (depth - 2) // 6

        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 16, n, stride=1, stochastic_depth=stochastic_depth, layer_offset=0,
                                       total_layers=3 * n)
        self.layer2 = self._make_layer(16, 32, n, stride=2, stochastic_depth=stochastic_depth, layer_offset=n,
                                       total_layers=3 * n)
        self.layer3 = self._make_layer(32, 64, n, stride=2, stochastic_depth=stochastic_depth, layer_offset=2 * n,
                                       total_layers=3 * n)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(64, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride, stochastic_depth, layer_offset, total_layers):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, stochastic_depth, layer_offset, total_layers))

        for i in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1, stochastic_depth, layer_offset + i, total_layers))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # Input normalization (equivalent to Rescaling in Keras)
        x = x / 255.0

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class WideNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, stochastic_depth_prob=0.0, layer_index=0, total_layers=1):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = torch.nn.Identity()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        self.stochastic_depth_prob = stochastic_depth_prob * layer_index / total_layers

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = torch.nn.functional.relu(out)

        if self.shortcut is not torch.nn.Identity():
            identity = self.shortcut(out)
        else:
            identity = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)

        # Apply stochastic depth
        if self.training and self.stochastic_depth_prob > 0:
            if torch.rand(1).item() < self.stochastic_depth_prob:
                return identity

        out += identity

        return out


class WideNet(torch.nn.Module):
    def __init__(self, depth=28, width=10, num_classes=10, dropout=0.0, stochastic_depth=0.0, activation='relu'):
        super().__init__()
        n = (depth - 4) // 6

        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(16, 16 * width, n, stride=1, stochastic_depth=stochastic_depth, layer_offset=0,
                                       total_layers=3 * n)
        self.layer2 = self._make_layer(16 * width, 32 * width, n, stride=2, stochastic_depth=stochastic_depth,
                                       layer_offset=n, total_layers=3 * n)
        self.layer3 = self._make_layer(32 * width, 64 * width, n, stride=2, stochastic_depth=stochastic_depth,
                                       layer_offset=2 * n, total_layers=3 * n)

        self.bn = torch.nn.BatchNorm2d(64 * width)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(64 * width, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride, stochastic_depth, layer_offset, total_layers):
        layers = []
        layers.append(WideNetBlock(in_channels, out_channels, stride, stochastic_depth, layer_offset, total_layers))

        for i in range(1, blocks):
            layers.append(WideNetBlock(out_channels, out_channels, 1, stochastic_depth, layer_offset + i, total_layers))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # Input normalization (equivalent to Rescaling in Keras)
        x = x / 255.0

        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = torch.nn.functional.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        if args.model == "resnet":
            model = ResNet(depth=args.depth, num_classes=10, dropout=args.dropout, activation=args.activation)
        elif args.model == "widenet":
            model = WideNet(depth=args.depth, width=args.width, num_classes=10, dropout=args.dropout,
                            activation=args.activation)
        else:
            raise ValueError("Unsupported model type: {}".format(args.model))

        self.model = model

    def forward(self, x):
        return self.model(x)


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
    cifar = CIFAR10()

    # TODO: Create the model and train it.
    # Define data augmentation transformations based on the chosen augmentation type.
    if args.augment == "basic":
        transform_train = torch.nn.Sequential(
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(32, padding=4)
        )
    elif args.augment == "autoaugment":
        transform_train = torch.nn.Sequential(
            v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10)
        )
    elif args.augment == "randaugment":
        transform_train = torch.nn.Sequential(
            v2.RandAugment()
        )
    elif args.augment == "cutmix":
        transform_train = torch.nn.Sequential(
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(32, padding=4),
            v2.CutMix(num_classes=10, alpha=1.0)
        )
    elif args.augment == "mixup":
        transform_train = torch.nn.Sequential(
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(32, padding=4),
            v2.MixUp(num_classes=10, alpha=1.0)
        )
    else:
        transform_train = torch.nn.Identity()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(args)
    model = model.to(device)

    train = ManualDataset(cifar.train, transform_train)
    dev = TransformedDataset(cifar.dev)
    test = TransformedDataset(cifar.test)

    # We now create the `torch.utils.data.DataLoader` instances. For the `train` dataset,
    # we create it manually, for `dev` we use the `TransformedDataset.dataloader`.
    train = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        num_workers=1, persistent_workers=True)
    dev = dev.dataloader(batch_size=args.batch_size, num_workers=1)
    test = test.dataloader(batch_size=args.batch_size, num_workers=1)

    optimizer = None

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum if args.momentum else 0,
            nesterov=args.momentum is not None
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate
        )

    scheduler = None

    if args.decay and args.learning_rate_final:
        num_batches = len(train) * args.epochs
        if args.decay == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.learning_rate / args.learning_rate,
                end_factor=args.learning_rate_final / args.learning_rate,
                total_iters=num_batches
            )
        elif args.decay == "exponential":
            gamma = (args.learning_rate_final / args.learning_rate) ** (1 / num_batches)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma
            )
        elif args.decay == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_batches,
                eta_min=args.learning_rate_final
            )

    model.configure(
        optimizer=optimizer,
        scheduler=scheduler,
        loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing) if args.label_smoothing > 0 else torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CIFAR10.LABELS)},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # for t in test:
    #     print(len(t))
    #     print(model.predict(t[0], data_with_labels=False))

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for probs in model.predict(test, data_with_labels=True):
            # print(probs)
            print(np.argmax(probs), file=predictions_file)

    # dev = dev_d.dataloader(batch_size=1, num_workers=1)
    with open(os.path.join(args.logdir, "cifar_competition_dev.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for probs in model.predict(dev, data_with_labels=True):
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
