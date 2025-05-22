#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torchvision.transforms.v2 as v2

import npfl138
npfl138.require_version("2425.5")
from npfl138.datasets.cags import CAGS
import torchmetrics
from torch_dataset import ManualDataset, TransformedDataset


# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--width", default=512, type=int, help="Linear layer size.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout.")
parser.add_argument("--label_smoothing", default=0.2, type=float, help="Label smoothing.")
parser.add_argument("--augment", default="basic", type=str, help="Augment data.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--learning_rate_final", default=1e-5, type=float, help="Final learning rate")
parser.add_argument("--fine_tune_epochs", default=40, type=int, help="Number of epochs for fine-tuning")
parser.add_argument("--fine_tune_lr", default=5e-4, type=float, help="Learning rate for fine-tuning")
parser.add_argument("--fine_tune_lr_final", default=0, type=float, help="Final learning rate for fine-tuning")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")


class Model(npfl138.TrainableModule):
    def __init__(self, backbone, args: argparse.Namespace) -> None:
        super().__init__()
        self.backbone = backbone

        # Create a more sophisticated classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.backbone.num_features),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(self.backbone.num_features, args.width),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(args.width),
            torch.nn.Dropout(args.dropout / 2),
            torch.nn.Linear(args.width, CAGS.LABELS)
        )

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Dropout(args.dropout),
        #     torch.nn.Linear(self.backbone.num_features, CAGS.LABELS)
        # )

        # Initial weights Xavier initialization for the classifier
        for m in self.classifier.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


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

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, 224, 224]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[1, 224, 224]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(CAGS.LABELS)`.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    cags = CAGS(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer. For an
    # input image, the model returns a tensor of shape `[batch_size, 1280]`.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])
    transform_base = torch.nn.Sequential(
        preprocessing)

    if args.augment == "basic":
        transform_train = torch.nn.Sequential(
            preprocessing,
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(CAGS.H, padding=4)
        )
    elif args.augment == "randaugment":
        transform_train = torch.nn.Sequential(
            preprocessing,
            v2.RandAugment()
        )
    elif args.augment == "cutmix":
        transform_train = torch.nn.Sequential(
            preprocessing,
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(CAGS.H, padding=4),
            v2.CutMix(num_classes=CAGS.LABELS, alpha=1.0)
        )
    elif args.augment == "mixup":
        transform_train = torch.nn.Sequential(
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(CAGS.H, padding=4),
            v2.MixUp(num_classes=CAGS.LABELS, alpha=1.0)
        )
    else:
        transform_train = transform_base

    # TODO: Create the model and train it.
    model = Model(efficientnetv2_b0, args)

    train = ManualDataset(cags.train, transform_train)
    dev = ManualDataset(cags.dev, transform_base)
    test = ManualDataset(cags.test, transform_base)

    # We now create the `torch.utils.data.DataLoader` instances. For the `train` dataset,
    # we create it manually, for `dev` we use the `TransformedDataset.dataloader`.
    train = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.threads, persistent_workers=args.threads>0)
    dev = torch.utils.data.DataLoader(
        dev, batch_size=args.batch_size, shuffle=False,
        num_workers=args.threads, persistent_workers=args.threads>0)
    test = torch.utils.data.DataLoader(
        test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.threads, persistent_workers=args.threads>0)

    # Freeze backbone layers if not fine-tuning.
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.fine_tune_epochs,
        eta_min=args.learning_rate_final
    )

    model.configure(
        optimizer=optimizer,
        scheduler=scheduler,
        loss=torch.nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing) if args.label_smoothing > 0 else torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CAGS.LABELS)},
        logdir=args.logdir,
    )

    # Train the classifier first
    print("Phase 1: Training classifier only, backbone frozen")
    model.fit(train, dev=dev, epochs=args.epochs - args.fine_tune_epochs)

    # PHASE 2: Fine-tune the entire model
    if args.fine_tune_epochs > 0:
        print("Phase 2: Fine-tuning the entire model")

        # Unfreeze the backbone
        for param in model.backbone.parameters():
            param.requires_grad = True

        # Create a new optimizer with a lower learning rate for fine-tuning
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.fine_tune_lr,
            weight_decay=args.weight_decay
        )

        # New scheduler for fine-tuning phase
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.fine_tune_epochs,
            eta_min=args.fine_tune_lr_final
        )

        # Reconfigure the model with the new optimizer and scheduler
        model.configure(
            optimizer=optimizer,
            scheduler=scheduler
        )

        # Continue training with the entire model unfrozen
        model.fit(train, dev=dev, epochs=args.fine_tune_epochs)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(np.argmax(prediction), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
