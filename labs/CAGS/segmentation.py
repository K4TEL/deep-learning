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

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=80, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dropout", default=0.25, type=float, help="Dropout.")
parser.add_argument("--label_smoothing", default=0.25, type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=5e-4, type=float, help="Learning rate")
parser.add_argument("--learning_rate_final", default=0, type=float, help="Final learning rate")
parser.add_argument("--fine_tune_epochs", default=20, type=int, help="Number of epochs for fine-tuning")
parser.add_argument("--fine_tune_lr", default=5e-4, type=float, help="Learning rate for fine-tuning")
parser.add_argument("--fine_tune_lr_final", default=0, type=float, help="Final learning rate for fine-tuning")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, cags: CAGS.Dataset, augmentation_fn=None) -> None:
        self.dataset = cags
        self.augmentation_fn = augmentation_fn

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        if not self.augmentation_fn:
            return self.dataset[index]["image"], self.dataset[index]["mask"]
        else:
            return self.augmentation_fn(self.dataset[index]["image"]), self.dataset[index]["mask"]


class Model(npfl138.TrainableModule):
    def __init__(self, backbone, args: argparse.Namespace) -> None:
        super().__init__()

        # TODO: Define the model
        # Apart from calling the model as in the classification task, you can call it using
        #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
        # obtaining (assuming the input images have 224x224 resolution):
        # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
        # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.

        # Define the model for segmentation
        # The backbone is EfficientNetV2-B0, and we'll use feature maps from different levels
        # to create a U-Net-like architecture for segmentation
        self.backbone = backbone

        # Decoder blocks that progressively upscale the feature maps
        self.segmentation = torch.nn.ModuleList()

        # Get sample features to determine dimensions
        dummy_input = torch.zeros(1, CAGS.C, CAGS.W, CAGS.H)
        output, features = self.backbone.forward_intermediates(dummy_input)
        feature_dims = [dummy_input.shape] + [f.shape for f in features]

        for i, ch in enumerate(feature_dims):
            print(i, ch)

        # Decoder path (from deepest to shallowest)
        in_channels = feature_dims[-1][1]

        # Create decoder blocks
        for i in range(len(feature_dims)-1, 0, -1):
            # Upsampling
            self.segmentation.append(torch.nn.ConvTranspose2d(
                in_channels, feature_dims[i - 1][1],
                kernel_size=2, stride=2
            ))

            # Skip connection processing
            self.segmentation.append(torch.nn.Sequential(
                torch.nn.Conv2d(feature_dims[i - 1][1] * 2, feature_dims[i - 1][1], kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(feature_dims[i - 1][1]),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout2d(args.dropout),
                torch.nn.Conv2d(feature_dims[i - 1][1], feature_dims[i - 1][1], kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(feature_dims[i - 1][1]),
                torch.nn.ReLU(inplace=True)
            ))

            in_channels = feature_dims[i - 1][1]

        for i, ch in enumerate(self.segmentation):
            print(i, ch)

        # Sigmoid activation for mask prediction
        self.sigmoid = torch.nn.Sigmoid()

        # Output layer - 1x1 convolution to get a single channel mask
        self.segmentation.append(torch.nn.Conv2d(feature_dims[0][1], 1, kernel_size=1))

    def forward(self, x):
        # Get features from the backbone
        _, encoder_features = self.backbone.forward_intermediates(x)

        # Start with the deepest feature map
        x = encoder_features[-1]

        # Decoder path
        for i in range(len(encoder_features)):
            x = self.segmentation[i * 2](x)

            if i==len(encoder_features)-1:
                break

            # Skip connection
            skip_connection = encoder_features[-(i + 2)]

            # Adjust dimensions if necessary
            if x.shape[2:] != skip_connection.shape[2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False
                )

            # Concatenate and process
            x = torch.cat([x, skip_connection], dim=1)
            x = self.segmentation[i * 2 + 1](x)

        # Final convolution and activation
        x = self.segmentation[-1](x)
        x = self.sigmoid(x)
        return x


class CustomLoss(torch.nn.Module):
    def __init__(self, label_smoothing: float) -> None:
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing) if label_smoothing > 0 \
            else torch.nn.CrossEntropyLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Ensure y_pred and y_true are of the same shape
        if y_pred.dim() == 4:  # (N, 1, H, W) -> (N, H*W)
            y_pred = y_pred.squeeze(1).reshape((-1, CAGS.H * CAGS.W))
        if y_true.dim() == 4:
            y_true = y_true.squeeze(1).reshape((-1, CAGS.H * CAGS.W))

        l = self.loss(y_pred, y_true)
        return l


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

    # Load the EfficientNetV2-B0 model without the classification layer.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    train = TorchDataset(cags.train, preprocessing)
    dev = TorchDataset(cags.dev, preprocessing)
    test = TorchDataset(cags.test, preprocessing)

    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dev, batch_size=args.batch_size, shuffle=False)
    test = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False)

    model = Model(efficientnetv2_b0, args)

    # Freeze backbone layers if not fine-tuning.
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.segmentation.parameters(),
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
        loss=CustomLoss(args.label_smoothing),
        metrics={"IoU": CAGS.MaskIoUMetric()},
        logdir=args.logdir,
    )

    # print(model)

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
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for mask in model.predict(test, data_with_labels=True):
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
