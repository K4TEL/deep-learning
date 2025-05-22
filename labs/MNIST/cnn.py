#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.4")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        label = example["label"]  # a torch.Tensor with a single integer representing the label
        return image, label  # return an (input, target) pair


# To implement the residual connections, you can use various approaches, for example:
# - you can create a specialized `torch.nn.Module` subclass representing a residual
#   connection that gets the inside layers as an argument, and implement its forward call.
#   This allows you to have the whole network in a single `torch.nn.Sequential`.
class ResidualBlock(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # a comma-separated list of the following layers:

        layers = []
        in_channels = MNIST.C

        residual = False
        L = args.cnn.replace("-[", "-").replace("]", ",]").split(",")
        ignore = []
        for i, layer in enumerate(L):
            if not layer:
                continue
            if i in ignore:
                continue
            name, *params = layer.split("-")
            params = [int(param) if param.isdigit() else param for param in params]

            if name == "R":
                residual = True
                params = ["-".join(layer.split("-")[1:])]

            if residual:
                for j in range(i + 1, len(L)):
                    next_name, *next_params = L[j].split("-")
                    if next_name == "]":
                        residual = False
                        ignore.append(j)
                        break
                    else:
                        params.append(L[j])
                        ignore.append(j)

            print(layer, name, params)

        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
            if name == "C":
                filters, kernel_size, stride, padding = params
                layers.append(torch.nn.Conv2d(in_channels, filters, kernel_size, stride, padding))
                layers.append(torch.nn.ReLU())
                in_channels = filters
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer **without bias** and activation,
        #   then add a batch normalization layer, and finally the ReLU activation.
            elif name == "CB":
                filters, kernel_size, stride, padding = params
                layers.append(torch.nn.Conv2d(in_channels, filters, kernel_size, stride, padding, bias=False))
                layers.append(torch.nn.BatchNorm2d(filters))
                layers.append(torch.nn.ReLU())
                in_channels = filters
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default padding of 0 (the "valid" padding).
            elif name == "M":
                pool_size, stride = params
                layers.append(torch.nn.MaxPool2d(pool_size, stride))
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearity of the last layer) should be added
        #   to the input (of this `R` layer).
            elif name == "R":
                sublayers = []
                for sublayer in params:
                    sub_name, *sub_params = sublayer.split("-")
                    sub_params = [int(param) if param.isdigit() else param for param in sub_params]
                    if sub_name == "C":
                        filters, kernel_size, stride, padding = sub_params
                        sublayers.append(torch.nn.Conv2d(in_channels, filters, kernel_size, stride, padding))
                        sublayers.append(torch.nn.ReLU())
                        in_channels = filters
                    elif sub_name == "CB":
                        filters, kernel_size, stride, padding = sub_params
                        sublayers.append(torch.nn.Conv2d(in_channels, filters, kernel_size, stride, padding, bias=False))
                        sublayers.append(torch.nn.BatchNorm2d(filters))
                        sublayers.append(torch.nn.ReLU())
                        in_channels = filters
                layers.append(ResidualBlock(sublayers))
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
            elif name == "F":
                layers.append(torch.nn.Flatten())
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and the specified size.
            elif name == "H":
                size, = params
                layers.append(torch.nn.LazyLinear(size))
                layers.append(torch.nn.ReLU())
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
            elif name == "D":
                dropout_rate, = params
                layers.append(torch.nn.Dropout(float(dropout_rate)))

        # You can assume the resulting network is valid; it is fine to crash if it is not.

        # TODO: Finally, add the final Linear output layer with `MNIST.LABELS` units.

        layers.append(torch.nn.LazyLinear(MNIST.LABELS))
        self.model = torch.nn.Sequential(*layers)

        # print(self)

        # It might be difficult to compute the number of features after the `F` layer. You can
        # nevertheless use the `torch.nn.LazyLinear` and `torch.nn.LazyConv2d` layers, which
        # do not require the number of input features to be specified in the constructor.
        # However, after the whole model is constructed, you must call the model once on a dummy input
        # so that the number of features is computed and the model parameters are initialized.
        # To that end, you can use for example
        #   self.eval()(torch.zeros(1, MNIST.C, MNIST.H, MNIST.W))
        # where the `self.eval()` is necessary to avoid the batchnorms to update their running statistics.
        self.model.eval()(torch.zeros(1, MNIST.C, MNIST.H, MNIST.W))

    def forward(self, x):
        return self.model(x)


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data and create dataloaders.
    mnist = MNIST()

    train = torch.utils.data.DataLoader(Dataset(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(mnist.dev), batch_size=args.batch_size)

    # Create the model and train it
    model = Model(args)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
