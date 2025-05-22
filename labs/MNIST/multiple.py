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
parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class SharedCNN(torch.nn.Module):
    def __init__(self):
        super(SharedCNN, self).__init__()
        self.rescale = torch.nn.Identity()  # Assuming input images are already in [0, 1]
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=0)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(20 * 6 * 6, 200)  # Updated input size based on image dimensions

    def forward(self, x):
        x = self.rescale(x)
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.nn.functional.relu(self.fc(x))
        return x


# Create a dataset from consecutive _pairs_ of original examples, assuming
# that the size of the original dataset is even.
class DatasetOfPairs(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self._dataset = dataset

    def __len__(self):
        # TODO: The new dataset has half the size of the original one.
        return len(self._dataset) // 2

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        # TODO: Given an `index`, generate an example composed of two input examples.
        # Notably, considering examples `self._dataset[2 * index]` and `self._dataset[2 * index + 1]`,
        # each being a dictionary with keys "image" and "label", return a pair `(input, output)` with
        # - `input` being a pair of images, each converted to `torch.float32` and divided by 255,
        # - `output` being a pair of labels.
        first_example = self._dataset[2 * index]
        second_example = self._dataset[2 * index + 1]

        # Each image is converted to `torch.float32` and divided by 255.
        first_image = first_example["image"].float() / 255.0
        second_image = second_example["image"].float() / 255.0

        # Each label remains unchanged.
        first_label = first_example["label"]
        second_label = second_example["label"]

        # Return a pair `(input, output)` with input being a pair of images,
        # and output being a pair of labels.
        return (first_image, second_image), (first_label, second_label)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # TODO: Create all layers required to implement the forward pass.
        self.shared_cnn = SharedCNN()
        self.fc_compare = torch.nn.Linear(400, 200)
        self.fc_out = torch.nn.Linear(200, 1)
        self.fc_digit = torch.nn.Linear(200, 10)

    def forward(
        self, first: torch.Tensor, second: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: Implement the forward pass of the model using the layers created in the constructor.
        #
        # The model starts by passing each input image through the same
        # module (with shared weights), which should perform
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - flattening layer,
        # - fully connected layer with 200 neurons and ReLU activation,
        # obtaining a 200-dimensional feature vector of each image.

        # Using the computed representations, the model should produce four outputs:
        # - first, compute _direct comparison_ whether the first digit is
        #   greater than the second, by
        #   - concatenating the two 200-dimensional image feature vectors,
        #   - processing them using another 200-neuron ReLU linear layer,
        #   - computing one output using a linear layer and the **sigmoid** activation;
        # - then, classify the computed representation FV of the first image using
        #   a linear layer into 10 classes;
        # - then, classify the computed representation FV of the second image using
        #   the same layer (identical, i.e., with shared weights) into 10 classes;
        # - finally, compute _indirect comparison_ whether the first digit
        #   is greater than second, by comparing the predictions from the above
        #   two outputs.
        # Process both images through the shared CNN
        fv1 = self.shared_cnn(first)
        fv2 = self.shared_cnn(second)

        # Direct comparison
        combined = torch.cat((fv1, fv2), dim=1)
        x = torch.nn.functional.relu(self.fc_compare(combined))
        direct_comparison = torch.sigmoid(self.fc_out(x)).squeeze()

        # Digit classifications
        digit_1 = self.fc_digit(fv1)
        digit_2 = self.fc_digit(fv2)

        # Indirect comparison
        digit1_class = torch.argmax(digit_1, dim=1)
        digit2_class = torch.argmax(digit_2, dim=1)
        indirect_comparison = (digit1_class > digit2_class).float()

        return direct_comparison, digit_1, digit_2, indirect_comparison

    def compute_loss(self, y_pred, y_true, *inputs):
        # The `compute_loss` method can override the loss computation of the model.
        # It is needed when there are multiple model outputs or multiple losses to compute.
        # We start by unpacking the multiple outputs of the model and the multiple targets.
        direct_comparison_pred, digit_1_pred, digit_2_pred, indirect_comparison_pred = y_pred
        digit_1_true, digit_2_true = y_true

        # TODO: Compute the required losses. Note that the `direct_comparison_pred` is
        # really a probability (sigmoid was applied), while the `digit_1_pred` and
        # `digit_2_pred` are logits of 10-class classification.
        # Ensure target has the same shape as the prediction
        direct_comparison_true = (digit_1_true > digit_2_true).float()

        direct_comparison_loss = torch.nn.functional.binary_cross_entropy(direct_comparison_pred,
                                                                          direct_comparison_true)
        digit_1_loss = torch.nn.functional.cross_entropy(digit_1_pred, digit_1_true)
        digit_2_loss = torch.nn.functional.cross_entropy(digit_2_pred, digit_2_true)

        return direct_comparison_loss + digit_1_loss + digit_2_loss

    def compute_metrics(self, y_pred, y_true, *inputs):
        # The `compute_metrics` can override metric computation for the model. We start by
        # unpacking the multiple outputs of the model and the multiple targets.
        direct_comparison_pred, digit_1_pred, digit_2_pred, indirect_comparison_pred = y_pred
        digit_1_true, digit_2_true = y_true

        direct_comparison_true = (digit_1_true > digit_2_true).float()

        # TODO: Update two metrics -- the `direct_comparison` and the `indirect_comparison`.
        self.metrics["direct_comparison"].update(direct_comparison_pred, direct_comparison_true.int())
        self.metrics["indirect_comparison"].update(indirect_comparison_pred, direct_comparison_true.int())

        # Finally, we return the dictionary of all the metric values.
        return {name: metric.compute() for name, metric in self.metrics.items()}


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

    train = torch.utils.data.DataLoader(DatasetOfPairs(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(DatasetOfPairs(mnist.dev), batch_size=args.batch_size)

    # Create the model and train it
    model = Model(args)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        metrics={
            # TODO: Create two binary accuracy metrics using `torchmetrics.Accuracy`:
            "direct_comparison": torchmetrics.Accuracy(task='binary'),
            "indirect_comparison": torchmetrics.Accuracy(task='binary'),
        },
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
