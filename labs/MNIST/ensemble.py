#!/usr/bin/env python3
import argparse

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.3")
from npfl138.datasets.mnist import MNIST
import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer_size", default=200, type=int, help="Size of the hidden layer.")
parser.add_argument("--models", default=5, type=int, help="Number of models.")
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


def main(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Load the data and create dataloaders.
    mnist = MNIST()

    train = torch.utils.data.DataLoader(Dataset(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(mnist.dev), batch_size=args.batch_size)

    # Create the models.
    models = []
    for model in range(args.models):
        models.append(npfl138.TrainableModule(torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(MNIST.H * MNIST.W * MNIST.C, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, MNIST.LABELS),
        )))

        models[-1].configure(
            optimizer=torch.optim.Adam(models[-1].parameters()),
            loss=torch.nn.CrossEntropyLoss(),
            metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)},
        )

        print("Training model {}: ".format(model + 1), end="", flush=True)
        models[-1].fit(train, epochs=args.epochs, console=0)
        print("Done")

    individual_accuracies, ensemble_accuracies = [], []
    for model in range(args.models):
        # TODO: Compute the accuracy on the dev set for the individual `models[model]`.
        individual_result = models[model].evaluate(dev)
        individual_accuracy = individual_result["test_accuracy"]

        # TODO: Compute the accuracy on the dev set for the ensemble `models[0:model+1]`.
        #
        # Generally you can choose one of the following approaches:
        # 1) For the given models to ensemble, create a new `npfl138.TrainableModule` subclass
        #    that in `forward` runs the given models and averages their outputs. Then you can
        #    configure the model with the required metric (and a loss) and use `model.evaluate`.
        # 2) Manually perform the averaging (using PyTorch or NumPy). In this case you do not
        #    need to construct the ensemble model at all; instead, call `model.predict`
        #    on the `dev` dataloader (with `data_with_labels=True` to indicate the dataloader
        #    also contains the labels), stack the predictions, and average the results.
        #    To measure accuracy, either do it completely manually or use `torchmetrics.Accuracy`.

        # Compute the accuracy on the dev set for the ensemble of models
        all_predictions = []
        true_labels = []

        # Perform predictions for each model in the ensemble
        for inputs, labels in dev:
            true_labels.extend(labels)
            ensemble_outputs = None
            for m in range(model + 1):
                predictions = models[m].predict(inputs)
                predictions = np.array(predictions)
                if ensemble_outputs is None:
                    ensemble_outputs = predictions
                else:
                    ensemble_outputs += predictions

            # Average the predictions over the ensemble
            averaged_predictions = ensemble_outputs / (model + 1)
            all_predictions.extend(torch.Tensor(averaged_predictions))

        # Concatenate predictions and true labels
        all_predictions = torch.vstack(all_predictions)
        true_labels = torch.hstack(true_labels)

        # Calculate ensemble accuracy
        accuracy_metric = torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)
        ensemble_accuracy = accuracy_metric(all_predictions.argmax(dim=1), true_labels)

        # Store the accuracies
        individual_accuracies.append(individual_accuracy)
        ensemble_accuracies.append(ensemble_accuracy)
    return individual_accuracies, ensemble_accuracies


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    individual_accuracies, ensemble_accuracies = main(main_args)
    for model, (individual_accuracy, ensemble_accuracy) in enumerate(zip(individual_accuracies, ensemble_accuracies)):
        print("Model {}, individual accuracy {:.2f}, ensemble accuracy {:.2f}".format(
            model + 1, 100 * individual_accuracy, 100 * ensemble_accuracy))
