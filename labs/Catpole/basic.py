#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.2")
from npfl138 import GymCartpoleDataset
import copy
# from torchmetrics import Accuracy, B
from pathlib import Path
import random

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--evaluate", default=False, action="store_true", help="Evaluate the given model")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--render", default=False, action="store_true", help="Render during evaluation")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs.")
parser.add_argument("--model", default="gym_cartpole_model.pt", type=str, help="Output model path.")


def evaluate_model(
    model: torch.nn.Module, seed: int = 42, episodes: int = 100, render: bool = False, report_per_episode: bool = False
) -> float:
    """Evaluate the given model on CartPole-v1 environment.

    Returns the average score achieved on the given number of episodes.
    """
    import gymnasium as gym

    # Create the environment
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env.reset(seed=seed)

    # Evaluate the episodes
    total_score = 0
    for episode in range(episodes):
        observation, score, done = env.reset()[0], 0, False
        while not done:
            prediction = model(torch.from_numpy(observation)).numpy(force=True)
            assert len(prediction) == 2, "The model must output two values."
            action = np.argmax(prediction)

            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated

        total_score += score
        if report_per_episode:
            print("The episode {} finished with score {}.".format(episode + 1, score))
    return total_score / episodes


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        # TODO: Create the model layers, with the last layer having 2 outputs.
        # To store a list of layers, you can use either `torch.nn.Sequential`
        # or `torch.nn.ModuleList`; you should *not* use a Python list.
        # Create the model layers using `torch.nn.Sequential`.
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 32),  # Input size 4, output size 16
            torch.nn.Sigmoid(),  # Activation function
            torch.nn.Linear(32, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, 2),  # Output size 2
            torch.nn.Sigmoid()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # TODO: Run your model. Because some inputs are on a CPU, you should
        # start by moving them to the `model.device`.
        inputs = inputs.to(self.device)
        return self.model(inputs)


def main(args: argparse.Namespace) -> torch.nn.Module | None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    if not args.evaluate:
        if args.batch_size is ...:
            raise ValueError("You must specify the batch size, either in the defaults or on the command line.")
        if args.epochs is ...:
            raise ValueError("You must specify the number of epochs, either in the defaults or on the command line.")

        # Create logdir name.
        args.logdir = os.path.join("logs", "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
        ))

        # Load the provided dataset. The `dataset.train` is a collection of 100 examples,
        # each being a pair of (inputs, label), where:
        # - `inputs` is a vector with `GymCartpoleDataset.FEATURES` floating point values,
        # - `label` is a gold 0/1 class index.

        large_source_file = "large_gym_cartpole_data.txt"

        if not Path(large_source_file).exists():
            dataset = GymCartpoleDataset()

            N = len(dataset.train)

            observations, labels = dataset.train.observations, dataset.train.labels

            # print(torch.max(observations))
            # print(torch.min(observations))

            o, l = copy.deepcopy(observations), copy.deepcopy(labels)
            for i in range(500):
                original = copy.deepcopy(observations)
                original[:, 0] += (i * 0.002)

                l = torch.hstack((l, labels))
                o = torch.vstack((o, original))

                l = torch.hstack((l, torch.logical_not(labels).type(torch.int32)))
                o = torch.vstack((o, -original))

                rand_pos = np.random.uniform(-4.8, 4.8, N)
                rand = copy.deepcopy(observations)
                rand[:, 0] = torch.from_numpy(rand_pos)

                l = torch.hstack((l, labels))
                o = torch.vstack((o, rand))

                l = torch.hstack((l, torch.logical_not(labels).type(torch.int32)))
                o = torch.vstack((o, -rand))

            observations, labels = o, l
            expanded_data = np.zeros((len(observations), 5))

            for i, (o, l) in enumerate(zip(observations, labels)):
                expanded_data[i, :] = np.hstack((o.numpy(), l.item()))

            # print(expanded_data)
            np.savetxt(large_source_file, expanded_data, delimiter=" ",
                       fmt=['%.3f', '%.3f', '%.3f', '%.3f', '%d'])

        dataset = GymCartpoleDataset(large_source_file)

        # print(len(dataset.train))
        # print(dataset.train.labels)

        dataset.train._labels = torch.nn.functional.one_hot(dataset.train.labels, 2)
        dataset.train._labels = dataset.train.labels.type(torch.float32)

        # print(dataset.train.labels)

        train = torch.utils.data.DataLoader(dataset.train, args.batch_size, shuffle=True)

        model = Model(args)

        # TODO: Configure the model for training.
        model.configure(
            optimizer=torch.optim.Adam(model.parameters(), lr=0.005),
            loss=torch.nn.BCELoss(),
            metrics={"accuracy": torchmetrics.classification.BinaryAccuracy()},
            logdir=args.logdir,
            # scheduler=
        )

        # TODO: Train the model. Note that you can pass a list of callbacks to the
        # `fit` method, each being a callable accepting the model, epoch, and logs.
        # Such callbacks are called after every epoch and if they modify the
        # logs dictionary, the values are logged on the console and to TensorBoard.
        model.fit(train, epochs=args.epochs, callbacks=[])

        # Save the model, both the hyperparameters and the parameters. If you
        # added additional arguments to the `Model` constructor beyond `args`,
        # you would have to add them to the `save_config` call below.
        model.save_config(f"{args.model}.json", args=args)
        model.save_weights(args.model)

    else:
        # Evaluating, either manually or in ReCodEx.
        model = Model(**Model.load_config(f"{args.model}.json"))
        model.load_weights(args.model)

        if args.recodex:
            return model
        else:
            score = evaluate_model(model, seed=args.seed, render=args.render, report_per_episode=True)
            print("The average score was {}.".format(score))


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
