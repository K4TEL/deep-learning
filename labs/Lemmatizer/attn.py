#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.9")
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=41, type=int, help="Random seed.")
parser.add_argument("--show_results_every_batch", default=10, type=int, help="Show results every given batch.")
parser.add_argument("--tie_embeddings", default=False, action="store_true", help="Tie target embeddings.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class WithAttention(torch.nn.Module):
    """A class adding Bahdanau attention to a given RNN cell."""
    def __init__(self, cell, attention_dim):
        super().__init__()
        self._cell = cell

        # TODO: Define
        # - `self._project_encoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs.
        # - `self._project_decoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs
        # - `self._output_layer` as a linear layer with `attention_dim` inputs and 1 output
        self._project_encoder_layer = torch.nn.Linear(self._cell.hidden_size, attention_dim)  #
        self._project_decoder_layer = torch.nn.Linear(self._cell.hidden_size, attention_dim)  #
        self._output_layer = torch.nn.Linear(attention_dim, 1)  #

    def setup_memory(self, encoded):
        self._encoded = encoded
        # TODO: Pass the `encoded` through the `self._project_encoder_layer` and store
        # the result as `self._encoded_projected`.
        self._encoded_projected = self._project_encoder_layer(encoded) #

    def forward(self, inputs, states):
        # TODO: Compute the attention.
        # - According to the definition, we need to project the encoder states, but we have
        #   already done that in `setup_memory`, so we just take `self._encoded_projected`.
        # - Compute projected decoder state by passing the given state through the `self._project_decoder_layer`.
        # - Sum the two projections. However, you have to deal with the fact that the first projection has
        #   shape `[batch_size, input_sequence_len, attention_dim]`, while the second projection has
        #   shape `[batch_size, attention_dim]`. The best solution is capable of creating the sum
        #   directly without creating any intermediate tensor.
        # - Pass the sum through the `torch.tanh` and then through the `self._output_layer`.
        # - Then, run softmax activation, generating `weights`.
        # - Multiply the original (non-projected) encoder states `self._encoded` with `weights` and sum
        #   the result in the axis corresponding to characters, generating `attention`. Therefore,
        #   `attention` is a fixed-size representation for every batch element, independently on
        #   how many characters the corresponding input word had.
        # - Finally, concatenate `inputs` and `attention` (in this order), and call the `self._cell`
        #   on this concatenated input and the `states`, returning the result.
        projected_decoder_state = self._project_decoder_layer(states)  #
        projected = self._encoded_projected + projected_decoder_state.unsqueeze(1)  # Add broadcasting dimension
        weights = torch.softmax(self._output_layer(torch.tanh(projected)), dim=1)  #
        attention = torch.sum(self._encoded * weights, dim=1)  #
        cell_input = torch.cat([inputs, attention], dim=1)  #
        return self._cell(cell_input, states)  #


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        self._source_vocab = train.words.char_vocab
        self._target_vocab = train.lemmas.char_vocab

        # TODO(lemmatizer_noattn): Define
        # - `self._source_embedding` as an embedding layer of source characters into `args.cle_dim` dimensions
        # - `self._source_rnn` as a bidirectional GRU with `args.rnn_dim` units processing embedded source chars
        self._source_embedding = torch.nn.Embedding(len(self._source_vocab), args.cle_dim)  #
        self._source_rnn = torch.nn.GRU(args.cle_dim, args.rnn_dim, batch_first=True, bidirectional=True)  #

        # TODO: Define
        # - `self._target_rnn_cell` as a `WithAttention` with `attention_dim=args.rnn_dim`, employing as the
        #   underlying cell the `torch.nn.GRUCell` with `args.rnn_dim`. The cell will process concatenated
        #   target character embeddings and the result of the attention mechanism.
        gru_cell = torch.nn.GRUCell(args.cle_dim + args.rnn_dim, args.rnn_dim)  #
        self._target_rnn_cell = WithAttention(gru_cell, args.rnn_dim)  #

        # TODO(lemmatizer_noattn): Then define
        # - `self._target_output_layer` as a linear layer into as many outputs as there are unique target chars
        self._target_output_layer = torch.nn.Linear(args.rnn_dim, len(self._target_vocab))  #

        if not args.tie_embeddings:
            # TODO(lemmatizer_noattn): Define the `self._target_embedding` as an embedding layer of the target
            # characters into `args.cle_dim` dimensions.
            self._target_embedding = torch.nn.Embedding(len(self._target_vocab), args.cle_dim) #
        else:
            assert args.cle_dim == args.rnn_dim, "When tying embeddings, cle_dim and rnn_dim must match."
            # TODO(lemmatizer_noattn): Create a function `self._target_embedding` computing the embedding of given
            # target characters. When called, use `torch.nn.functional.embedding` to suitably
            # index the shared embedding matrix `self._target_output_layer.weight`
            # multiplied by the square root of `args.rnn_dim`.
            self._target_embedding = lambda inputs: (args.rnn_dim ** 0.5) * torch.nn.functional.embedding(  #
                inputs, self._target_output_layer.weight)  #

        self._show_results_every_batch = args.show_results_every_batch
        self._batches = 0

    def forward(self, words: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(words)
        if targets is not None:
            return self.decoder_training(encoded, targets)
        else:
            return self.decoder_prediction(encoded, max_length=words.shape[1] + 10)

    def encoder(self, words: torch.Tensor) -> torch.Tensor:
        # TODO(lemmatizer_noattn): Embed the inputs using `self._source_embedding`.
        embedded = self._source_embedding(words)  #

        # TODO: Run the `self._source_rnn` on the embedded sequences, correctly handling
        # padding. Newly, the result should be encoding of every sequence element,
        # summing results in the opposite directions.
        forms_len = torch.sum(words != MorphoDataset.PAD, dim=-1).cpu()  # Calculate lengths before padding
        # Pack sequence for RNN to ignore padding
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, forms_len, batch_first=True, enforce_sorted=False)  #
        # Run RNN
        hidden, _ = self._source_rnn(packed)  #
        # Unpack sequence
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)  #
        # Sum the outputs of the forward and backward directions
        # hidden shape: [batch_size, seq_len, 2 * rnn_dim]
        # Split into forward and backward states and sum them
        hidden = sum(torch.chunk(hidden, 2, dim=-1))  #
        # hidden shape: [batch_size, seq_len, rnn_dim]
        return hidden  #

    def decoder_training(self, encoded: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # TODO(lemmatizer_noattn): Generate inputs for the decoder, which are obtained from `targets` by
        # - prepending `MorphoDataset.BOW` as the first element of every batch example,
        # - dropping the last element of `targets`.
        decoder_inputs = torch.nn.functional.pad(targets[:, :-1], (1, 0), value=MorphoDataset.BOW)  #

        # TODO: Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn_cell` on the `encoded` input.
        self._target_rnn_cell.setup_memory(encoded)  #

        # TODO: Process the generated inputs by
        # - the `self._target_embedding` layer to obtain embeddings,
        # - repeatedly call the `self._target_rnn_cell` on the sequence of embedded
        #   inputs and the previous states, starting with state `encoded[:, 0]`,
        #   obtaining outputs for all target hidden states,
        # - the `self._target_output_layer` to obtain logits,
        # - finally, permute dimensions so that the logits are in the dimension 1,
        # and return the result.
        embedded_inputs = self._target_embedding(decoder_inputs)  #
        batch_size, seq_len, _ = embedded_inputs.shape  #
        # Use the first encoder hidden state as the initial decoder state. Note: For BiGRU, encoder output is summed, so encoded[:, 0] has dim rnn_dim.
        states = encoded[:, 0]  #
        outputs = []  #
        for i in range(seq_len):  #
            # Pass current input embedding and previous state to the attention cell
            states = self._target_rnn_cell(embedded_inputs[:, i], states)  #
            outputs.append(states)  #
        # Stack the outputs along the sequence dimension
        outputs_tensor = torch.stack(outputs, dim=1)  # shape: [batch_size, seq_len, rnn_dim]
        # Apply the output layer to get logits
        logits = self._target_output_layer(outputs_tensor)  # shape: [batch_size, seq_len, target_vocab_size]
        # Permute to [batch_size, target_vocab_size, seq_len] for CrossEntropyLoss
        logits = logits.permute(0, 2, 1)  #
        return logits  #


    def decoder_prediction(self, encoded: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = encoded.shape[0]

        # TODO(decoder_training): Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn_cell` on the `encoded` input.
        self._target_rnn_cell.setup_memory(encoded)  #

        # TODO: Define the following variables, that we will use in the cycle:
        # - `index`: the time index, initialized to 0;
        # - `inputs`: a tensor of shape `[batch_size]` containing the `MorphoDataset.BOW` symbols,
        # - `states`: initial RNN state from the encoder, i.e., `encoded[:, 0]`.
        # - `results`: an empty list, where generated outputs will be stored;
        # - `result_lengths`: a tensor of shape `[batch_size]` filled with `max_length`,
        index = 0
        inputs = torch.full([batch_size], MorphoDataset.BOW, dtype=torch.long, device=encoded.device) # Use torch.long for embedding lookup
        states = encoded[:, 0] #
        results = []
        result_lengths = torch.full([batch_size], max_length, dtype=torch.long, device=encoded.device) # Use torch.long


        while index < max_length and torch.any(result_lengths == max_length):
            # TODO(lemmatizer_noattn):
            # - First embed the `inputs` using the `self._target_embedding` layer.
            # - Then call `self._target_rnn_cell` using two arguments, the embedded `inputs`
            #   and the current `states`. The call returns a single tensor, which you should
            #   store as both a new `hidden` and a new `states`.
            # - Pass the outputs through the `self._target_output_layer`.
            # - Generate the most probable prediction for every batch example.
            embedded_input = self._target_embedding(inputs)  #
            # `_target_rnn_cell` returns the new hidden state
            hidden = states = self._target_rnn_cell(embedded_input, states)  #
            logits = self._target_output_layer(hidden)  #
            # Get the index of the highest probability prediction
            predictions = logits.argmax(dim=-1)  #

            # Store the predictions in the `results` and update the `result_lengths`
            # by setting it to current `index` if an EOW was generated for the first time.
            results.append(predictions)
            result_lengths[(predictions == MorphoDataset.EOW) & (result_lengths > index)] = index + 1

            # TODO(lemmatizer_noattn): Finally,
            # - set `inputs` to the `predictions`,
            # - increment the `index` by one.
            inputs = predictions  # Use current prediction as next input
            index += 1  #

        results = torch.stack(results, dim=1)
        return results

    def compute_metrics(self, y_pred, y, *xs):
        if self.training:  # In training regime, convert logits to most likely predictions.
            y_pred = y_pred.argmax(dim=-2)
        # Compare the lemmas with the predictions using exact match accuracy.
        y_pred = y_pred[:, :y.shape[-1]]
        y_pred = torch.nn.functional.pad(y_pred, (0, y.shape[-1] - y_pred.shape[-1]), value=MorphoDataset.PAD)
        self.metrics["accuracy"].update(torch.all((y_pred == y) | (y == MorphoDataset.PAD), dim=-1))
        return {name: metric.compute() for name, metric in self.metrics.items()}  # Return all metrics.

    def train_step(self, xs, y):
        result = super().train_step(xs, y)

        self._batches += 1
        if self._show_results_every_batch and self._batches % self._show_results_every_batch == 0:
            self.log_console("{}: {} -> {}".format(
                self._batches,
                "".join(self._source_vocab.strings(xs[0][0][xs[0][0] != MorphoDataset.PAD].numpy(force=True))),
                "".join(self._target_vocab.strings(self.predict_step((xs[0][:1],))[0]))))

        return result

    def test_step(self, xs, y):
        with torch.no_grad():
            y_pred = self.forward(*xs)
            return self.compute_metrics(y_pred, y, *xs)

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            batch = self.forward(*xs)
            # Trim the predictions at the first EOW
            batch = [lemma[(lemma == MorphoDataset.EOW).cumsum(-1) == 0] for lemma in batch]
            return [lemma.numpy(force=True) for lemma in batch] if as_numpy else batch


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: MorphoDataset.Dataset, training: bool) -> None:
        super().__init__(dataset)
        self._training = training

    def transform(self, example):
        # TODO(lemmatizer_noattn): Return `example["words"]` as inputs and `example["lemmas"]` as targets.
        return example["words"], example["lemmas"] #

    def collate(self, batch):
        # Construct a single batch, where `batch` is a list of examples generated by `transform`.
        words, lemmas = zip(*batch)
        # TODO(lemmatizer_noattn): The `words` are a list of list of strings. Flatten it into a single list of strings
        # and then map the characters to their indices using the `self.dataset.words.char_vocab` vocabulary.
        # Then create a tensor by padding the words to the length of the longest one in the batch.
        word_indices = [self.dataset.words.char_vocab.indices(form) for sentence in words for form in sentence]
        # Pad sequences and convert to tensor
        padded_words = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(indices) for indices in word_indices],
            batch_first=True,
            padding_value=MorphoDataset.PAD
        )
        # TODO(lemmatizer_noattn): Process `lemmas` analogously to `words`, but use `self.dataset.lemmas.char_vocab`,
        # and additionally, append `MorphoDataset.EOW` to the end of each lemma.
        lemmas = [self.dataset.lemmas.char_vocab.indices(lemma) + [MorphoDataset.EOW] for sentence in lemmas for lemma
                  in sentence]
        lemma_indices = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(lemma) for lemma in lemmas], batch_first=True)
        # Pad sequences and convert to tensor
        padded_lemmas = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(indices) for indices in lemma_indices],
            batch_first=True,
            padding_value=MorphoDataset.PAD
        )
        # TODO(lemmatizer_noattn): Return a pair (inputs, targets), where
        # - the inputs are words during inference and (words, lemmas) pair during training;
        # - the targets are lemmas.
        if self._training:
            # During training, decoder needs shifted lemmas as input
            # The forward method handles the shifting (prepending BOW, removing last)
            # So we pass the padded_lemmas (with EOW) which will be processed by decoder_training
            inputs = (padded_words, padded_lemmas)
        else:
            # During inference/testing, only words are needed as input
            inputs = padded_words

        targets = padded_lemmas  # Targets are always the padded lemmas (with EOW)
        return inputs, targets


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

    # Load the data.
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Prepare the data for training.
    train = TrainableDataset(morpho.train, training=True).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev, training=False).dataloader(batch_size=args.batch_size)

    # Create the model and train.
    model = Model(args, morpho.train)

    model.configure(
        # TODO(lemmatizer_noattn): Create the Adam optimizer.
        optimizer=torch.optim.Adam(model.parameters()), #
        # TODO(lemmatizer_noattn): Use the usual `torch.nn.CrossEntropyLoss` loss function. Additionally,
        # pass `ignore_index=morpho.PAD` to the constructor so that the padded
        # tags are ignored during the loss computation.
        loss=torch.nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD), #
        # TODO(lemmatizer_noattn): Create a `torchmetrics.MeanMetric()` metric, where we will manually
        # collect lemmatization accuracy.
        metrics={"accuracy": torchmetrics.MeanMetric()}, #
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return all metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items()}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
