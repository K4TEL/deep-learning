#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.8")
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--label_smoothing", default=0.0, type=float, help="Label smoothing.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=219, type=int, help="Random seed.")
parser.add_argument("--show_predictions", default=False, action="store_true", help="Show predicted tag sequences.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        self._show_predictions = args.show_predictions
        # TODO: Compute the transition matrix `A` of shape `[num_tags, num_tags]`, so
        # that `A[i, j]` is 0 or 1 depending on whether the tag `j` is allowed to follow
        # the tag `i` (according to our BIO encoding; not necessarily in the data).
        # The tag strings can be obtained by calling `list(train.tags.string_vocab)`.
        # Compute the transition matrix `A` for BIO constraints.
        num_tags = len(train.tags.string_vocab)
        tag_vocab_list = list(train.tags.string_vocab)
        tag_to_idx = {tag: idx for idx, tag in enumerate(tag_vocab_list)}
        A = torch.zeros(num_tags, num_tags, dtype=torch.float32)

        O_idx = tag_to_idx.get("O", -1)
        # Assuming PAD index is MorphoDataset.PAD (typically 0)
        PAD_idx = MorphoDataset.PAD

        for i, prev_tag_str in enumerate(tag_vocab_list):
            # Handle PAD transitions: Only PAD can follow PAD
            if i == PAD_idx:
                if PAD_idx >= 0:
                    A[i, PAD_idx] = 1
                continue

            for j, next_tag_str in enumerate(tag_vocab_list):
                # Nothing can transition to PAD except PAD itself
                if j == PAD_idx:
                    continue

                is_token_next = next_tag_str.startswith("[P")
                is_O_next = next_tag_str == "O"
                is_B_next = next_tag_str.startswith("B-")
                is_I_next = next_tag_str.startswith("I-")
                next_type = next_tag_str[2:] if (is_B_next or is_I_next) else None

                is_token_prev = prev_tag_str.startswith("[P")
                is_O_prev = prev_tag_str == "O"
                is_B_prev = prev_tag_str.startswith("B-")
                is_I_prev = prev_tag_str.startswith("I-")
                prev_type = prev_tag_str[2:] if (is_B_prev or is_I_prev) else None

                # Rule 1: O can follow O, B-TYPE, I-TYPE
                if is_O_next and (is_O_prev or is_B_prev or is_I_prev):
                    A[i, j] = 1

                # Rule 2: B-TYPE can follow O, B-TYPE2, I-TYPE2
                elif is_B_next and (is_O_prev or is_B_prev or is_I_prev):
                    A[i, j] = 1

                # Rule 3: I-TYPE can follow B-TYPE or I-TYPE (same type)
                elif is_I_next:
                    if (is_B_prev and prev_type == next_type) or \
                            (is_I_prev and prev_type == next_type):
                        A[i, j] = 1

                # Rule 2: [PAD] can follow O, B-TYPE2, I-TYPE2
                elif is_token_next and (is_O_prev or is_B_prev or is_I_prev):
                    A[i, j] = 1

                # Rule 2: [PAD] can follow O, B-TYPE2, I-TYPE2
                elif is_token_prev and (is_O_next or is_B_next or is_I_next):
                    A[i, j] = 0

        # Store log transitions for Viterbi, adding epsilon for numerical stability
        self.register_buffer("_log_A", torch.log(A + 1e-9))

        # Constraint for start: Cannot start with I-
        # We simulate this by assuming a virtual O tag before the sequence starts.
        # The allowed first tags are those that can follow O.
        start_constraint = torch.full((num_tags,), -float('inf'), dtype=torch.float32)
        if O_idx != -1:
            # Get allowed transitions from O, convert back to log space
            # Ensure O_idx is valid before indexing
            if O_idx < self._log_A.shape[0]:
                start_constraint = self._log_A[O_idx, :].clone()
            # Ensure PAD cannot be the first tag unless the sequence is empty (handled later)
            if PAD_idx >= 0 and PAD_idx < num_tags:
                start_constraint[PAD_idx] = -float('inf')
        else:  # Fallback if O tag doesn't exist (unlikely for NER)
            for j, tag_str in enumerate(tag_vocab_list):
                if not tag_str.startswith("I-") and j != PAD_idx:
                    start_constraint[j] = 0  # log(1) = 0
        self.register_buffer("_log_start_constraint", start_constraint)

        # Create all needed layers.
        # TODO(tagger_we): Create a `torch.nn.Embedding` layer, embedding the word ids
        # from `train.words.string_vocab` to dimensionality `args.we_dim`.
        self._word_embedding = torch.nn.Embedding(len(train.words.string_vocab), args.we_dim)

        # TODO(tagger_we): Create an RNN layer, either `torch.nn.LSTM` or `torch.nn.GRU` depending
        # on `args.rnn`. The layer should be bidirectional (`bidirectional=True`) with
        # dimensionality `args.rnn_dim`. During the model computation, the layer will
        # process the word embeddings generated by the `self._word_embedding` layer,
        # and we will sum the outputs of forward and backward directions.
        self._word_rnn = getattr(torch.nn, args.rnn)(
            input_size=args.we_dim,
            hidden_size=args.rnn_dim,
            batch_first=True,
            bidirectional=True
        )

        # TODO(tagger_we): Create an output linear layer (`torch.nn.Linear`) processing the RNN output,
        # producing logits for tag prediction; `train.tags.string_vocab` is the tag vocabulary.
        self._output_layer = torch.nn.Linear(args.rnn_dim, len(train.tags.string_vocab))

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        # TODO(tagger_we): Start by embedding the `word_ids` using the word embedding layer.
        hidden = self._word_embedding(word_ids)

        # TODO(tagger_we): Process the embedded words through the RNN layer. Because the sentences
        # have different length, you have to use `torch.nn.utils.rnn.pack_padded_sequence`
        # to construct a variable-length `PackedSequence` from the input. You need to compute
        # the length of each sentence in the batch (by counting non-`MorphoDataset.PAD` tokens);
        # note that these lengths must be on CPU, so you might need to use the `.cpu()` method.
        # Finally, also pass `batch_first=True` and `enforce_sorted=False` to the call.
        word_lengths = torch.sum(word_ids != MorphoDataset.PAD, dim=-1).cpu()

        valid_lengths = word_lengths[word_lengths > 0]
        if len(valid_lengths) == 0:  # All sequences have length 0
            # Return zero logits matching the expected shape
            batch_size, seq_len = word_ids.shape
            num_tags = self._output_layer.out_features
            return torch.zeros(batch_size, num_tags, seq_len, device=word_ids.device)

        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden, word_lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self._word_rnn(packed)

        # TODO(tagger_we): Pass the `PackedSequence` through the RNN, choosing the appropriate output.
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)

        # TODO(tagger_we): Unpack the RNN output using the `torch.nn.utils.rnn.pad_packed_sequence` with
        # `batch_first=True` argument. Then sum the outputs of forward and backward directions.
        forward, backward = torch.chunk(hidden, 2, dim=-1)
        hidden = forward + backward

        # TODO(tagger_we): Pass the RNN output through the output layer. Such an output has a shape
        # `[batch_size, sequence_length, num_tags]`, but the loss and the metric expect
        # the `num_tags` dimension to be in front (`[batch_size, num_tags, sequence_length]`),
        # so you need to reorder the dimensions.
        hidden = self._output_layer(hidden).permute(0, 2, 1)

        return hidden

    def constrained_decoding(self, logits: torch.Tensor, word_ids: torch.Tensor) -> torch.Tensor:
        # TODO: Perform constrained decoding, i.e., produce the most likely BIO-encoded
        # valid sequence. In a valid sequence, all tags are `O`, `B-TYPE`, `I-TYPE`, and
        # the `I-TYPE` tag must follow either `B-TYPE` or `I-TYPE` tag. This correctness
        # can be implemented by checking that every neighboring pair of tags is valid
        # according to the transition matrix `self._A`, plus the sequence cannot start
        # with an "I-" tag -- a possible solution is to consider a tag sequence to be
        # prefixed by a virtual "O" tag during decoding. Finally, the tags for padding
        # tokens must be `MorphoDataset.PAD`s.
        batch_size, num_tags, seq_len = logits.shape
        device = logits.device
        pad_idx = MorphoDataset.PAD

        # Use log probabilities for numerical stability
        log_probs = torch.log_softmax(logits, dim=1)
        lengths = torch.sum(word_ids != pad_idx, dim=1)

        # Initialize Viterbi scores and backpointers
        scores = torch.full((batch_size, seq_len, num_tags), -float('inf'), device=device)
        backpointers = torch.zeros((batch_size, seq_len, num_tags), dtype=torch.long, device=device)

        # Handle sequences with length 0 explicitly
        if seq_len == 0:
            return torch.zeros((batch_size, 0), dtype=torch.long, device=device)

        # Initialization step (t=0)
        # Ensure start constraints are applied correctly even for batches with 0-length sequences
        mask_len_gt_0 = (lengths > 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        initial_log_probs = log_probs[:, :, 0]  # [B, N]
        initial_scores = initial_log_probs + self._log_start_constraint.unsqueeze(0)  # [B, N]
        scores[:, 0, :] = torch.where(mask_len_gt_0.squeeze(-1), initial_scores,
                                      -float('inf'))  # Apply only where length > 0

        # Recursion step (t=1 to seq_len-1)
        for t in range(1, seq_len):
            # score(t, j) = emission(t, j) + max_i( score(t-1, i) + transition(i, j) )
            prev_scores = scores[:, t - 1, :].unsqueeze(2)  # [B, N_prev, 1]
            transitions = self._log_A.unsqueeze(0)  # [1, N_prev, N_curr] -> broadcasts
            score_candidates = prev_scores + transitions  # [B, N_prev, N_curr]
            best_prev_scores, backpointers[:, t, :] = torch.max(score_candidates, dim=1)  # [B, N_curr], [B, N_curr]
            current_scores = log_probs[:, :, t] + best_prev_scores  # [B, N_curr]

            # Only update scores for sequences that are long enough for this timestep
            mask_len_gt_t = (lengths > t).unsqueeze(1)  # [B, 1]
            scores[:, t, :] = torch.where(mask_len_gt_t, current_scores, -float('inf'))

        # Backtracking
        best_paths = torch.full((batch_size, seq_len), pad_idx, dtype=torch.long, device=device)  # Initialize with PAD

        for b in range(batch_size):
            length = lengths[b].item()
            if length == 0:
                continue

            last_valid_idx = length - 1
            # Find the best tag for the last *actual* token
            best_paths[b, last_valid_idx] = torch.argmax(scores[b, last_valid_idx, :])

            # Backtrack from the last actual token
            for t in range(last_valid_idx, 0, -1):
                current_tag = best_paths[b, t]
                # Check bounds before indexing backpointers
                if current_tag < num_tags:
                    best_paths[b, t - 1] = backpointers[b, t, current_tag]
                else:
                    # This should not happen if logic is correct, but as a safeguard:
                    best_paths[b, t - 1] = pad_idx

        # The path is already initialized with PAD, so masking isn't strictly needed again
        # but ensures correctness if initialization logic changes.
        # mask = (word_ids != pad_idx)
        # best_paths.masked_fill_(~mask, pad_idx)

        return best_paths


    def compute_metrics(self, y_pred, y, word_ids):
        self.metrics["accuracy"].update(y_pred, y)
        if self.training:
            return {"accuracy": self.metrics["accuracy"].compute()}

        # Perform greedy decoding.
        predictions_greedy = y_pred.argmax(dim=1)
        predictions_greedy.masked_fill_(word_ids == MorphoDataset.PAD, MorphoDataset.PAD)
        self.metrics["f1_greedy"].update(predictions_greedy, y)

        # TODO: Perform constrained decoding by calling `self.constrained_decoding`
        # on `y_pred` and `word_ids`.
        predictions = self.constrained_decoding(y_pred, word_ids)
        self.metrics["f1_constrained"].update(predictions, y)

        if self._show_predictions:
            # Get labels from the metric object
            labels = getattr(self.metrics["f1_constrained"], '_labels', None)
            if labels:
                print("--- Predictions (Constrained) ---")
                for i in range(predictions.shape[0]):
                    length = torch.sum(word_ids[i] != MorphoDataset.PAD).item()
                    pred_tags = predictions[i][:length].cpu().numpy()
                    print(*[labels[tag] for tag in pred_tags])
                print("---------------------------------")
            else:
                print("Labels not found for prediction display.")

        return {name: metric.compute() for name, metric in self.metrics.items()}

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            # Perform constrained decoding.
            batch = self.constrained_decoding(self.forward(*xs), *xs)
            if as_numpy:
                batch = [example.numpy(force=True) for example in batch]
            # Trim the padding tags from the predictions.
            batch = [example[example != MorphoDataset.PAD] for example in batch]
            return batch


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO(tagger_we): Construct a single example, each consisting of the following pair:
        # - a PyTorch tensor of integer ids of input words as input,
        # - a PyTorch tensor of integer tag ids as targets.
        # To create the ids, use `string_vocab` of `self.dataset.words` and `self.dataset.tags`.
        word_ids = torch.tensor(self.dataset.words.string_vocab.indices(example["words"]))
        tag_ids = torch.tensor(self.dataset.tags.string_vocab.indices(example["tags"]))
        return word_ids, tag_ids

    def collate(self, batch):
        # Construct a single batch, where `batch` is a list of examples
        # generated by `transform`.
        word_ids, tag_ids = zip(*batch)
        # TODO(tagger_we): Combine `word_ids` into a single tensor, padding shorter
        # sequences to length of the longest sequence in the batch with zeros
        # using `torch.nn.utils.rnn.pad_sequence` with `batch_first=True` argument.
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        # TODO(tagger_we): Process `tag_ids` analogously to `word_ids`.
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        return word_ids, tag_ids


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
    morpho = MorphoDataset("czech_cnec", max_sentences=args.max_sentences)

    # Prepare the data for training.
    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)

    # Create the model and train.
    model = Model(args, morpho.train)

    model.configure(
        # TODO(tagger_we): Create the Adam optimizer.
        optimizer=torch.optim.Adam(model.parameters()),
        # TODO: Use `torch.nn.CrossEntropyLoss` to instantiate the loss function.
        # Pass `ignore_index=morpho.PAD` to the constructor to ignore padding tags
        # during loss computation; also pass `label_smoothing=args.label_smoothing`.
        loss=torch.nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD, label_smoothing=args.label_smoothing),
        metrics={
            # TODO(tagger_we): Create a `torchmetrics.Accuracy` metric, passing "multiclass" as
            # the first argument, `num_classes` set to the number of unique tags, and
            # again `ignore_index=morpho.PAD` to ignore the padded tags.
            "accuracy": torchmetrics.Accuracy("multiclass", num_classes=len(morpho.train.tags.string_vocab), ignore_index=MorphoDataset.PAD),
            # TODO: Create a `npfl138.metrics.BIOEncodingF1Score` for constrained decoding and also
            # for greedy decoding, passing both a `list(morpho.train.tags.string_vocab)`
            # and `ignore_index=morpho.PAD`.
            "f1_constrained": npfl138.metrics.BIOEncodingF1Score(list(morpho.train.tags.string_vocab), ignore_index=MorphoDataset.PAD),
            "f1_greedy": npfl138.metrics.BIOEncodingF1Score(list(morpho.train.tags.string_vocab), ignore_index=MorphoDataset.PAD),
        },
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
