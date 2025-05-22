#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.7.2")
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer
from npfl138.trainable_module import TrainableModule

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--cle_dim", default=256, type=int, help="CLE embedding dimension.") # Increased CLE dim
parser.add_argument("--rnn", default="GRU", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=512, type=int, help="RNN layer dimension.") # Increased RNN dim
parser.add_argument("--we_dim", default=256, type=int, help="Word embedding dimension.") # Increased WE dim
parser.add_argument("--word_masking", default=0.2, type=float, help="Mask words with the given probability.") # Added slight masking

parser.add_argument("--learning_rate", default=1e-2, type=float, help="Learning rate")
parser.add_argument("--learning_rate_final", default=1e-18, type=float, help="Final learning rate")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")


# --- Model Definition (Adapted from tagger_cle.py) ---
class Model(TrainableModule):
    class MaskElements(torch.nn.Module):
        """A layer randomly masking elements with a given value."""
        def __init__(self, mask_probability, mask_value):
            super().__init__()
            self._mask_probability = mask_probability
            self._mask_value = mask_value

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if self.training and self._mask_probability:
                mask = torch.rand_like(inputs, dtype=torch.float32, device=inputs.device) < self._mask_probability
                # Use clone() to avoid modifying the original tensor if it's needed elsewhere
                inputs = inputs.clone().masked_fill(mask, self._mask_value)
            return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset, morpho_analyzer: MorphoAnalyzer | None = None) -> None:
        super().__init__()
        self._args = args
        self._morpho_analyzer = morpho_analyzer # Store analyzer if provided

        # Word Masking
        self._word_masking = self.MaskElements(args.word_masking, MorphoDataset.UNK) #

        # Character Embeddings (CLE)
        self._char_embedding = torch.nn.Embedding(len(train.words.char_vocab), args.cle_dim, padding_idx=MorphoDataset.PAD) #
        self._char_rnn = torch.nn.GRU(args.cle_dim, args.cle_dim, batch_first=True, bidirectional=True) #

        # Word Embeddings (WE)
        self._word_embedding = torch.nn.Embedding(len(train.words.string_vocab), args.we_dim, padding_idx=MorphoDataset.PAD) #

        # TODO: Optionally incorporate MorphoAnalyzer features here
        # Example: Add an embedding layer for known lemmas/tags from the analyzer.
        # This requires modifying the TrainableDataset and collate function as well.
        # For simplicity, this example does not integrate the analyzer features into the model directly,
        # but it could be a valuable extension.
        analyzer_feature_dim = 0 # Placeholder - adjust if using analyzer features

        # Main RNN Layer
        rnn_input_dim = args.we_dim + 2 * args.cle_dim + analyzer_feature_dim # Input includes WE, CLE, and optional analyzer features
        self._word_rnn = getattr(torch.nn, args.rnn)(
            rnn_input_dim, args.rnn_dim, batch_first=True, bidirectional=True
        ) #

        # Output Layer
        # The output dimension must match the number of unique tags in the training set
        self._output_layer = torch.nn.Linear(2 * args.rnn_dim, len(train.tags.string_vocab)) # Bidirectional RNN output is summed or concatenated; here we assume concatenation (adjust if summing)

    def forward(self, word_ids: torch.Tensor, unique_words: torch.Tensor, word_indices: torch.Tensor, # Add other inputs if needed (e.g., analyzer features)
                ) -> torch.Tensor:
        # --- Character-Level Embeddings ---
        char_embeddings = self._char_embedding(unique_words) # Shape: [num_unique_words, max_word_length, cle_dim]
        unique_words_len = torch.sum(unique_words != MorphoDataset.PAD, dim=-1).cpu().clamp(min=1) #
        packed_chars = torch.nn.utils.rnn.pack_padded_sequence(char_embeddings, unique_words_len, batch_first=True, enforce_sorted=False) #
        _, hidden = self._char_rnn(packed_chars) #
        # hidden shape is [num_layers * num_directions, batch_size (num_unique_words), hidden_dim (cle_dim)]
        # Concatenate the last hidden states of forward and backward directions
        # Hidden[0] is forward layer 1, Hidden[1] is backward layer 1 for GRU
        char_level_embeddings = torch.cat((hidden[0], hidden[1]), dim=-1) # Shape: [num_unique_words, 2 * cle_dim]

        # Map unique word CLEs back to sentence structure
        # Use PAD index 0 for embedding lookup, assuming PAD word_index corresponds to PAD unique_word
        sentence_char_embeddings = torch.nn.functional.embedding(word_indices, char_level_embeddings, padding_idx=0) # Shape: [batch_size, max_sentence_length, 2 * cle_dim]


        # --- Word-Level Embeddings ---
        masked_word_ids = self._word_masking(word_ids) #
        word_embeddings = self._word_embedding(masked_word_ids) # Shape: [batch_size, max_sentence_length, we_dim]

        # --- Combine Embeddings ---
        # TODO: Concatenate analyzer features if used
        combined_embeddings = torch.cat([word_embeddings, sentence_char_embeddings], dim=-1) # Shape: [batch_size, max_sentence_length, we_dim + 2*cle_dim + analyzer_feature_dim]

        # --- Word-Level RNN ---
        sentence_lens = torch.sum(word_ids != MorphoDataset.PAD, dim=-1).cpu().clamp(min=1) #
        packed = torch.nn.utils.rnn.pack_padded_sequence(combined_embeddings, sentence_lens, batch_first=True, enforce_sorted=False) #
        packed_output, _ = self._word_rnn(packed) #
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) # Shape: [batch_size, max_sentence_length, 2 * rnn_dim]

        # --- Output Layer ---
        # Option 1: Sum forward and backward outputs (as in tagger_cle) - requires adjusting Linear layer input dim
        # forward_out, backward_out = torch.chunk(output, 2, dim=-1)
        # summed_output = forward_out + backward_out # Shape: [batch_size, max_sentence_length, rnn_dim]
        # logits = self._output_layer(summed_output) # Shape: [batch_size, max_sentence_length, num_tags]

        # Option 2: Use concatenated output (more common)
        logits = self._output_layer(output) # Shape: [batch_size, max_sentence_length, num_tags]

        # Permute for CrossEntropyLoss: [batch_size, num_tags, max_sentence_length]
        logits = logits.permute(0, 2, 1) #

        return logits


# --- Dataset Preparation (Adapted from tagger_cle.py) ---
class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: MorphoDataset.Dataset, morpho_analyzer: MorphoAnalyzer | None = None):
        super().__init__(dataset) # Pass the original MorphoDataset.Dataset
        self._morpho_analyzer = morpho_analyzer # Store analyzer if provided

    def transform(self, example: MorphoDataset.Element): #
        # Convert words and tags to IDs using the vocabulary from the original dataset
        word_ids = torch.tensor(self.dataset.words.string_vocab.indices(example["words"]), dtype=torch.long) #
        tag_ids = torch.tensor(self.dataset.tags.string_vocab.indices(example["tags"]), dtype=torch.long) #

        # TODO: Extract features from MorphoAnalyzer if used
        # Example: Get possible tags/lemmas for each word
        analyzer_features = []
        if self._morpho_analyzer:
            for word in example["words"]:
                analyses = self._morpho_analyzer.get(word)
                # Process analyses into features (e.g., count, presence of specific tag types)
                # This needs careful design based on how you want to use the features
                analyzer_features.append(...) # Replace with actual feature extraction

        # Return word IDs, original words (for CLE), tag IDs, and optional analyzer features
        # Ensure the collate function handles any additional returned items
        return word_ids, example["words"], tag_ids # Add analyzer_features if used

    def collate(self, batch):
        # Adjust based on what `transform` returns
        word_ids, words, tag_ids = zip(*batch) # Unpack batch; add analyzer_features if used

        # Pad word IDs
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True, padding_value=MorphoDataset.PAD) #

        # Get CLE inputs
        unique_words, words_indices = self.dataset.cle_batch(words) #
        words_indices = words_indices.long() # Ensure indices are long

        # Pad tag IDs
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True, padding_value=MorphoDataset.PAD) #

        # TODO: Collate analyzer features if used
        # Example: Pad or combine analyzer features for the batch

        # Return tuple: (inputs), targets
        # Inputs must match the Model.forward signature
        inputs = (word_ids, unique_words, words_indices) # Add collated analyzer_features if used
        return inputs, tag_ids


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

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # Prepare datasets for training
    print("Preparing datasets...", flush=True)
    # Pass analyzer to dataset if used
    train_dataset = TrainableDataset(morpho.train, analyses)
    dev_dataset = TrainableDataset(morpho.dev, analyses)
    test_dataset = TrainableDataset(morpho.test, analyses)

    train_loader = train_dataset.dataloader(batch_size=args.batch_size, shuffle=True)
    dev_loader = dev_dataset.dataloader(batch_size=args.batch_size)
    # Use a larger batch size for prediction if memory allows
    test_loader = test_dataset.dataloader(batch_size=args.batch_size * 2)

    # TODO: Create the model and train it.
    model = Model(args, morpho.train, analyses)

    num_batches = len(train_loader) * args.epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters())

    # Configure the model for training
    model.configure(
        optimizer=optimizer,  #
        loss=torch.nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD),  #
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_batches, eta_min=args.learning_rate_final),
        metrics={"accuracy": torchmetrics.Accuracy(
            task="multiclass",
            num_classes=len(morpho.train.tags.string_vocab),
            ignore_index=MorphoDataset.PAD)},  #
        logdir=args.logdir,
    )

    # Train the model
    print("Starting training...", flush=True)
    logs = model.fit(train_loader, dev=dev_loader, epochs=args.epochs)  #
    print("Training finished.", flush=True)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set. The following code assumes you use the same
        # output structure as in `tagger_we`, i.e., that for each sentence, the predictions are
        # a Numpy vector of shape `[num_tags, sentence_len_or_more]`, where `sentence_len_or_more`
        # is the length of the corresponding batch. (FYI, if you instead used the `packed` variant,
        # the prediction for each sentence is a vector of shape `[exactly_sentence_len, num_tags]`.)
        predictions = model.predict(test_loader, data_with_labels=True, as_numpy=False)

        # for predicted_tags, words in zip(predictions, morpho.test.words.strings):
        #     for predicted_tag in predicted_tags[:, :len(words)].argmax(axis=0):
        #         print(morpho.train.tags.string_vocab.string(predicted_tag), file=predictions_file)
        #     print(file=predictions_file)

        # Process predictions batch by batch
        example_index = 0
        for batch_input, _ in test_loader:
            # Get the predictions corresponding to this batch
            batch_predictions = predictions[example_index: example_index + len(batch_input[0])]
            # Get original word sequences for length information (using the test MorphoDataset.Dataset)
            original_words = [morpho.test[i]["words"] for i in
                              range(example_index, example_index + len(batch_input[0]))]

            for i, prediction_tensor in enumerate(batch_predictions):
                # prediction_tensor shape: [num_tags, sequence_length] (after model's permute)
                sentence_len = len(original_words[i])
                # Get tag indices for the actual length of the sentence
                predicted_indices = torch.argmax(prediction_tensor[:, :sentence_len],
                                                 dim=0)  # Get index of max logit for each position
                # Convert indices back to tag strings
                for tag_index in predicted_indices.cpu().numpy():  # Move to CPU and convert to numpy for iteration
                    print(morpho.train.tags.string_vocab.string(tag_index), file=predictions_file)
                print(file=predictions_file)  # Newline after each sentence

            example_index += len(batch_input[0])


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
