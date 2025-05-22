#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy

import torch
import torchaudio.models.decoder

import npfl138
from torch.utils.data._utils.collate import default_collate

npfl138.require_version("2425.12")
from npfl138.datasets.homr_dataset import HOMRDataset
from npfl138 import TransformedDataset, TrainableModule
import torchvision

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate.")  # Adjusted default
parser.add_argument("--rnn_dim", default=512, type=int, help="RNN dimension.")  # Adjusted default
parser.add_argument("--rnn_layers", default=3, type=int, help="Number of RNN layers.")  # Added from reference
parser.add_argument("--cnn_filters", default=[16, 32, 64], type=int, nargs="+",
                    help="CNN filters per layer.")  # Adjusted default
parser.add_argument("--cosine_decay", default=True, action="store_true", help="Use cosine learning rate decay.")
parser.add_argument("--ctc_beam_size", default=10, type=int, help="Beam size for CTC decoder.")  # Added for CTC Decoder
parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay for optimizer.")


class HOMRModel(TrainableModule):
    def __init__(self, args: argparse.Namespace, num_classes: int, mark_names: list[str]):
        """Initialize the HOMR model utilizing TrainableModule as base.

        Args:
            args: Command line arguments
            num_classes: Number of output classes
            mark_names: List of mark names, where mark_names[0] is the blank token.
        """
        super().__init__()
        self.args = args

        # CNN layers
        cnn_layers = []
        in_channels = 1  # Grayscale images

        # Add CNN blocks based on the specified filters
        for filters in args.cnn_filters:
            cnn_layers.extend([
                torch.nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(filters),  # Added BatchNorm
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout2d(args.dropout)
            ])
            in_channels = filters
        self.cnn = torch.nn.Sequential(*cnn_layers)

        # Calculate flattened CNN output size for RNN
        # Assuming input image height is 128 (as per HOMRDatasetTransform)
        # Each MaxPool2d halves the height dimension.
        cnn_output_height = 128 // (2 ** len(args.cnn_filters))
        self.rnn_input_size = args.cnn_filters[-1] * cnn_output_height

        # RNN layers (Bidirectional LSTM)
        self.rnn = torch.nn.LSTM(
            self.rnn_input_size,
            args.rnn_dim,
            num_layers=args.rnn_layers,  # Using args.rnn_layers
            batch_first=True,
            bidirectional=True,
            dropout=args.dropout if args.rnn_layers > 1 else 0  # Dropout only between LSTM layers
        )

        # Output projection
        self.fc = torch.nn.Linear(args.rnn_dim * 2, num_classes)  # *2 for bidirectional

        # CTC Decoder
        # Assuming mark_names[0] (e.g., "[PAD]") is the blank token, consistent with CTCLoss(blank=0)
        if torch.cuda.is_available():

            self._ctc_decoder = torchaudio.models.decoder.cuda_ctc_decoder(

                mark_names, beam_size=args.ctc_beam_size

            )

        else:
            self._ctc_decoder = torchaudio.models.decoder.ctc_decoder(
                lexicon=None,  # No lexicon needed for character/mark-level recognition
                tokens=mark_names,
                blank_token=mark_names[0],  # Crucial: must match blank index in CTCLoss
                sil_token=mark_names[0],  # Using blank as SIL, or choose another non-data token
                nbest=1,
                beam_size=args.ctc_beam_size,
                log_add=True  # Typically better for numerical stability
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            images: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Output logits of shape [batch_size, sequence_length, num_classes]
        """
        batch_size, _, _, _ = images.size()

        # Pass through CNN
        # Input: (B, C_in, H, W) e.g. (B, 1, 128, W_img)
        cnn_out = self.cnn(images)
        # Output: (B, C_out, H_out, W_out) e.g. (B, 64, 128/(2^N_cnn), W_img/(2^N_cnn))

        # Reshape for RNN:
        # Current shape: (batch_size, channels_cnn, height_cnn, width_cnn)
        # Desired shape for RNN (batch_first=True): (batch_size, sequence_length, features)
        # Here, sequence_length is width_cnn, and features are channels_cnn * height_cnn

        # (B, C_out, H_out, W_out) -> (B, W_out, C_out, H_out)
        rnn_input = cnn_out.permute(0, 3, 1, 2)
        # (B, W_out, C_out, H_out) -> (B, W_out, C_out * H_out)
        rnn_input = rnn_input.contiguous().view(batch_size, rnn_input.size(1), -1)
        # rnn_input shape: (batch_size, sequence_length_after_cnn, features_for_rnn)

        # Pass through RNN
        # rnn_input: (batch_size, seq_len, rnn_input_size)
        rnn_out, _ = self.rnn(rnn_input)
        # rnn_out: (batch_size, seq_len, rnn_dim * 2)

        # Pass through output layer
        # logits: (batch_size, seq_len, num_classes)
        logits = self.fc(rnn_out)

        return logits

    def _get_input_lengths(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculates the input lengths for CTC loss/decoding based on logits shape.
        Assumes that the sequence dimension of logits (dim 1) is the unpadded sequence length.
        """
        batch_size = logits.size(0)
        sequence_length = logits.size(1)  # This is W_out from CNN
        return torch.full((batch_size,), sequence_length, dtype=torch.int32, device=logits.device)

    def compute_loss(self, y_pred_logits: torch.Tensor, y_true_targets: torch.Tensor,
                     images: torch.Tensor) -> torch.Tensor:
        """Compute CTC loss.

        Args:
            y_pred_logits: Model predictions (raw logits) of shape [batch_size, seq_len, num_classes]
            y_true_targets: Target labels of shape [batch_size, max_target_len]
            images: Input tensor (not directly used for loss calculation here, but part of TrainableModule API)

        Returns:
            CTC loss value
        """
        # Logits to log_probabilities
        # y_pred_log_probs: (batch_size, seq_len, num_classes)
        y_pred_log_probs = torch.nn.functional.log_softmax(y_pred_logits, dim=2)

        # Permute for CTCLoss: (seq_len, batch_size, num_classes)
        y_pred_for_ctc = y_pred_log_probs.permute(1, 0, 2)

        # Get input lengths (sequence lengths from the model's perspective)
        # This is the width of the CNN output, which is y_pred_logits.size(1)
        input_lengths = self._get_input_lengths(y_pred_logits)

        # Get target lengths (actual lengths of the target sequences, excluding padding)
        # Assuming 0 is the padding token for targets
        target_lengths = torch.sum(y_true_targets != 0, dim=1).to(device=y_true_targets.device)

        # Filter out targets that have length 0 (if any, though unlikely for this dataset)
        # CTCLoss requires target_lengths > 0
        valid_indices = target_lengths > 0
        if not valid_indices.all():
            # This part is a safeguard. Ideally, all target_lengths should be > 0.
            # If some are 0, CTCLoss will error. We filter them out.
            y_pred_for_ctc = y_pred_for_ctc[:, valid_indices, :]
            y_true_targets = y_true_targets[valid_indices]
            input_lengths = input_lengths[valid_indices]
            target_lengths = target_lengths[valid_indices]
            if y_true_targets.numel() == 0:  # No valid targets left
                return torch.tensor(0.0, device=self.device, requires_grad=True)

        return self.loss(y_pred_for_ctc, y_true_targets, input_lengths, target_lengths)

    def ctc_decoding(self, logits: torch.Tensor, input_lengths: torch.Tensor) -> list[torch.Tensor]:
        """Performs CTC decoding on the model's output logits.

        Args:
            logits: Raw output logits from the model (batch_size, seq_len, num_classes).
            input_lengths: Tensor of sequence lengths for each item in the batch (batch_size,).

        Returns:
            A list of tensors, where each tensor contains the decoded token indices for one batch item.
        """
        # Apply log_softmax if decoder expects log_probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        # Decoder expects emissions: (batch, time, num_tokens)
        hypotheses = self._ctc_decoder(log_probs, input_lengths)

        # Extract token indices from hypotheses (top hypothesis)
        # hypotheses is a list of lists of Hypos. We take the first (and only if nbest=1) Hypo's tokens.
        decoded_sequences = []
        for i in range(log_probs.size(0)):  # Iterate over batch
            if hypotheses[i]:  # Check if hypotheses were generated for this item
                # print(hypotheses[i][0].tokens)
                decoded_sequences.append(torch.Tensor(hypotheses[i][0].tokens).to(device=logits.device))
            else:  # Handle cases where decoder might return empty for very short/problematic inputs
                decoded_sequences.append(torch.tensor([], dtype=torch.long, device=logits.device))
        return decoded_sequences

    def compute_metrics(self, y_pred_logits: torch.Tensor, y_true_targets: torch.Tensor, images: torch.Tensor) -> dict[
        str, torch.Tensor]:
        """Compute metrics, typically using CTC decoding for predictions.

        Args:
            y_pred_logits: Model predictions (raw logits)
            y_true_targets: Target labels
            images: Input tensor

        Returns:
            Dictionary of metrics
        """
        input_lengths = self._get_input_lengths(y_pred_logits)

        # Perform CTC decoding only if not training, or if explicitly desired for training metrics
        # For validation/testing, always use the proper decoding.
        # For training, one might use greedy decoding for speed, but beam search is better.
        # The reference model computes full decoding even for training metrics.

        # Use ctc_decoding to get predicted sequences
        # Note: self.training is a bool indicating if the model is in training mode
        if not self.training or self.args.ctc_beam_size > 1:  # Use beam search for eval or if beam > 1
            decoded_predictions = self.ctc_decoding(y_pred_logits, input_lengths)
        else:  # Use greedy decoding for training if beam_size is 1 (or for speed)
            # Greedy decoding:
            pred_indices = y_pred_logits.argmax(dim=2)  # (batch, seq_len)
            # Basic de-duplication and blank removal for greedy (simplified CTC decode)
            decoded_predictions = []
            for i in range(pred_indices.size(0)):
                seq = pred_indices[i]
                length = input_lengths[i]
                # Filter blanks (assuming blank is 0) and remove consecutive duplicates
                last_char = -1
                greedy_decoded_seq = []
                for k in range(length):  # Only up to actual sequence length
                    char_index = seq[k].item()
                    if char_index != 0 and char_index != last_char:  # Not blank and not same as last
                        greedy_decoded_seq.append(char_index)
                    if char_index != 0:  # Update last_char only if not blank
                        last_char = char_index
                    elif char_index == 0:  # Reset last_char if blank encountered
                        last_char = -1
                decoded_predictions.append(
                    torch.tensor(greedy_decoded_seq, dtype=torch.long, device=y_pred_logits.device))

        results = {}
        for name, metric in self.metrics.items():
            # Ensure metric is on the same device
            metric.to(y_pred_logits.device)
            # metric.update expects list of tensors (predictions) and tensor (targets)
            metric.update(decoded_predictions, y_true_targets)
            results[name] = metric.compute()
        return results

    def predict_step(self, batch_images: tuple[torch.Tensor], as_numpy: bool = True):
        """Perform prediction on a batch using CTC decoding.

        Args:
            batch_images: A tuple containing the input image batch.
            as_numpy: Whether to convert tensors to numpy arrays.

        Returns:
            Predictions (list of sequences)
        """
        images_tensor = batch_images[0]  # Unpack if it's a tuple
        self.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():
            # Forward pass to get logits
            logits = self.forward(images_tensor)

            # Get input lengths for the decoder
            input_lengths = self._get_input_lengths(logits)

            # Perform CTC decoding
            decoded_batch = self.ctc_decoding(logits, input_lengths)

            if as_numpy:
                return [seq.cpu().numpy() for seq in decoded_batch]
            return decoded_batch


class HOMRDatasetTransform(TransformedDataset):
    """Dataset transformation for HOMR dataset."""

    def __init__(self, dataset, target_height=128):
        super().__init__(dataset)
        self.target_height = target_height

    def transform(self, item):
        """Transform a single dataset item.
        Args:
            item: Dictionary with 'image' and 'marks' fields
        Returns:
            Transformed item
        """
        image, marks = item["image"], item["marks"]
        # Image is [1, H, W], uint8. Convert to float and normalize.
        image = image.float() / 255.0

        # Resize image to fixed height while preserving aspect ratio (done in collate)
        # Here, we just ensure it's a tensor.
        return {"image": image, "marks": marks}

    def collate(self, batch):
        """Collate a batch of items.
        Args:
            batch: List of items to collate
        Returns:
            Collated batch (images_tensor, targets_tensor)
        """
        pad_token_idx = HOMRDataset.MARK_NAMES.index("[PAD]")

        # Pad marks to equal length within batch
        max_marks_length = 0
        if batch:  # Ensure batch is not empty
            max_marks_length = max(len(item["marks"]) for item in batch)

        processed_marks = []
        for item in batch:
            marks = item["marks"]
            padding = torch.full((max_marks_length - len(marks),), pad_token_idx, dtype=torch.long)
            processed_marks.append(torch.cat([marks, padding]))

        targets = torch.stack(processed_marks) if batch else torch.empty(0, max_marks_length, dtype=torch.long)

        # Pad images to equal width and fixed height within batch
        images = [item["image"] for item in batch]
        if not images:  # Handle empty batch case
            # Return empty tensors with expected number of dimensions
            # For images: (B, C, H, W) -> (0, 1, self.target_height, 0)
            # For targets: (B, MaxLen) -> (0, max_marks_length)
            return (torch.empty(0, 1, self.target_height, 0),
                    torch.empty(0, max_marks_length, dtype=torch.long))

        max_image_width = max(img.shape[-1] for img in images)

        processed_images = []
        for img in images:
            # Pad width to max_image_width (pad with 0 for normalized float images, or 1.0 if background is white)
            # Original padding was 255 for uint8, so 1.0 for float normalized.
            padding_width = max_image_width - img.shape[-1]
            padded_img = torch.nn.functional.pad(img, (0, padding_width, 0, 0), value=1.0)

            # Resize to (target_height, max_image_width)
            # torchvision.transforms.Resize expects PIL or (C, H, W) tensor.
            # Our image is (1, H_orig, W_padded)
            resized_img = torchvision.transforms.functional.resize(
                padded_img,
                [self.target_height, max_image_width],
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            processed_images.append(resized_img)

        images_tensor = torch.stack(processed_images)

        return images_tensor, targets


class EditDistanceMetric(torch.nn.Module):
    """Compute edit distance metric between predicted and target sequences."""

    def __init__(self, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer("total_distance", torch.tensor(0.0))
        self.register_buffer("total_length", torch.tensor(0.0))

    def reset(self):
        self.total_distance.zero_()
        self.total_length.zero_()

    def update(self, decoded_preds: list[torch.Tensor], targets: torch.Tensor):
        """Update metric with a batch of predictions and targets.
        Args:
            decoded_preds: A list of 1D tensors, each tensor is a sequence of predicted token indices.
            targets: A 2D tensor of shape (batch_size, max_target_length) with ground truth token indices.
        """
        self.total_distance = self.total_distance.to(targets.device)
        self.total_length = self.total_length.to(targets.device)

        for pred_seq, target_seq_padded in zip(decoded_preds, targets):
            # pred_seq is already a list/tensor of token IDs from CTC decoder (blanks removed, not repeated)
            pred_filtered = pred_seq.tolist()  # Convert to list of ints

            # Filter padding from target sequence
            target_filtered = [t.item() for t in target_seq_padded if t.item() != self.ignore_index]

            distance = self._levenshtein_distance(pred_filtered, target_filtered)

            self.total_distance += distance
            # Normalize by the length of the true target sequence
            # If target_filtered is empty, Levenshtein distance is len(pred_filtered).
            # total_length should reflect the sum of lengths of reference sequences.
            if len(target_filtered) > 0:
                self.total_length += len(target_filtered)
            elif len(pred_filtered) > 0:  # If target is empty but pred is not, this is pure insertion
                self.total_length += len(pred_filtered)  # Consider this for normalization, or stick to target_length
                # Standard WER/CER uses target length.

    def compute(self) -> torch.Tensor:
        if self.total_length == 0:
            return torch.tensor(1.0, device=self.total_distance.device)  # Max error if no valid targets
        return self.total_distance / self.total_length

    def _levenshtein_distance(self, s1: list[int], s2: list[int]) -> int:
        """Compute the Levenshtein distance between two sequences."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if not s2:  # len(s2) == 0
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


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
    # - "image", a `[1, HEIGHT, WIDTH]` tensor of `torch.uint8` values in [0-255] range,
    # - "marks", a `[num_marks]` tensor with indices of marks on the image.
    # Using `decode_on_demand=True` loads just the raw dataset (~500MB of undecoded PNG images)
    # and then decodes them on every access. Using `decode_on_demand=False` decodes the images
    # during loading, resulting in much faster access, but requires ~5GB of memory.
    homr = HOMRDataset(decode_on_demand=True)

    # TODO: Create the model and train it
    # Create transformed datasets
    train_dataset = HOMRDatasetTransform(homr.train)
    dev_dataset = HOMRDatasetTransform(homr.dev)
    test_dataset = HOMRDatasetTransform(homr.test)

    # Create data loaders
    train_loader = train_dataset.dataloader(batch_size=args.batch_size, shuffle=True)
    dev_loader = dev_dataset.dataloader(batch_size=args.batch_size)
    test_loader = test_dataset.dataloader(batch_size=args.batch_size)

    # Initialize model
    # HOMRDataset.MARKS is the number of classes (including [PAD] at index 0)
    # HOMRDataset.MARK_NAMES is the list of token names
    model = HOMRModel(args, num_classes=HOMRDataset.MARKS, mark_names=HOMRDataset.MARK_NAMES)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)  # Using AdamW

    scheduler = None
    if args.cosine_decay:
        # Total steps for CosineAnnealingLR
        num_training_steps = args.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,  # Number of iterations
            eta_min=1e-8  # Minimum learning rate
        )

    # Configure the training process
    # CTCLoss blank index must match the one used for the decoder and data padding.
    # HOMRDataset.MARK_NAMES[0] is '[PAD]', so blank=0 is correct.
    pad_token_idx = HOMRDataset.MARK_NAMES.index("[PAD]")
    model.configure(
        optimizer=optimizer,
        scheduler=scheduler,  # Pass the scheduler instance
        loss=torch.nn.CTCLoss(blank=pad_token_idx, zero_infinity=True),
        metrics={
            "edit_distance": EditDistanceMetric(ignore_index=pad_token_idx),
        },
        logdir=args.logdir,
        # clip_grad_norm_value=1.0 # Optional: gradient clipping
    )

    # Training loop
    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    # Generate test set predictions
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(test_loader, data_with_labels=True)

        for sequence in predictions:
            # Convert indices to mark names, filtering out padding tokens
            # print(sequence)
            mark_sequence = [HOMRDataset.MARK_NAMES[mark] for mark in sequence.astype(numpy.int32) if mark != 0]
            print(" ".join(mark_sequence), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
