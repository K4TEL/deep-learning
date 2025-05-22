#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchaudio.models.decoder
import torchmetrics

import npfl138
npfl138.require_version("2425.8")
from npfl138.datasets.common_voice_cs import CommonVoiceCs

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate.")
parser.add_argument("--rnn_dim", default=256, type=int, help="RNN dimension.")
parser.add_argument("--rnn_layers", default=3, type=int, help="Number of RNN layers.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--ctc_beam", default=1, type=int, help="CTC beam size.")
parser.add_argument("--cosine_decay", default=False, action="store_true", help="Use cosine learning rate decay.")



class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: CommonVoiceCs.Dataset) -> None:
        super().__init__()
        # TODO: Define the model.
        # Define the model architecture with bidirectional LSTMs
        self.rnn_dim = args.rnn_dim
        self.dropout_rate = args.dropout
        
        # Create a list of LSTM layers
        self._rnn = torch.nn.ModuleList([
            torch.nn.LSTM(CommonVoiceCs.MFCC_DIM, args.rnn_dim, batch_first=True, bidirectional=True)
        ])
        
        # Add additional LSTM layers
        self._rnn.extend([
            torch.nn.LSTM(args.rnn_dim, args.rnn_dim, batch_first=True, bidirectional=True) 
            for _ in range(args.rnn_layers - 1)
        ])
        
        # Dropout for regularization
        self._dropout_layer = torch.nn.Dropout(args.dropout)
        
        # Output projection layer
        self._output_layer = torch.nn.Linear(args.rnn_dim, len(CommonVoiceCs.LETTER_NAMES))
        
        # Initialize CTC decoder based on available hardware
        if torch.cuda.is_available():
            self._ctc_decoder = torchaudio.models.decoder.cuda_ctc_decoder(
                CommonVoiceCs.LETTER_NAMES, beam_size=args.ctc_beam
            )
        else:
            self._ctc_decoder = torchaudio.models.decoder.ctc_decoder(
                blank_token=CommonVoiceCs.LETTER_NAMES[0],  # [PAD] as blank token
                sil_token=" ",  # Space as silent token
                tokens=CommonVoiceCs.LETTER_NAMES,
                beam_size=args.ctc_beam,
                log_add=True
            )


    def forward(self, mfccs: torch.Tensor, mfccs_lengths: torch.Tensor) -> torch.Tensor:
        # TODO: Compute the output of the model.
        # Pack padded sequence for efficient computation
        hidden = torch.nn.utils.rnn.pack_padded_sequence(
            mfccs, mfccs_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process through LSTM layers
        for i, rnn in enumerate(self._rnn):
            residual = hidden
            hidden, _ = rnn(hidden)
            
            # Split and combine bidirectional outputs
            forward, backward = torch.chunk(hidden.data, 2, dim=-1)
            hidden = self._dropout_layer(forward + backward)
            
            # Add residual connection after first layer
            if i > 0:
                hidden += residual.data
                
            # Repack the sequence
            hidden = torch.nn.utils.rnn.PackedSequence(hidden, *residual[1:])
        
        # Unpack the sequence back to padded form
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)
        
        # Project to output dimension and apply log softmax
        outputs = self._output_layer(hidden)
        log_probs = outputs.log_softmax(dim=-1)
        
        return log_probs

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, mfccs: torch.Tensor, mfccs_lengths: torch.Tensor) -> torch.Tensor:
        # TODO: Compute the loss, most likely using the `torch.nn.CTCLoss` class.
        # Compute CTC loss
        # Note: CTC loss expects inputs in (time, batch, classes) format
        y_pred_transposed = y_pred.permute(1, 0, 2)
        
        # Get target lengths by counting non-zero elements
        target_lengths = torch.sum(y_true != 0, dim=1)
        
        return self.loss(
            y_pred_transposed,
            y_true,
            mfccs_lengths,
            target_lengths
        )

    def ctc_decoding(self, y_pred: torch.Tensor, mfccs: torch.Tensor, mfccs_lengths: torch.Tensor) -> list[torch.Tensor]:
        # TODO: Compute predictions, either using manual CTC decoding, or you can use:
        # - `torchaudio.models.decoder.ctc_decoder`, which is CPU-based decoding with
        #   rich functionality;
        #   - note that you need to provide `blank_token` and `sil_token` arguments
        #     and they must be valid tokens. For `blank_token`, you need to specify
        #     the token whose index corresponds to the blank token index;
        #     for `sil_token`, you can use also the blank token index (by default,
        #     `sil_token` has ho effect on the decoding apart from being added as the
        #     first and the last token of the predictions unless it is a blank token).
        # - `torchaudio.models.decoder.cuda_ctc_decoder`, which is faster GPU-based
        #   decoder with limited functionality.
        # Use the CTC decoder to get predicted sequences
        hypotheses = self._ctc_decoder(y_pred, mfccs_lengths)
        
        # Extract token indices from hypotheses
        return [torch.as_tensor(hypothesis[0].tokens) for hypothesis in hypotheses]

    def compute_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, mfccs: torch.Tensor, mfccs_lengths: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # TODO: Compute predictions using the `ctc_decoding`. Consider computing it
        # only when `self.training==False` to speed up training.
        # Get predicted sequences
        predictions = self.ctc_decoding(y_pred, mfccs, mfccs_lengths)
        self.metrics["edit_distance"].update(predictions, y_true)
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            # Perform constrained decoding.
            batch = self.ctc_decoding(self.forward(*xs), *xs)
            if as_numpy:
                batch = [example.numpy(force=True) for example in batch]
            return batch


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO: Prepare a single example. The structure of the inputs then has to be reflected
        # in the `forward`, `compute_loss`, and `compute_metrics` methods; right now, there are
        # just `...` instead of the input arguments in the definition of the mentioned methods.
        #
        # Note that while the `CommonVoiceCs.LETTER_NAMES` do not explicitly contain a blank token,
        # the [PAD] token can be employed as a blank token.
        mfccs = example["mfccs"]
        
        # Convert sentence characters to token indices
        # Note: Using the first token (index 0) as padding/blank
        tag_ids = torch.tensor(
            [CommonVoiceCs.LETTER_NAMES.index(char) for char in example["sentence"]], 
            dtype=torch.int64
        )
        
        return mfccs, tag_ids

    def collate(self, batch):
        # TODO: Construct a single batch from a list of individual examples.
        # Split batch into features and targets
        mfccs, tag_ids = zip(*batch)
        
        # Get sequence lengths
        mfccs_lengths = torch.tensor([len(sequence) for sequence in mfccs], dtype=torch.int32)
        
        # Pad sequences to equal length within batch
        mfccs_padded = torch.nn.utils.rnn.pad_sequence(mfccs, batch_first=True)
        print(tag_ids[0].shape, tag_ids[-1].shape)
        if tag_ids[0].shape[0] > 0 and tag_ids[-1].shape[0] > 0:
            tag_ids_padded = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True)
        else:
            tag_ids = tuple(torch.zeros(10) * len(tag_ids))
            print(tag_ids[0].shape, tag_ids[-1].shape)
            tag_ids_padded = torch.zeros((len(tag_ids), 10))
        print(tag_ids_padded.shape)
        
        return (mfccs_padded, mfccs_lengths), tag_ids_padded


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

    # Load the data.
    common_voice = CommonVoiceCs()

    train = TrainableDataset(common_voice.train).dataloader(args.batch_size, shuffle=True)
    dev = TrainableDataset(common_voice.dev).dataloader(args.batch_size)
    test = TrainableDataset(common_voice.test).dataloader(args.batch_size)

    # TODO: Create the model and train it
    model = Model(args, common_voice.train)
    
    # Configure optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Setup learning rate scheduler if cosine decay is enabled
    schedule = None
    if args.cosine_decay:
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs * len(train),
            eta_min=1e-8
        )
    
    # Configure the training process
    model.configure(
        optimizer=optimizer,
        scheduler=schedule,
        loss=torch.nn.CTCLoss(blank=0, zero_infinity=True),
        metrics={
            "edit_distance": common_voice.EditDistanceMetric(ignore_index=0),
        },
        logdir=args.logdir,
    )

    # Train the model
    model.fit(train, dev=dev, epochs=args.epochs)
    
    for t in test:
    	print(t)
    	break

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "speech_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the CommonVoice sentences.
        predictions = model.predict(test, data_with_labels=True)

        for sentence in predictions:
            print("".join(CommonVoiceCs.LETTER_NAMES[char] for char in sentence), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
