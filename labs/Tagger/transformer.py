#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.10")
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--transformer_dropout", default=0., type=float, help="Transformer dropout.")
parser.add_argument("--transformer_expansion", default=4, type=int, help="Transformer FFN expansion factor.")
parser.add_argument("--transformer_heads", default=4, type=int, help="Transformer heads.")
parser.add_argument("--transformer_layers", default=2, type=int, help="Transformer layers.")
parser.add_argument("--seed", default=46, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(npfl138.TrainableModule):
    class FFN(torch.nn.Module):
        def __init__(self, dim: int, expansion: int) -> None:
            super().__init__()
            # TODO: Create the required layers -- first a ReLU-activated dense
            # layer with `dim * expansion` units, followed by a dense layer
            # with `dim` units without an activation.
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(dim, dim * expansion),
                torch.nn.ReLU(),
                torch.nn.Linear(dim * expansion, dim),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # TODO: Execute the FFN Transformer layer.
            return self.layers(inputs)

    class SelfAttention(torch.nn.Module):
        def __init__(self, dim: int, heads: int) -> None:
            super().__init__()
            self.dim, self.heads = dim, heads
            # TODO: Create weight matrices W_Q, W_K, W_V, and W_O; each a module parameter
            # `torch.nn.Parameter` of shape `[dim, dim]`. The weights should be initialized using
            # the `torch.nn.init.xavier_uniform_` in the same order the matrices are listed above.
            self.W_Q = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(dim, dim)))
            self.W_K = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(dim, dim)))
            self.W_V = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(dim, dim)))
            self.W_O = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(dim, dim)))

        def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # TODO: Execute the self-attention layer.
            #
            # Start by computing Q, K and V. In all cases:
            # - first multiply `inputs` by the corresponding weight matrix W_Q/W_K/W_V,
            # - reshape via `torch.reshape` to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - permute dimensions via `torch.permute` to `[batch_size, heads, max_sentence_len, dim // heads]`.

            batch_size, max_sentence_len, _ = inputs.shape
            head_dim = self.dim // self.heads

            Q = inputs @ self.W_Q
            Q = torch.reshape(Q, [batch_size, max_sentence_len, self.heads, head_dim])
            Q = torch.permute(Q, [0, 2, 1, 3])

            K = inputs @ self.W_K
            K = torch.reshape(K, [batch_size, max_sentence_len, self.heads, head_dim])
            K = torch.permute(K, [0, 2, 1, 3])

            V = inputs @ self.W_V
            V = torch.reshape(V, [batch_size, max_sentence_len, self.heads, head_dim])
            V = torch.permute(V, [0, 2, 1, 3])

            # TODO: Continue by computing the self-attention weights as Q @ K^T,
            # normalizing by the square root of `dim // heads`.
            attention_weights = (Q @ K.transpose(-2, -1)) / (head_dim ** 0.5)

            # TODO: Apply the softmax, but including a suitable mask ignoring all padding words.
            # The original `mask` is a bool matrix of shape `[batch_size, max_sentence_len]`
            # indicating which words are valid (nonzero value) or padding (zero value).
            # To mask an input to softmax, replace it by -1e9 (theoretically we should use
            # minus infinity, but `torch.exp(-1e9)` is also zero because of limited precision).
            attention_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, max_sentence_len]
            attention_mask = attention_mask & mask.unsqueeze(1).unsqueeze(
                -1)  # [batch_size, 1, max_sentence_len, max_sentence_len]
            attention_weights = attention_weights.masked_fill(~attention_mask, -1e9)
            attention_probs = torch.softmax(attention_weights, dim=-1)

            # TODO: Finally,
            # - take a weighted combination of values V according to the computed attention
            #   (using a suitable matrix multiplication),
            # - permute the result to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - reshape to `[batch_size, max_sentence_len, dim]`,
            # - multiply the result by the W_O matrix.
            context = attention_probs @ V  # [batch_size, heads, max_sentence_len, head_dim]
            context = torch.permute(context, [0, 2, 1, 3])  # [batch_size, max_sentence_len, heads, head_dim]
            context = torch.reshape(context,
                                    [batch_size, max_sentence_len, self.dim])  # [batch_size, max_sentence_len, dim]
            output = context @ self.W_O  # [batch_size, max_sentence_len, dim]

            return output

    class PositionalEmbedding(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # TODO: Compute the sinusoidal positional embeddings. Assuming the embeddings have
            # a shape `[max_sentence_len, dim]` with `dim` even, and for `0 <= i < dim/2`:
            # - the value on index `[pos, i]` should be
            #     `sin(pos / 10_000 ** (2 * i / dim))`
            # - the value on index `[pos, dim/2 + i]` should be
            #     `cos(pos / 10_000 ** (2 * i / dim))`
            # - the `0 <= pos < max_sentence_len` is the sentence index.
            # This order is the same as in the visualization on the slides, but
            # different from the original paper where `sin` and `cos` interleave.
            _, max_sentence_len, dim = inputs.shape

            embeddings = (
                    torch.arange(max_sentence_len, dtype=torch.float32)[:, None] /
                    (10_000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            )
            embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
            return embeddings

    class Transformer(torch.nn.Module):
        def __init__(self, layers: int, dim: int, expansion: int, heads: int, dropout: float) -> None:
            super().__init__()
            # TODO: Create:
            # - the positional embedding layer;
            # - the required number of transformer layers, each consisting of
            #   - a layer normalization and a self-attention layer followed by a dropout layer,
            #   - a layer normalization and a FFN layer followed by a dropout layer.
            # During ReCodEx evaluation, the order of layer creation is not important,
            # but if you want to get the same results as on the course website, create
            # the layers in the order they are called in the `forward` method.
            self.positional_embedding = Model.PositionalEmbedding()
            self.dropout = torch.nn.Dropout(dropout)

            # Create transformer layers
            self.layers = []
            for _ in range(layers):
                # Self-attention sublayer
                self_attention_norm = torch.nn.LayerNorm(dim)
                self_attention = Model.SelfAttention(dim, heads)

                # FFN sublayer
                ffn_norm = torch.nn.LayerNorm(dim)
                ffn = Model.FFN(dim, expansion)

                self.layers.append((self_attention_norm, self_attention, ffn_norm, ffn))

            # Convert list to ModuleList to handle parameters properly
            self.layers = torch.nn.ModuleList([
                torch.nn.ModuleList([sa_norm, sa, ffn_norm, ffn])
                for sa_norm, sa, ffn_norm, ffn in self.layers
            ])

        def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # TODO: First compute the positional embeddings.
            pos_embeddings = self.positional_embedding(inputs)

            # TODO: Add the positional embeddings to the `inputs` and then
            # perform the given number of transformer layers, composed of
            # - a self-attention sub-layer, followed by
            # - a FFN sub-layer.
            # In each sub-layer, pass the input through LayerNorm, then compute
            # the corresponding operation, apply dropout, and finally add this result
            # to the original sub-layer input. Note that the given `mask` should be
            # passed to the self-attention operation to ignore the padding words.
            x = inputs + pos_embeddings

            # Apply transformer layers
            for self_attention_norm, self_attention, ffn_norm, ffn in self.layers:
                # Self-attention sublayer
                residual = x
                x = self_attention_norm(x)
                x = self_attention(x, mask)
                x = self.dropout(x)
                x = residual + x

                # FFN sublayer
                residual = x
                x = ffn_norm(x)
                x = ffn(x)
                x = self.dropout(x)
                x = residual + x

            return x

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        # Create all needed layers.
        # TODO(tagger_we): Create a `torch.nn.Embedding` layer, embedding the word ids
        # from `train.words.string_vocab` to dimensionality `args.we_dim`.
        self._word_embedding = torch.nn.Embedding(len(train.words.string_vocab), args.we_dim)

        # TODO: Create a `Model.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        self._transformer = Model.Transformer(
            layers=args.transformer_layers,
            dim=args.we_dim,
            expansion=args.transformer_expansion,
            heads=args.transformer_heads,
            dropout=args.transformer_dropout
        )

        # TODO(tagger_we): Create an output linear layer (`torch.nn.Linear`) processing the RNN output,
        # producing logits for tag prediction; `train.tags.string_vocab` is the tag vocabulary.
        self._output_layer = torch.nn.Linear(args.we_dim, len(train.tags.string_vocab))

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        # TODO(tagger_we): Start by embedding the `word_ids` using the word embedding layer.
        hidden = self._word_embedding(word_ids)

        # TODO: Process the embedded words through the transformer. As the second argument,
        # pass the attention mask `word_ids != MorphoDataset.PAD`.
        hidden = self._transformer(hidden, word_ids != MorphoDataset.PAD)

        # TODO(tagger_we): Pass `hidden` through the output layer. Such an output has a shape
        # `[batch_size, sequence_length, num_tags]`, but the loss and the metric expect
        # the `num_tags` dimension to be in front (`[batch_size, num_tags, sequence_length]`),
        # so you need to reorder the dimensions.
        hidden = self._output_layer(hidden)
        # Reorder dimensions: [batch_size, sequence_length, num_tags] -> [batch_size, num_tags, sequence_length]
        hidden = hidden.permute(0, 2, 1)

        return hidden


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO(tagger_we): Construct a single example, each consisting of the following pair:
        # - a PyTorch tensor of integer ids of input words as input,
        # - a PyTorch tensor of integer tag ids as targets.
        # To create the ids, use `string_vocab` of `self.dataset.words` and `self.dataset.tags`.
        word_ids = torch.tensor(self.dataset.words.string_vocab.indices(example["words"]), dtype=torch.int64)
        tag_ids = torch.tensor(self.dataset.tags.string_vocab.indices(example["tags"]), dtype=torch.int64)
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
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Prepare the data for training.
    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)

    # Create the model and train.
    model = Model(args, morpho.train)

    model.configure(
        # TODO(tagger_we): Create the Adam optimizer.
        optimizer=torch.optim.Adam(model.parameters()),
        # TODO(tagger_we): Use the usual `torch.nn.CrossEntropyLoss` loss function. Additionally,
        # pass `ignore_index=morpho.PAD` to the constructor so that the padded
        # tags are ignored during the loss computation. Note that the loss
        # expects the input to be of shape `[batch_size, num_tags, sequence_length]`.
        loss=torch.nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD),
        # TODO(tagger_we): Create a `torchmetrics.Accuracy` metric, passing "multiclass" as
        # the first argument, `num_classes` set to the number of unique tags, and
        # again `ignore_index=morpho.PAD` to ignore the padded tags.
        metrics={"accuracy": torchmetrics.Accuracy(
            "multiclass",
            num_classes=len(morpho.train.tags.string_vocab),
            ignore_index=MorphoDataset.PAD
        )},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development and training losses for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if "loss" in metric}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
