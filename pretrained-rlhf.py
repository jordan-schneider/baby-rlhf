import logging
from pathlib import Path
from typing import Callable, Generator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchtext.data.utils import get_tokenizer  # type: ignore
from torchtext.vocab import Vocab, build_vocab_from_iterator  # type: ignore
from torchtyping import TensorType  # type: ignore

from hh_data import load_hh_data


def binarize(data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out = []
    for better, worse in data:
        out.append((f"GOOD:{better}", f"BAD:{better}"))
        out.append((f"BAD:{worse}", f"GOOD:{worse}"))
    return out


def log_pref_loss(returns: TensorType["batch", 2]) -> TensorType["batch"]:
    return torch.log(1 + torch.exp(returns[:, 1] - returns[:, 0]))


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super().__init__()
        # decoder_layer = TransformerDecoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     layer_norm_eps=layer_norm_eps,
        #     batch_first=batch_first,
        #     norm_first=norm_first,
        # )
        # decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        # self.model = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.d_model = d_model

    def forward(
        self,
        inputs: TensorType["batch", "seqlen"],
        mask: TensorType["seqlen", "seqlen"],
    ) -> torch.Tensor:
        return inputs


class RewardModel(nn.Module):
    def __init__(self, language_model: DecoderOnlyTransformer, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.language_model = language_model
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=language_model.d_model
        )
        self.unembedding = nn.Linear(language_model.d_model, vocab_size)
        self.reward_layer = nn.Linear(language_model.d_model, 1)

    def forward(
        self,
        inputs: TensorType["batch", "seqlen"],
        mask: TensorType["seqlen", "seqlen"],
    ) -> Tuple[TensorType["batch", "seqlen"], TensorType["batch", "seqlen", "vocab"]]:
        embedded = self.embedding(inputs)
        rewards = self.reward_layer(self.language_model(embedded, mask)).squeeze(-1)

        no_mask = torch.ones((inputs.shape[1], inputs.shape[1]), dtype=torch.bool)
        token_probs = functional.softmax(
            self.unembedding(self.language_model(embedded, no_mask)), dim=-1
        )
        return rewards, token_probs


def token_iterator(
    tokenizer: Callable[[str], List[str]], prefs: List[Tuple[str, str]]
) -> Generator[List[str], None, None]:
    for better, worse in prefs:
        yield tokenizer(better)
        yield tokenizer(worse)


def pretrain(
    model: RewardModel,
    optim: torch.optim.Optimizer,
    data: List[Tuple[str, str]],
    tokenizer: Callable[[str], List[str]],
    vocab: Vocab,
) -> RewardModel:
    binarized_data = binarize(data)
    optim.zero_grad()
    for better, worse in binarized_data:
        # TODO: Add masking for rewards
        tokenized_better = torch.tensor(
            vocab.lookup_indices(tokenizer(better)), dtype=torch.long
        ).view(1, -1)
        tokenized_worse = torch.tensor(
            vocab.lookup_indices(tokenizer(worse)), dtype=torch.long
        ).view(1, -1)
        mask = torch.ones(
            (tokenized_better.shape[1], tokenized_better.shape[1]), dtype=torch.bool
        ).triu()
        better_reward, better_probs = model(tokenized_better, mask)
        worse_reward, worse_probs = model(tokenized_worse, mask)

        # TODO: Language lab doesn't use any masking for language model loss. Why?
        preference_loss: torch.Tensor = log_pref_loss(
            torch.stack(
                [torch.sum(better_reward, dim=1), torch.sum(worse_reward, dim=1)], dim=1
            )
        )
        language_model_loss = functional.binary_cross_entropy(
            better_probs,
            functional.one_hot(tokenized_better, num_classes=len(vocab)).float(),
        )
        loss = preference_loss + language_model_loss
        loss.backward()
        logging.info(f"loss: {loss.item()}")
        optim.step()
        optim.zero_grad()
    return model


def main():
    print("Loading data")
    prefs = load_hh_data(Path("./hh-rlhf/helpful-base/train.jsonl"))[:100]
    print("Tokenizing")
    tokenizer = get_tokenizer("spacy", "en_core_web_sm")
    tokens = list(token_iterator(tokenizer, prefs))
    vocab = build_vocab_from_iterator(tokens + [["GOOD", "BAD"]])
    print(vocab["good"])
    print("Initializing model")
    model = RewardModel(
        DecoderOnlyTransformer(d_model=8, nhead=2), vocab_size=len(vocab)
    )
    optim = torch.optim.Adam(model.parameters())
    print("Pretraining")
    pretrain(model, optim, prefs, tokenizer, vocab)


main()
