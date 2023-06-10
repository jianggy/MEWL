# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Embedder(nn.Module):
    def __init__(self, embedding_size, vocab_sizes):
        super(Embedder, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_sizes = vocab_sizes
        self.embedders = nn.ModuleList([
            nn.Linear(vocab_size, self.embedding_size, bias=False)
            for vocab_size in self.vocab_sizes
        ])

    @property
    def output_size(self):
        return self.embedding_size * len(self.vocab_sizes)

    def forward(self, x):
        base = 0
        output = []
        for i, embedder in enumerate(self.embedders):
            # print(x[..., base:base + self.vocab_sizes[i]])
            # print(embedder(x[..., base:base + self.vocab_sizes[i]]).shape)
            output.append(embedder(x[..., base:base + self.vocab_sizes[i]]))
            base += self.vocab_sizes[i]
        return torch.cat(output, dim=-1)


def get_position_embedding(sequence_length,
                           hidden_size,
                           max_timescale=10000.0,
                           min_timescale=2.0):
    pos_seq = torch.arange(sequence_length)
    freqs = torch.arange(0, hidden_size, min_timescale)
    inv_freq = 1 / (max_timescale**(freqs / hidden_size))
    sinusoid_inp = torch.einsum("i,j->ij", pos_seq, inv_freq)
    pos_embed = torch.cat([torch.sin(sinusoid_inp),
                           torch.cos(sinusoid_inp)], -1)
    pos_embed = pos_embed.unsqueeze(0)
    return pos_embed


def rel_shift(position_logits):
    # batch, nhead, seq_len, seq_len
    batch_size, num_heads, t1, t2 = position_logits.shape
    to_pad = torch.zeros([batch_size, num_heads, t1, 1],
                         device=position_logits.device,
                         dtype=position_logits.dtype)
    position_logits = torch.cat([to_pad, position_logits], dim=-1)
    position_logits = position_logits.view(batch_size, num_heads, t2 + 1, t1)
    position_logits = position_logits[:, :, 1:, :]
    return position_logits.view(batch_size, num_heads, t1, t2)