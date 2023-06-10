import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import clip
import math
import einops

from PIL import Image

import pytorch_lightning as pl
import torchmetrics
from .consts import task_names


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class CLIPTrans(pl.LightningModule):
    def __init__(self, *, clip_model, d_model=512, num_layers=6, nhead=8, dim_feedforward=1024, n_classes=5, dropout=0.1, **kwargs):
        super().__init__()
        self.clip_model = clip.load(clip_model)[0]
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.d_model = d_model

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, activation='gelu', dropout=dropout, norm_first=True)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.pos_embedding = PositionalEmbedding(d_model)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )
        self.train_accs = nn.ModuleList([torchmetrics.Accuracy(
            task='multiclass', num_classes=n_classes) for _ in range(len(task_names))])
        self.val_accs = nn.ModuleList([torchmetrics.Accuracy(
            task='multiclass', num_classes=n_classes) for _ in range(len(task_names))])
        self.test_accs = nn.ModuleList([torchmetrics.Accuracy(
            task='multiclass', num_classes=n_classes) for _ in range(len(task_names))])

        # save hyperparam
        self.save_hyperparameters()

    def forward(self, images, texts):
        '''
            images: (batch_size, 6 + 1, 3, 224, 224)
            texts: (batch_size, 6 + 5, 77)
        '''
        batch_size = images.shape[0]
        images_input = einops.rearrange(
            images, 'b c n h w -> (b c) n h w', c=7)
        texts_input = einops.rearrange(texts, 'b c d -> (b c) d', c=11)

        image_features = self.clip_model.encode_image(
            images_input)  # (batch_size * 7, 512)
        text_features = self.clip_model.encode_text(
            texts_input)  # (batch_size * 11, 512)

        image_features = einops.rearrange(
            image_features, '(b c) d -> b c d', b=batch_size, c=7)
        text_features = einops.rearrange(
            text_features, '(b c) d -> b c d', b=batch_size, c=11)

        # inputs like cls, image_1, text_1, sep, image_2, text_2, sep, ..., image_6, sep, text 6, text 7, ..., text 11
        image_contexts = image_features[:, :6, :]
        text_contexts = text_features[:, :6, :]
        context_inputs = torch.stack([image_contexts, text_contexts, self.sep_token.expand(
            batch_size, 6, -1)], dim=2)  # (batch_size, 6, 3, d_model)
        # random permute
        if self.training:
            context_inputs = context_inputs[:, torch.randperm(6), :, :]

        context_inputs = einops.rearrange(
            context_inputs, 'b c d e -> b (c d) e')  # (batch_size, 18, d_model)

        image_query = image_features[:, 6:, :]
        text_query = text_features[:, 6:, :]

        transformer_inputs = torch.cat([self.cls_token.expand(batch_size, 1, -1),
                                        context_inputs, image_query,
                                        self.sep_token.expand(batch_size, 1, -1), text_query], dim=1)  # (batch_size, 18 + 6 + 5, d_model)

        transformer_inputs = transformer_inputs + \
            self.pos_embedding(transformer_inputs)

        transformer_outputs = self.transformer(transformer_inputs)

        cls_output = transformer_outputs[:, 0, :]

        return self.mlp_head(cls_output)

    def training_step(self, batch, batch_idx):
        images, texts, labels, task_idx = batch
        logits = self(images, texts)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss, sync_dist=True)

        for i in range(len(task_names)):
            task_mask = (task_idx == i)
            if task_mask.sum() > 0:
                task_logits = logits[task_mask]
                task_labels = labels[task_mask]
                acc = self.train_accs[i](task_logits, task_labels)

        return loss

    def validation_step(self, batch, batch_idx):
        images, texts, labels, task_idx = batch
        logits = self(images, texts)
        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss, sync_dist=True)

        for i in range(len(task_names)):
            task_mask = (task_idx == i)
            if task_mask.sum() > 0:
                task_logits = logits[task_mask]
                task_labels = labels[task_mask]
                acc = self.val_accs[i](task_logits, task_labels)

        return loss

    def test_step(self, batch, batch_idx):
        images, texts, labels, task_idx = batch
        logits = self(images, texts)
        loss = F.cross_entropy(logits, labels)
        self.log('test_loss', loss, sync_dist=True)

        for i in range(len(task_names)):
            task_mask = (task_idx == i)
            if task_mask.sum() > 0:
                task_logits = logits[task_mask]
                task_labels = labels[task_mask]
                acc = self.test_accs[i](task_logits, task_labels)

        return loss

    def training_epoch_end(self, outputs):
        for i in range(len(task_names)):
            self.log(
                f'train_acc_{task_names[i]}', self.train_accs[i].compute(), sync_dist=True)

    def validation_epoch_end(self, outputs):
        for i in range(len(task_names)):
            self.log(f'val_acc_{task_names[i]}',
                     self.val_accs[i].compute(), sync_dist=True)

    def test_epoch_end(self, outputs):
        for i in range(len(task_names)):
            self.log(
                f'test_acc_{task_names[i]}', self.test_accs[i].compute(), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-4, weight_decay=0.01)
        return optimizer


class CLIPTransWOTextEncoder(pl.LightningModule):
    def __init__(self, *, clip_model, d_model=512, num_layers=6, nhead=8, dim_feedforward=1024, n_classes=5, dropout=0.1, token_length=16, **kwargs):
        super().__init__()
        self.clip_model = clip.load(clip_model)[0]
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.d_model = d_model
        # tokenizer pad to length
        self.token_length = token_length

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, activation='gelu', dropout=dropout, norm_first=True)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.token_embedding = self.clip_model.token_embedding
        self.pos_embedding = PositionalEmbedding(d_model)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )
        self.train_accs = nn.ModuleList([torchmetrics.Accuracy(
            task='multiclass', num_classes=n_classes) for _ in range(len(task_names))])
        self.val_accs = nn.ModuleList([torchmetrics.Accuracy(
            task='multiclass', num_classes=n_classes) for _ in range(len(task_names))])
        self.test_accs = nn.ModuleList([torchmetrics.Accuracy(
            task='multiclass', num_classes=n_classes) for _ in range(len(task_names))])

        # save hyperparam
        self.save_hyperparameters()

    def forward(self, images, texts):
        '''
            images: (batch_size, 6 + 1, 3, 224, 224)
            texts: (batch_size, 6 + 5, 77)
        '''
        batch_size = images.shape[0]
        images_input = einops.rearrange(
            images, 'b c n h w -> (b c) n h w', c=7)
        texts_input = einops.rearrange(texts, 'b c d -> (b c) d', c=11)

        image_features = self.clip_model.encode_image(
            images_input)  # (batch_size * 7, 512)
        # (batch_size * 11, self.token_length, 512)
        text_features = self.token_embedding(texts_input)

        image_features = einops.rearrange(
            image_features, '(b c) d -> b c d', b=batch_size, c=7)  # (batch_size, 7, 512)
        # (batch_size, 11, self.token_length, 512)
        text_features = einops.rearrange(
            text_features, '(b c) t d -> b c t d', b=batch_size, c=11)

        # inputs like cls, image_1, text_1s, sep, image_2, text_2s, sep, ..., image_6, sep, text 6s, text 7s, ..., text 11s
        image_contexts = image_features[:, :6, :]  # (batch_size, 6, 512)
        # (batch_size, 6, self.token_length, 512)
        text_contexts = text_features[:, :6, :, :]
        # context_inputs should be (batch_size, 6, 1 + self.token_length + 1, d_model)
        context_inputs = torch.cat([image_contexts.unsqueeze(2),
                                    text_contexts,
                                    self.sep_token.expand(batch_size, 6, 1, -1)
                                    ], dim=2)  # (batch_size, 6, 1 + self.token_length + 1, d_model)

        # random permute
        if self.training:
            context_inputs = context_inputs[:, torch.randperm(6), :, :]

        context_inputs = einops.rearrange(
            context_inputs, 'b c l d -> b (c l) d') # (batch_size, 6 * (self.token_length + 2), d_model)

        image_query = image_features[:, 6:, :]  # (batch_size, 1, 512)
        text_query = text_features[:, 6:, :]   # (batch_size, 5, self.token_length, 512)
        text_query = einops.rearrange(
            text_query, 'b c t d -> b (c t) d') # (batch_size, 5 * self.token_length, 512)
    
        transformer_inputs = torch.cat([self.cls_token.expand(batch_size, 1, -1),
                                        context_inputs, image_query,
                                        self.sep_token.expand(batch_size, 1, -1), text_query], dim=1)
        # (batch_size, 6 * (self.token_length + 2) + 1 + 5 * self.token_length, d_model)

        transformer_inputs = transformer_inputs + \
            self.pos_embedding(transformer_inputs)

        transformer_outputs = self.transformer(transformer_inputs)

        cls_output = transformer_outputs[:, 0, :]

        return self.mlp_head(cls_output)

    def training_step(self, batch, batch_idx):
        images, texts, labels, task_idx = batch
        logits = self(images, texts)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss, sync_dist=True)

        for i in range(len(task_names)):
            task_mask = (task_idx == i)
            if task_mask.sum() > 0:
                task_logits = logits[task_mask]
                task_labels = labels[task_mask]
                acc = self.train_accs[i](task_logits, task_labels)

        return loss

    def validation_step(self, batch, batch_idx):
        images, texts, labels, task_idx = batch
        logits = self(images, texts)
        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss, sync_dist=True)

        for i in range(len(task_names)):
            task_mask = (task_idx == i)
            if task_mask.sum() > 0:
                task_logits = logits[task_mask]
                task_labels = labels[task_mask]
                acc = self.val_accs[i](task_logits, task_labels)

        return loss

    def test_step(self, batch, batch_idx):
        images, texts, labels, task_idx = batch
        logits = self(images, texts)
        loss = F.cross_entropy(logits, labels)
        self.log('test_loss', loss, sync_dist=True)

        for i in range(len(task_names)):
            task_mask = (task_idx == i)
            if task_mask.sum() > 0:
                task_logits = logits[task_mask]
                task_labels = labels[task_mask]
                acc = self.test_accs[i](task_logits, task_labels)

        return loss

    def training_epoch_end(self, outputs):
        for i in range(len(task_names)):
            self.log(
                f'train_acc_{task_names[i]}', self.train_accs[i].compute(), sync_dist=True)

    def validation_epoch_end(self, outputs):
        for i in range(len(task_names)):
            self.log(f'val_acc_{task_names[i]}',
                     self.val_accs[i].compute(), sync_dist=True)

    def test_epoch_end(self, outputs):
        for i in range(len(task_names)):
            self.log(
                f'test_acc_{task_names[i]}', self.test_accs[i].compute(), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-4, weight_decay=0.01)
        return optimizer
