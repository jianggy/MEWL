import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# import clip
import math
import einops

from PIL import Image

import pytorch_lightning as pl
import torchmetrics
from .consts import task_names

from .embedder import Embedder, get_position_embedding, rel_shift


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


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dropout,
                 position_encodings,
                 use_relative_positions,
                 device=None,
                 dtype=None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.use_relative_positions = use_relative_positions

        self.head_dim = self.d_model // self.nhead
        assert self.head_dim * self.nhead == self.d_model, "d_model not divisible by nhead"

        self.register_buffer("position_encodings", position_encodings)

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.val_linear = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(d_model, d_model)

        if self.use_relative_positions:
            self.r_w_bias = nn.Parameter(torch.randn(self.nhead,
                                                     self.head_dim))
            self.r_r_bias = nn.Parameter(torch.randn(self.nhead,
                                                     self.head_dim))
            torch.nn.init.xavier_normal_(self.r_w_bias)
            torch.nn.init.xavier_normal_(self.r_r_bias)

    def forward(self, query, key, val, attn_mask=None, **kwargs):
        batch = query.shape[0]
        q_len = query.shape[1]
        k_len = key.shape[1]
        v_len = val.shape[1]
        scale = math.sqrt(query.shape[-1])
        if (self.position_encodings is not None
                and not self.use_relative_positions):
            query += self.position_encodings[:, :q_len, :]
            key += self.position_encodings[:, :k_len, :]
            val += self.position_encodings[:, :v_len, :]
        query = self.query_linear(query).view(batch, -1, self.nhead,
                                              self.head_dim).transpose(1, 2)
        key = self.key_linear(key).view(batch, -1, self.nhead,
                                        self.head_dim).transpose(1, 2)
        val = self.val_linear(val).view(batch, -1, self.nhead,
                                        self.head_dim).transpose(1, 2)
        if self.use_relative_positions:
            scores1 = torch.matmul(
                query + self.r_w_bias.unsqueeze(1).unsqueeze(0),
                key.transpose(-2, -1))
            rel_encodings = self.key_linear(
                self.position_encodings[:, :k_len, :]).view(
                    1, -1, self.nhead, self.head_dim).transpose(1, 2)
            scores2 = torch.matmul(
                query + self.r_r_bias.unsqueeze(1).unsqueeze(0),
                rel_encodings.transpose(-2, -1))
            scores2 = rel_shift(scores2)
            scores = (scores1 + scores2) / scale
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) / scale
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attn_mask == 1, -1e9)
        attn_scores = F.softmax(scores, dim=-1)
        attn_scores = self.attn_dropout(attn_scores)
        attn_output = torch.matmul(attn_scores, val)
        output = attn_output.transpose(1, 2).contiguous().view(
            batch, -1, self.d_model)
        return self.final_linear(output)


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5,
                 norm_first=False,
                 position_encodings=None,
                 use_relative_positions=False,
                 device=None,
                 dtype=None):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first
        self.use_relative_positions = use_relative_positions
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.register_buffer("position_encodings", position_encodings)

        self.self_attn = MultiHeadAttention(
            self.d_model,
            self.nhead,
            dropout=self.dropout,
            position_encodings=self.position_encodings,
            use_relative_positions=self.use_relative_positions,
            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward,
                                 **factory_kwargs)
        self.dropoutff = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model,
                                 **factory_kwargs)

        self.norm_first = self.norm_first
        self.norm1 = nn.LayerNorm(self.d_model,
                                  eps=self.layer_norm_eps,
                                  **factory_kwargs)
        self.norm2 = nn.LayerNorm(self.d_model,
                                  eps=self.layer_norm_eps,
                                  **factory_kwargs)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self.sa_block(self.norm1(x), src_mask,
                                  src_key_padding_mask)
            x = x + self.ff_block(self.norm2(x))
        else:
            x = self.norm1(x +
                           self.sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self.ff_block(x))

        return x

    # self-attention block
    def sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x,
                           x,
                           x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)
        return self.dropout1(x)

    # feed forward block
    def ff_block(self, x):
        x = self.linear2(self.dropoutff(self.activation(self.linear1(x))))
        return self.dropout2(x)

class Aloe(pl.LightningModule):
    def __init__(self, *, d_model=512, num_layers=6, nhead=8, dim_feedforward=1024, n_classes=5, dropout=0.1, token_length=16, head_size=144, seed=0, **kwargs):
        super().__init__()
        pl.seed_everything(seed)

        # MAX_SEQ_LEN = 2048
        self.d_monet_latent = 16
        self.head_size = head_size

        # self.clip_model = clip.load(clip_model)[0]
        # for param in self.clip_model.parameters():
        #     param.requires_grad = False

        self.d_model = d_model
        # tokenizer pad to length
        self.token_length = token_length
        self.max_seq_len = 134

        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, activation='gelu', dropout=dropout, norm_first=True)
        self.use_relative_positions = True
        self.register_buffer(
            "position_encodings",
            get_position_embedding(self.max_seq_len, d_model))


        self.encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                norm_first=True,
                activation=F.relu, 
                # batch_first=True, 
                position_encodings=self.position_encodings,
                dropout=dropout, 
                use_relative_positions=self.use_relative_positions)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model))

        # self.embed_dim = 128
        self.embedder = nn.Embedding(num_embeddings=30, embedding_dim=d_model) #100
        # self.token_embedding = self.clip_model.token_embedding
        
        # self.embedder = Embedder(self.embed_dim, 100)
        
        # self.pos_embedding = PositionalEmbedding(d_model)

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(d_model),
        #     nn.Linear(d_model, n_classes)
        # )
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, self.head_size),
            nn.GELU(),
            nn.Linear(self.head_size, 5)
        )

        self.vis_embed = nn.Parameter(torch.randn(1, 1, d_model))
        # self.cls = nn.Parameter(torch.randn((1, 1, d_model)))
        self.lang_embed = nn.Parameter(torch.randn(1, 1, d_model))
    
        self.vision_proj = nn.Linear(self.d_monet_latent, d_model)

        self.shuffle_objects = True

        # self.pos_embed = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, d_model))

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
        # images_input = einops.rearrange(
        #     images, 'b c n h w -> (b c) n h w', c=7)
        texts_input = einops.rearrange(texts, 'b c d -> (b c) d', c=11)

        # image_features = self.clip_model.encode_image(
        #     images_input)  # (batch_size * 7, 512)
        # (batch_size * 11, self.token_length, 512)
        image_features = images
        text_features = self.embedder(texts_input)

        # (batch_size, 11, self.token_length, 512)
        text_features = einops.rearrange(
            text_features, '(b c) t d -> b c t d', b=batch_size, c=11)
        # print(image_features.shape, text_features.shape)
        if self.training and self.shuffle_objects:
            # image_features = einops.rearrange(
            # image_features, '(b c o) d -> b c o d', b=batch_size, c=7, o=1)
            image_features = image_features[:, :, torch.randperm(
                image_features.shape[2]), :]
        image_features = einops.rearrange(
            image_features, 'b c o d -> b (c o) d')
        # print(image_features.shape, text_features.shape)
        image_features = self.vision_proj(image_features)
        # print(image_features.shape, text_features.shape)
        image_features = einops.rearrange(
            image_features, 'b (c o) d -> b c o d', c=7)
        
        image_features += self.vis_embed
        text_features += self.lang_embed

        # inputs like cls, image_1, text_1s, sep, image_2, text_2s, sep, ..., image_6, sep, text 6s, text 7s, ..., text 11s
        image_contexts = image_features[:, :6, :, :]  # (batch_size, 6, 512)
        # (batch_size, 6, self.token_length, 512)
        text_contexts = text_features[:, :6, :, :]
        # context_inputs should be (batch_size, 6, 1 + self.token_length + 1, d_model)
        # print(image_contexts.shape, text_contexts.shape)
        context_inputs = torch.cat([image_contexts,
                                    text_contexts,
                                    self.sep_token.expand(batch_size, 6, 1, -1)
                                    ], dim=2)  # (batch_size, 6, 1 + self.token_length + 1, d_model)
        # print(context_inputs.shape)
        # random permute
        if self.training:
            context_inputs = context_inputs[:, torch.randperm(6), :, :]

        context_inputs = einops.rearrange(
            context_inputs, 'b c l d -> b (c l) d') # (batch_size, 6 * (self.token_length + 2), d_model)

        # print(context_inputs.shape)

        image_query = image_features[:, 6:, :, :]  # (batch_size, 1, obj_num, 512)
        text_query = text_features[:, 6:, :, :]   # (batch_size, 5, self.token_length, 512)
        image_query = einops.rearrange(
            image_query, 'b c t d -> b (c t) d')
        text_query = einops.rearrange(
            text_query, 'b c t d -> b (c t) d') # (batch_size, 5 * self.token_length, 512)

    
        transformer_inputs = torch.cat([self.cls_token.expand(batch_size, 1, -1),
                                        context_inputs, image_query,
                                        self.sep_token.expand(batch_size, 1, -1), text_query], dim=1)
        # (batch_size, 6 * (self.token_length + 2) + 1 + 5 * self.token_length, d_model)

        # print(transformer_inputs.shape)
        # exit(0)

        # transformer_inputs = transformer_inputs
        #  + \
        #     self.pos_embedding(transformer_inputs)

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


# class Aloe(pl.LightningModule):
#     def __init__(self, *, clip_model, d_model=512, num_layers=6, nhead=8, dim_feedforward=1024, n_classes=5, dropout=0.1, **kwargs):
#         super().__init__()
        
#         MAX_SEQ_LEN = 2048
        
#         self.clip_model = clip.load(clip_model)[0]
#         for param in self.clip_model.parameters():
#             param.requires_grad = False

#         self.d_model = d_model

#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, activation='gelu', dropout=dropout, norm_first=True)

#         self.transformer = nn.TransformerEncoder(
#             encoder_layer=self.encoder_layer, num_layers=num_layers)

#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
#         self.sep_token = nn.Parameter(torch.randn(1, 1, d_model))

#         self.pos_embedding = PositionalEmbedding(d_model)

#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, n_classes)
#         )

#         self.vis_embed = nn.Parameter(torch.randn(1, 1, d_model))
#         self.cls = nn.Parameter(torch.randn((1, 1, d_model)))
#         self.lang_embed = nn.Parameter(torch.randn(1, 1, d_model))
    
#         self.vision_proj = nn.Linear(d_model, d_model)

#         self.pos_embed = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, d_model))

#         self.train_accs = nn.ModuleList([torchmetrics.Accuracy(
#             task='multiclass', num_classes=n_classes) for _ in range(len(task_names))])
#         self.val_accs = nn.ModuleList([torchmetrics.Accuracy(
#             task='multiclass', num_classes=n_classes) for _ in range(len(task_names))])
#         self.test_accs = nn.ModuleList([torchmetrics.Accuracy(
#             task='multiclass', num_classes=n_classes) for _ in range(len(task_names))])

#         # save hyperparam
#         self.save_hyperparameters()

#     def forward(self, images, texts):
#         '''
#             images: (batch_size, 6 + 1, 3, 128, 128)
#             texts: (batch_size, 6 + 5, 77)
#         '''
#         batch_size = images.shape[0]
#         images_input = einops.rearrange(
#             images, 'b c n h w -> (b c) n h w', c=7)
#         texts_input = einops.rearrange(texts, 'b c d -> (b c) d', c=11)

#         image_features = self.clip_model.encode_image(
#             images_input)  # (batch_size * 7, 512)
#         text_features = self.clip_model.encode_text(
#             texts_input)  # (batch_size * 11, 512)

#         image_features = einops.rearrange(
#             image_features, '(b c) d -> b c o d', b=batch_size, c=7, o=1)
#         if self.shuffle_objects:
#             image_features = image_features[:, :, torch.randperm(
#                 image_features.shape[2]), :]
#         image_features = self.vision_proj(image_features)
#         text_features = einops.rearrange(
#             text_features, '(b c) d -> b c d', b=batch_size, c=11)

#         # inputs like cls, image_1, text_1, sep, image_2, text_2, sep, ..., image_6, sep, text 6, text 7, ..., text 11
#         image_contexts = image_features[:, :6, :]
#         text_contexts = text_features[:, :6, :]
#         context_inputs = torch.stack([image_contexts, text_contexts, self.sep_token.expand(
#             batch_size, 6, -1)], dim=2)  # (batch_size, 6, 3, d_model)
#         # random permute
#         if self.training:
#             context_inputs = context_inputs[:, torch.randperm(6), :, :]

#         context_inputs = einops.rearrange(
#             context_inputs, 'b c d e -> b (c d) e')  # (batch_size, 18, d_model)

#         image_query = image_features[:, 6:, :]
#         text_query = text_features[:, 6:, :]

#         transformer_inputs = torch.cat([self.cls_token.expand(batch_size, 1, -1),
#                                         context_inputs, image_query,
#                                         self.sep_token.expand(batch_size, 1, -1), text_query], dim=1)  # (batch_size, 18 + 6 + 5, d_model)

#         transformer_inputs = transformer_inputs + \
#             self.pos_embedding(transformer_inputs)

#         transformer_outputs = self.transformer(transformer_inputs)

#         cls_output = transformer_outputs[:, 0, :]

#         return self.mlp_head(cls_output)

#     def training_step(self, batch, batch_idx):
#         images, texts, labels, task_idx = batch
#         logits = self(images, texts)
#         loss = F.cross_entropy(logits, labels)
#         self.log('train_loss', loss, sync_dist=True)

#         for i in range(len(task_names)):
#             task_mask = (task_idx == i)
#             if task_mask.sum() > 0:
#                 task_logits = logits[task_mask]
#                 task_labels = labels[task_mask]
#                 acc = self.train_accs[i](task_logits, task_labels)

#         return loss

#     def validation_step(self, batch, batch_idx):
#         images, texts, labels, task_idx = batch
#         logits = self(images, texts)
#         loss = F.cross_entropy(logits, labels)
#         self.log('val_loss', loss, sync_dist=True)

#         for i in range(len(task_names)):
#             task_mask = (task_idx == i)
#             if task_mask.sum() > 0:
#                 task_logits = logits[task_mask]
#                 task_labels = labels[task_mask]
#                 acc = self.val_accs[i](task_logits, task_labels)

#         return loss

#     def test_step(self, batch, batch_idx):
#         images, texts, labels, task_idx = batch
#         logits = self(images, texts)
#         loss = F.cross_entropy(logits, labels)
#         self.log('test_loss', loss, sync_dist=True)

#         for i in range(len(task_names)):
#             task_mask = (task_idx == i)
#             if task_mask.sum() > 0:
#                 task_logits = logits[task_mask]
#                 task_labels = labels[task_mask]
#                 acc = self.test_accs[i](task_logits, task_labels)

#         return loss

#     def training_epoch_end(self, outputs):
#         for i in range(len(task_names)):
#             self.log(
#                 f'train_acc_{task_names[i]}', self.train_accs[i].compute(), sync_dist=True)

#     def validation_epoch_end(self, outputs):
#         for i in range(len(task_names)):
#             self.log(f'val_acc_{task_names[i]}',
#                      self.val_accs[i].compute(), sync_dist=True)

#     def test_epoch_end(self, outputs):
#         for i in range(len(task_names)):
#             self.log(
#                 f'test_acc_{task_names[i]}', self.test_accs[i].compute(), sync_dist=True)

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             self.parameters(), lr=1e-4, weight_decay=0.01)
#         return optimizer


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
