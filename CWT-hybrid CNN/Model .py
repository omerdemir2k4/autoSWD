#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model.py — CWT‑Hybrid CNN + Multi‑Kernel Conv1D with SE Attention (Seq→Token)

This module defines the exact architecture used in our study. It processes
(L, 32, 32, 2) sequences of time–frequency patches, encodes each channel via a
shared lightweight 2‑D CNN, aggregates temporal context using multi‑kernel
Conv1D branches, applies Squeeze‑and‑Excitation (SE) channel attention, and
outputs per‑time‑step probabilities with a TimeDistributed classifier.

Import in training/eval scripts:
    from model import ModelConfig, build_model
    cfg = ModelConfig(sequence_len=100, proj_dim=128, kernel_sizes=(51, 7))
    model = build_model(cfg)
    model.summary()

Author: Bekir Arda Yıldırım
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Dense, Dropout, TimeDistributed, Conv1D, Conv2D,
    BatchNormalization, MaxPooling2D, Flatten, Lambda, Concatenate,
    GlobalAveragePooling1D, Multiply, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    sequence_len: int = 100               # tokens per sequence (L)
    proj_dim: int = 128                   # latent dim per token
    kernel_sizes: Tuple[int, ...] = (51, 7)
    dropout: float = 0.30
    l2_reg: float = 1e-3
    learning_rate: float = 1e-3
    input_height: int = 32
    input_width: int = 32
    input_channels: int = 2               # expects 2 channels

    def input_shape(self) -> Tuple[int, int, int, int]:
        return (self.sequence_len, self.input_height, self.input_width, self.input_channels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def f1_metric(y_true, y_pred):
    y_true_f = K.cast(y_true, 'float32')
    y_pred_l = K.cast(K.round(y_pred), 'float32')
    tp = K.sum(y_true_f * y_pred_l)
    fp = K.sum((1. - y_true_f) * y_pred_l)
    fn = K.sum(y_true_f * (1. - y_pred_l))
    precision = tp / (tp + fp + K.epsilon())
    recall    = tp / (tp + fn + K.epsilon())
    return 2. * (precision * recall) / (precision + recall + K.epsilon())


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def build_cnn_encoder(input_shape=(32, 32, 1), proj_dim=128, l2_reg=1e-3, dropout=0.30) -> Model:
    """Lightweight CNN encoder used per token & per channel (shared).

    Conv(32,3) → BN → MaxPool(2) → Conv(64,3) → BN → MaxPool(2) → Flatten →
    Dense(proj_dim) → Dropout
    """
    inp = Input(shape=input_shape)
    x = Conv2D(32, 3, activation='swish', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(64, 3, activation='swish', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)

    x = Flatten()(x)
    x = Dense(proj_dim, activation='swish', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout)(x)
    return Model(inp, x, name='cnn_encoder')


def squeeze_excite_block(x, ratio: int = 16):
    se = GlobalAveragePooling1D(name='se_squeeze')(x)
    reduced = max(1, x.shape[-1] // ratio)
    se = Dense(reduced, activation='swish', use_bias=False, name='se_reduce')(se)
    se = Dense(x.shape[-1], activation='sigmoid', use_bias=False, name='se_expand')(se)
    se = Reshape((1, x.shape[-1]))(se)
    return Multiply(name='se_reweight')([x, se])


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig) -> Model:
    """Build full Seq→Token architecture.

    Input  : (L, 32, 32, 2)
    Output : (L, 1) token‑level probabilities (sigmoid)
    """
    assert cfg.input_channels == 2, "This implementation expects 2 channels."

    inp = Input(cfg.input_shape(), name='input_seq_patches')

    # split channels: (..., 2) → (..., 1) × 2
    ch0 = Lambda(lambda t: t[..., 0:1], name='split_ch0')(inp)
    ch1 = Lambda(lambda t: t[..., 1:2], name='split_ch1')(inp)

    # shared encoder per token
    encoder = build_cnn_encoder(
        input_shape=(cfg.input_height, cfg.input_width, 1),
        proj_dim=cfg.proj_dim, l2_reg=cfg.l2_reg, dropout=cfg.dropout
    )
    enc0 = TimeDistributed(encoder, name='td_enc_ch0')(ch0)  # (L, proj_dim)
    enc1 = TimeDistributed(encoder, name='td_enc_ch1')(ch1)  # (L, proj_dim)

    # multi‑kernel temporal aggregation
    branches = []
    for k in cfg.kernel_sizes:
        c0 = Conv1D(cfg.proj_dim, k, padding='same', activation='swish',
                    kernel_regularizer=l2(cfg.l2_reg), name=f'conv1d_ch0_k{k}')(enc0)
        c1 = Conv1D(cfg.proj_dim, k, padding='same', activation='swish',
                    kernel_regularizer=l2(cfg.l2_reg), name=f'conv1d_ch1_k{k}')(enc1)
        branches.append(Dropout(cfg.dropout, name=f'drop_ch0_k{k}')(c0))
        branches.append(Dropout(cfg.dropout, name=f'drop_ch1_k{k}')(c1))

    # concatenate temporal features + raw encodings (minimal skip concat)
    x = Concatenate(axis=-1, name='concat_multi_kernel')([*branches, enc0, enc1])

    # channel attention over temporal feature maps
    x = squeeze_excite_block(x)

    # token‑wise head
    x = TimeDistributed(
        Dense(cfg.proj_dim, activation='swish', kernel_regularizer=l2(cfg.l2_reg)),
        name='td_dense'
    )(x)
    x = Dropout(cfg.dropout, name='td_dropout')(x)

    out = TimeDistributed(
        Dense(1, activation='sigmoid', kernel_regularizer=l2(cfg.l2_reg)),
        name='td_logits'
    )(x)

    model = Model(inp, out, name='cwt_hybrid_cnn_seq2token')
    model.compile(optimizer=Adam(cfg.learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1_metric])
    return model


# ---------------------------------------------------------------------------
# Standalone summary
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = ModelConfig()
    model = build_model(cfg)
    model.summary(line_length=120)
