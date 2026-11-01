# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FluidInference möbius, models/stt/parakeet-ctc-0.6b-ja/coreml/conversion/individual_components.py
# (https://github.com/FluidInference/mobius), Copyright FluidInference, licensed under the Apache License 2.0
# (https://www.apache.org/licenses/LICENSE-2.0). Changed: the encoder and the CTC head are fused into one module
# (one Core ML call per window), the head is AI4Bharat's multilingual ConvASRDecoder (5,633 outputs, the
# language mask applied by the caller), and the input is an 80-bin mel with an int32 length.
"""Torch wrappers that trace cleanly for Core ML."""
from __future__ import annotations

import torch


class EncoderCTCWrapper(torch.nn.Module):
    """mel [1, 80, T] + length [1] (int32) → raw CTC logits [1, T/8, 5633].

    Bypasses the CTC decoder's forward(), which masks per language and applies log_softmax: möbius found
    log_softmax inside the head gives garbage values in Core ML, and greedy decoding only needs an argmax over
    the language's columns, which the host takes.
    """

    def __init__(self, encoder: torch.nn.Module, ctc_decoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.ctc_decoder = ctc_decoder

    def forward(self, mel: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(audio_signal=mel, length=length.to(dtype=torch.long))
        return self.ctc_decoder.decoder_layers(encoded).transpose(1, 2)


@torch.no_grad()
def rescale_residual_stream(encoder: torch.nn.Module, scale: float) -> int:
    """Scale the Conformer's residual stream by `scale` without changing what the encoder computes (ours, not möbius).

    Measured in Torch on real speech, the stream reaches |x| ≈ 380 at the layer norms, so x² overflows fp16
    (65,504). Tried with scale 1/16 against the Neural Engine's 2 % frame-label drift: no change (98.2 % → 98.25 %),
    so that drift is not layer-norm overflow and the option is off by default. Every layer norm is scale-invariant, so with
    the stream multiplied by `scale` everywhere the output is the same up to the norms' epsilon:
      - the input: pos_enc.xscale (√d_model) × scale
      - each branch's last projection (feed_forward1/2.linear2, self_attn.linear_out, conv.pointwise_conv2),
        weight and bias × scale, so what it adds to the stream is scaled
      - each layer's norm_out weight and bias × scale, except the last layer's, so the encoder output is unchanged.
    Returns the number of tensors changed.
    """
    changed = 0
    # RelPositionalEncoding multiplies its input by .xscale when set (√d_model here; None when xscaling is off).
    encoder.pos_enc.xscale = (encoder.pos_enc.xscale or 1.0) * scale
    layers = list(encoder.layers)
    for i, layer in enumerate(layers):
        for mod in (layer.feed_forward1.linear2, layer.feed_forward2.linear2, layer.self_attn.linear_out, layer.conv.pointwise_conv2):
            mod.weight.mul_(scale)
            if mod.bias is not None:
                mod.bias.mul_(scale)
            changed += 1
        if i < len(layers) - 1:
            layer.norm_out.weight.mul_(scale)
            layer.norm_out.bias.mul_(scale)
            changed += 1
    return changed
