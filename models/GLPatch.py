import torch
import torch.nn as nn

from layers.decomp import DECOMP
# from layers.network_glpatch_v5 import GLPatchNetwork  # version 5
# from layers.network_glpatch import GLPatchNetwork     # version 8

from layers.network_glpatch import GLPatchNetwork    # version 9
from layers.revin import RevIN


class Model(nn.Module):
    """
    GLPatch v9: Global-Local Patch model with cross-channel variate-wise gating.

    Builds on GLPatch v8 (xPatch + inter-patch gating + bottleneck fusion) with:
      1. VariateWiseGating (VGM): XLinear-inspired cross-channel interaction
         after both streams are computed, using trend output as global token.
      2. Trend-as-global-token: principled hub for cross-channel information,
         grounded in the EMA decomposition (trend = slow persistent signal).
      3. Full GatingBlock fusion: per-dimension gating on cat([seasonal, trend])
         replaces the constrained bottleneck (H→32→H, g∈[0.1,0.9]).

    Drop-in replacement for v8 — identical configs, decomposition, RevIN,
    training pipeline.  Only addition: passes enc_in to GLPatchNetwork.
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        seq_len = configs.seq_len
        pred_len = configs.pred_len
        c_in = configs.enc_in

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        self.revin = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        self.ma_type = configs.ma_type
        alpha = configs.alpha
        beta = configs.beta

        self.decomp = DECOMP(self.ma_type, alpha, beta)

        # vgm_hidden: cap on VGM GatingBlock hidden dim.
        # Default 256 keeps Traffic (C=862) and Electricity (C=321) efficient.
        # Can be overridden via configs.vgm_hidden if needed.
        # NEW
        vgm_emb_dim = getattr(configs, 'vgm_emb_dim', 64)
        
        self.net = GLPatchNetwork(
            seq_len, pred_len, patch_len, stride, padding_patch,
            channel=c_in,
        )

    def forward(self, x):
        # x: [Batch, Input, Channel]

        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x
