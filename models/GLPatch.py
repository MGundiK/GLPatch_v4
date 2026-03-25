import torch
import torch.nn as nn

from layers.decomp import DECOMP
# from layers.network_glpatch_v5 import GLPatchNetwork # use version 5
# from layers.network_glpatch_v9 import GLPatchNetwork # use version 9

from layers.network_glpatch import GLPatchNetwork # use  version 8
from layers.revin import RevIN


class Model(nn.Module):
    """
    GLPatch: Global-Local Patch model for Long-Term Time Series Forecasting.
    
    Combines the best of xPatch (EMA decomposition, dual-stream, overlapping
    patches, depthwise-separable convolutions) with GLCN innovations (aggregate
    Conv1D, inter-patch gating, multiscale local convolutions).
    
    Drop-in replacement for xPatch — uses identical configs, decomposition,
    RevIN normalization, and training pipeline.
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        c_in = configs.enc_in

        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha
        beta = configs.beta

        self.decomp = DECOMP(self.ma_type, alpha, beta)
        self.net = GLPatchNetwork(seq_len, pred_len, patch_len, stride,
                                  padding_patch)

    def forward(self, x):
        # x: [Batch, Input, Channel]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x
