import torch.nn as nn

from layers.decomp import DECOMP
from layers.network_glpatch import GLPatchNetwork
from layers.revin import RevIN


class Model(nn.Module):
    """
    GLPatch v9: patching (seasonal) + XLinear TGM/VGM (trend) hybrid.

    New config args vs v8 (all optional — safe defaults provided):
        d_model  (int): trend embedding dim            default 64
        t_ff     (int): TGM GatingBlock hidden dim     default 2*d_model
        c_ff     (int): VGM GatingBlock hidden dim     default min(2*enc_in, 256)
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        seq_len       = configs.seq_len
        pred_len      = configs.pred_len
        c_in          = configs.enc_in
        patch_len     = configs.patch_len
        stride        = configs.stride
        padding_patch = configs.padding_patch

        self.revin       = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        self.ma_type = configs.ma_type
        self.decomp  = DECOMP(self.ma_type, configs.alpha, configs.beta)

        # New v9 hyperparameters — read from configs with safe defaults
        d_model = getattr(configs, 'd_model', 64)
        t_ff    = getattr(configs, 't_ff',    2 * d_model)
        c_ff    = getattr(configs, 'c_ff',    min(2 * c_in, 256))

        self.net = GLPatchNetwork(
            seq_len, pred_len, patch_len, stride, padding_patch,
            channel=c_in,
            d_model=d_model,
            t_ff=t_ff,
            c_ff=c_ff,
        )

    def forward(self, x):
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
