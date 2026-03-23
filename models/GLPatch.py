import torch.nn as nn

from layers.decomp import DECOMP
from layers.network_glpatch import GLPatchNetwork
from layers.revin import RevIN


class Model(nn.Module):
    """
    GLPatch v9.1: patching (seasonal) + XLinear TGM/mean-pool-VGM (trend).

    New config args vs v8:
        d_model  (int): trend embedding dim,        default 64
        t_ff     (int): TGM+VGM GatingBlock hidden, default 2*d_model
        (c_ff removed — mean-pool VGM needs no bottleneck hyperparameter)
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

        d_model = getattr(configs, 'd_model', 64)
        t_ff    = getattr(configs, 't_ff',    2 * d_model)

        self.net = GLPatchNetwork(
            seq_len, pred_len, patch_len, stride, padding_patch,
            channel=c_in,
            d_model=d_model,
            t_ff=t_ff,
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
