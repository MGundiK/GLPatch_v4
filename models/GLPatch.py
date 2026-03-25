import torch.nn as nn

from layers.decomp import DECOMP
from layers.network_glpatch import GLPatchNetwork
from layers.revin import RevIN


class Model(nn.Module):
    """
    GLPatch v9.2: xPatch MLP trend stream + mean-pool VGM.

    Identical to v8 except for one addition: VariateWiseGating applied to
    the MLP trend output before fusion. The MLP trend stream is unchanged.

    New config arg vs v8:
        vgm_ff (int): VGM GatingBlock hidden dim, default = pred_len
                      Scales with pred_len only, not with C.
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

        vgm_ff = getattr(configs, 'vgm_ff', 64)

        self.net = GLPatchNetwork(
            seq_len, pred_len, patch_len, stride, padding_patch,
            channel=c_in,
            vgm_ff=vgm_ff,
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
