import torch.nn as nn

from layers.decomp import DECOMP
from layers.network_glpatch_tgm_only import GLPatchNetwork
from layers.revin import RevIN


class Model(nn.Module):
    """
    GLPatch TGM-only ablation wrapper.

    Runs TGM trend stream with no VGM (use_vgm=False).
    Seasonal stream and fusion identical to v9.

    Purpose: isolate whether ETT regressions in v9 vs v8 come from
    replacing the xPatch MLP trend with TGM, or from the VGM itself.

    Drop-in replacement — same configs as v8/v9.
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
            use_vgm=False,      # TGM only — no cross-channel interaction
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
