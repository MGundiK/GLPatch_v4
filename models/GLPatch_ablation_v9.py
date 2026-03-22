import torch
import torch.nn as nn

from layers.decomp import DECOMP
from layers.network_glpatch_ablation_v9 import GLPatchNetworkAblation
from layers.revin import RevIN


class Model(nn.Module):
    """
    GLPatch Ablation model wrapper — v9 extended.

    Supports all v8 ablation flags plus v9 additions:
        use_vgm                (default 0 = off → v8 behavior)
        use_gating_block_fusion(default 0 = off → v8 behavior)
        vgm_hidden             (default 256)

    Ablation matrix for v9 components:
        A: use_vgm=0, use_gating_block_fusion=0  →  v8 baseline
        B: use_vgm=1, use_gating_block_fusion=0  →  VGM only
        C: use_vgm=0, use_gating_block_fusion=1  →  GatingBlock fusion only
        D: use_vgm=1, use_gating_block_fusion=1  →  full v9

    Can also be combined with v8 flags:
        use_gating=0, use_fusion=0               →  pure xPatch baseline
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
        self.decomp = DECOMP(self.ma_type, configs.alpha, configs.beta)

        # ---- v8 ablation flags (defaults = v8 production values) ----
        use_gating = getattr(configs, 'use_gating', 1) == 1
        use_fusion = getattr(configs, 'use_fusion', 1) == 1
        gate_position = getattr(configs, 'gate_position', 'pre_pointwise')
        res_alpha_init = getattr(configs, 'res_alpha_init', 0.05)
        gate_hidden_dim = getattr(configs, 'gate_hidden_dim', 32)
        gate_min = getattr(configs, 'gate_min', 0.1)
        gate_max = getattr(configs, 'gate_max', 0.9)
        gate_reduction = getattr(configs, 'gate_reduction', 4)

        # ---- v9 ablation flags (defaults = OFF → preserves v8 behavior) ----
        use_vgm = getattr(configs, 'use_vgm', 0) == 1
        use_gating_block_fusion = getattr(
            configs, 'use_gating_block_fusion', 0) == 1
        vgm_hidden = getattr(configs, 'vgm_hidden', 256)

        self.net = GLPatchNetworkAblation(
            seq_len, pred_len, patch_len, stride, padding_patch,
            # v8 flags
            use_gating=use_gating,
            use_fusion=use_fusion,
            gate_position=gate_position,
            res_alpha_init=res_alpha_init,
            gate_hidden_dim=gate_hidden_dim,
            gate_min=gate_min,
            gate_max=gate_max,
            gate_reduction=gate_reduction,
            # v9 flags
            use_vgm=use_vgm,
            use_gating_block_fusion=use_gating_block_fusion,
            channel=c_in,
            vgm_hidden=vgm_hidden,
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
