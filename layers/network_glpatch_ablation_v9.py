import torch
from torch import nn


# ============================================================
# Primitives
# ============================================================

class GatingBlock(nn.Module):
    """
    XLinear-style gating block (v9 addition).
    Linear → ReLU → Linear → Sigmoid → ⊗ input.
    """
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super(GatingBlock, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class InterPatchGating(nn.Module):
    """GLCN-inspired inter-patch gating module (unchanged from v8)."""
    def __init__(self, patch_num, reduction=4):
        super(InterPatchGating, self).__init__()
        hidden = max(patch_num // reduction, 2)
        self.mlp = nn.Sequential(
            nn.Linear(patch_num, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, patch_num),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = x.mean(dim=2)
        w = self.mlp(w).unsqueeze(2)
        return x * w


class VariateWiseGating(nn.Module):
    """
    XLinear-inspired Variate-Wise Gating Module (VGM) — v9 addition.

    Uses trend output as global token for cross-channel seasonal interaction.
    vgm_hidden caps the GatingBlock hidden dim for large-channel datasets.
    """
    def __init__(self, channel, pred_len, vgm_hidden=256):
        super(VariateWiseGating, self).__init__()
        d_variate = 2 * channel
        hf = min(d_variate, vgm_hidden)
        self.gating = GatingBlock(d_variate, hf)
        self.proj = nn.Linear(2 * pred_len, pred_len)

    def forward(self, s, t_glob):
        """s, t_glob: [B, C, pred_len] → returns s_enhanced [B, C, pred_len]"""
        B, C, P = s.shape
        ex_emb = torch.cat([s, t_glob], dim=1)            # [B, 2C, P]
        ex_atten = self.gating(ex_emb.permute(0, 2, 1))   # [B, P, 2C]
        ex_atten = ex_atten.permute(0, 2, 1)              # [B, 2C, P]
        cross_info = ex_atten[:, C:, :]                   # [B, C, P]
        s_enhanced = torch.cat([s, cross_info], dim=-1)   # [B, C, 2P]
        return self.proj(s_enhanced)                      # [B, C, P]


# ============================================================
# Ablation network
# ============================================================

class GLPatchNetworkAblation(nn.Module):
    """
    GLPatch ablation network supporting v8 and v9 components.

    Ablation controls (v8 — existing)
    ----------------------------------
        use_gating:              Enable inter-patch gating
        use_fusion:              Enable adaptive stream fusion
        gate_position:           'pre_depthwise' | 'pre_pointwise' | 'post_pointwise'
        res_alpha_init:          Initial value for gating residual blend
        gate_hidden_dim:         Bottleneck dim for v8 fusion gate (-1 = full-rank)
        gate_min / gate_max:     Constraint bounds for v8 gate
        gate_reduction:          Reduction ratio for inter-patch gating MLP

    Ablation controls (v9 — new, default OFF to preserve v8 behavior)
    -------------------------------------------------------------------
        use_vgm:                 Enable Variate-Wise Gating (cross-channel)
        use_gating_block_fusion: Replace v8 bottleneck with GatingBlock fusion
        channel:                 Number of input channels (required for VGM)
        vgm_hidden:              Cap on VGM GatingBlock hidden dim (default 256)

    Typical ablation matrix for v9 components:
        A: use_vgm=F, use_gating_block_fusion=F  →  v8 baseline
        B: use_vgm=T, use_gating_block_fusion=F  →  VGM only
        C: use_vgm=F, use_gating_block_fusion=T  →  GatingBlock fusion only
        D: use_vgm=T, use_gating_block_fusion=T  →  full v9

    Combined with existing v8 ablation flags:
        use_gating=F, use_fusion=F                →  pure xPatch baseline
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 # v8 ablation flags
                 use_gating=True, use_fusion=True,
                 gate_position='pre_pointwise',
                 res_alpha_init=0.05,
                 gate_hidden_dim=32,
                 gate_min=0.1, gate_max=0.9,
                 gate_reduction=4,
                 # v9 ablation flags (default OFF → v8 behavior)
                 use_vgm=False,
                 use_gating_block_fusion=False,
                 channel=1,
                 vgm_hidden=256):
        super(GLPatchNetworkAblation, self).__init__()

        # Store flags
        self.use_gating = use_gating
        self.use_fusion = use_fusion
        self.gate_position = gate_position
        self.gate_min = gate_min
        self.gate_max = gate_max
        self.use_vgm = use_vgm and (channel > 1)
        self.use_gating_block_fusion = use_gating_block_fusion
        self.channel = channel

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1

        # ================================================================
        # Non-linear Stream (Seasonality)
        # ================================================================

        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        self.fc2 = nn.Linear(self.dim, patch_len)

        # Inter-patch gating (conditional on use_gating)
        if use_gating:
            self.inter_patch_gate = InterPatchGating(self.patch_num,
                                                     reduction=gate_reduction)
            self.res_alpha = nn.Parameter(torch.tensor(float(res_alpha_init)))

        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # ================================================================
        # Linear Stream (Trend)
        # ================================================================
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # ================================================================
        # [v9] Variate-Wise Gating (conditional on use_vgm)
        # ================================================================
        if self.use_vgm:
            self.vgm = VariateWiseGating(channel, pred_len, vgm_hidden)

        # ================================================================
        # Stream Fusion
        # ================================================================
        if use_fusion:
            if use_gating_block_fusion:
                # [v9] Full GatingBlock fusion
                self.fusion_gate = GatingBlock(pred_len * 2, pred_len)
                self.fusion_proj = nn.Linear(pred_len * 2, pred_len)
            else:
                # [v8] Bottleneck fusion gate
                if gate_hidden_dim == -1:
                    # Full-rank gate (ablation: no bottleneck)
                    self.gate_fc = nn.Linear(pred_len * 2, pred_len)
                    nn.init.normal_(self.gate_fc.weight, std=0.01)
                    nn.init.zeros_(self.gate_fc.bias)
                    self.full_rank = True
                else:
                    gate_hidden = min(gate_hidden_dim, pred_len)
                    self.gate_compress_s = nn.Linear(pred_len, gate_hidden)
                    self.gate_compress_t = nn.Linear(pred_len, gate_hidden)
                    self.gate_expand = nn.Linear(gate_hidden, pred_len)
                    nn.init.normal_(self.gate_compress_s.weight, std=0.01)
                    nn.init.normal_(self.gate_compress_t.weight, std=0.01)
                    nn.init.normal_(self.gate_expand.weight, std=0.01)
                    nn.init.zeros_(self.gate_compress_s.bias)
                    nn.init.zeros_(self.gate_compress_t.bias)
                    nn.init.zeros_(self.gate_expand.bias)
                    self.full_rank = False

        self.fc8 = nn.Linear(pred_len, pred_len)

    def _apply_inter_patch_gating(self, s):
        s_base = s
        s_gated = self.inter_patch_gate(s)
        return s_base + self.res_alpha * (s_gated - s_base)

    def forward(self, s, t):
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        B, C, I = s.shape
        s = torch.reshape(s, (B * C, I))
        t = torch.reshape(t, (B * C, I))

        # ---- Non-linear Stream ----

        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        if self.use_gating and self.gate_position == 'pre_depthwise':
            s = self._apply_inter_patch_gating(s)

        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        res = self.fc2(res)
        s = s + res

        if self.use_gating and self.gate_position == 'pre_pointwise':
            s = self._apply_inter_patch_gating(s)

        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        if self.use_gating and self.gate_position == 'post_pointwise':
            s = self._apply_inter_patch_gating(s)

        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)             # s: [B*C, pred_len]

        # ---- Linear Stream ----
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)
        t = self.fc7(t)             # t: [B*C, pred_len]

        # ---- [v9] Variate-Wise Gating ----
        if self.use_vgm:
            s_3d = s.reshape(B, C, self.pred_len)
            t_3d = t.reshape(B, C, self.pred_len)
            s_3d = self.vgm(s_3d, t_3d)
            s = s_3d.reshape(B * C, self.pred_len)

        # ---- Stream Fusion ----
        if self.use_fusion:
            if self.use_gating_block_fusion:
                # [v9] GatingBlock fusion
                combined = torch.cat([s, t], dim=-1)
                fused = self.fusion_gate(combined)
                x = self.fusion_proj(fused)
            else:
                # [v8] Bottleneck fusion
                if self.full_rank:
                    gate = torch.sigmoid(
                        self.gate_fc(torch.cat([s, t], dim=-1)))
                else:
                    gate = torch.sigmoid(
                        self.gate_expand(
                            self.gate_compress_s(s) + self.gate_compress_t(t)
                        )
                    )
                gate_range = self.gate_max - self.gate_min
                gate = gate * gate_range + self.gate_min
                x = gate * s + (1 - gate) * t
        else:
            # Static fusion (xPatch style)
            x = s + t

        x = self.fc8(x)

        x = torch.reshape(x, (B, C, self.pred_len))
        x = x.permute(0, 2, 1)

        return x
