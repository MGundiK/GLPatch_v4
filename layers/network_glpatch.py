import torch
from torch import nn


# ============================================================
# Primitives
# ============================================================

class GatingBlock(nn.Module):
    """XLinear-style gating block. Linear → ReLU → Linear → Sigmoid → ⊗ input."""
    def __init__(self, d_model, hidden_dim):
        super(GatingBlock, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class InterPatchGating(nn.Module):
    """GLCN-inspired inter-patch gating (unchanged from v8)."""
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
    Variate-Wise Gating Module (VGM) — v9 addition.

    WHY THE FIRST ATTEMPT FAILED
    -----------------------------
    The previous VGM operated on pred_len-dimensional prediction outputs with
    a Linear(2C) layer gating across all channels. For Traffic (C=862) this
    meant a 1724-dim linear layer operating directly on predictions — it could
    only inject noise. More fundamentally, XLinear's VGM operates in compact
    d_model embedding space BEFORE the prediction head, not on pred_len outputs.

    THIS REDESIGN
    -------------
    1. Compact embedding space:
       Projects pred_len → vgm_emb_dim (default 64) before cross-channel
       interaction. GatingBlock sees 2*vgm_emb_dim vectors — small and constant
       regardless of C. Mirrors XLinear operating on d_model embeddings.

    2. Scalable cross-channel aggregation:
       Uses mean-pooled trend as global context hub rather than a Linear(2C)
       layer. Each channel gates on [own_seasonal_emb, global_trend_context].
       GatingBlock size = 2*E regardless of whether C=7 or C=862.

    3. Zero-initialized residual:
       out_proj starts at zero — VGM begins as exact identity, adds cross-
       channel signal gradually as training proceeds. No destabilization at init.
    """
    def __init__(self, pred_len, vgm_emb_dim=64):
        super(VariateWiseGating, self).__init__()
        self.vgm_emb_dim = vgm_emb_dim

        self.s_proj = nn.Linear(pred_len, vgm_emb_dim)
        self.t_proj = nn.Linear(pred_len, vgm_emb_dim)
        # GatingBlock on [own_seasonal_emb, global_trend_ctx]: size 2E, constant
        self.gating = GatingBlock(2 * vgm_emb_dim, vgm_emb_dim)
        # Zero-init: VGM starts as identity residual
        self.out_proj = nn.Linear(vgm_emb_dim, pred_len)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, s, t_glob):
        """s, t_glob: [B, C, pred_len] → returns [B, C, pred_len]"""
        s_emb = self.s_proj(s)                                  # [B, C, E]
        t_emb = self.t_proj(t_glob)                             # [B, C, E]

        # Mean-pool trend across channels → global context hub
        global_ctx = t_emb.mean(dim=1, keepdim=True).expand_as(t_emb)  # [B, C, E]

        # Each channel gates independently on its own + global context
        combined = torch.cat([s_emb, global_ctx], dim=-1)      # [B, C, 2E]
        gated = self.gating(combined)                           # [B, C, 2E]
        cross_info = gated[..., :self.vgm_emb_dim]             # [B, C, E]

        # Zero-init residual: starts at 0, learns cross-channel delta
        delta = self.out_proj(cross_info)                       # [B, C, pred_len]
        return s + delta


# ============================================================
# Main network
# ============================================================

class GLPatchNetwork(nn.Module):
    """
    GLPatch v9 — VGM added to v8's proven backbone.

    One change over v8
    ------------------
    VariateWiseGating (VGM): after both streams compute per-channel predictions,
    each channel enriches its seasonal output with a global trend-derived
    cross-channel context, before the v8 bottleneck fusion.

    Everything else is IDENTICAL to v8:
    - Inter-patch gating (GLCN, pre-pointwise, alpha=0.05)
    - Bottleneck fusion (H→32→H, gate constrained [0.1, 0.9], same init)
    - Same weight inits, same architecture

    Note: GatingBlock stream fusion (improvement 3 from original plan) is NOT
    included. The v8 bottleneck fusion has careful initialization that keeps
    training stable; the GatingBlock fusion used PyTorch default init which
    destabilized early training. Revert to v8 fusion only.

    Args:
        seq_len     (int): input sequence length
        pred_len    (int): prediction horizon
        patch_len   (int): patch size
        stride      (int): patch stride
        padding_patch (str): 'end' to pad input before patching
        channel     (int): number of input channels C
        vgm_emb_dim (int): embedding dim for VGM cross-channel interaction (default 64)
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 channel=1, vgm_emb_dim=64):
        super(GLPatchNetwork, self).__init__()

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1

        # ================================================================
        # Non-linear Stream (Seasonality) — identical to v8
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

        self.inter_patch_gate = InterPatchGating(self.patch_num, reduction=4)
        self.res_alpha = nn.Parameter(torch.tensor(0.05))

        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # ================================================================
        # Linear Stream (Trend) — identical to xPatch / v8
        # ================================================================
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # ================================================================
        # [v9] Variate-Wise Gating
        # ================================================================
        self.use_vgm = (channel > 1)
        if self.use_vgm:
            self.vgm = VariateWiseGating(pred_len, vgm_emb_dim)

        # ================================================================
        # Bottleneck Fusion — identical to v8
        # ================================================================
        gate_hidden = min(32, pred_len)
        self.gate_compress_s = nn.Linear(pred_len, gate_hidden)
        self.gate_compress_t = nn.Linear(pred_len, gate_hidden)
        self.gate_expand = nn.Linear(gate_hidden, pred_len)

        nn.init.normal_(self.gate_compress_s.weight, std=0.01)
        nn.init.normal_(self.gate_compress_t.weight, std=0.01)
        nn.init.normal_(self.gate_expand.weight, std=0.01)
        nn.init.zeros_(self.gate_compress_s.bias)
        nn.init.zeros_(self.gate_compress_t.bias)
        nn.init.zeros_(self.gate_expand.bias)

        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        B, C, I = s.shape
        s = s.reshape(B * C, I)
        t = t.reshape(B * C, I)

        # ---- Non-linear Stream ----
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        s = self.fc1(s);  s = self.gelu1(s);  s = self.bn1(s)
        res = s
        s = self.conv1(s); s = self.gelu2(s); s = self.bn2(s)
        res = self.fc2(res)
        s = s + res

        s_base = s
        s_gated = self.inter_patch_gate(s)
        s = s_base + self.res_alpha * (s_gated - s_base)

        s = self.conv2(s); s = self.gelu3(s); s = self.bn3(s)
        s = self.flatten1(s)
        s = self.fc3(s);  s = self.gelu4(s);  s = self.fc4(s)  # [B*C, P]

        # ---- Linear Stream ----
        t = self.fc5(t);  t = self.avgpool1(t);  t = self.ln1(t)
        t = self.fc6(t);  t = self.avgpool2(t);  t = self.ln2(t)
        t = self.fc7(t)                                         # [B*C, P]

        # ---- [v9] Variate-Wise Gating ----
        if self.use_vgm:
            s_3d = s.reshape(B, C, self.pred_len)
            t_3d = t.reshape(B, C, self.pred_len)
            s_3d = self.vgm(s_3d, t_3d)
            s = s_3d.reshape(B * C, self.pred_len)

        # ---- Bottleneck Fusion (v8) ----
        gate = torch.sigmoid(
            self.gate_expand(
                self.gate_compress_s(s) + self.gate_compress_t(t)
            )
        )
        gate = gate * 0.8 + 0.1

        x = gate * s + (1 - gate) * t
        x = self.fc8(x)

        x = x.reshape(B, C, self.pred_len)
        x = x.permute(0, 2, 1)
        return x
