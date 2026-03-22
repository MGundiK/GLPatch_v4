import torch
from torch import nn


# ============================================================
# Primitives
# ============================================================

class GatingBlock(nn.Module):
    """
    XLinear-style gating block.

    MLP with sigmoid produces a per-dimension multiplicative gate applied to
    the input. Richer than a scalar gate — every feature dimension gets its
    own context-dependent suppression / amplification.

    Architecture: Linear → ReLU → (Dropout) → Linear → Sigmoid → ⊗ input
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
    """
    GLCN-inspired inter-patch gating module (unchanged from v8).

    Captures global patch-level dynamics:
    GlobalAvgPool (over features) → MLP → Sigmoid → element-wise scaling.
    """
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
        # x: [B*C, patch_num, patch_len]
        w = x.mean(dim=2)           # GAP over features → [B*C, patch_num]
        w = self.mlp(w).unsqueeze(2)  # weights → [B*C, patch_num, 1]
        return x * w


class VariateWiseGating(nn.Module):
    """
    XLinear-inspired Variate-Wise Gating Module (VGM) — v9 addition.

    Breaks channel independence by letting every channel's seasonal forecast
    attend to all other channels via a trend-derived global token.

    Design choices:
      - trend output t [B, C, P] serves as the global token (hub), since the
        trend stream captures slow persistent structure — a principled choice
        grounded in the EMA decomposition.
      - GatingBlock operates on the variate dimension (2C), so each prediction
        timestep decides independently how to mix cross-channel information.
      - Hidden dim is capped at vgm_hidden (default 256) so Traffic/Electricity
        (C=862/321) don't blow up parameter counts.
      - Output projection fuses [temporal, cross-variate] back to pred_len.

    Args:
        channel   (int): number of input channels C
        pred_len  (int): prediction horizon P
        vgm_hidden(int): cap on GatingBlock hidden dim (default 256)
    """
    def __init__(self, channel, pred_len, vgm_hidden=256):
        super(VariateWiseGating, self).__init__()
        d_variate = 2 * channel
        hf = min(d_variate, vgm_hidden)
        # GatingBlock sees [B, pred_len, 2C] — gates across the variate dim
        self.gating = GatingBlock(d_variate, hf)
        # Fuse temporal + cross-variate into pred_len
        self.proj = nn.Linear(2 * pred_len, pred_len)

    def forward(self, s, t_glob):
        """
        s:       seasonal features  [B, C, pred_len]
        t_glob:  trend global token [B, C, pred_len]

        Returns: s_enhanced [B, C, pred_len]
        """
        B, C, P = s.shape

        # Stack seasonal + trend-glob along channel dim
        ex_emb = torch.cat([s, t_glob], dim=1)            # [B, 2C, P]

        # Variate-wise gating: transpose so GatingBlock gates across 2C
        ex_atten = self.gating(ex_emb.permute(0, 2, 1))   # [B, P, 2C]
        ex_atten = ex_atten.permute(0, 2, 1)              # [B, 2C, P]

        # Extract cross-variate information carried in the trend-glob half
        cross_info = ex_atten[:, C:, :]                   # [B, C, P]

        # Fuse: concatenate temporal + cross-variate, project back
        s_enhanced = torch.cat([s, cross_info], dim=-1)   # [B, C, 2P]
        s_enhanced = self.proj(s_enhanced)                # [B, C, P]

        return s_enhanced


# ============================================================
# Main network
# ============================================================

class GLPatchNetwork(nn.Module):
    """
    GLPatch v9 — XLinear-inspired cross-channel gating on top of v8.

    Three improvements over v8
    --------------------------
    1. VariateWiseGating (VGM):
       After both streams are computed, reshapes back to [B, C, pred_len] and
       applies cross-channel gating before fusion. Directly addresses Solar,
       Traffic, and Weather long-horizon failures caused by channel independence.

    2. Trend-as-global-token:
       The VGM uses the trend stream output as its global hub token rather
       than a learned ones-initialised parameter. Trend captures slow, persistent
       per-channel structure — a principled choice aligned with the EMA
       decomposition philosophy. No extra parameters required.

    3. Full GatingBlock stream fusion:
       Replaces the constrained bottleneck gate (H→32→H, g∈[0.1,0.9]) with
       XLinear's GatingBlock on cat([s, t]). Per-dimension gating, no
       hardcoded constraint, hidden dim = pred_len (still regularised vs v7's
       full-rank Linear(pred_len, pred_len) gates).

    Everything else is identical to v8.

    Args:
        seq_len    (int): input sequence length
        pred_len   (int): prediction horizon
        patch_len  (int): patch size
        stride     (int): patch stride
        padding_patch (str): 'end' to pad input before patching
        channel    (int): number of input channels C (needed for VGM)
        vgm_hidden (int): cap on VGM GatingBlock hidden dim (default 256)
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 channel=1, vgm_hidden=256):
        super(GLPatchNetwork, self).__init__()

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1
        self.channel = channel

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

        # Inter-patch gating (v8, unchanged)
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
        # [v9 — Improvement 1+2] Variate-Wise Gating
        # Only active when C > 1; for C=1 the module is a no-op (VGM with
        # a single channel degenerates to self-gating, harmless but pointless,
        # so we skip it for univariate datasets).
        # ================================================================
        self.use_vgm = (channel > 1)
        if self.use_vgm:
            self.vgm = VariateWiseGating(channel, pred_len, vgm_hidden)

        # ================================================================
        # [v9 — Improvement 3] Full GatingBlock Stream Fusion
        # Replaces: bottleneck (H→32→H) + hardcoded constraint [0.1, 0.9]
        # With:     GatingBlock(2*pred_len, pred_len) + linear projection
        #
        # Parameter comparison (pred_len=720):
        #   v8:  (720*32 + 32*720)*2 + 32*720 = ~115K
        #   v9:  2*(1440*720) + (1440*720) = ~3.1M
        # Still far less than v5-v7 full-rank gates (~1M for fusion alone
        # at pred_len=720), and the GatingBlock's ReLU bottleneck provides
        # natural regularisation without a hardcoded constraint.
        # ================================================================
        self.fusion_gate = GatingBlock(pred_len * 2, pred_len)
        self.fusion_proj = nn.Linear(pred_len * 2, pred_len)

        # Final projection (v8 style)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        # s: seasonality [B, T, C]
        # t: trend       [B, T, C]

        s = s.permute(0, 2, 1)   # [B, C, T]
        t = t.permute(0, 2, 1)   # [B, C, T]

        B, C, I = s.shape

        # Flatten channels for per-channel (CI) processing
        s = s.reshape(B * C, I)   # [B*C, T]
        t = t.reshape(B * C, I)   # [B*C, T]

        # ---- Non-linear Stream ----

        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        res = self.fc2(res)
        s = s + res

        # Inter-patch gating (v8)
        s_base = s
        s_gated = self.inter_patch_gate(s)
        s = s_base + self.res_alpha * (s_gated - s_base)

        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

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
        # Break out of B*C to enable cross-channel interaction.
        # Trend output serves as the global token (hub) — principled by
        # decomposition: trend = slow global signal per channel.
        if self.use_vgm:
            s_3d = s.reshape(B, C, self.pred_len)   # [B, C, P]
            t_3d = t.reshape(B, C, self.pred_len)   # [B, C, P]

            s_3d = self.vgm(s_3d, t_3d)            # [B, C, P]

            s = s_3d.reshape(B * C, self.pred_len)  # back to [B*C, P]

        # ---- [v9] Full GatingBlock Stream Fusion ----
        # Per-dimension gating on cat([s, t]) — no hardcoded constraint.
        combined = torch.cat([s, t], dim=-1)        # [B*C, 2*pred_len]
        fused = self.fusion_gate(combined)           # [B*C, 2*pred_len]
        x = self.fusion_proj(fused)                 # [B*C, pred_len]

        x = self.fc8(x)

        x = x.reshape(B, C, self.pred_len)
        x = x.permute(0, 2, 1)                     # [B, pred_len, C]

        return x
