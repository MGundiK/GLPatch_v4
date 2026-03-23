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


class PatchCrossChannel(nn.Module):
    """
    Cross-channel gating in patch embedding space — v9 addition.

    WHY THIS PLACEMENT IS CORRECT
    ------------------------------
    The v9.0 VGM failed because it operated on pred_len-dimensional prediction
    outputs. XLinear's cross-channel interaction happens in compact d_model
    embedding space BEFORE the prediction head — that is why it works and
    scales. GLPatch's natural equivalent is the patch embedding space:
    after fc1/gelu/bn1, each channel has a [patch_num, dim] representation
    where dim = patch_len² = 256.

    Operating here gives us:
      - GatingBlock size = 2*dim = 512 — constant regardless of C.
        C=7 (ETTm1) and C=862 (Traffic) see identical module sizes.
        No hardcoded thresholds, no scaling problems.
      - Rich features: 256-dim embeddings carry more cross-channel signal
        than 16-dim post-CNN features or pred_len-dim predictions.
      - Temporal locality: cross-channel gating at patch level, so each
        patch position independently decides how much to use global context.
        More fine-grained than mixing at the prediction level.

    HOW IT WORKS (matching XLinear's VGM philosophy)
    -------------------------------------------------
    1. Each channel's patch embedding is enriched with a global context
       vector — the mean of all channels' embeddings at each patch position.
       Mean-pool is stable in 256-dim space (even for C=7) unlike in 1- or
       16-dim prediction space.
    2. GatingBlock(2*dim) gates [own_embedding, global_context] — each
       patch token decides per-feature how much global context to absorb.
    3. Zero-initialized out_proj: module starts as exact identity (same as
       v8). Cross-channel signal is added only as training discovers it
       useful. Cannot hurt at initialization.

    Args:
        dim (int): patch embedding dimension (patch_len² = 256 by default)
    """
    def __init__(self, dim):
        super(PatchCrossChannel, self).__init__()
        # GatingBlock size 2*dim — constant, C-independent
        self.gating = GatingBlock(2 * dim, dim)
        # Zero-init: starts as identity residual
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, s, B, C):
        """
        s:       [B*C, patch_num, dim]
        B, C:    batch size and channel count captured before B*C flatten
        Returns: [B*C, patch_num, dim]  (same shape, cross-channel enriched)
        """
        BC, N, D = s.shape

        # Expose channel dimension
        s_4d = s.reshape(B, C, N, D)                           # [B, C, N, D]

        # Global context: mean over channels at each patch position
        # Stable in D=256-dim space regardless of C
        glob = s_4d.mean(dim=1, keepdim=True).expand_as(s_4d) # [B, C, N, D]

        # Each token gates on [own_embedding, global_context]
        combined = torch.cat([s_4d, glob], dim=-1)             # [B, C, N, 2D]
        combined_flat = combined.reshape(B * C * N, 2 * D)

        gated = self.gating(combined_flat)                     # [B*C*N, 2D]
        gated = gated.reshape(B, C, N, 2 * D)

        # Own-embedding half after gating — what this channel retains
        cross_info = gated[..., :D]                            # [B, C, N, D]

        # Zero-init residual: delta starts at 0, grows as training finds use
        delta = self.out_proj(cross_info)                      # [B, C, N, D]

        return (s_4d + delta).reshape(BC, N, D)


# ============================================================
# Main network
# ============================================================

class GLPatchNetwork(nn.Module):
    """
    GLPatch v9 — patch-level cross-channel gating on v8's backbone.

    Single architectural addition over v8
    --------------------------------------
    PatchCrossChannel (PCCM): inserted after patch embedding (fc1/gelu/bn1)
    in the seasonal stream. Each channel's patch embeddings are enriched with
    a global cross-channel context before the depthwise CNN processes them.

    This mirrors XLinear's design philosophy:
      - XLinear: cross-channel interaction in d_model embedding space
      - GLPatch v9: cross-channel interaction in patch embedding space (dim=256)

    Both operate BEFORE the prediction head in compact embedding space.
    This is fundamentally different from v9.0 which operated on pred_len
    prediction outputs — the wrong space.

    Key properties:
      - GatingBlock size = 2*dim = 512, independent of C (C=7 or C=862, same)
      - Zero-init residual: v9 == v8 at initialization
      - Placed before depthwise CNN: cross-channel signal informs local
        temporal feature extraction, not just the final prediction

    Everything else is identical to v8:
      - Inter-patch gating (GLCN, pre-pointwise, alpha=0.05)
      - Bottleneck fusion (H→32→H, gate constrained [0.1, 0.9], same inits)
      - All other layers, norms, activations

    Args:
        seq_len     (int): input sequence length
        pred_len    (int): prediction horizon
        patch_len   (int): patch size
        stride      (int): patch stride
        padding_patch (str): 'end' to pad input before patching
        channel     (int): number of input channels C
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 channel=1):
        super(GLPatchNetwork, self).__init__()

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len          # 256 for patch_len=16
        self.patch_num = (seq_len - patch_len) // stride + 1
        self.channel = channel

        # ================================================================
        # Non-linear Stream (Seasonality)
        # ================================================================
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch embedding (from xPatch)
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # [v9] Cross-channel gating in patch embedding space
        # Placed here: after embedding, before CNN, in 256-dim space
        # Only meaningful when C > 1
        self.use_pccm = (channel > 1)
        if self.use_pccm:
            self.pccm = PatchCrossChannel(self.dim)

        # CNN Depthwise (from xPatch)
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual (from xPatch)
        self.fc2 = nn.Linear(self.dim, patch_len)

        # Inter-patch gating (v8, unchanged — within-channel, pre-pointwise)
        self.inter_patch_gate = InterPatchGating(self.patch_num, reduction=4)
        self.res_alpha = nn.Parameter(torch.tensor(0.05))

        # CNN Pointwise (from xPatch)
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten head (from xPatch)
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
        s = s.permute(0, 2, 1)   # [B, C, T]
        t = t.permute(0, 2, 1)

        B, C, I = s.shape

        s = s.reshape(B * C, I)
        t = t.reshape(B * C, I)

        # ---- Non-linear Stream ----
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [B*C, patch_num, patch_len]

        # Patch embedding → [B*C, patch_num, dim=256]
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        # [v9] Cross-channel gating in 256-dim embedding space
        # Zero-init: starts as identity. B and C passed to reshape back.
        if self.use_pccm:
            s = self.pccm(s, B, C)

        res = s

        # Depthwise CNN: [B*C, patch_num, dim] → [B*C, patch_num, patch_len]
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual: fc2 maps dim → patch_len to match conv1 output
        res = self.fc2(res)
        s = s + res

        # Inter-patch gating (v8): within-channel, pre-pointwise
        s_base = s
        s_gated = self.inter_patch_gate(s)
        s = s_base + self.res_alpha * (s_gated - s_base)

        # Pointwise CNN
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten head → [B*C, pred_len]
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # ---- Linear Stream ----
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)
        t = self.fc7(t)

        # ---- Bottleneck Fusion (v8) ----
        gate = torch.sigmoid(
            self.gate_expand(
                self.gate_compress_s(s) + self.gate_compress_t(t)
            )
        )
        gate = gate * 0.8 + 0.1   # constrain to [0.1, 0.9]

        x = gate * s + (1 - gate) * t
        x = self.fc8(x)

        x = x.reshape(B, C, self.pred_len)
        x = x.permute(0, 2, 1)
        return x
