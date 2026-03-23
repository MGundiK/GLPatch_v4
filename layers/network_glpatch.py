import torch
from torch import nn


# ============================================================
# Primitives
# ============================================================

class GatingBlock(nn.Module):
    """
    XLinear-style gating block.
    Linear → ReLU → Linear → Sigmoid → ⊗ input.
    Used in both TGM and VGM.
    """
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


class TrendStream(nn.Module):
    """
    XLinear-style trend stream: Linear embedding + TGM + VGM + head.

    Replaces xPatch's 3-layer MLP + AvgPool trend stream with XLinear's
    full temporal+variate gating pipeline. This is the core architectural
    contribution of v9 — the seasonal stream (patching) is unchanged.

    TGM (Time-wise Gating Module)
    ------------------------------
    Each channel's linear embedding is concatenated with a learnable global
    token [1, C, d_model], initialized to ones (following XLinear exactly).
    GatingBlock gates across the 2*d_model temporal dimension, learning which
    features to amplify and suppress. Output is split back into:
      - origin_atten: temporally-enriched embedding per channel
      - glob_atten:   updated global token carrying temporal context

    VGM (Variate-wise Gating Module)
    ----------------------------------
    The original embedding and updated global token are stacked along the
    channel dimension [B, 2C, d_model], permuted to [B, d_model, 2C], and
    passed through a GatingBlock(2C, c_ff). This gates across channels at
    every feature position — each of the d_model features independently
    decides how much cross-channel information to absorb.

    The cross-channel half of the output is concatenated with origin_atten
    to form the final per-channel representation [B, C, 2*d_model], which
    the prediction head maps to [B, C, pred_len].

    Efficiency
    ----------
    TGM: O(B × C × d_model²) — cheap, d_model is small (default 64)
    VGM: O(B × d_model × C²) — c_ff is capped (default min(2C, 256))
         For Traffic C=862: B=4, d_model=64, C=862 → 4×64×1724 = 441K ops
         vs PCCM's B×C×N×D² = 4×862×90×512 = 160M ops. 360× faster.

    Args:
        seq_len  (int): input sequence length
        pred_len (int): prediction horizon
        channel  (int): number of input channels C
        d_model  (int): trend embedding dimension (default 64)
        t_ff     (int): TGM GatingBlock hidden dim (default 2*d_model)
        c_ff     (int): VGM GatingBlock hidden dim (default min(2C, 256))
    """
    def __init__(self, seq_len, pred_len, channel, d_model, t_ff, c_ff):
        super(TrendStream, self).__init__()
        self.d_model = d_model
        self.channel = channel

        # Linear projection: full sequence → compact embedding
        self.projection = nn.Linear(seq_len, d_model)

        # Learnable global token — one per channel, ones init (XLinear)
        self.glob_token = nn.Parameter(torch.ones(1, channel, d_model))

        # TGM: gate on [emb, glob_token] in 2*d_model space
        self.tgm = GatingBlock(2 * d_model, t_ff)

        # VGM: gate across channels in 2C space (permuted to [B, d_model, 2C])
        self.vgm = GatingBlock(2 * channel, c_ff)

        # Prediction head: [B, C, 2*d_model] → [B, C, pred_len]
        self.head = nn.Linear(2 * d_model, pred_len)

    def forward(self, t):
        """
        t: [B, C, seq_len]
        returns: [B*C, pred_len]
        """
        B, C, _ = t.shape

        # Linear embed: [B, C, d_model]
        emb = self.projection(t)

        # TGM ─────────────────────────────────────────────────
        glob = self.glob_token.expand(B, -1, -1)            # [B, C, d_model]
        en_emb = torch.cat([emb, glob], dim=-1)             # [B, C, 2*d_model]
        en_atten = self.tgm(en_emb)                         # [B, C, 2*d_model]
        origin_atten = en_atten[:, :, :self.d_model]        # [B, C, d_model]
        glob_atten   = en_atten[:, :, self.d_model:]        # [B, C, d_model]

        # VGM ─────────────────────────────────────────────────
        # Stack original emb with updated global token along channel dim
        ex_emb  = torch.cat([emb, glob_atten], dim=1)       # [B, 2C, d_model]
        ex_atten = self.vgm(ex_emb.permute(0, 2, 1))        # [B, d_model, 2C]
        glob_cross = ex_atten[:, :, C:]                     # [B, d_model, C]

        # Combine temporal + cross-channel → [B, C, 2*d_model]
        en = torch.cat(
            [origin_atten, glob_cross.permute(0, 2, 1)], dim=-1
        )

        # Head → [B, C, pred_len] → [B*C, pred_len]
        return self.head(en).reshape(B * C, -1)


# ============================================================
# Main network
# ============================================================

class GLPatchNetwork(nn.Module):
    """
    GLPatch v9 — patching + XLinear TGM/VGM hybrid.

    Architecture
    ------------
    SEASONAL STREAM  (local temporal patterns — GLPatch strength)
        Patching → embed → depthwise CNN → inter-patch gating →
        pointwise CNN → MLP head
        Channel-independent throughout, O(B*C) complexity.

    TREND STREAM  (global context + cross-channel — XLinear strength)
        Linear(seq_len → d_model) → TGM → VGM → head
        TGM: per-channel temporal gating with learnable global token
        VGM: cross-channel variate-wise gating using updated global token
        Replaces xPatch's 3-layer MLP + AvgPool trend stream.

    FUSION  (v8 bottleneck, proven stable)
        H→32→H bottleneck gate, constrained [0.1, 0.9]

    Why this division of labour
    ---------------------------
    Patching is excellent at local temporal patterns but has two weaknesses:
    (1) no natural way to model global sequence context — patches are local
        windows, and there's no attention or global token to summarize the
        full history; (2) channel independence means no cross-channel signal.
    XLinear's TGM+VGM directly addresses both: the global token + TGM
    captures full-sequence temporal context, VGM handles cross-channel.
    By putting XLinear where xPatch's weak 3-layer MLP trend stream was,
    we fix both weaknesses without touching the seasonal stream at all.

    New hyperparameters vs v8
    -------------------------
        d_model  (int): trend embedding dim, default 64
        t_ff     (int): TGM hidden dim, default 2*d_model
        c_ff     (int): VGM hidden dim, default min(2*channel, 256)

    Args:
        seq_len   (int): input sequence length
        pred_len  (int): prediction horizon
        patch_len (int): patch length
        stride    (int): patch stride
        padding_patch (str): 'end' to pad before patching
        channel   (int): number of input channels C
        d_model   (int): trend embedding dimension
        t_ff      (int): TGM GatingBlock hidden dim
        c_ff      (int): VGM GatingBlock hidden dim
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 channel=1, d_model=64, t_ff=None, c_ff=None):
        super(GLPatchNetwork, self).__init__()

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1

        # Resolve defaults
        if t_ff is None:
            t_ff = 2 * d_model
        if c_ff is None:
            c_ff = min(2 * channel, 256)

        # ================================================================
        # Seasonal stream — identical to v8
        # ================================================================
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        self.fc1   = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1   = nn.BatchNorm1d(self.patch_num)

        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2   = nn.BatchNorm1d(self.patch_num)

        self.fc2 = nn.Linear(self.dim, patch_len)

        self.inter_patch_gate = InterPatchGating(self.patch_num, reduction=4)
        self.res_alpha = nn.Parameter(torch.tensor(0.05))

        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3   = nn.BatchNorm1d(self.patch_num)

        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3   = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4   = nn.Linear(pred_len * 2, pred_len)

        # ================================================================
        # Trend stream — XLinear TGM + VGM (replaces xPatch MLP)
        # ================================================================
        self.trend_stream = TrendStream(
            seq_len, pred_len, channel, d_model, t_ff, c_ff
        )

        # ================================================================
        # Bottleneck fusion — identical to v8
        # ================================================================
        gate_hidden = min(32, pred_len)
        self.gate_compress_s = nn.Linear(pred_len, gate_hidden)
        self.gate_compress_t = nn.Linear(pred_len, gate_hidden)
        self.gate_expand     = nn.Linear(gate_hidden, pred_len)

        nn.init.normal_(self.gate_compress_s.weight, std=0.01)
        nn.init.normal_(self.gate_compress_t.weight, std=0.01)
        nn.init.normal_(self.gate_expand.weight,     std=0.01)
        nn.init.zeros_(self.gate_compress_s.bias)
        nn.init.zeros_(self.gate_compress_t.bias)
        nn.init.zeros_(self.gate_expand.bias)

        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        # s: [B, T, C]  t: [B, T, C]
        s = s.permute(0, 2, 1)   # [B, C, T]
        t = t.permute(0, 2, 1)   # [B, C, T]

        B, C, I = s.shape

        # ---- Trend stream (needs B,C separate for VGM) ----
        t_out = self.trend_stream(t)   # [B*C, pred_len]

        # ---- Seasonal stream (channel-independent, flatten to B*C) ----
        s = s.reshape(B * C, I)

        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        s = self.fc1(s);   s = self.gelu1(s);  s = self.bn1(s)
        res = s
        s = self.conv1(s); s = self.gelu2(s);  s = self.bn2(s)
        res = self.fc2(res)
        s = s + res

        s_base  = s
        s_gated = self.inter_patch_gate(s)
        s = s_base + self.res_alpha * (s_gated - s_base)

        s = self.conv2(s); s = self.gelu3(s);  s = self.bn3(s)
        s = self.flatten1(s)
        s = self.fc3(s);   s = self.gelu4(s);  s = self.fc4(s)  # [B*C, P]

        # ---- Bottleneck fusion (v8) ----
        gate = torch.sigmoid(
            self.gate_expand(
                self.gate_compress_s(s) + self.gate_compress_t(t_out)
            )
        )
        gate = gate * 0.8 + 0.1   # constrain to [0.1, 0.9]

        x = gate * s + (1 - gate) * t_out
        x = self.fc8(x)

        x = x.reshape(B, C, self.pred_len)
        x = x.permute(0, 2, 1)   # [B, pred_len, C]
        return x
