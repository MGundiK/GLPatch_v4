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


class TrendStream(nn.Module):
    """
    XLinear-inspired trend stream with mean-pool VGM.

    TGM (Time-wise Gating Module) — unchanged from v9.0
    -------------------------------------------------------
    Linear(seq_len → d_model) embeds each channel's full sequence.
    Concatenated with a learnable global token [1, C, d_model] (ones init),
    passed through GatingBlock(2*d_model, t_ff), then split into:
      - origin_atten: temporally-enriched per-channel embedding
      - glob_atten:   updated global token carrying temporal summary

    VGM (Variate-wise Gating Module) — redesigned
    -----------------------------------------------
    Previous design: cat([emb, glob_atten], dim=1) → [B, 2C, d_model]
    → permute → GatingBlock(2C, c_ff). This required c_ff ≥ 2C to avoid
    lossy bottleneck. Traffic (2C=1724) with c_ff=256 threw away 85% of
    cross-channel information, causing +5-10% MAE regression.

    New design: mean-pool glob_atten across channels → global context hub.
    Each channel gates on [own_origin_atten, global_ctx] independently.
    GatingBlock size = 2*d_model — constant, C-independent.

    Why mean-pool is correct here (unlike in pred_len space):
      - glob_atten is d_model-dimensional (64) per channel, not scalar.
        Mean over C of 64-dim vectors is a stable, information-rich summary
        regardless of whether C=7 or C=862.
      - The TGM has already encoded each channel's temporal structure into
        glob_atten. Mean-pooling these summaries gives a meaningful global
        temporal context, not just noise.
      - Each channel gates independently: [own_temporal, global_temporal]
        via GatingBlock(2*d_model). This is the same size as TGM — no new
        hyperparameter, no C-dependent scaling.

    Complexity:
      TGM: O(B × C × d_model²)  — same as before
      VGM: O(B × C × d_model²)  — same as TGM, replaces O(B × d_model × C²)
      For Traffic C=862: ~26M ops vs previous ~480M ops. 18× cheaper.
      No c_ff hyperparameter needed.

    Args:
        seq_len  (int): input sequence length
        pred_len (int): prediction horizon
        channel  (int): number of input channels C
        d_model  (int): embedding dimension
        t_ff     (int): GatingBlock hidden dim for both TGM and VGM
    """
    def __init__(self, seq_len, pred_len, channel, d_model, t_ff):
        super(TrendStream, self).__init__()
        self.d_model = d_model
        self.channel = channel

        # Full-sequence linear projection per channel
        self.projection = nn.Linear(seq_len, d_model)

        # Learnable global token — ones init (XLinear)
        self.glob_token = nn.Parameter(torch.ones(1, channel, d_model))

        # TGM: temporal gating on [emb, glob_token]
        self.tgm = GatingBlock(2 * d_model, t_ff)

        # VGM: per-channel gating on [own_temporal, global_ctx]
        # Size 2*d_model — constant, independent of C
        self.vgm = GatingBlock(2 * d_model, t_ff)

        # Prediction head: [B, C, 2*d_model] → [B, C, pred_len]
        self.head = nn.Linear(2 * d_model, pred_len)

    def forward(self, t):
        """t: [B, C, seq_len]  →  returns [B*C, pred_len]"""
        B, C, _ = t.shape

        # Linear embed: [B, C, d_model]
        emb = self.projection(t)

        # ── TGM ──────────────────────────────────────────────
        glob = self.glob_token.expand(B, -1, -1)            # [B, C, d_model]
        en_emb   = torch.cat([emb, glob], dim=-1)           # [B, C, 2*d_model]
        en_atten = self.tgm(en_emb)                         # [B, C, 2*d_model]
        origin_atten = en_atten[:, :, :self.d_model]        # [B, C, d_model]
        glob_atten   = en_atten[:, :, self.d_model:]        # [B, C, d_model]

        # ── VGM (mean-pool) ───────────────────────────────────
        # Global context: mean of updated global tokens across channels
        # Stable in d_model=64 space regardless of C (no bottleneck)
        global_ctx = glob_atten.mean(dim=1, keepdim=True)   # [B, 1, d_model]
        global_ctx = global_ctx.expand_as(origin_atten)     # [B, C, d_model]

        # Each channel gates independently on [own_temporal, global_ctx]
        vgm_in  = torch.cat([origin_atten, global_ctx], dim=-1)  # [B, C, 2*d_model]
        vgm_out = self.vgm(vgm_in)                               # [B, C, 2*d_model]

        # ── Head ─────────────────────────────────────────────
        # [B, C, pred_len] → [B*C, pred_len]
        return self.head(vgm_out).reshape(B * C, -1)


# ============================================================
# Main network
# ============================================================

class GLPatchNetwork(nn.Module):
    """
    GLPatch v9.1 — patching + XLinear TGM/mean-pool-VGM hybrid.

    Architecture
    ------------
    SEASONAL STREAM  (local temporal patterns — GLPatch strength)
        patch → embed → depthwise CNN → inter-patch gating →
        pointwise CNN → MLP head  [B*C, pred_len]
        Channel-independent, O(B*C). Identical to v8.

    TREND STREAM  (global context + cross-channel — XLinear strength)
        Linear(seq_len → d_model) → TGM → mean-pool VGM → head
        [B*C, pred_len]
        GatingBlock size = 2*d_model for both TGM and VGM.
        Scales to any C with no bottleneck.

    FUSION  (v8 bottleneck, proven stable)
        H→32→H, gate constrained [0.1, 0.9]

    Hyperparameters vs v8
    ---------------------
        d_model  (int): trend embedding dim, default 64
        t_ff     (int): GatingBlock hidden dim for TGM+VGM, default 2*d_model
        (c_ff removed — no longer needed with mean-pool VGM)
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 channel=1, d_model=64, t_ff=None):
        super(GLPatchNetwork, self).__init__()

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1

        if t_ff is None:
            t_ff = 2 * d_model

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
        # Trend stream — TGM + mean-pool VGM
        # ================================================================
        self.trend_stream = TrendStream(
            seq_len, pred_len, channel, d_model, t_ff
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
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        B, C, I = s.shape

        # ---- Trend stream ----
        t_out = self.trend_stream(t)        # [B*C, pred_len]

        # ---- Seasonal stream (CI) ----
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
        gate = gate * 0.8 + 0.1

        x = gate * s + (1 - gate) * t_out
        x = self.fc8(x)

        x = x.reshape(B, C, self.pred_len)
        x = x.permute(0, 2, 1)
        return x
