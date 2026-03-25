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
    Mean-pool VGM operating on the MLP trend stream output.

    PARAMETER SCALING FIX (v9.3)
    ----------------------------
    Previous default: vgm_ff=pred_len → out_proj was Linear(pred_len, pred_len).
    At pred_len=720 this adds 2.6M parameters, causing overfit on small-C
    datasets (ETTh1 has ~8700 training samples with C=7).

    Fix: vgm_ff is capped at 64 by default. out_proj is now a bottleneck:
        pred_len → vgm_ff → pred_len   (two small linears, not one huge one)

    Parameter comparison (pred_len=720):
        Before: GatingBlock(1440,720) + Linear(720,720) = 2,594,880
        After:  GatingBlock(1440,64)  + Linear(720,64) + Linear(64,720) = 704,944

    The GatingBlock input is still 2*pred_len (full cross-channel context),
    only the bottleneck dimension is capped. This preserves representational
    richness in the gating while dramatically reducing out_proj parameters.

    ZERO-INIT RESIDUAL
    ------------------
    out_proj[-1] (the last linear) is zero-initialized: VGM starts as exact
    identity at initialization regardless of pred_len. Cannot hurt at init.

    Args:
        pred_len (int): prediction horizon
        vgm_ff   (int): bottleneck dim for GatingBlock and out_proj (default 64)
    """
    def __init__(self, pred_len, vgm_ff=64):
        super(VariateWiseGating, self).__init__()
        self.gating  = GatingBlock(2 * pred_len, vgm_ff)
        # Bottleneck projection: pred_len → vgm_ff → pred_len
        self.out_proj = nn.Sequential(
            nn.Linear(pred_len, vgm_ff),
            nn.GELU(),
            nn.Linear(vgm_ff, pred_len),
        )
        # Zero-init the last layer only — identity residual at init
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, t):
        """
        t: [B, C, pred_len]  →  [B, C, pred_len]  (residual addition)
        """
        # Mean-pool MLP outputs across channels → global forecast context
        global_ctx = t.mean(dim=1, keepdim=True).expand_as(t)  # [B, C, pred_len]

        # Each channel gates on [own_forecast, global_ctx]
        combined = torch.cat([t, global_ctx], dim=-1)           # [B, C, 2*pred_len]
        gated    = self.gating(combined)                        # [B, C, 2*pred_len]

        # Own-forecast half after gating
        cross_info = gated[..., :t.shape[-1]]                   # [B, C, pred_len]

        # Zero-init residual: starts at 0, grows as training finds cross-channel signal
        return t + self.out_proj(cross_info)                    # [B, C, pred_len]


# ============================================================
# Main network
# ============================================================

class GLPatchNetwork(nn.Module):
    """
    GLPatch v9.2 — xPatch MLP trend stream + mean-pool VGM.

    What changed vs v8
    ------------------
    One addition: VariateWiseGating (VGM) inserted after the xPatch trend
    stream MLP, before fusion. Everything else — seasonal stream, MLP trend
    stream, bottleneck fusion — is byte-for-byte identical to v8.

    Why this design (ablation conclusion)
    --------------------------------------
    The TGM+VGM approach in v9.0/v9.1 replaced xPatch's 3-layer MLP trend
    stream with TGM. Ablation showed the regression on ETTh/ETTm was caused
    by this replacement — TGM-only loses to v8 on ETT (5W 10L MSE). The
    xPatch MLP with AvgPool has better inductive bias for small-C datasets:
    progressive bottleneck (seq_len→4P→2P→P) and implicit low-pass filtering.

    The VGM itself was never the problem — v9mp beats TGM-only on ETTh/ETTm
    (partial recovery). So the fix is: keep the proven MLP, add VGM on top.

    Architecture
    ------------
    SEASONAL STREAM  (identical to v8):
        patch → embed → depthwise CNN → inter-patch gating →
        pointwise CNN → MLP head  →  s [B*C, pred_len]

    TREND STREAM (xPatch MLP, identical to v8):
        fc5 → AvgPool → LN → fc6 → AvgPool → LN → fc7  →  t [B*C, pred_len]

    VGM (new):
        reshape t to [B, C, pred_len]
        mean-pool across C → global_ctx [B, C, pred_len]
        GatingBlock(2*pred_len) on [own, global_ctx]
        zero-init residual addition
        reshape back to [B*C, pred_len]

    FUSION (identical to v8):
        bottleneck H→32→H, gate ∈ [0.1, 0.9]

    New hyperparameter vs v8
    ------------------------
        vgm_ff (int): VGM GatingBlock hidden dim, default pred_len
                      Only scales with pred_len, not C.
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 channel=1, vgm_ff=None):
        super(GLPatchNetwork, self).__init__()

        self.pred_len      = pred_len
        self.patch_len     = patch_len
        self.stride        = stride
        self.padding_patch = padding_patch
        self.dim           = patch_len * patch_len
        self.patch_num     = (seq_len - patch_len) // stride + 1

        # ── Seasonal stream — identical to v8 ─────────────────
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

        self.fc2   = nn.Linear(self.dim, patch_len)

        self.inter_patch_gate = InterPatchGating(self.patch_num, reduction=4)
        self.res_alpha        = nn.Parameter(torch.tensor(0.05))

        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3   = nn.BatchNorm1d(self.patch_num)

        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3   = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4   = nn.Linear(pred_len * 2, pred_len)

        # ── Trend stream — identical to xPatch / v8 ───────────
        self.fc5      = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1      = nn.LayerNorm(pred_len * 2)

        self.fc6      = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2      = nn.LayerNorm(pred_len // 2)

        self.fc7      = nn.Linear(pred_len // 2, pred_len)

        # ── VGM — cross-channel on MLP trend output ───────────
        # Only meaningful for multivariate; skip for C=1
        self.use_vgm = (channel > 1)
        if self.use_vgm:
            self.vgm = VariateWiseGating(pred_len, vgm_ff)

        # ── Bottleneck fusion — identical to v8 ───────────────
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
        s = s.permute(0, 2, 1)   # [B, C, T]
        t = t.permute(0, 2, 1)   # [B, C, T]

        B, C, I = s.shape

        s = s.reshape(B * C, I)
        t = t.reshape(B * C, I)

        # ── Seasonal stream ────────────────────────────────────
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
        s = self.fc3(s);   s = self.gelu4(s);  s = self.fc4(s)   # [B*C, P]

        # ── Trend stream (xPatch MLP, identical to v8) ─────────
        t = self.fc5(t);   t = self.avgpool1(t);  t = self.ln1(t)
        t = self.fc6(t);   t = self.avgpool2(t);  t = self.ln2(t)
        t = self.fc7(t)                                            # [B*C, P]

        # ── VGM — cross-channel enrichment of trend output ─────
        if self.use_vgm:
            t_3d = t.reshape(B, C, self.pred_len)   # [B, C, P]
            t_3d = self.vgm(t_3d)                   # [B, C, P] + cross-channel delta
            t    = t_3d.reshape(B * C, self.pred_len)

        # ── Bottleneck fusion (v8) ─────────────────────────────
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
