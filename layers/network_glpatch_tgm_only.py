import torch
from torch import nn


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
    Trend stream with configurable VGM for ablation.

    use_vgm=True  → full v9: TGM + mean-pool VGM, head input = 2*d_model
    use_vgm=False → TGM only:             no VGM, head input =   d_model

    TGM parameters are IDENTICAL in both cases — the only difference is
    whether origin_atten is enriched with cross-channel context before the
    head. This isolates the VGM contribution cleanly.

    When use_vgm=False:
      - glob_atten is computed by TGM (needed to update glob_token) but
        discarded — it is NOT passed to the head.
      - head receives only origin_atten [B, C, d_model].
      - This is the correct ablation: TGM temporal processing intact,
        cross-channel interaction surgically removed.
    """
    def __init__(self, seq_len, pred_len, channel, d_model, t_ff,
                 use_vgm=True):
        super(TrendStream, self).__init__()
        self.d_model  = d_model
        self.channel  = channel
        self.use_vgm  = use_vgm

        self.projection = nn.Linear(seq_len, d_model)
        self.glob_token = nn.Parameter(torch.ones(1, channel, d_model))
        self.tgm        = GatingBlock(2 * d_model, t_ff)

        if use_vgm:
            self.vgm  = GatingBlock(2 * d_model, t_ff)
            head_in   = 2 * d_model
        else:
            head_in   = d_model          # origin_atten only

        self.head = nn.Linear(head_in, pred_len)

    def forward(self, t):
        """t: [B, C, seq_len]  →  [B*C, pred_len]"""
        B, C, _ = t.shape

        emb = self.projection(t)                                # [B, C, d_model]

        # TGM — always runs
        glob     = self.glob_token.expand(B, -1, -1)           # [B, C, d_model]
        en_emb   = torch.cat([emb, glob], dim=-1)              # [B, C, 2*d_model]
        en_atten = self.tgm(en_emb)                            # [B, C, 2*d_model]
        origin_atten = en_atten[:, :, :self.d_model]           # [B, C, d_model]
        glob_atten   = en_atten[:, :, self.d_model:]           # [B, C, d_model]

        if self.use_vgm:
            # Mean-pool VGM
            global_ctx = glob_atten.mean(dim=1, keepdim=True).expand_as(origin_atten)
            vgm_in  = torch.cat([origin_atten, global_ctx], dim=-1)  # [B, C, 2*d_model]
            vgm_out = self.vgm(vgm_in)                               # [B, C, 2*d_model]
            return self.head(vgm_out).reshape(B * C, -1)
        else:
            # TGM only — cross-channel interaction removed
            return self.head(origin_atten).reshape(B * C, -1)


class GLPatchNetwork(nn.Module):
    """
    GLPatch ablation network: TGM-only vs full v9 (TGM + VGM).

    Controlled by use_vgm flag:
        use_vgm=True  → identical to production v9 mean-pool
        use_vgm=False → TGM-only ablation (no cross-channel interaction)

    Everything else — seasonal stream, fusion — is identical between
    both variants and identical to v9 production.

    Purpose: isolate whether ETT regression in v9 vs v8 comes from:
        (A) replacing xPatch MLP trend with TGM  ← tested here with use_vgm=False
        (B) the VGM's cross-channel interaction  ← delta between use_vgm=F and T
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 channel=1, d_model=64, t_ff=None, use_vgm=True):
        super(GLPatchNetwork, self).__init__()

        self.pred_len     = pred_len
        self.patch_len    = patch_len
        self.stride       = stride
        self.padding_patch = padding_patch
        self.dim          = patch_len * patch_len
        self.patch_num    = (seq_len - patch_len) // stride + 1

        if t_ff is None:
            t_ff = 2 * d_model

        # ── Seasonal stream — identical to v8/v9 ──────────────
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

        # ── Trend stream — configurable ────────────────────────
        self.trend_stream = TrendStream(
            seq_len, pred_len, channel, d_model, t_ff, use_vgm=use_vgm
        )

        # ── Bottleneck fusion — identical to v8/v9 ─────────────
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

        t_out = self.trend_stream(t)        # [B*C, pred_len]

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
