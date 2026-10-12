# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the MSNet / MAROB-2500 authors. See the LICENSE file.
# ------------------------------------------------------------------------
# MSNet -- Maritime-Structure NAFNet. The first architecture here that exploits the MARITIME
# SCENE, not just the blur. Maritime images are strongly HORIZONTAL (sky band on top, horizon
# line, sea below; waves are horizontal). So:
#   (1) HORIZONTAL context aggregation -- a wide 1xK depthwise strip gathers along-row context
#       (a blurred sea/sky/horizon row is best restored from its horizontal neighbourhood), and
#   (2) HEIGHT/ROW conditioning -- a per-row gate from the row-mean + vertical coordinate lets the
#       net treat sky (top) / horizon / sea (bottom) differently (builds on the user's
#       HeightAwareGate + SeaSkyRowAttention, but modulates AGGREGATED context, not just x).
# ZERO-INIT output (identity floor) so worst case == NAFNet+aug. Spatial ops only -> resolution-
# robust (unlike FSNet's FFT). Param-matched (enc trimmed).
# ------------------------------------------------------------------------
import torch
import torch.nn as nn

from basicsr.models.archs.NAFNet_arch import NAFBlock, NAFNet
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base


class MaritimeContextBlock(nn.Module):
    def __init__(self, c, strip_len=65, horizontal=True, use_gate=True, band=1,
                 multiscale=False, gate_zero_init=True, isotropic=False):
        super().__init__()
        self.horizontal = horizontal   # True=(bandxK) horizontal (maritime); False=(Kxband) vertical (control)
        self.use_gate = use_gate       # False = ablate the height/row gate
        self.multiscale = bool(multiscale) and not isotropic  # isotropic square control takes precedence
        self.norm = LayerNorm2d(c)
        self.pw_in = nn.Conv2d(c, c, 1)
        band = band | 1                 # force odd for symmetric same-padding
        if isotropic:
            # ISOTROPIC (square) control: side ~ sqrt(strip_len) so its AREA ~ the 1xK strip's tap count
            # (65->8x8, 33->6x6, 17->4x4, 9->3x3). padding='same' handles even sides cleanly (no size bug).
            side = max(2, int(round(strip_len ** 0.5)))
            self.iso_side = side
            self.strip_h = nn.Conv2d(c, c, (side, side), padding='same', groups=c, bias=True)
        elif self.multiscale:
            # PARALLEL depthwise strips at several horizontal lengths: short=fine waves .. long=horizon.
            # channels are SPLIT across the lengths, so total params ~ one depthwise strip (param-matched).
            lens = [max(3, int(round(f * strip_len)) | 1) for f in (0.14, 0.33, 0.66, 1.0)]
            ng = len(lens); base = c // ng
            self.ch_splits = [base] * (ng - 1) + [c - base * (ng - 1)]
            self.strips = nn.ModuleList()
            for gc, L in zip(self.ch_splits, lens):
                kk = (band, L) if horizontal else (L, band)
                pp = (band // 2, L // 2) if horizontal else (L // 2, band // 2)
                self.strips.append(nn.Conv2d(gc, gc, kk, padding=pp, groups=gc, bias=True))
            self.ms_lens = lens
        else:
            k = (band, strip_len) if horizontal else (strip_len, band)
            p = (band // 2, strip_len // 2) if horizontal else (strip_len // 2, band // 2)
            self.strip_h = nn.Conv2d(c, c, k, padding=p, groups=c, bias=True)
        # per-row HEIGHT-aware gate from [row-mean context, vertical coordinate]
        if use_gate:
            hidden = max(c // 8, 4)
            self.gate = nn.Sequential(
                nn.Conv2d(c + 1, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, c, 1))
            if gate_zero_init:
                nn.init.zeros_(self.gate[-1].weight)
                nn.init.zeros_(self.gate[-1].bias)   # neutral 0.5 start -- but diagnosed to stay DEAD
            # gate_zero_init=False: default init so the gate can actually LEARN height/texture; the
            # identity floor is still held by the zero-init pw_out/gamma below (whole branch = 0 at init).
        self.pw_out = nn.Conv2d(c, c, 1)      # NORMAL init -- do NOT zero it. Zeroing BOTH pw_out AND gamma
                                              # starves the branch's gradient (product of two zeros) so it
                                              # stays dead forever (verified: R17 gamma/pw_out==0 after 200k).
                                              # NAFNet zero-inits only the LayerScale; we now do the same.
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)  # gamma=0 => identity floor;
                                              # gamma still gets gradient from pw_out(feat)!=0 and wakes up.

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.norm(x)
        h = self.pw_in(y)
        if self.multiscale:
            chunks = torch.split(h, self.ch_splits, dim=1)
            hctx = torch.cat([s(ch) for s, ch in zip(self.strips, chunks)], dim=1)   # multi-scale strips
        else:
            hctx = self.strip_h(h)                          # single strip context (B,C,H,W)
        if self.use_gate:
            row = y.mean(dim=3, keepdim=True)               # (B,C,H,1) per-row context
            ycoord = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, 1)
            gate = torch.sigmoid(self.gate(torch.cat([row, ycoord], dim=1)))  # (B,C,H,1) per-row height gate
            feat = hctx * gate                              # context modulated by height/row
        else:
            feat = hctx
        return x + self.gamma * self.pw_out(feat)


def _interleave(nafblocks, c, every, strip_len, horizontal, use_gate, band=1,
                multiscale=False, gate_zero_init=True, isotropic=False):
    seq = []
    for i, blk in enumerate(nafblocks):
        seq.append(blk)
        if every > 0 and (i + 1) % every == 0:
            seq.append(MaritimeContextBlock(c, strip_len=strip_len, horizontal=horizontal,
                                            use_gate=use_gate, band=band, multiscale=multiscale,
                                            gate_zero_init=gate_zero_init, isotropic=isotropic))
    return nn.Sequential(*seq)


class MSNet(NAFNet):
    def __init__(self, img_channel=3, width=32, middle_blk_num=1,
                 enc_blk_nums=(1, 1, 1, 25), dec_blk_nums=(1, 1, 1, 1),
                 mar_stages=(0, 1, 2), strip_len=65, deep_every=6,
                 horizontal=True, use_gate=True, band=1,
                 multiscale=False, gate_zero_init=True, isotropic=False):
        super().__init__(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                         enc_blk_nums=list(enc_blk_nums), dec_blk_nums=list(dec_blk_nums))
        chan = width
        chans = []
        for _ in enc_blk_nums:
            chans.append(chan)
            chan *= 2
        # one MaritimeContextBlock appended to each of the shallow encoder stages (where the
        # sky/sea/horizon layout is spatially resolved). strip length shrinks with resolution.
        for s in mar_stages:
            blocks = list(self.encoders[s])
            sl = max(9, (strip_len >> s) | 1)  # force ODD (same-padding): 65,33,17 at stages 0,1,2
            blocks.append(MaritimeContextBlock(chans[s], strip_len=sl, horizontal=horizontal,
                                               use_gate=use_gate, band=band, multiscale=multiscale,
                                               gate_zero_init=gate_zero_init, isotropic=isotropic))
            self.encoders[s] = nn.Sequential(*blocks)
        # plus a few in the deep stage (interleaved)
        deep = list(self.encoders[-1])
        self.encoders[-1] = _interleave(deep, chans[-1], deep_every, 9, horizontal, use_gate, band=band,
                                        multiscale=multiscale, gate_zero_init=gate_zero_init,
                                        isotropic=isotropic)


class MSNetLocal(Local_Base, MSNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        MSNet.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == "__main__":
    base = NAFNet(width=32, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
    bp = sum(p.numel() for p in base.parameters())
    for enc in [(1, 1, 1, 25), (1, 1, 1, 26), (1, 1, 1, 24)]:
        m = MSNet(width=32, enc_blk_nums=enc)
        p = sum(x.numel() for x in m.parameters())
        print(f"enc={enc}: {p/1e6:.4f}M  (base28={bp/1e6:.4f}M, {(p-bp)/1e3:+.1f}K)")
