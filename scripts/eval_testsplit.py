#!/usr/bin/env python
"""
Camera-ready: recompute all main-table numbers on the HELD-OUT TEST split (250 imgs),
same 3-region protocol (DILATE=15) + whole-image SSIM, best-val checkpoints.
Run (from the NAFNet tree root, or set NAFNET_ROOT): CUDA_VISIBLE_DEVICES=0 python eval_testsplit.py
Then swap the printed overall/SSIM/bg/vessel into tab:main / tab:aug / tab:abl (currently on valid).
"""
import os, sys, csv, glob
os.chdir(os.environ.get("NAFNET_ROOT", os.getcwd())); sys.path.insert(0, ".")  # BasicSR/NAFNet tree
import numpy as np, cv2, torch
from basicsr.models.archs import define_network
try:
    from skimage.metrics import structural_similarity as sk_ssim
except Exception:
    sk_ssim = None

DS = os.environ.get("MAROB_ROOT", "datasets/MAROB2500")  # your copy of the benchmark (README: Data availability)
SPLIT = "test"                       # <-- the only change vs the valid eval
TB, TS, MAN = f"{DS}/{SPLIT}/blur", f"{DS}/{SPLIT}/sharp", f"{DS}/split_manifest.csv"
DIL = 15; dev = "cuda:0"

def rd(p): return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR).astype(np.float32)/255., cv2.COLOR_BGR2RGB)
def mpsnr(a, b, m):
    m = m.astype(bool)
    if m.sum() == 0: return None
    e = ((a[m]-b[m])**2).mean(); return 100. if e <= 0 else float(10*np.log10(1./e))
def ssim_full(a, b):
    if sk_ssim is None: return float("nan")
    return float(sk_ssim(a, b, channel_axis=2, data_range=1.0))

# (name, network_g dict, checkpoint) -- best-validation checkpoints used in the paper
MODELS = [
 ("NAFNet+aug", dict(type="NAFNet",img_channel=3,width=32,middle_blk_num=1,enc_blk_nums=[1,1,1,28],dec_blk_nums=[1,1,1,1]),
                "experiments/R2-NAFNet-aug-hardevalv3-200k/models/net_g_200000.pth"),
 ("MSNet",      dict(type="MSNet",img_channel=3,width=32,middle_blk_num=1,enc_blk_nums=[1,1,1,26],dec_blk_nums=[1,1,1,1],
                     mar_stages=[0,1,2],strip_len=65,deep_every=6),
                "experiments/R26-MSNet-FIXED-hardevalv3-200k/models/net_g_190000.pth"),
 ("MIMO-UNet+", dict(type="MIMOUNetWrap",num_res=20,base_channel=32),
                "experiments/S3-MIMO-msloss-hardevalv3-200k/models/net_g_190000.pth"),
 ("Stripformer", dict(type="Stripformer"),
                "experiments/S1-Stripformer-maritime-200k/models/net_g_200000.pth"),
 # add Restormer once the fair X6 checkpoint is transferred back from remote:
 # ("Restormer", dict(type="RestormerWrap"), "experiments/X6-Restormer-fair-maritime-200k/models/net_g_XXXXXX.pth"),
]

rows = {os.path.basename(r["blur_dst"]): r for r in csv.DictReader(open(MAN)) if r["split"] == SPLIT}
files = sorted(f for f in os.listdir(TB) if f in rows)
print(f"=== TEST-split eval ({len(files)} imgs) ===", flush=True)

def load(cfg, ck):
    net = define_network(dict(cfg)); sd = torch.load(ck, map_location="cpu"); sd = sd.get("params", sd)
    net.load_state_dict(sd, strict=True); return net.eval().to(dev)

def infer(net, blur, pad=16):
    H, W, _ = blur.shape; bt = torch.from_numpy(blur.transpose(2,0,1))[None].to(dev)
    ph, pw = (pad-H%pad)%pad, (pad-W%pad)%pad
    with torch.no_grad():
        o = net(torch.nn.functional.pad(bt, (0,pw,0,ph), mode='reflect'))
        if isinstance(o,(list,tuple)): o = o[-1]
        o = o[:,:,:H,:W].clamp(0,1)[0].cpu().numpy().transpose(1,2,0)
    del bt; torch.cuda.empty_cache(); return o

for name, cfg, ck in MODELS:
    if not os.path.exists(ck): print(f"  {name}: checkpoint missing ({ck}) -- skip"); continue
    net = load(cfg, ck); ov=[]; ss=[]; bg=[]; sh=[]
    for fn in files:
        r = rows[fn]; blur, sharp = rd(TB+"/"+fn), rd(TS+"/"+fn); H, W, _ = blur.shape
        out = infer(net, blur)
        ov.append(mpsnr(out, sharp, np.ones((H,W), bool))); ss.append(ssim_full(out, sharp))
        if r.get("has_ship")=="1" and os.path.exists(r.get("mask_path","")):
            mk = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mk is not None and mk.shape == (H,W):
                m = (mk>0); m = cv2.dilate(m.astype(np.uint8), np.ones((DIL,DIL),np.uint8)).astype(bool)
                if m.sum()>0: sh.append(mpsnr(out,sharp,m)); bg.append(mpsnr(out,sharp,~m))
    print(f"  {name:12s} overall={np.nanmean(ov):.3f}  SSIM={np.nanmean(ss):.4f}  "
          f"bg={np.mean(bg):.3f}  vessel={np.mean(sh):.3f}", flush=True)
    del net; torch.cuda.empty_cache()
print("Done. Swap these TEST numbers into the tables and drop the [NOTE: test split] marker.")
