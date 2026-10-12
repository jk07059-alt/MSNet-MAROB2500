#!/usr/bin/env python
"""
5-seed robustness eval on the TEST split (run when all 10 runs reach net_g_200000).
Baseline = R2-NAFNet-aug-seed{1..5}, MSNet = R26-MSNet-seed{1..5}, all @200k (batch 16, 1 GPU).
Same protocol as tab:main: overall = BasicSR psnr, bg/vessel = masked PSNR (dilate 15), single-pass pad-16.
Reports per-seed overall/bg/vessel, baseline & MSNet mean+/-std, and the per-seed vessel GAIN mean+/-std.
NOTE: seeds are effective batch 16 (tab:main is batch 24), so absolute means need not match tab:main;
the robust quantities are the GAIN (MSNet-baseline) and the run-to-run STD.
Run (from the NAFNet tree root, or set NAFNET_ROOT): CUDA_VISIBLE_DEVICES=0 python seed_eval_5seed.py
"""
import os, sys, csv, numpy as np, cv2, torch
os.chdir(os.environ.get("NAFNET_ROOT", os.getcwd())); sys.path.insert(0, ".")  # BasicSR/NAFNet tree
from basicsr.models.archs import define_network
from basicsr.metrics import calculate_psnr as bpsnr
DS = os.environ.get("MAROB_ROOT", "datasets/MAROB2500")  # your copy of the benchmark (README: Data availability)
TB, TS, MAN = f"{DS}/test/blur", f"{DS}/test/sharp", f"{DS}/split_manifest.csv"
DIL = 15; dev = "cuda:0"

NAF = dict(type="NAFNet",img_channel=3,width=32,middle_blk_num=1,enc_blk_nums=[1,1,1,28],dec_blk_nums=[1,1,1,1])
MS  = dict(type="MSNet",img_channel=3,width=32,middle_blk_num=1,enc_blk_nums=[1,1,1,26],dec_blk_nums=[1,1,1,1],
           mar_stages=[0,1,2],strip_len=65,deep_every=6)
SEEDS = {
 "baseline": [(f"R2-NAFNet-aug-seed{s}-200k", NAF) for s in (1,2,3,4,5)],
 "MSNet":    [(f"R26-MSNet-seed{s}-200k",     MS ) for s in (1,2,3,4,5)],
}

def rd(p): return cv2.cvtColor(cv2.imread(p,cv2.IMREAD_COLOR).astype(np.float32)/255., cv2.COLOR_BGR2RGB)
def mpsnr(a,b,m):
    m=m.astype(bool)
    if m.sum()==0: return None
    e=((a[m]-b[m])**2).mean(); return 100. if e<=0 else float(10*np.log10(1./e))
def load(cfg,ck):
    net=define_network(dict(cfg)); sd=torch.load(ck,map_location="cpu"); sd=sd.get("params",sd)
    net.load_state_dict(sd,strict=True); return net.eval().to(dev)
def full(net,b,pad=16):
    H,W,_=b.shape; t=torch.from_numpy(b.transpose(2,0,1))[None].to(dev)
    ph,pw=(pad-H%pad)%pad,(pad-W%pad)%pad
    with torch.no_grad():
        o=net(torch.nn.functional.pad(t,(0,pw,0,ph),mode='reflect'))
        if isinstance(o,(list,tuple)): o=o[-1]
        o=o[:,:,:H,:W].clamp(0,1)[0].cpu().numpy().transpose(1,2,0)
    torch.cuda.empty_cache(); return o

rows={os.path.basename(r["blur_dst"]):r for r in csv.DictReader(open(MAN)) if r["split"]=="test"}
files=sorted(f for f in os.listdir(TB) if f in rows)
print(f"=== 5-seed TEST eval ({len(files)} imgs), all @200k, single-pass pad-16 ===", flush=True)

def eval_ckpt(cfg, ckpt):
    net=load(cfg,ckpt); ov=[];bg=[];sh=[]
    for fn in files:
        r=rows[fn]; blur,sharp=rd(TB+"/"+fn),rd(TS+"/"+fn); H,W,_=blur.shape
        out=full(net,blur)
        ov.append(bpsnr((out*255.).astype(np.float32),(sharp*255.).astype(np.float32),crop_border=0,test_y_channel=False))
        if r.get("has_ship")=="1" and os.path.exists(r.get("mask_path","")):
            mk=cv2.imread(r["mask_path"],cv2.IMREAD_GRAYSCALE)
            if mk is not None and mk.shape==(H,W):
                m=cv2.dilate((mk>0).astype(np.uint8),np.ones((DIL,DIL),np.uint8)).astype(bool)
                if m.sum()>0: sh.append(mpsnr(out,sharp,m)); bg.append(mpsnr(out,sharp,~m))
    del net; torch.cuda.empty_cache()
    return np.mean(ov), np.mean(bg), np.mean(sh)

res={}   # role -> list of (overall,bg,vessel) per seed
for role, lst in SEEDS.items():
    res[role]=[]
    for name,cfg in lst:
        ck=f"experiments/{name}/models/net_g_200000.pth"
        if not os.path.exists(ck): print(f"  {name}: MISSING {ck} -- skip", flush=True); continue
        try:
            o,b,s=eval_ckpt(cfg,ck); res[role].append((o,b,s))
            print(f"  {name:30s} overall={o:.3f}  bg={b:.3f}  vessel={s:.3f}", flush=True)
        except Exception as e:
            print(f"  {name}: ERROR {e}", flush=True)

def stat(arr): a=np.array(arr); return a.mean(), a.std(ddof=1) if len(a)>1 else 0.0
print("\n=== aggregate (mean +/- std over seeds) ===", flush=True)
for role in ("baseline","MSNet"):
    R=res[role]
    if not R: print(f"  {role}: no seeds"); continue
    for i,k in enumerate(("overall","bg","vessel")):
        m,sd=stat([r[i] for r in R]); print(f"  {role:9s} {k:8s} = {m:.3f} +/- {sd:.3f}  (n={len(R)})", flush=True)
# paired-by-seed vessel gain (MSNet_i - baseline_i)
if len(res["baseline"])==len(res["MSNet"]) and res["baseline"]:
    for i,k in enumerate(("overall","bg","vessel")):
        gains=[res["MSNet"][j][i]-res["baseline"][j][i] for j in range(len(res["MSNet"]))]
        m,sd=stat(gains); print(f"  GAIN(MSNet-baseline) {k:8s} = {m:+.3f} +/- {sd:.3f}", flush=True)
print("\nNOTE: seeds are effective batch 16 (tab:main batch 24); use GAIN and STD, not absolute means.", flush=True)
print("DONE", flush=True)
