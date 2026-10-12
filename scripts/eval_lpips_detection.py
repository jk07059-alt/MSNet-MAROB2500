#!/usr/bin/env python
"""
R1.6: perceptual metric (LPIPS) + QUANTITATIVE downstream detection (YOLOv8) on the
held-out TEST split (250 imgs). Same pad-16 single-pass inference + DILATE=15 vessel mask.
Convolutional models only (single-pass safe); Transformer baselines need tiled inference (TODO).
Run (from the NAFNet tree root, or set NAFNET_ROOT): CUDA_VISIBLE_DEVICES=0 python eval_lpips_detection.py
"""
import os, sys, csv
os.chdir(os.environ.get("NAFNET_ROOT", os.getcwd())); sys.path.insert(0, ".")  # BasicSR/NAFNet tree
import numpy as np, cv2, torch
from basicsr.models.archs import define_network
import lpips as lpips_lib
from ultralytics import YOLO

DS  = os.environ.get("MAROB_ROOT", "datasets/MAROB2500")  # your copy of the benchmark (README: Data availability)
TB, TS, MAN = f"{DS}/test/blur", f"{DS}/test/sharp", f"{DS}/split_manifest.csv"
DIL = 15; dev = "cuda:0"            # CUDA_VISIBLE_DEVICES remaps, so cuda:0 = chosen GPU
YOLO_W = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")   # ultralytics auto-downloads if absent
BOAT_CLS = {8}                     # COCO 'boat'
CONF = 0.25

def rd(p): return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR).astype(np.float32)/255., cv2.COLOR_BGR2RGB)

# (name, cfg, checkpoint); None cfg = identity (blurred input / sharp GT handled separately)
MODELS = [
 ("NAFNet+aug", dict(type="NAFNet",img_channel=3,width=32,middle_blk_num=1,enc_blk_nums=[1,1,1,28],dec_blk_nums=[1,1,1,1]),
                "experiments/R2-NAFNet-aug-hardevalv3-200k/models/net_g_200000.pth"),
 ("MSNet",      dict(type="MSNet",img_channel=3,width=32,middle_blk_num=1,enc_blk_nums=[1,1,1,26],dec_blk_nums=[1,1,1,1],
                     mar_stages=[0,1,2],strip_len=65,deep_every=6),
                "experiments/R26-MSNet-FIXED-hardevalv3-200k/models/net_g_190000.pth"),
 ("MIMO-UNet+", dict(type="MIMOUNetWrap",num_res=20,base_channel=32),
                "experiments/S3-MIMO-msloss-hardevalv3-200k/models/net_g_190000.pth"),
]

rows = {os.path.basename(r["blur_dst"]): r for r in csv.DictReader(open(MAN)) if r["split"] == "test"}
files = sorted(f for f in os.listdir(TB) if f in rows)
print(f"=== LPIPS + detection eval (TEST, {len(files)} imgs) ===", flush=True)

loss_fn = lpips_lib.LPIPS(net='alex', verbose=False).to(dev)
yolo = YOLO(YOLO_W)

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

def lpips_of(out, sharp):
    a = torch.from_numpy(out.transpose(2,0,1))[None].to(dev)*2-1
    b = torch.from_numpy(sharp.transpose(2,0,1))[None].to(dev)*2-1
    with torch.no_grad(): return float(loss_fn(a, b).item())

def vessel_mask(r, H, W):
    if r.get("has_ship")!="1" or not os.path.exists(r.get("mask_path","")): return None
    mk = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
    if mk is None or mk.shape != (H,W): return None
    m = cv2.dilate((mk>0).astype(np.uint8), np.ones((DIL,DIL),np.uint8)).astype(bool)
    return m if m.sum()>0 else None

def detected(img_rgb01, mask):
    """YOLO on restored image; hit = a boat-class box overlapping the vessel mask."""
    bgr = cv2.cvtColor((img_rgb01*255).clip(0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    res = yolo.predict(bgr, conf=CONF, verbose=False, device=0)[0]
    for b in res.boxes:
        if int(b.cls.item()) in BOAT_CLS:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(mask.shape[1],x2), min(mask.shape[0],y2)
            if x2>x1 and y2>y1 and mask[y1:y2, x1:x2].any(): return True
    return False

# precompute masks + per-image sharp/blur once
cache = {}
for fn in files:
    r = rows[fn]; blur, sharp = rd(TB+"/"+fn), rd(TS+"/"+fn); H,W,_ = blur.shape
    cache[fn] = (blur, sharp, vessel_mask(r, H, W))

def run_method(name, produce):
    lp=[]; hit=0; nves=0
    for fn in files:
        blur, sharp, m = cache[fn]
        out = produce(blur, sharp)
        lp.append(lpips_of(out, sharp))
        if m is not None:
            nves += 1
            if detected(out, m): hit += 1
    det = 100.0*hit/nves if nves else float('nan')
    print(f"  {name:12s} LPIPS={np.mean(lp):.4f}   detection-rate={det:.1f}%  ({hit}/{nves} vessels)", flush=True)

# references
run_method("Sharp(GT)", lambda blur, sharp: sharp)
run_method("Blurred",   lambda blur, sharp: blur)
# restoration models
for name, cfg, ck in MODELS:
    if not os.path.exists(ck): print(f"  {name}: checkpoint missing -- skip"); continue
    net = load(cfg, ck)
    run_method(name, lambda blur, sharp, net=net: infer(net, blur))
    del net; torch.cuda.empty_cache()
print("Done.")
