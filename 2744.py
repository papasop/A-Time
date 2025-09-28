# -*- coding: utf-8 -*-
# One-cell Colab: Abell 2744 κ-map → linear vs nonlinear ψ/μ, ROI stats, two nulls (phase & block)
import os, sys, urllib.request, csv
import numpy as np, numpy.fft as fft
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- optional deps ----------
try:
    from astropy.io import fits
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "astropy"])
    from astropy.io import fits

try:
    from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure, label
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---------- paths & data ----------
OUT_DIR   = "/content/out"; os.makedirs(OUT_DIR, exist_ok=True)
FITS_PATH = "/content/kappa_map.fits"
URL       = "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"

if not os.path.exists(FITS_PATH):
    print("Downloading κ map:", URL, flush=True)
    urllib.request.urlretrieve(URL, FITS_PATH)
else:
    print("File exists:", FITS_PATH, flush=True)

# ---------- utils ----------
def clean_field(a):
    # 兼容 numpy 2.x：使用 asarray 允许拷贝
    a = np.asarray(a, dtype=np.float64)
    bad = ~np.isfinite(a)
    if bad.any():
        a = a.copy()
        a[bad] = 0.0
    return a

def downsample_mean(a, factor=1):
    if factor <= 1: return a
    m, n = a.shape
    m2, n2 = m//factor, n//factor
    a = a[:m2*factor, :n2*factor]
    return a.reshape(m2, factor, n2, factor).mean(axis=(1,3))

def safe_gauss(a, sigma_pix=1.0):
    if sigma_pix <= 0: return a
    if _HAS_SCIPY:
        return gaussian_filter(a, sigma=sigma_pix, mode='wrap')
    # Fallback：频域高斯
    Ar = fft.rfftn(a, s=a.shape)
    nx, ny = a.shape
    kx = fft.fftfreq(nx) * (2*np.pi)
    ky = fft.rfftfreq(ny) * (2*np.pi)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    H = np.exp(-0.5 * (KX*KX + KY*KY) * (sigma_pix**2))
    return fft.irfftn(Ar*H, s=a.shape)

def dilate_mask(mask, r=6):
    if r <= 0: return mask
    if _HAS_SCIPY:
        st = generate_binary_structure(2, 1)
        m = mask.copy()
        for _ in range(int(r)):
            m = binary_dilation(m, st)
        return m
    # 无 SciPy：环绕边界的方形结构元膨胀
    rad = int(r); k = 2*rad + 1
    x = np.pad(mask.astype(np.uint8), rad, mode='wrap')
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(x, (k, k))
        return (win.max(axis=(2,3))).astype(bool)
    except Exception:
        out = np.zeros_like(mask, dtype=np.uint8)
        base = mask.astype(np.uint8)
        for dx in range(-rad, rad+1):
            for dy in range(-rad, rad+1):
                out |= np.roll(np.roll(base, dx, axis=0), dy, axis=1)
        return out.astype(bool)

def spectral_green_poisson(rhs, eps=1e-12):
    # 解 u：Δu = rhs（周期边界），直流项设 0
    RH = fft.rfftn(rhs, s=rhs.shape)
    nx, ny = rhs.shape
    kx = fft.fftfreq(nx) * (2*np.pi)
    ky = fft.rfftfreq(ny) * (2*np.pi)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX*KX + KY*KY
    G  = np.zeros_like(K2)
    m  = K2 > eps
    G[m] = -1.0 / K2[m]
    U  = G * RH
    return fft.irfftn(U, s=rhs.shape)

def hessian_from_psi(psi):
    Psi_r = fft.rfftn(psi, s=psi.shape)
    nx, ny = psi.shape
    kx = fft.fftfreq(nx) * (2*np.pi)
    ky = fft.rfftfreq(ny) * (2*np.pi)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    Dxx, Dyy, Dxy = -(KX**2), -(KY**2), -(KX*KY)
    psi_xx = fft.irfftn(Psi_r*Dxx, s=psi.shape)
    psi_yy = fft.irfftn(Psi_r*Dyy, s=psi.shape)
    psi_xy = fft.irfftn(Psi_r*Dxy, s=psi.shape)
    return psi_xx, psi_yy, psi_xy

def mu_from_psi(psi, eps=1e-9):
    psi_xx, psi_yy, psi_xy = hessian_from_psi(psi)
    kappa  = 0.5*(psi_xx + psi_yy)
    gamma1 = 0.5*(psi_xx - psi_yy)
    gamma2 = psi_xy
    den = (1.0 - kappa)**2 - (gamma1*gamma1 + gamma2*gamma2)
    den = np.where(np.isfinite(den), den, np.inf)
    den = np.where(np.abs(den) < eps, np.sign(den)*eps, den)
    return 1.0/den

def robust_stats(x):
    x = x[np.isfinite(x)]
    if x.size == 0: return (np.nan, np.nan)
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med) + 1e-12)
    return med, mad

def energy_and_grad(psi, kappa, lam):
    # E = ||Δψ - 2κ||^2 + lam * ||∇ψ||^2，返回数据项及全梯度（周期）
    Psi_r = fft.rfftn(psi, s=psi.shape)
    nx, ny = psi.shape
    kx = fft.fftfreq(nx) * (2*np.pi)
    ky = fft.rfftfreq(ny) * (2*np.pi)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX*KX + KY*KY

    lap   = fft.irfftn(-K2*Psi_r, s=psi.shape)
    resid = lap - 2.0*kappa
    E_data = float(np.mean(resid*resid))

    resid_r = fft.rfftn(resid, s=resid.shape)
    grad_data = fft.irfftn(2.0*(K2**2) * resid_r, s=resid.shape)   # 2 Δ^2 ψ
    grad_reg  = -2.0*lam * lap                                     # -2λΔψ
    return E_data, (grad_data + grad_reg)

def nonlinear_refine_roi(psi0, kappa, roi, steps=60, tau0=4e-2, lam=0.05, clip_u=0.04, verbose=True):
    psi  = psi0.copy()
    tau  = float(tau0)
    last = None
    for it in range(1, steps+1):
        E, g = energy_and_grad(psi, kappa, lam)
        upd  = -tau * g
        if clip_u is not None:
            upd = np.clip(upd, -clip_u, clip_u)
        # 仅 ROI 更新
        upd = np.where(roi, upd, 0.0)
        psi_try = psi + upd
        E_try, _ = energy_and_grad(psi_try, kappa, lam)
        if (last is None) or (E_try <= E + 1e-9):
            psi, last = psi_try, E_try
        else:
            tau *= 0.5
            # 小步再试一次（避免卡死）
            psi_try2 = psi + np.where(roi, np.clip(-0.5*upd, -clip_u, clip_u), 0.0)
            last = energy_and_grad(psi_try2, kappa, lam)[0]
            psi  = psi_try2
        if verbose and (it == 1 or it % 10 == 0):
            print(f"[nonlin] it={it:02d} tau={tau:.2e} |E_data={last:.4g}", flush=True)
    return psi

def phase_randomize(field, rng):
    F = fft.rfftn(field, s=field.shape)
    mag = np.abs(F)
    ph  = rng.uniform(0, 2*np.pi, size=F.shape)
    ph[0,:] = 0.0; ph[:,0] = 0.0
    return fft.irfftn(mag*np.exp(1j*ph), s=field.shape)

def block_surrogate(field, rng, tile=64, rotate=True, flip=True):
    # 图像域块置乱 + 可选旋转/翻转，用于更“苛刻”的零假设
    a = field.copy()
    nx, ny = a.shape
    tx = max(8, min(tile, nx))
    ty = max(8, min(tile, ny))
    px = nx // tx; py = ny // ty
    a = a[:px*tx, :py*ty]
    # 分块
    blocks = []
    for i in range(px):
        for j in range(py):
            blk = a[i*tx:(i+1)*tx, j*ty:(j+1)*ty]
            if rotate:
                k = rng.integers(0, 4)
                blk = np.rot90(blk, int(k))
            if flip and rng.random() < 0.5:
                blk = np.flipud(blk)
            if flip and rng.random() < 0.5:
                blk = np.fliplr(blk)
            blocks.append(blk)
    rng.shuffle(blocks)
    # 回填
    out = np.zeros_like(a)
    t = 0
    for i in range(px):
        for j in range(py):
            out[i*tx:(i+1)*tx, j*ty:(j+1)*ty] = blocks[t]; t += 1
    # 若裁掉了边缘，做环绕拼回原尺寸
    if out.shape != field.shape:
        padx = field.shape[0] - out.shape[0]
        pady = field.shape[1] - out.shape[1]
        out = np.pad(out, ((0,padx),(0,pady)), mode='wrap')
    return out

def morans_I(img, mask):
    # 简化的四邻域 Moran's I（mask 内计算）
    M = mask & np.isfinite(img)
    x = img[M]
    if x.size < 2: return np.nan
    mu = x.mean()
    num = 0.0; den = np.sum((x - mu)**2) + 1e-12
    W = 0
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nbr = np.roll(img, shift=(dx,dy), axis=(0,1))
        Mnbr = M & np.roll(M, shift=(dx,dy), axis=(0,1))
        num += np.sum( (img[Mnbr]-mu)*(nbr[Mnbr]-mu) )
        W   += np.count_nonzero(Mnbr)
    if W == 0: return np.nan
    return (x.size / W) * (num / den)

def max_cluster_area(binary_mask):
    # 统计最大连通簇面积（8 邻域）；无 SciPy 时用 BFS 兜底
    B = (binary_mask.astype(bool)).copy()
    if not B.any(): return 0
    if _HAS_SCIPY:
        lbl, n = label(B)  # 默认 8 邻域
        if n == 0: return 0
        counts = np.bincount(lbl.ravel())
        return int(counts[1:].max())  # 跳过背景 0
    # BFS 兜底
    visited = np.zeros_like(B, dtype=bool)
    H, W = B.shape
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    mmax = 0
    for i in range(H):
        for j in range(W):
            if B[i,j] and not visited[i,j]:
                # flood fill
                stack = [(i,j)]; visited[i,j] = True; cnt = 0
                while stack:
                    x,y = stack.pop(); cnt += 1
                    for dx,dy in nbrs:
                        xx = (x+dx) % H; yy = (y+dy) % W  # wrap
                        if B[xx,yy] and not visited[xx,yy]:
                            visited[xx,yy] = True
                            stack.append((xx,yy))
                mmax = max(mmax, cnt)
    return mmax

# ---------- config ----------
DS            = 2            # 下采样因子（1/2/3…）
SIGMA_SMOOTH  = 1.0          # 像素高斯平滑
ROI_RULE      = ("abs", 10.) # ("abs", MU_THR) 或 ("mad", k)  → |μ|>median+k*MAD
R_DILATE      = 6            # ROI 膨胀像素
NL_STEPS      = 60
NL_TAU_INIT   = 4.0e-2
NL_LAMBDA     = 0.05
NL_CLIP       = 0.04
NULL_MODE     = "phase"      # "phase" 或 "block"
N_PERM        = 60
BLOCK_TILE    = 64           # 块置乱的块大小
Q_LIST        = [0.95, 0.975, 0.99]  # 三个分位阈值
RNG           = np.random.default_rng(2025)

# ---------- load κ ----------
with fits.open(FITS_PATH) as hdul:
    cands = []
    for i,h in enumerate(hdul):
        d = getattr(h, "data", None)
        if d is None: continue
        arr = np.asarray(d)
        if arr.ndim==2 and np.isfinite(arr).sum()>0 and np.nanstd(arr)>1e-12:
            cands.append((arr.size, i, arr))
    if not cands:
        raise RuntimeError("No usable 2D image HDU in FITS.")
    cands.sort(reverse=True)
    _, hdu_idx, kappa0 = cands[0]
kappa0 = clean_field(kappa0)
print(f"Loaded κ map: {kappa0.shape} (picked HDU[{hdu_idx}])", flush=True)

# ---------- preprocess ----------
kappa = downsample_mean(kappa0, DS)
kappa = safe_gauss(kappa, SIGMA_SMOOTH)

# 线性解与 μ
psi_lin = spectral_green_poisson(2.0*kappa)
mu_lin  = mu_from_psi(psi_lin)
med_mu, mad_mu = robust_stats(mu_lin)
print(f"Downsampled: {kappa.shape} | μ_lin robust: median={med_mu:.3g}, MAD={mad_mu:.3g}", flush=True)

# ROI
if ROI_RULE[0] == "abs":
    thr_mu = float(ROI_RULE[1])
    roi0 = np.abs(mu_lin) > thr_mu
else:
    k = float(ROI_RULE[1])
    roi0 = np.abs(mu_lin) > (med_mu + k*mad_mu)
roi  = dilate_mask(roi0, r=R_DILATE)
roiN = int(np.count_nonzero(roi))
print(f"ROI pixels: {roiN} (rule: {ROI_RULE}, dilate={R_DILATE}px)", flush=True)

if roiN == 0:
    print("\n[Early stop] ROI is empty; nothing to refine. Exiting gracefully.", flush=True)
    # 仍然输出空图与 CSV
    FIG_PATH = os.path.join(OUT_DIR, "fig_overview.png")
    CSV_PATH = os.path.join(OUT_DIR, "summary_metrics.csv")
    fig,ax = plt.subplots(1,1,figsize=(5,4)); ax.imshow(kappa, cmap='coolwarm'); ax.set_title("κ"); plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=140); plt.show()
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["when","note"]); w.writerow([datetime.utcnow().isoformat(),"empty ROI"])
    print("Artifacts:\n -", FIG_PATH, "\n -", CSV_PATH)
    raise SystemExit

# ---------- nonlinear refinement (obs) ----------
print("\n[1] Nonlinear refinement (obs)…", flush=True)
psi_nl = nonlinear_refine_roi(psi_lin, kappa, roi,
                              steps=NL_STEPS, tau0=NL_TAU_INIT,
                              lam=NL_LAMBDA, clip_u=NL_CLIP)

mu_nl = mu_from_psi(psi_nl)
dmu   = np.abs(mu_nl) - np.abs(mu_lin)
dmu   = np.where(np.isfinite(dmu), dmu, 0.0)

# 观测统计
d_in  = dmu[roi]
d_in  = d_in[np.isfinite(d_in)]
if d_in.size == 0:
    print("\n[Early stop] Δ|μ| in ROI is empty/NaN-only; exiting.", flush=True)
    raise SystemExit

thr_vals   = [float(np.quantile(d_in, q)) for q in Q_LIST]
masks_obs  = [((dmu>=t) & roi) for t in thr_vals]
areas_obs  = [int(m.sum()) for m in masks_obs]
clusters_ob= [max_cluster_area(m) for m in masks_obs]
moran_obs  = morans_I(dmu, roi)

print("\n[2] Obs Δ|μ| thresholds & stats:", flush=True)
for q,t,a,c in zip(Q_LIST, thr_vals, areas_obs, clusters_ob):
    print(f"  q={q:.3f}: thr={t:.3g} | area={a} | maxCluster={c}", flush=True)
print(f"  Moran's I (Δ|μ| within ROI): {moran_obs:.3e}", flush=True)

# ---------- permutation null ----------
print("\n[3] Permutation null …", flush=True)
hits_area   = np.zeros(len(Q_LIST), dtype=int)
hits_cl     = np.zeros(len(Q_LIST), dtype=int)
hits_moran  = 0
areas_null  = np.zeros((N_PERM, len(Q_LIST)), dtype=int)
cl_null     = np.zeros((N_PERM, len(Q_LIST)), dtype=int)
moran_null  = np.zeros(N_PERM)

for i in range(N_PERM):
    if NULL_MODE == "phase":
        ks = phase_randomize(kappa, RNG)
    elif NULL_MODE == "block":
        ks = block_surrogate(kappa, RNG, tile=BLOCK_TILE, rotate=True, flip=True)
    else:
        # 交替
        ks = phase_randomize(kappa, RNG) if (i%2==0) else block_surrogate(kappa, RNG, tile=BLOCK_TILE, rotate=True, flip=True)
    psi_lin_s = spectral_green_poisson(2.0*ks)
    mu_lin_s  = mu_from_psi(psi_lin_s)
    psi_nl_s  = nonlinear_refine_roi(psi_lin_s, ks, roi,
                                     steps=max(10, NL_STEPS//6), tau0=NL_TAU_INIT,
                                     lam=NL_LAMBDA, clip_u=NL_CLIP, verbose=False)
    mu_nl_s = mu_from_psi(psi_nl_s)
    dmu_s   = np.abs(mu_nl_s) - np.abs(mu_lin_s)
    dmu_s   = np.where(np.isfinite(dmu_s), dmu_s, 0.0)

    for j,t in enumerate(thr_vals):
        M = ((dmu_s >= t) & roi)
        a = int(M.sum()); areas_null[i,j] = a
        c = max_cluster_area(M); cl_null[i,j] = c
        if a >= areas_obs[j]: hits_area[j]  += 1
        if c >= clusters_ob[j]: hits_cl[j]  += 1
    mi = morans_I(dmu_s, roi); moran_null[i] = mi
    if np.isfinite(mi) and np.isfinite(moran_obs) and (mi >= moran_obs): hits_moran += 1

    if (i+1) % max(1, N_PERM//6) == 0 or i==0:
        msg = " | ".join([f"q{int(q*100)}%: area_hits={hits_area[j]}, cl_hits={hits_cl[j]}" for j,q in enumerate(Q_LIST)])
        print(f"  perm {i+1:03d}/{N_PERM} | {msg} | Moran_hits={hits_moran}", flush=True)

# p-values (+1 无偏)
p_area  = (hits_area + 1)  / (N_PERM + 1)
p_cl    = (hits_cl   + 1)  / (N_PERM + 1)
p_moran = (hits_moran + 1) / (N_PERM + 1)

# ---------- plots ----------
fig = plt.figure(figsize=(14,7))
ax1 = plt.subplot(2,3,1); im1=ax1.imshow(kappa, cmap='coolwarm'); ax1.set_title("κ (downsampled)"); plt.colorbar(im1, ax=ax1, fraction=0.046)
ax2 = plt.subplot(2,3,2); im2=ax2.imshow(mu_lin, cmap='magma', vmin=-50, vmax=50); ax2.set_title("μ_lin (clip v±50)"); plt.colorbar(im2, ax=ax2, fraction=0.046)
ax3 = plt.subplot(2,3,3); im3=ax3.imshow(roi, cmap='Greens'); ax3.set_title(f"ROI (rule={ROI_RULE}, dil={R_DILATE})"); plt.colorbar(im3, ax=ax3, fraction=0.046)
ax4 = plt.subplot(2,3,4); im4=ax4.imshow(np.abs(mu_nl)-np.abs(mu_lin), cmap='inferno'); ax4.set_title("Δ|μ| (obs)"); plt.colorbar(im4, ax=ax4, fraction=0.046)
ax5 = plt.subplot(2,3,5)
for t in thr_vals:
    ax5.contour(((dmu>=t)&roi), levels=[0.5], colors='w', linewidths=1.0, alpha=0.8)
ax5.imshow(dmu, cmap='inferno'); ax5.set_title("Contours @ q=95/97.5/99 %")
ax6 = plt.subplot(2,3,6)
for j,q in enumerate(Q_LIST):
    ax6.hist(areas_null[:,j], bins=20, alpha=0.5, label=f"q={int(q*100)} null")
    ax6.axvline(areas_obs[j], ls='--', label=f"obs={areas_obs[j]}")
ax6.legend(fontsize=8); ax6.set_title("Area null hist")
plt.tight_layout()
FIG_PATH = os.path.join(OUT_DIR, "fig_overview.png")
plt.savefig(FIG_PATH, dpi=160); plt.show()

# ---------- CSV ----------
CSV_PATH = os.path.join(OUT_DIR, "summary_metrics.csv")
with open(CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["when","shape","DS","sigma","ROI_RULE","dilate","steps","tau0","lambda","clip","null","perms","tile"])
    w.writerow([datetime.utcnow().isoformat(), f"{kappa.shape}", DS, SIGMA_SMOOTH, str(ROI_RULE),
                R_DILATE, NL_STEPS, NL_TAU_INIT, NL_LAMBDA, NL_CLIP, NULL_MODE, N_PERM, BLOCK_TILE])
    w.writerow([])
    w.writerow(["q","thr","area_obs","area_p","cl_obs","cl_p","moran_obs","moran_p"])
    for q,t,a,pa,c,pc in zip(Q_LIST, thr_vals, areas_obs, p_area, clusters_ob, p_cl):
        w.writerow([q, t, a, float(pa), c, float(pc), float(moran_obs), float(p_moran)])
print("\n[Summary]", flush=True)
for q,a,pa,c,pc in zip(Q_LIST, areas_obs, p_area, clusters_ob, p_cl):
    print(f"  q={q:.3f}: area_obs={a}  p≈{pa:.3f} | maxCluster={c}  p≈{pc:.3f}", flush=True)
print(f"  Moran's I (obs) p≈{p_moran:.3f}", flush=True)
print("\nArtifacts:\n -", FIG_PATH, "\n -", CSV_PATH, flush=True)
