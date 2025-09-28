# -*- coding: utf-8 -*-
# One-cell Colab: Abell 2744 κ-map → linear vs nonlinear ψ, μ, ROI stats with phase-random null
# NumPy 2.0 兼容（np.asarray 修复）+ rfftn/irfftn 显式 axes，避免 DeprecationWarning

import os, sys, urllib.request, csv
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- Robust deps ----------
try:
    from astropy.io import fits
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "astropy"])
    from astropy.io import fits

try:
    from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure, maximum_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---------- Paths ----------
OUT_DIR = "/content/out"; os.makedirs(OUT_DIR, exist_ok=True)
FITS_PATH = "/content/kappa_map.fits"
URL = "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"

if not os.path.exists(FITS_PATH):
    print("Downloading κ map:", URL, flush=True)
    urllib.request.urlretrieve(URL, FITS_PATH)
else:
    print("File exists:", FITS_PATH, flush=True)

# ---------- Small utils ----------
def clean_field(a):
    # NumPy 2.0-safe：允许必要拷贝
    a = np.asarray(a, dtype=np.float64)
    bad = ~np.isfinite(a)
    if bad.any():
        med = np.nanmedian(a)
        if not np.isfinite(med):
            med = 0.0
        if not a.flags.writeable:
            a = a.copy()
        a[bad] = med
    return a

def downsample_mean(a, factor):
    if factor <= 1: return a
    m, n = a.shape
    m2, n2 = m//factor, n//factor
    a = a[:m2*factor, :n2*factor]
    return a.reshape(m2, factor, n2, factor).mean(axis=(1,3))

def robust_stats(x):
    x = x[np.isfinite(x)]
    if x.size == 0: return (np.nan, np.nan)
    med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-12
    return med, 1.4826*mad

def safe_gauss(a, sigma_pix=1.0):
    if sigma_pix <= 0: return a
    if _HAS_SCIPY:
        return gaussian_filter(a, sigma=sigma_pix, mode='wrap')
    # 频域高斯（显式 axes）
    Ar = fft.rfftn(a, s=a.shape, axes=(0,1))
    nx, ny = a.shape
    kx = fft.fftfreq(nx) * (2*np.pi)
    ky = fft.rfftfreq(ny) * (2*np.pi)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX*KX + KY*KY
    H = np.exp(-0.5 * K2 * (sigma_pix**2))
    return fft.irfftn(Ar * H, s=a.shape, axes=(0,1))

def dilate_mask(mask, r=6):
    if r <= 0: return mask
    if _HAS_SCIPY:
        st = generate_binary_structure(2, 1)
        m = mask.copy()
        for _ in range(r):
            m = binary_dilation(m, st)
        return m
    # fallback：最大值滤波
    rad = int(r); size = 2*rad+1
    mm = maximum_filter(mask.astype(np.uint8), size=size, mode='wrap')
    return mm.astype(bool)

def phase_randomize(field, rng):
    # |F| 保持，相位随机；显式 axes 确保 NumPy 2.0 兼容
    F = fft.rfftn(field, s=field.shape, axes=(0,1))
    mag = np.abs(F)
    ph  = rng.uniform(0, 2*np.pi, size=F.shape)
    ph[0, :] = 0.0
    ph[:, 0] = 0.0
    Fr = mag * np.exp(1j*ph)
    return fft.irfftn(Fr, s=field.shape, axes=(0,1))

# ---------- FFT-based operators (axes=(0,1)) ----------
def spectral_poisson(rhs, eps=1e-12):
    RH = fft.rfftn(rhs, s=rhs.shape, axes=(0,1))
    nx, ny = rhs.shape
    kx = fft.fftfreq(nx) * (2*np.pi)
    ky = fft.rfftfreq(ny) * (2*np.pi)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX*KX + KY*KY
    G = np.zeros_like(K2)
    mask = K2 > eps
    G[mask] = -1.0 / K2[mask]
    return fft.irfftn(G * RH, s=rhs.shape, axes=(0,1))

def hessian_from_psi(psi):
    Psi_r = fft.rfftn(psi, s=psi.shape, axes=(0,1))
    nx, ny = psi.shape
    kx = fft.fftfreq(nx) * (2*np.pi)
    ky = fft.rfftfreq(ny) * (2*np.pi)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    Dxx = -(KX**2); Dyy = -(KY**2); Dxy = -(KX*KY)
    psi_xx = fft.irfftn(Psi_r*Dxx, s=psi.shape, axes=(0,1))
    psi_yy = fft.irfftn(Psi_r*Dyy, s=psi.shape, axes=(0,1))
    psi_xy = fft.irfftn(Psi_r*Dxy, s=psi.shape, axes=(0,1))
    return psi_xx, psi_yy, psi_xy

def mu_from_psi(psi, eps=1e-9):
    psi_xx, psi_yy, psi_xy = hessian_from_psi(psi)
    kappa = 0.5*(psi_xx + psi_yy)
    gamma1 = 0.5*(psi_xx - psi_yy)
    gamma2 = psi_xy
    den = ((1.0 - kappa)**2 - (gamma1*gamma1 + gamma2*gamma2))
    den = np.where(np.isfinite(den), den, np.inf)
    den = np.where(np.abs(den) < eps, np.sign(den)*eps, den)
    return 1.0/den

# ---------- Nonlinear energy ----------
def energy_and_grad(psi, kappa, lam):
    Psi_r = fft.rfftn(psi, s=psi.shape, axes=(0,1))
    nx, ny = psi.shape
    kx = fft.fftfreq(nx) * (2*np.pi)
    ky = fft.rfftfreq(ny) * (2*np.pi)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX*KX + KY*KY

    lap = fft.irfftn(-K2 * Psi_r, s=psi.shape, axes=(0,1))
    resid = lap - 2.0*kappa
    E_data = float(np.mean(resid*resid))

    resid_r = fft.rfftn(resid, s=resid.shape, axes=(0,1))
    grad_data = fft.irfftn( 2.0*(K2**2) * resid_r, s=resid.shape, axes=(0,1))
    grad_reg  = -2.0*lam * lap
    return E_data, (grad_data + grad_reg)

def nonlinear_refine_roi(psi0, kappa, roi, steps=60, tau0=4e-2, lam=0.05, clip_u=0.04):
    psi = psi0.copy()
    tau = float(tau0)
    last_E = None
    for it in range(1, steps+1):
        E_data, g = energy_and_grad(psi, kappa, lam)
        upd = -tau * g
        # 只在 ROI 内更新，并裁剪更新幅度（抑制溢出）
        upd = np.where(roi, np.clip(upd, -clip_u, clip_u), 0.0)

        psi_try = psi + upd
        E_try, _ = energy_and_grad(psi_try, kappa, lam)
        if (last_E is None) or (E_try <= E_data + 1e-12):
            psi = psi_try; last_E = E_try
        else:
            tau *= 0.5
            psi = psi + np.where(roi, np.clip(-0.5*upd, -clip_u, clip_u), 0.0)
            last_E, _ = energy_and_grad(psi, kappa, lam)

        if it == 1 or (it % 10 == 0):
            print(f"[nonlin] it={it:02d} tau={tau:.2e} |E_data={last_E:.4g}", flush=True)
    return psi

# ---------- Load κ map (robust HDU picker) ----------
with fits.open(FITS_PATH) as hdul:
    cands = []
    for i, h in enumerate(hdul):
        data = getattr(h, "data", None)
        if data is None: continue
        arr = np.asarray(data)
        if arr.ndim == 2 and np.isfinite(arr).any():
            cands.append((arr.size, i, arr))
    if not cands:
        raise RuntimeError("No valid 2D image HDU in FITS.")
    cands.sort(reverse=True)
    _, hdu_idx, kappa0 = cands[0]

kappa0 = clean_field(kappa0)
print("Loaded κ map:", kappa0.shape, f"(picked HDU[{hdu_idx}])", flush=True)

# ---------- Config (tunable) ----------
DS            = 2          # 下采样（2 会更快；=1 用全分辨率）
SIGMA_SMOOTH  = 1.0        # Gaussian 平滑半径（像素）
MU_THR        = 10.0       # ROI 阈值：|μ_lin| > MU_THR
R_DILATE      = 6          # ROI 膨胀像素
NL_STEPS      = 60
NL_TAU_INIT   = 4.0e-2
NL_LAMBDA     = 0.05
NL_CLIP       = 0.04
N_PERM        = 60         # 置换次数
RNG           = np.random.default_rng(2025)

# ---------- Preprocess ----------
kappa = downsample_mean(kappa0, DS) if DS>1 else kappa0.copy()
kappa = safe_gauss(kappa, SIGMA_SMOOTH)
psi_lin = spectral_poisson(2.0*kappa)
mu_lin  = mu_from_psi(psi_lin)

med_mu, mad_mu = robust_stats(mu_lin)
print(f"Downsampled: {kappa.shape} | μ_lin robust: median={med_mu:.3g}, MAD={mad_mu:.3g}", flush=True)

roi0 = np.abs(mu_lin) > MU_THR
roi  = dilate_mask(roi0, r=R_DILATE)
print(f"ROI pixels: {int(roi.sum())} (|μ_lin|>{MU_THR}, dilate={R_DILATE}px)", flush=True)

# ---------- Nonlinear (obs) ----------
print("\n[1] Nonlinear refinement (obs)…", flush=True)
psi_nl = nonlinear_refine_roi(psi_lin, kappa, roi,
                              steps=NL_STEPS, tau0=NL_TAU_INIT,
                              lam=NL_LAMBDA, clip_u=NL_CLIP)
mu_nl = mu_from_psi(psi_nl)
dmu   = np.abs(mu_nl) - np.abs(mu_lin)

# 阈值分位（ROI 内）
thr_pcts = [95.0, 97.5, 99.0]
thr_vals = [float(np.percentile(dmu[roi], p)) for p in thr_pcts]
areas_obs = [int(np.count_nonzero((dmu >= thr) & roi)) for thr in thr_vals]

print("\n[2] Obs Δ|μ| thresholds:")
for p, thr, a in zip(thr_pcts, thr_vals, areas_obs):
    print(f"  P{p:>4.1f}: thr={thr:.3g} | area={a}", flush=True)

# ---------- Permutation null ----------
print("\n[3] Permutation null …", flush=True)
hits = np.zeros(3, dtype=int)
areas_perm = []

for i in range(1, N_PERM+1):
    ks = phase_randomize(kappa, RNG)
    psi_lin_s = spectral_poisson(2.0*ks)
    mu_lin_s  = mu_from_psi(psi_lin_s)
    psi_nl_s  = nonlinear_refine_roi(psi_lin_s, ks, roi,
                                     steps=max(10, NL_STEPS//6),
                                     tau0=NL_TAU_INIT, lam=NL_LAMBDA, clip_u=NL_CLIP)
    mu_nl_s = mu_from_psi(psi_nl_s)
    dmu_s   = np.abs(mu_nl_s) - np.abs(mu_lin_s)

    areas = [int(np.count_nonzero((dmu_s >= thr) & roi)) for thr in thr_vals]
    areas_perm.append(areas)
    for j in range(3):
        if areas[j] >= areas_obs[j]:
            hits[j] += 1

    if (i % 10) == 0 or i == 1:
        print(f"  perm {i:03d}/{N_PERM} | hits: P95%={hits[0]} | P97.5%={hits[1]} | P99%={hits[2]}", flush=True)

areas_perm = np.array(areas_perm, dtype=int)
pvals = (hits + 1) / (N_PERM + 1)  # add-one 无偏估计

# ---------- Figures ----------
fig = plt.figure(figsize=(13,6))
ax1 = plt.subplot(2,3,1); im1 = ax1.imshow(kappa, cmap='coolwarm'); ax1.set_title("κ (downsampled)")
plt.colorbar(im1, ax=ax1, fraction=0.046)
ax2 = plt.subplot(2,3,2); im2 = ax2.imshow(mu_lin, cmap='magma', vmin=-50, vmax=50); ax2.set_title("μ_lin (clip)")
plt.colorbar(im2, ax=ax2, fraction=0.046)
ax3 = plt.subplot(2,3,3); im3 = ax3.imshow(roi, cmap='Greens'); ax3.set_title(f"ROI (|μ_lin|>{MU_THR}, dil={R_DILATE})")
plt.colorbar(im3, ax=ax3, fraction=0.046)

ax4 = plt.subplot(2,3,4); im4 = ax4.imshow(dmu, cmap='inferno'); ax4.set_title("Δ|μ| = |μ_nl|-|μ_lin|")
plt.colorbar(im4, ax=ax4, fraction=0.046)
ax5 = plt.subplot(2,3,5)
for thr in thr_vals:
    ax5.contour(((dmu>=thr)&roi).astype(float), levels=[0.5], colors='w', linewidths=1.0, alpha=0.8)
im5 = ax5.imshow(dmu, cmap='inferno'); ax5.set_title("Contours @ P95/97.5/99")
plt.colorbar(im5, ax=ax5, fraction=0.046)

ax6 = plt.subplot(2,3,6)
labels = ["P95","P97.5","P99"]; colors = ["C0","C1","C2"]
for j,(lab,c) in enumerate(zip(labels, colors)):
    ax6.hist(areas_perm[:,j], bins=20, alpha=0.5, label=f"{lab} null", color=c)
    ax6.axvline(areas_obs[j], color=c, linestyle='--', label=f"{lab} obs={areas_obs[j]}")
ax6.set_title("Null hist (areas in ROI)")
ax6.legend(fontsize=8)
plt.tight_layout()
FIG_PATH = os.path.join(OUT_DIR, "fig_overview.png")
plt.savefig(FIG_PATH, dpi=160)
plt.show()

# ---------- CSV ----------
CSV_PATH = os.path.join(OUT_DIR, "summary_metrics.csv")
with open(CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["when","shape","DS","sigma","MU_THR","R_DILATE","NL_STEPS","NL_TAU_INIT","NL_LAMBDA","NL_CLIP","N_PERM"])
    # 注：UTC 在新 Python 建议使用 timezone-aware；这里输出字符串即可
    w.writerow([datetime.utcnow().isoformat(), f"{kappa.shape}", DS, SIGMA_SMOOTH, MU_THR, R_DILATE,
                NL_STEPS, NL_TAU_INIT, NL_LAMBDA, NL_CLIP, N_PERM])
    w.writerow([])
    w.writerow(["pct","thr","area_obs","hits","p_est"])
    for p, thr, a, h, pv in zip(thr_pcts, thr_vals, areas_obs, hits, pvals):
        w.writerow([p, thr, a, h, pv])

print("\n[Summary]")
for p, a, h, pv in zip(thr_pcts, areas_obs, hits, pvals):
    print(f"  P{p:>4.1f}: obs_area={a} | perm_hits={h}/{N_PERM} → p≈{pv:.3f}")

print("\nArtifacts:")
print(" -", FIG_PATH)
print(" -", CSV_PATH)
