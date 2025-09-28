# ==== Optimized Time-Layered GR Validation — Colab Single Cell (Abell 2744) ====
# 内容： κ→gate_width (Stage A) ; FFT-Poisson ψ→gate_width (Stage B) ; 引入 magnification μ 权重 (Stage C)
# 优化： ψ 的对数压缩正规化；μ 的对数压缩；参数扫描(gamma_k/gamma_psi×sigma)；分箱曲线；Jackknife 误差
# 约束：仅 matplotlib；每图独立；不设颜色；最后 plt.show() + print 汇总

# 0) 依赖
import os, sys, subprocess, urllib.request, warnings
warnings.filterwarnings("ignore")
def ensure(pkg):
    try:
        __import__(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        __import__(pkg)
for p in ["astropy", "numpy", "matplotlib", "scipy"]:
    ensure(p)

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import ttest_ind
from scipy.ndimage import gaussian_filter

# 1) 路径与数据
OUT_DIR = "/content/out"; os.makedirs(OUT_DIR, exist_ok=True)
FITS_PATH = "/content/kappa_map.fits"
URL = "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"
if not os.path.exists(FITS_PATH):
    print("Downloading κ map:", URL, flush=True)
    urllib.request.urlretrieve(URL, FITS_PATH)
else:
    print("File exists:", FITS_PATH, flush=True)

hdul = fits.open(FITS_PATH)
header = hdul[0].header
kappa_map = hdul[0].data.astype(np.float64)
hdul.close()

print("Header info (key fields):", {k: header.get(k, None) for k in ['NAXIS1','NAXIS2','CRPIX1','CRPIX2','CDELT1','CDELT2']})
print("Data shape:", kappa_map.shape)
print("Data mean/median/std:", float(np.mean(kappa_map)), float(np.median(kappa_map)), float(np.std(kappa_map)))
print("Kappa range:", float(np.nanmin(kappa_map)), float(np.nanmax(kappa_map)))

# 2) 全局参数
z_background = np.array([8.0, 8.5, 9.0, 10.0])
c = 3e8
L = 1e25            # ~10 Mpc
T = L / c           # ~3.33e16 s (~1 Gyr)
alpha = 0.3
n_samples = 500
rng = np.random.default_rng(42)

# 3) 基本函数
def gr_redshift(z_grav, z_cosmo):
    return (1 + z_grav) * (1 + z_cosmo) - 1

def hann_mean_from_width(width, T):
    # Hann 积分 = width/2；返回 <w> = (∫w dt)/T
    return (width / 2.0) / T

def fft_poisson_psi_from_kappa(kappa):
    H,W = kappa.shape
    kx = np.fft.fftfreq(W)*2*np.pi
    ky = np.fft.fftfreq(H)*2*np.pi
    kx2, ky2 = np.meshgrid(kx**2, ky**2)
    k2 = kx2 + ky2
    K_hat = np.fft.fft2(kappa)
    psi_hat = np.zeros_like(K_hat, dtype=np.complex128)
    mask = k2!=0
    psi_hat[mask] = -2.0 * K_hat[mask] / k2[mask]
    psi = np.fft.ifft2(psi_hat).real
    return psi

def kaiser_squires_shear_from_kappa(kappa):
    H,W = kappa.shape
    kx = np.fft.fftfreq(W)*2*np.pi
    ky = np.fft.fftfreq(H)*2*np.pi
    kxg, kyg = np.meshgrid(kx, ky)
    k2 = kxg**2 + kyg**2
    K_hat = np.fft.fft2(kappa)
    Z = np.zeros_like(K_hat, dtype=np.complex128)
    mask = k2!=0
    P1 = (kxg**2 - kyg**2) / k2
    P2 = (2 * kxg * kyg) / k2
    gamma1_hat = Z.copy(); gamma2_hat = Z.copy()
    gamma1_hat[mask] = P1[mask] * K_hat[mask]
    gamma2_hat[mask] = P2[mask] * K_hat[mask]
    gamma1 = np.fft.ifft2(gamma1_hat).real
    gamma2 = np.fft.ifft2(gamma2_hat).real
    return gamma1, gamma2

def ttest_contrast(F_mean, z_grav_lens, z_background, n_samples):
    z_gr_samples = []
    z_layer_samples = []
    for z_bg in z_background:
        z_gr = gr_redshift(z_grav_lens, z_bg)
        z_layer = z_gr * F_mean
        z_gr_samples.append(rng.normal(z_gr, 0.005*max(z_gr,1e-6), n_samples))
        z_layer_samples.append(rng.normal(z_layer, 0.005*max(z_layer,1e-6), n_samples))
    z_gr_samples = np.concatenate(z_gr_samples); z_layer_samples = np.concatenate(z_layer_samples)
    t_stat, p_value = ttest_ind(z_layer_samples, z_gr_samples, alternative='greater', equal_var=False)
    return float(t_stat), float(p_value)

# 4) κ 统计（winsorize 正 κ）
kpos = kappa_map[kappa_map>0]
if kpos.size == 0:
    raise RuntimeError("No positive kappa values found.")
p1,p99 = np.percentile(kpos, [1,99])
kpos_clip = np.clip(kpos, p1, p99)
kappa_avg = float(np.mean(kpos_clip))
z_grav_lens = kappa_avg
print(f"\nAverage κ (winsorized pos 1–99%): {kappa_avg:.4f} -> z_grav≈{z_grav_lens:.4f}")

# 5) 可视化 κ
plt.figure(figsize=(10,8))
plt.imshow(kappa_map, origin='lower')
plt.colorbar(label='Kappa (Convergence)')
plt.title('Abell 2744 Kappa Map (HST Frontier Fields v4)')
plt.xlabel('Pixel X'); plt.ylabel('Pixel Y')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'fig01_kappa_map.png'), dpi=150); plt.show()

# --------- Stage A：norm κ → gate_width_map（增强：分位裁剪+平滑+归一） ---------
sigma_A = 2.0
kpos_map = np.where(kappa_map>0, kappa_map, 0.0)
kappa_sm_A = gaussian_filter(kpos_map, sigma=sigma_A)
lo_A, hi_A = np.percentile(kappa_sm_A[kappa_sm_A>0], [5,95]) if np.any(kappa_sm_A>0) else (0.0,1.0)
kappa_sm_clip_A = np.clip(kappa_sm_A, lo_A, hi_A)
den_A = (hi_A-lo_A) if (hi_A>lo_A) else 1.0
norm_kappa_A = np.zeros_like(kappa_sm_clip_A)
mask_A = kappa_sm_clip_A>0
norm_kappa_A[mask_A] = np.clip((kappa_sm_clip_A[mask_A]-lo_A)/den_A, 0.0, 1.0)
gamma_k = 0.2
gate_width_A = gamma_k * norm_kappa_A * T
F_map_A = 1.0 + alpha * hann_mean_from_width(gate_width_A, T)

weights_k = kappa_sm_clip_A.clip(min=0)
w_sum_A = weights_k.sum() if weights_k.sum()>0 else 1.0
F_mean_A = float((F_map_A * weights_k).sum() / w_sum_A)

# Jackknife (A)
H,W = kappa_map.shape
B = 10; bx, by = H//B, W//B
F_blocks_A = []
for i in range(B):
    for j in range(B):
        sub = kappa_map[i*bx:(i+1)*bx, j*by:(j+1)*by]
        sub_pos = np.where(sub>0, sub, 0.0)
        if (sub_pos>0).sum()<100: continue
        sm = gaussian_filter(sub_pos, sigma=sigma_A)
        slo,shi = np.percentile(sm[sm>0],[5,95]) if np.any(sm>0) else (0.0,1.0)
        smc = np.clip(sm, slo, shi)
        denb = (shi-slo) if (shi>slo) else 1.0
        nk = np.zeros_like(smc); m = smc>0
        nk[m] = np.clip((smc[m]-slo)/denb, 0.0, 1.0)
        gw = gamma_k * nk * T
        Fm = 1.0 + alpha * hann_mean_from_width(gw, T)
        wk = smc.clip(min=0); s = wk.sum() if wk.sum()>0 else 1.0
        F_blocks_A.append(float((Fm*wk).sum()/s))
F_blocks_A = np.array(F_blocks_A) if len(F_blocks_A)>0 else np.array([])

plt.figure(figsize=(8,5))
plt.hist(F_blocks_A, bins=15)
plt.xlabel('F_mean (blocks)'); plt.ylabel('Count'); plt.title('Stage A Jackknife F distribution')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'figA_jackknife_hist.png'), dpi=150); plt.show()

# κ 分箱 (A)
kp = kappa_map[kappa_map>0]
edges = np.percentile(kp, [0,10,30,50,70,90,100])
bin_rows_A = []
for lo2,hi2 in zip(edges[:-1], edges[1:]):
    slab_mask = (kappa_map>=lo2)&(kappa_map<hi2)
    if slab_mask.sum()<500: continue
    F_bin = F_map_A[slab_mask]
    bin_rows_A.append((lo2,hi2,float(F_bin.mean()),float(F_bin.std())))

# --------- Stage B：ψ（FFT–Poisson）→ 对数压缩正规化 → gate_width_map ---------
psi = fft_poisson_psi_from_kappa(kappa_map)
psi_sm = gaussian_filter(np.maximum(psi, 0.0), sigma=2.0)
plo, phi = np.percentile(psi_sm[psi_sm>0], [10, 99]) if np.any(psi_sm>0) else (0.0,1.0)
psi_clip = np.clip(psi_sm, plo, phi)
psi_log = np.log1p(psi_clip / max(psi_clip.mean(), 1e-9))
psi_log = gaussian_filter(psi_log, sigma=2.0)
p2, p98 = np.percentile(psi_log, [2, 98])
norm_psi = np.clip((psi_log - p2) / max(p98 - p2, 1e-9), 0.0, 1.0)

gamma_psi = 0.3
gate_width_B = gamma_psi * norm_psi * T
F_map_B = 1.0 + alpha * hann_mean_from_width(gate_width_B, T)

# κ 权重下的平均（与 A 一致以便可比）
F_mean_B = float((F_map_B * weights_k).sum() / w_sum_A)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(psi, origin='lower'); plt.title('Lens potential ψ (raw)'); plt.colorbar()
plt.subplot(1,2,2); plt.imshow(F_map_B, origin='lower'); plt.title('F_map (Stage B, ψ-based)'); plt.colorbar()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'figB_psi_Fmap.png'), dpi=150); plt.show()

# --------- Stage C：magnification μ 权重（对数压缩，均值归一） ---------
gamma1, gamma2 = kaiser_squires_shear_from_kappa(kappa_map)
mu = 1.0 / np.clip((1.0 - kappa_map)**2 - (gamma1**2 + gamma2**2), 1e-3, 1e6)
mu_sm = gaussian_filter(mu, sigma=2.0)
mlo, mhi = np.percentile(mu_sm, [5,95])
mu_clip = np.clip(mu_sm, mlo, mhi)
mu_log = np.log1p(mu_clip)                      # 对数压缩
mu_w = mu_log / max(mu_log.mean(),1e-9)         # 均值归一
w_mu_sum = mu_w.sum() if mu_w.sum()>0 else 1.0

F_mean_A_mu = float((F_map_A * mu_w).sum() / w_mu_sum)
F_mean_B_mu = float((F_map_B * mu_w).sum() / w_mu_sum)

# 8) 结果打印
def print_stage_rows(tag, F_mean):
    print(f"\n{tag}")
    print("Background z | Traditional GR z | Time-Layered z | F_mean")
    for z_bg in z_background:
        z_gr = gr_redshift(z_grav_lens, z_bg)
        z_layer = z_gr * F_mean
        print(f"{z_bg:4.1f}        | {z_gr:10.3f}        | {z_layer:10.3f}    | {F_mean:6.4f}")

print_stage_rows("=== Stage A (κ→gate_width, κ-weighted) ===", F_mean_A)
print_stage_rows("=== Stage B (ψ→gate_width, κ-weighted) ===", F_mean_B)
print_stage_rows("=== Stage C (μ-weighted averages) — A-map ===", F_mean_A_mu)
print_stage_rows("=== Stage C (μ-weighted averages) — B-map ===", F_mean_B_mu)

# 9) t 检验
tA,  pA  = ttest_contrast(F_mean_A,   z_grav_lens, z_background, n_samples)
tB,  pB  = ttest_contrast(F_mean_B,   z_grav_lens, z_background, n_samples)
tAμ, pAμ = ttest_contrast(F_mean_A_mu,z_grav_lens, z_background, n_samples)
tBμ, pBμ = ttest_contrast(F_mean_B_mu,z_grav_lens, z_background, n_samples)

# 10) A/B/C 的分箱曲线（给出均值±std）
def plot_binned_curve(xmap, Fmap, title, fname):
    x = xmap[xmap>0].ravel()
    F = Fmap[xmap>0].ravel()
    qs = np.percentile(x, [0,10,30,50,70,90,100])
    centers, means, stds = [], [], []
    for lo,hi in zip(qs[:-1], qs[1:]):
        sel = (x>=lo)&(x<hi)
        if sel.sum()<500: continue
        centers.append(0.5*(lo+hi))
        Fi = F[sel]
        means.append(float(Fi.mean()))
        stds.append(float(Fi.std()))
    plt.figure(figsize=(8,5))
    plt.errorbar(centers, means, yerr=stds, fmt='o-')
    plt.xlabel('bin center')
    plt.ylabel('F mean ± std')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,fname), dpi=150); plt.show()

plot_binned_curve(kappa_map.clip(min=0), F_map_A, "Stage A: F vs κ (binned)", "figA_F_vs_kappa.png")
plot_binned_curve(np.maximum(psi,0),       F_map_B, "Stage B: F vs ψ (binned)", "figB_F_vs_psi.png")
plot_binned_curve(mu_w,                    F_map_B, "Stage C: F (B-map) vs μ (binned)", "figC_F_vs_mu.png")

# 11) 参数扫描：gamma×sigma（二维网格），展示 F_mean 与 Jackknife std
def jackknife_F_mean(kmap, sigma, gamma, mode="kappa"):
    # mode="kappa" 用 norm κ；mode="psi" 用 ψ（对数压缩流程）
    if mode=="kappa":
        sm = gaussian_filter(np.where(kmap>0,kmap,0.0), sigma=sigma)
        if not np.any(sm>0): return np.nan, np.nan
        lo,hi = np.percentile(sm[sm>0],[5,95]); smc=np.clip(sm,lo,hi)
        den=(hi-lo) if (hi>lo) else 1.0
        nk = np.zeros_like(smc); m=smc>0; nk[m]=np.clip((smc[m]-lo)/den,0,1)
        gw = gamma * nk * T
        Fm = 1.0 + alpha * hann_mean_from_width(gw, T)
        wk = smc.clip(min=0); s = wk.sum() if wk.sum()>0 else 1.0
        F_mean = float((Fm*wk).sum()/s)
        # Jackknife
        H,W = kmap.shape; B=8; bx,by=H//B, W//B
        Fs=[]
        for i in range(B):
            for j in range(B):
                sub = kmap[i*bx:(i+1)*bx, j*by:(j+1)*by]
                subp = np.where(sub>0,sub,0.0)
                if (subp>0).sum()<100: continue
                sm2=gaussian_filter(subp, sigma=sigma)
                if not np.any(sm2>0): continue
                slo,shi=np.percentile(sm2[sm2>0],[5,95]); smc2=np.clip(sm2,slo,shi)
                denb=(shi-slo) if (shi>slo) else 1.0
                nk2=np.zeros_like(smc2); m2=smc2>0; nk2[m2]=np.clip((smc2[m2]-slo)/denb,0,1)
                gw2=gamma * nk2 * T
                Fm2=1.0 + alpha * hann_mean_from_width(gw2, T)
                wk2=smc2.clip(min=0); s2=wk2.sum() if wk2.sum()>0 else 1.0
                Fs.append(float((Fm2*wk2).sum()/s2))
        Fs=np.array(Fs) if len(Fs)>0 else np.array([np.nan])
        return F_mean, float(np.nanstd(Fs))
    else:
        psi0 = fft_poisson_psi_from_kappa(kmap)
        psi_sm = gaussian_filter(np.maximum(psi0,0.0), sigma=sigma)
        if not np.any(psi_sm>0): return np.nan, np.nan
        plo,phi=np.percentile(psi_sm[psi_sm>0],[10,99]); psi_clip=np.clip(psi_sm,plo,phi)
        psi_log = np.log1p(psi_clip / max(psi_clip.mean(),1e-9))
        psi_log = gaussian_filter(psi_log, sigma=2.0)
        p2,p98=np.percentile(psi_log,[2,98]); norm=np.clip((psi_log-p2)/max(p98-p2,1e-9),0,1)
        gw = gamma * norm * T
        Fm = 1.0 + alpha * hann_mean_from_width(gw, T)
        wk = np.clip(gaussian_filter(np.where(kmap>0,kmap,0.0), sigma=sigma), 0, None)
        s = wk.sum() if wk.sum()>0 else 1.0
        F_mean = float((Fm*wk).sum()/s)
        # Jackknife
        H,W = kmap.shape; B=8; bx,by=H//B, W//B
        Fs=[]
        for i in range(B):
            for j in range(B):
                sub = kmap[i*bx:(i+1)*bx, j*by:(j+1)*by]
                if (sub>0).sum()<100: continue
                psi0s = fft_poisson_psi_from_kappa(sub)
                psi_sms = gaussian_filter(np.maximum(psi0s,0.0), sigma=sigma)
                if not np.any(psi_sms>0): continue
                pl,ph=np.percentile(psi_sms[psi_sms>0],[10,99]); ps=np.clip(psi_sms,pl,ph)
                lg=np.log1p(ps / max(ps.mean(),1e-9))
                lg=gaussian_filter(lg, sigma=2.0)
                q2,q98=np.percentile(lg,[2,98]); nm=np.clip((lg-q2)/max(q98-q2,1e-9),0,1)
                gw2=gamma * nm * T
                Fm2=1.0 + alpha * hann_mean_from_width(gw2, T)
                wk2=np.clip(gaussian_filter(np.where(sub>0,sub,0.0), sigma=sigma),0,None)
                s2=wk2.sum() if wk2.sum()>0 else 1.0
                Fs.append(float((Fm2*wk2).sum()/s2))
        Fs=np.array(Fs) if len(Fs)>0 else np.array([np.nan])
        return F_mean, float(np.nanstd(Fs))

gammas = [0.1, 0.2, 0.3, 0.4]
sigmas  = [1.0, 2.0, 3.0, 4.0]

Fgrid_A, Egrid_A = np.zeros((len(sigmas),len(gammas))), np.zeros((len(sigmas),len(gammas)))
Fgrid_B, Egrid_B = np.zeros((len(sigmas),len(gammas))), np.zeros((len(sigmas),len(gammas)))

for si,sg in enumerate(sigmas):
    for gi,gm in enumerate(gammas):
        fm, se = jackknife_F_mean(kappa_map, sigma=sg, gamma=gm, mode="kappa")
        Fgrid_A[si,gi] = fm; Egrid_A[si,gi] = se
        fm, se = jackknife_F_mean(kappa_map, sigma=sg, gamma=gm, mode="psi")
        Fgrid_B[si,gi] = fm; Egrid_B[si,gi] = se

# 简单成像（非 seaborn）
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(Fgrid_A, origin='lower'); plt.xticks(range(len(gammas)), gammas); plt.yticks(range(len(sigmas)), sigmas)
plt.xlabel('gamma_k'); plt.ylabel('sigma'); plt.title('Stage A: F_mean (grid)'); plt.colorbar()
plt.subplot(1,2,2); plt.imshow(Egrid_A, origin='lower'); plt.xticks(range(len(gammas)), gammas); plt.yticks(range(len(sigmas)), sigmas)
plt.xlabel('gamma_k'); plt.ylabel('sigma'); plt.title('Stage A: Jackknife std (grid)'); plt.colorbar()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'figScan_A.png'), dpi=150); plt.show()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(Fgrid_B, origin='lower'); plt.xticks(range(len(gammas)), gammas); plt.yticks(range(len(sigmas)), sigmas)
plt.xlabel('gamma_psi'); plt.ylabel('sigma'); plt.title('Stage B: F_mean (grid)'); plt.colorbar()
plt.subplot(1,2,2); plt.imshow(Egrid_B, origin='lower'); plt.xticks(range(len(gammas)), gammas); plt.yticks(range(len(sigmas)), sigmas)
plt.xlabel('gamma_psi'); plt.ylabel('sigma'); plt.title('Stage B: Jackknife std (grid)'); plt.colorbar()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'figScan_B.png'), dpi=150); plt.show()

# 12) 汇总
print("\n==== SUMMARY ====")
print(f"⟨κ⟩ (winsorized, pos) = {kappa_avg:.4f}  -> z_grav≈{z_grav_lens:.4f}")
print(f"Stage A: F_mean_A(κ-wt) = {F_mean_A:.6f} | Jackknife: N={len(F_blocks_A)}, mean={np.nanmean(F_blocks_A):.6f}, std={np.nanstd(F_blocks_A):.6f}")
print(f"Stage B: F_mean_B(κ-wt) = {F_mean_B:.6f}")
print(f"Stage C: F_Aμ = {F_mean_A_mu:.6f}  |  F_Bμ = {F_mean_B_mu:.6f}")
tA,  pA  = ttest_contrast(F_mean_A,    z_grav_lens, z_background, n_samples)
tB,  pB  = ttest_contrast(F_mean_B,    z_grav_lens, z_background, n_samples)
tAμ, pAμ = ttest_contrast(F_mean_A_mu, z_grav_lens, z_background, n_samples)
tBμ, pBμ = ttest_contrast(F_mean_B_mu, z_grav_lens, z_background, n_samples)
print(f"T-test (layer>GR):  A: t={tA:.3f}, p={pA:.2e} | B: t={tB:.3f}, p={pB:.2e} | Aμ: t={tAμ:.3f}, p={pAμ:.2e} | Bμ: t={tBμ:.3f}, p={pBμ:.2e}")
print(f"Scans saved: figScan_A.png / figScan_B.png")
print(f"Saved figures to: {OUT_DIR}/")
