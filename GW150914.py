# ============================================================
# Time-budget κ (kappa) — Two-detector LOSC injection–recovery (one-click)
#   - Upload or detect two files: H-H1_LOSC_4_V2-*.hdf5, L-L1_LOSC_4_V2-*.hdf5
#   - Gate → FFT → banded robust coherence → coarse→fine κ scan
#   - Quadratic peak, Fisher curvature 1σ, Wilks ΔlogL=0.5 CI (auto expand)
#   - Injection on L1: phase ramp exp(+i 2π f κ_inj * SHIFT_SIGN)
#   - Reports baseline, injection, differential Δκ̂ and per-band κ(f)
# ============================================================

import os, sys, time, math, h5py, numpy as np

# -------------------- User knobs (you可调) --------------------
GATE_CENTER_S  = 16.4    # 并合在这份 32s 记录 ~16.4 s，先锁定这里
GATE_WIDTH_S   = 3.0     # 2.0→3.0 更稳
GATE_FLOOR     = 1e-3    # 门外地板（避免 w=0 退化）

FMIN, FMAX     = 35.0, 350.0   # 频带范围（优先稳健）
NBANDS         = 32            # 分带数（24/32/48）
REG_EPS        = 3e-3          # 带内幅度正则，抑制弱带

# 扫描与CI
KAPPA_SPAN0    = (-0.006, 0.006)  # 粗扫 ±6ms 覆盖广
NK_COARSE      = 241
FINE_HALF0     = 1.5e-3           # 细扫半宽（初值，会根据峰值自动扩展）
NK_FINE        = 501
AUTO_WILKS_EXPAND = True
WILKS_LEVEL    = 0.5

# 注入设置（第二轮会改）
KAPPA_INJ      = 0.0          # s；第一轮 baseline 为 0，第二轮你改成 +1e-3、+2e-3 等
SHIFT_SIGN     = +1           # +1/-1 符号叉验
SNR_SCALE      = 1.0          # 仅用于门内整体幅度缩放（谨慎使用）

# Bootstrap（频带重采样）——默认关，打开会变慢
DO_BOOTSTRAP   = False
BOOT_NB        = 300
# -------------------------------------------------------------

def _is_colab():
    try:
        import google.colab  # noqa
        return True
    except:
        return False

def tukey_window(N, alpha=0.1):
    # 纯 numpy 的 Tukey（避免 scipy.signal.tukey 依赖）
    n = np.arange(N)
    w = np.ones(N)
    L = int(alpha*(N-1)/2.0)
    if L > 0:
        w[:L+1] = 0.5*(1 + np.cos(np.pi*((2*n[:L+1]/(alpha*(N-1))) - 1)))
        w[-(L+1):] = w[:L+1][::-1]
    return w

def load_losc_hdf5(fn):
    with h5py.File(fn, 'r') as f:
        keys = list(f.keys())
        if 'strain' not in f:
            raise KeyError("HDF5不含 'strain' 组")
        ds = f['strain/Strain']
        x  = np.array(ds[:], dtype=np.float64)
        fs = None
        # 尝试 meta
        try:
            fs = float(f['meta']['SampleRate'][()])
        except:
            # 退化推断：LOSC V2 多为 4096 Hz 且整段 32 s
            fs = 4096.0
        # gps 仅信息
        try:
            gps_start = int(f['meta']['GPSstart'][()])
        except:
            gps_start = 0
        return x, fs, gps_start, keys, ds.shape

def ensure_two_files():
    # 尝试自动发现
    cand = [fn for fn in os.listdir('.') if fn.endswith('.hdf5') and ('LOSC' in fn)]
    fn_H = None
    fn_L = None
    for s in cand:
        if 'H-H1_LOSC_4_V2' in s: fn_H = s
        if 'L-L1_LOSC_4_V2' in s: fn_L = s

    if (fn_H is None) or (fn_L is None):
        if _is_colab():
            from google.colab import files
            print("请一次或多次上传两个 LOSC HDF5：H1 与 L1。\n示例：H-H1_LOSC_4_V2-*.hdf5  和  L-L1_LOSC_4_V2-*.hdf5")
            uploaded = files.upload()
            # 再次扫描
            cand = [fn for fn in os.listdir('.') if fn.endswith('.hdf5') and ('LOSC' in fn)]
            for s in cand:
                if 'H-H1_LOSC_4_V2' in s: fn_H = s
                if 'L-L1_LOSC_4_V2' in s: fn_L = s

    if (fn_H is None) or (fn_L is None):
        raise RuntimeError("❌ 未同时检测到 H1 与 L1，请重新上传。")
    return fn_H, fn_L

def make_gate(N, fs, center_s, width_s, floor=1e-3):
    t = np.arange(N)/fs
    c = center_s
    w = width_s
    gate = np.full(N, floor, dtype=np.float64)
    L = max(0.0, c - w/2)
    R = min(N/fs, c + w/2)
    m = (t>=L) & (t<=R)
    if m.sum() > 4:
        tau = (t[m]-L)/max(w, 1e-9)
        gate[m] = np.maximum(floor, 0.5 - 0.5*np.cos(2*np.pi*tau))
    return gate

def rfft_with_freqs(x, fs):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), d=1.0/fs)
    return X, f

def band_slices(freqs, fmin, fmax, nbands):
    mask = (freqs>=fmin) & (freqs<=fmax)
    idx  = np.where(mask)[0]
    if len(idx) < nbands:
        nbands = max(4, len(idx)//8)
    edges = np.linspace(idx[0], idx[-1], nbands+1).astype(int)
    bands = [(edges[i], edges[i+1]) for i in range(nbands) if edges[i+1]>edges[i]+1]
    return bands

def inject_phase_ramp(Xf, freqs, kappa, sign=+1):
    # X(f) -> X(f)*exp(+i 2π f κ * sign)
    phase = np.exp(1j*2*np.pi*freqs*kappa*sign)
    return Xf * phase

def loglike_profile(Hf, Lf, freqs, k_grid, bands, reg=3e-3):
    # 稳健带内相干：sum_b Re{ Σ_f [ Hf conj(Lf) / (|Hf||Lf|+reg) * e^{i2π f κ} ] }
    LL = np.zeros_like(k_grid, dtype=np.float64)
    for (a,b) in bands:
        H = Hf[a:b]; L = Lf[a:b]; f = freqs[a:b]
        denom = np.maximum(np.abs(H)*np.abs(L), 0) + reg
        C = (H * np.conj(L)) / denom
        # 预备 e^{i2π f κ} 的核
        # 对每个 κ: Re Σ C(f) e^{i 2π f κ}
        # 直接向量化：K x F -> 复杂度可接受
        phase = np.exp(1j*2*np.pi*np.outer(k_grid, f))
        S = np.real(phase @ C)
        LL += S
    return LL

def quad_peak(K, Y, ipeak, w=3):
    i0 = max(0, ipeak-w); i1 = min(len(K), ipeak+w+1)
    k = K[i0:i1]; y = Y[i0:i1]
    if len(k) < 3:
        return K[ipeak], np.nan
    A = np.vstack([k**2, k, np.ones_like(k)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a,b,c = coef
    if a <= 0:
        return K[ipeak], np.nan
    k_hat = -b/(2*a)
    curv  = 2*a
    return k_hat, curv

def wilks_ci(K, Y, k_hat, y_peak, level=0.5):
    # 找 ΔlogL = level 的两侧交点（线性插值，保证长度一致）
    th = y_peak - level
    # 左交点：在 K<x0 区域找 Y 上穿点
    idxL = np.where((K[:-1] < k_hat) & (Y[:-1] <= th) & (Y[1:] >= th))[0]
    kL = np.nan
    if len(idxL)>0:
        i = idxL[-1]
        xL,xR,yL,yR = K[i],K[i+1],Y[i],Y[i+1]
        a = (th - yL)/max(yR - yL, 1e-18)
        kL = xL + a*(xR - xL)
    # 右交点：在 K>x0 区域找 Y 下穿点
    idxR = np.where((K[:-1] > k_hat) & (Y[:-1] >= th) & (Y[1:] <= th))[0]
    kR = np.nan
    if len(idxR)>0:
        i = idxR[0]
        xL,xR,yL,yR = K[i],K[i+1],Y[i],Y[i+1]
        a = (th - yL)/max(yR - yL, 1e-18)
        kR = xL + a*(xR - xL)
    return kL, kR

def per_band_kappa(Hf, Lf, freqs, bands, reg=3e-3):
    # 每带上用局部相位斜率估 κ 的“近似器”（快速粗估）
    ks = []
    centers = []
    for (a,b) in bands:
        H = Hf[a:b]; L = Lf[a:b]; f = freqs[a:b]
        denom = np.maximum(np.abs(H)*np.abs(L), 0) + reg
        C = (H * np.conj(L)) / denom
        # arg(C) ~ -2π f κ  的最小二乘直线拟合
        ang = np.unwrap(np.angle(C))
        A = np.vstack([f, np.ones_like(f)]).T
        slope, offset = np.linalg.lstsq(A, -ang, rcond=None)[0]
        k_est = slope/(2*np.pi)
        ks.append(k_est)
        centers.append(float((f[0]+f[-1])/2))
    return np.array(centers), np.array(ks)

def run_once(fn_H, fn_L, kappa_inj=0.0, shift_sign=+1,
             gate_center=16.4, gate_width=3.0, gate_floor=1e-3,
             fmin=35.0, fmax=350.0, nbands=32, reg=3e-3,
             span0=(-0.006,0.006), nk0=241, fine_half=1.5e-3, nk1=501,
             auto_expand=True, level=0.5, snr_scale=1.0):

    h1, fsH, gpsH, keysH, shpH = load_losc_hdf5(fn_H)
    l1, fsL, gpsL, keysL, shpL = load_losc_hdf5(fn_L)
    if abs(fsH - fsL) > 1e-6: raise ValueError("H1/L1 采样率不一致")

    N = min(len(h1), len(l1))
    h1 = h1[:N].copy()
    l1 = l1[:N].copy()
    fs = fsH

    # 粗对齐（最大互相关，避免大幅错位）
    # 注意：这里只做整点粗移，后续 κ 扫描做亚采样精细
    def rough_align(x, y, maxshift=8192):
        # 找 y 相对 x 的最佳整数移位（xcorr）
        M = min(maxshift, N//4)
        best = 0; best_val = -1e99
        for s in range(-M, M+1):
            if s>=0:
                v = np.dot(x[:N-s], y[s:N])
            else:
                v = np.dot(x[-s:N], y[:N+s])
            if v>best_val:
                best_val = v; best = s
        return best
    s = rough_align(h1, l1, maxshift=8192)
    # 把 L1 平移 s 样点（整点），负 s 表示向前移（提早）
    if s>0:
        l1 = np.r_[l1[s:], np.zeros(s)]
    elif s<0:
        l1 = np.r_[np.zeros(-s), l1[:N+s]]

    # 门 + 可选全局SNR缩放
    gate = make_gate(N, fs, gate_center, gate_width, gate_floor)
    w = gate * tukey_window(N, alpha=0.2)
    h1 = (h1 * w) * snr_scale
    l1 = (l1 * w) * snr_scale

    # 频域
    Hf, freqs = rfft_with_freqs(h1, fs)
    Lf, _     = rfft_with_freqs(l1, fs)

    # 注入：对 L1 施加频域相位斜坡（符号可叉验）
    if abs(kappa_inj) > 0:
        Lf = inject_phase_ramp(Lf, freqs, kappa_inj, sign=shift_sign)

    # 频带划分
    bands = band_slices(freqs, fmin, fmax, nbands)

    # 粗扫
    K0 = np.linspace(span0[0], span0[1], int(nk0))
    LL0 = loglike_profile(Hf, Lf, freqs, K0, bands, reg=reg)
    i0  = int(np.argmax(LL0))
    k_peak0 = K0[i0]

    # 细扫（以粗峰为中心）
    fine_half = max(fine_half, (K0[1]-K0[0])*2)
    K1 = np.linspace(k_peak0 - fine_half, k_peak0 + fine_half, int(nk1))
    LL1 = loglike_profile(Hf, Lf, freqs, K1, bands, reg=reg)
    i1  = int(np.argmax(LL1))
    k_hat, curv = quad_peak(K1, LL1, i1, w=5)
    y_peak = np.interp(k_hat, K1, LL1) if np.isfinite(k_hat) else LL1[i1]

    # Fisher/曲率 1σ
    sigma_curv = (1.0/np.sqrt(curv)) if (curv>0 and np.isfinite(curv)) else np.nan

    # Wilks ΔlogL=0.5 区间（自动扩展）
    def try_wilks(K, Y, k_hat, y_peak, expand=True):
        kL, kR = wilks_ci(K, Y, k_hat, y_peak, level=level)
        if (not np.isfinite(kL) or not np.isfinite(kR)) and expand:
            w = (K[-1]-K[0])
            K2 = np.linspace(k_hat-2*w, k_hat+2*w, len(K))
            Y2 = loglike_profile(Hf, Lf, freqs, K2, bands, reg=reg)
            kL2, kR2 = wilks_ci(K2, Y2, k_hat, np.max(Y2), level=level)
            return (K2, Y2, kL2, kR2)
        return (K, Y, kL, kR)

    K_use, LL_use, kL, kR = try_wilks(K1, LL1, k_hat, y_peak, expand=AUTO_WILKS_EXPAND)

    # 频带 κ(f)
    centers, kbands = per_band_kappa(Hf, Lf, freqs, bands, reg=reg)

    return {
        "N": N, "fs": fs, "gate_center": gate_center, "gate_width": gate_width,
        "fmin": fmin, "fmax": fmax, "nbands": len(bands),
        "kappa_grid": K_use, "ll_grid": LL_use,
        "kappa_hat": float(k_hat), "curvature": float(curv),
        "sigma_curv": float(sigma_curv),
        "kappa_wilks": (float(kL) if np.isfinite(kL) else np.nan,
                        float(kR) if np.isfinite(kR) else np.nan),
        "kb_centers": centers, "kb_est": kbands
    }

# ----------------------------- main -----------------------------
if __name__ == "__main__":
    fn_H, fn_L = ensure_two_files()
    print(f"发现两台站数据：\nH1 = {fn_H}\nL1 = {fn_L}")

    # —— Baseline（未注入）
    out0 = run_once(fn_H, fn_L,
                    kappa_inj=0.0, shift_sign=SHIFT_SIGN,
                    gate_center=GATE_CENTER_S, gate_width=GATE_WIDTH_S, gate_floor=GATE_FLOOR,
                    fmin=FMIN, fmax=FMAX, nbands=NBANDS, reg=REG_EPS,
                    span0=KAPPA_SPAN0, nk0=NK_COARSE, fine_half=FINE_HALF0, nk1=NK_FINE,
                    auto_expand=AUTO_WILKS_EXPAND, level=WILKS_LEVEL, snr_scale=SNR_SCALE)

    # —— Injection（把 L1 注入 +κ_inj 相位斜坡）
    KAPPA_INJ_RUN = 1.0e-3   # ★建议先 1–2 ms 做演示；之后你可改
    out1 = run_once(fn_H, fn_L,
                    kappa_inj=KAPPA_INJ_RUN, shift_sign=SHIFT_SIGN,
                    gate_center=GATE_CENTER_S, gate_width=GATE_WIDTH_S, gate_floor=GATE_FLOOR,
                    fmin=FMIN, fmax=FMAX, nbands=NBANDS, reg=REG_EPS,
                    span0=KAPPA_SPAN0, nk0=NK_COARSE, fine_half=FINE_HALF0, nk1=NK_FINE,
                    auto_expand=AUTO_WILKS_EXPAND, level=WILKS_LEVEL, snr_scale=SNR_SCALE)

    # —— 汇总
    def fmt_ci_wilks(out):
        kL,kR = out["kappa_wilks"]
        if np.isfinite(kL) and np.isfinite(kR):
            return f"[{kL:.9e}, {kR:.9e}]  (half ~ {(kR-kL)/2:.3e})"
        return "未闭合（已尝试扩展；可加大 span 或增宽门/提升 SNR）"

    print("\n=== Baseline (no injection) ===")
    print(f"kappa_hat(fine)   = {out0['kappa_hat']:.9e} s")
    print(f"~1σ (curvature)   = ±{out0['sigma_curv']:.3e} s")
    print(f"~1σ (Wilks)       = {fmt_ci_wilks(out0)}")

    print("\n=== Injection (L1 → phase ramp) ===")
    print(f"kappa_hat(fine)   = {out1['kappa_hat']:.9e} s")
    print(f"~1σ (curvature)   = ±{out1['sigma_curv']:.3e} s")
    print(f"~1σ (Wilks)       = {fmt_ci_wilks(out1)}")

    dK = out1["kappa_hat"] - out0["kappa_hat"]
    print("\n=== Differential estimate (cancels geometry/base delay) ===")
    print(f"Δkappa_hat = {dK:.9e} s")
    print(f"Injected   = {KAPPA_INJ_RUN:.9e} s")
    # 简易一致性判据：如果两轮都给得出曲率 1σ，就把 1σ 合成下
    if np.isfinite(out0['sigma_curv']) and np.isfinite(out1['sigma_curv']):
        sigma_diff = math.hypot(out0['sigma_curv'], out1['sigma_curv'])
        print(f"~1σ (curv,diff)  = ±{sigma_diff:.3e} s  → |Δκ̂-κ_inj| = {abs(dK-KAPPA_INJ_RUN):.3e} s")
    else:
        print("~1σ (curv,diff)  = n/a（至少一轮曲率未可靠）")

    # 频带 κ(f)
    def dump_kf(out, tag):
        centers = out["kb_centers"].astype(int)
        ks = [f"{v:.6e}" for v in out["kb_est"]]
        print(f"\nκ(f) — {tag}")
        print(f"band centers (Hz): {centers.tolist()}")
        print(f"κ estimates (s)  : {ks}")

    dump_kf(out0, "baseline")
    dump_kf(out1, "injection")

    print("\nInterpretation tips:")
    print("- Baseline：理想 kappa_hat≈0；CI 覆盖 0；κ(f) 近常数。")
    print("- 注入–回收：Δκ̂ 与 κ_inj 在 1σ（曲率或 Wilks 半宽）内一致；如符号不合，改 SHIFT_SIGN 复跑。")
    print("- 若 Wilks 未闭合：增大 KAPPA_SPAN0、FINE_HALF0 或加宽 GATE_WIDTH_S、提高 SNR/减弱REG。")
