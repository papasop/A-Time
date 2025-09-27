# =========================
# QED time-layered demo (Colab-ready)
# 外部时间直接嵌入（Hann 窗），并进行稳健频域统计
# =========================
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window
from scipy.stats import permutation_test

# -------- 参数区（可改） --------
fs        = 5_000_000.0          # 采样率 (Hz)
T         = 0.040                # 总时长 (s)
N         = int(fs*T)
dt        = 1.0/fs
f0        = 50_000.0             # 载频 50 kHz
n_minus   = 1.0
n_plus    = 1.12                 # Δn=0.12
t_step    = 20e-6                # 时间折射阶跃时刻
W         = 4e-3                 # Hann 门宽 (s)
gate_delay= 0.5                  # 门中心在阶跃后 W*gate_delay
alpha_bw  = 3.5                  # 自适应带宽：bw ≈ alpha/(2W)
nperseg_ratio = 0.5              # Welch 分段长度比例（相对于 N）
realizations   = 120             # Monte Carlo 重复
perm_resamples = 5000            # 置换检验重采样次数
noise_std      = 0.01            # 高斯噪声幅度

# -------- 工具函数 --------
rng = np.random.default_rng(2025)
t   = np.arange(N)*dt
df_FFT = 1.0/T

def hann_gate(t, center, width):
    half = width/2
    w = np.zeros_like(t)
    m = (t>=center-half) & (t<=center+half)
    if np.any(m):
        w[m] = 0.5 - 0.5*np.cos(2*np.pi*(t[m]-center+half)/width)
    return w

def synth_signal(n_before, n_after):
    """时间折射：f(t)= f0*(n_before/n(t)); n(t)为阶跃"""
    n_t = np.where(t < t_step, n_before, n_after)
    f_t = f0 * (n_minus / n_t)
    phase = 2*np.pi*np.cumsum(f_t)*dt
    x = np.cos(phase) + noise_std*rng.standard_normal(N)
    return x

def welch_psd(x, fs, nperseg_ratio=0.5, window='hann'):
    nperseg = max(256, int(N*nperseg_ratio))
    noverlap = nperseg//2
    f, Pxx = welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                   detrend='constant', return_onesided=True, scaling='density')
    return f, Pxx  # Pxx: V^2/Hz

def band_power_density(f, P, f_center, bw):
    """积分功率：∫_band P(f) df。若 band 与另一带重合则可在上层处理。"""
    lo, hi = f_center-bw, f_center+bw
    m = (f>=max(0.0,lo)) & (f<=hi) & np.isfinite(P)
    if not np.any(m):
        return 0.0
    df = np.mean(np.diff(f))
    return float(np.sum(P[m])*df)

def metrics_R_S_C(x, fs, f0_hint, n_ratio, gate_center, W, alpha_bw, nperseg_ratio=0.5):
    """计算 R'、S、质心 C；用 Welch 估计频谱；两带重合时触发 50–50 规则。"""
    # 外部时间：Hann 门
    g  = hann_gate(t, gate_center, W)
    xw = (x - x.mean()) * g

    f, P = welch_psd(xw, fs, nperseg_ratio=nperseg_ratio, window='hann')

    # 带宽与中心
    bw   = max(6*df_FFT, alpha_bw/(2*W))   # 至少 > 6*Δf_res，且随 W 自适应
    f0_c = f0_hint
    f_red_c = n_ratio * f0_c

    # 如果两带过近（重合/强重叠），启用“50–50 分配”以确保 R'_null≈0.5
    if abs(f_red_c - f0_c) < 3.0*bw:
        P_band = band_power_density(f, P, f0_c, bw)
        Pc = Pr = 0.5 * P_band
    else:
        Pc = band_power_density(f, P, f0_c,  bw)
        Pr = band_power_density(f, P, f_red_c, bw)

    denom = Pc + Pr
    if denom <= 0:
        Rprime = 0.5
        S = 0.0
    else:
        Rprime = Pr/denom
        S      = (Pr-Pc)/denom

    # 质心（仅在 gated 段上）
    df = np.mean(np.diff(f))
    Pn = np.maximum(P, 0.0)
    Ptot = np.sum(Pn)*df
    if Ptot > 0:
        C = float(np.sum(f*Pn)*df / Ptot)
    else:
        C = np.nan

    return dict(f=f, P=P, bw=bw, Rprime=Rprime, S=S, C=C)

def one_run(gated=True, n_after=n_plus):
    x = synth_signal(n_minus, n_after)
    gate_center = t_step + gate_delay*W if gated else (t_step - 5*W)
    n_ratio = n_minus/n_after
    m = metrics_R_S_C(x, fs, f0, n_ratio, gate_center, W, alpha_bw, nperseg_ratio)
    return m

def many_runs(realizations=100, perm=5000):
    R_step, R_null = [], []
    S_step, S_null = [], []
    C_step, C_null = [], []

    vis = None
    for _ in range(realizations):
        m_null = one_run(gated=True,  n_after=n_minus)  # null：n_after = n_minus
        m_step = one_run(gated=True,  n_after=n_plus)   # step：n_after = n_plus
        R_null.append(m_null['Rprime']); S_null.append(m_null['S']); C_null.append(m_null['C'])
        R_step.append(m_step['Rprime']); S_step.append(m_step['S']); C_step.append(m_step['C'])
        vis = m_step

    # 单尾置换：step > null
    def stat(x, y): return np.mean(x) - np.mean(y)
    p_R = permutation_test((R_step, R_null), stat, n_resamples=perm, alternative='greater').pvalue
    p_S = permutation_test((S_step, S_null), stat, n_resamples=perm, alternative='greater').pvalue
    # 质心希望更接近 f_red（距离更小），故比较 (|C-f0| - |C-f_red|) 的改善；这里直接比较 step 的 (f0-C) > null 的（越大越偏向红移）
    f_red_pred = f0*(n_minus/n_plus)
    step_gain  = np.array([abs(f0 - c) - abs(f_red_pred - c) for c in C_step])
    null_gain  = np.array([abs(f0 - c) - abs(f_red_pred - c) for c in C_null])
    p_C = permutation_test((step_gain, null_gain), stat, n_resamples=perm, alternative='greater').pvalue

    out = dict(
        R_step=np.mean(R_step), R_null=np.mean(R_null), dR=np.mean(R_step)-np.mean(R_null), p_R=p_R,
        S_step=np.mean(S_step), S_null=np.mean(S_null), dS=np.mean(S_step)-np.mean(S_null), p_S=p_S,
        C_step=np.nanmean(C_step), C_null=np.nanmean(C_null), p_C=p_C,
        bw=vis['bw'] if vis else np.nan,
        f=vis['f'] if vis else np.array([]), P=vis['P'] if vis else np.array([]),
    )
    return out

# -------- 运行 --------
df_res = df_FFT
gap    = f0 - f0*(n_minus/n_plus)
bw_the = alpha_bw/(2*W)

print("=== Meta (resolution & bands) ===")
print(f"fs = {fs:,.3f} Hz | T = {T:.3f} s | N = {N}")
print(f"Δf_res = {df_res:.3f} Hz (FFT 1/T)")
print(f"gap = {gap:.1f} Hz | bw ~ alpha/(2W) = {bw_the:.1f} Hz | gate W = {W:.3f} s")
print(f"f0 = {f0:.1f} Hz | f_red(pred) = {f0*(n_minus/n_plus):.1f} Hz")

# 小 pilot（给你一个感觉）
for W_try in [0.003, 0.004, 0.005]:
    W = W_try
    res_p = many_runs(realizations=16, perm=1000)
    print(f"Pilot W={W*1e3:5.1f} ms | ΔR'={res_p['dR']:+.4f} (p={res_p['p_R']:.3f}) | ΔS={res_p['dS']:+.4f} (p={res_p['p_S']:.3f})")

# 恢复首选 W
W = 4e-3
res = many_runs(realizations=realizations, perm=perm_resamples)

print("\n=== Hold-out (strict statistics) ===")
print(f"bands: f0={f0:.1f} ± {res['bw']:.1f} Hz | f_red={f0*(n_minus/n_plus):.1f} ± {res['bw']:.1f} Hz")
print("\n=== A) ΔR'（红/载功率比差 + 置换）===")
print(f"R':  step={res['R_step']:.4f} vs null={res['R_null']:.4f} | ΔR'={res['dR']:+.4f} | p={res['p_R']:.4f}")
print("\n=== B) ΔS（对称有界差 + 置换）===")
print(f"S:   step={res['S_step']:+.4f} vs null={res['S_null']:+.4f} | ΔS={res['dS']:+.4f} | p={res['p_S']:.4f}")
print("\n=== C) 质心 C（更接近 f_red 的改进，单尾）===")
print(f"C:   step={res['C_step']:.1f} Hz vs null={res['C_null']:.1f} Hz | p_C={res['p_C']:.4f}")

# -------- 可视化（一步进 vs 一空） --------
# 取一组代表性 realization 用于画图
x_null = synth_signal(n_minus, n_minus)
x_step = synth_signal(n_minus, n_plus)
gc     = t_step + gate_delay*W
gn     = hann_gate(t, gc, W)
xw_null= (x_null - x_null.mean())*gn
xw_step= (x_step - x_step.mean())*gn

fN, PN = welch_psd(xw_null, fs, nperseg_ratio, window='hann')
fS, PS = welch_psd(xw_step, fs, nperseg_ratio, window='hann')
PSn = PS/np.max(PS); PNn = PN/np.max(PN)

plt.figure(figsize=(10,5))
plt.plot(fS/1e3, PSn, label='step (gated) — normalized')
plt.plot(fN/1e3, PNn, label='null (gated) — normalized', alpha=0.8)
f_red_pred = f0*(n_minus/n_plus)
for fc, lab in [(f0,'f0'), (f_red_pred, "f_red")]:
    plt.axvline(fc/1e3, ls='--', lw=1, color='k')
    plt.fill_betweenx([0,1.05], (fc-res['bw'])/1e3, (fc+res['bw'])/1e3, color='gray', alpha=0.08)
plt.ylim(0,1.05); plt.xlim(35,60)
plt.xlabel('Frequency (kHz)'); plt.ylabel('Normalized PSD')
plt.title('Welch PSD with external-time Hann gate (one realization)')
plt.legend()
plt.grid(alpha=0.2)
plt.show()
