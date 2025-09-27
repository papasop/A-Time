# ==== Temporal Refraction: Step + OU Noise ====
# Version: Dual metrics + Post-step-only + Hann gate + Adaptive BW + High-res Permutation
# Colab-ready: plots via matplotlib; prints summary at the end.

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq

# -----------------------------
# 1) Global parameters
# -----------------------------
rng = np.random.default_rng(2025)

c = 1.0
n0 = 1.50
carrier_f = 50.0
omega0 = 2*np.pi*carrier_f

T_total = 2.0
fs = 4000.0
t = np.arange(0, T_total, 1/fs)
N = t.size

# Gaussian pulse
t0_env = 1.0
tau = 0.15
envelope = np.exp(-0.5*((t - t0_env)/tau)**2)

# Temporal step
step_time = 1.0
Delta_n_list = [0.08, 0.10, 0.12]   # ~5–8% redshift
offset_after_step = 0.06            # gate center = step_time + this
gate_width_list = [0.50, 0.30, 0.20, 0.12]  # avoid too-tight windows

# OU "quantum-like" noise
noise_sigma = 0.004
theta = 30.0
smooth = 0.003

# Monte Carlo & permutation
M = 400          # trials per (Δn, gate)
perm_B = 10000   # permutations for p-value (high resolution)

# Adaptive spectral bands
min_bw = 3.0
alpha_bw = 0.4   # bw = max(min_bw, alpha_bw * |f0 - fred|)


# -----------------------------
# 2) Helpers
# -----------------------------
def ou_noise(T, dt, theta=40.0, sigma=0.003):
    X = np.zeros_like(T)
    coef = np.exp(-theta*dt)
    sdev = sigma*np.sqrt(1 - np.exp(-2*theta*dt))
    for i in range(1, len(T)):
        X[i] = coef*X[i-1] + sdev*rng.standard_normal()
    return X

def build_index(time, Delta_n, step_t, noise_sigma=0.003, theta=30.0, smooth=0.003):
    base = n0*np.ones_like(time)
    step = 0.5*(1 + np.tanh((time - step_t)/smooth))
    n_det = base + Delta_n*step
    dt = time[1]-time[0]
    eta = ou_noise(time, dt, theta=theta, sigma=noise_sigma)
    return n_det + eta, eta

def synthesize_field(time, Delta_n, step_t, noise_sigma=0.003, theta=30.0):
    n_t, eta = build_index(time, Delta_n, step_t, noise_sigma=noise_sigma, theta=theta, smooth=smooth)
    n_before = n0
    n_after  = n0 + Delta_n
    scale_step = np.where(time < step_t, 1.0, n_before / n_after)  # temporal boundary redshift
    det_t = n0 + Delta_n*np.where(time>=step_t, 1.0, 0.0)
    corr = 1.0 - (n_t - det_t) / n0  # linearized ω correction
    omega_t = omega0 * scale_step * corr
    dt = time[1]-time[0]
    phase = np.cumsum(omega_t) * dt
    field = envelope * np.cos(phase)
    return field, n_t, omega_t

def hann_gate(time, center, width):
    """Hann window centered at 'center' with total width 'width' (0 outside)."""
    half = width/2.0
    left, right = center - half, center + half
    gate = np.zeros_like(time, dtype=float)
    idx = np.where((time >= left) & (time <= right))[0]
    if idx.size > 0:
        m = idx.size
        # Hann: 0.5 - 0.5 cos(2π k/(m-1))
        k = np.arange(m)
        gate[idx] = 0.5 - 0.5*np.cos(2*np.pi*k/(max(m-1,1)))
    return gate

def apply_time_gate(field, time, center, width):
    return field * hann_gate(time, center, width)

def spectrum_metrics(field, Delta_n):
    F = rfft(field)
    freqs = rfftfreq(N, d=1/fs)
    P = np.abs(F)**2

    f0 = carrier_f
    f1 = carrier_f * (n0/(n0 + Delta_n)) if Delta_n > 0 else carrier_f

    # Adaptive bandwidth
    bw = max(min_bw, alpha_bw * abs(f0 - f1))

    def band_power(f_c, bw):
        m = (freqs > (f_c - bw)) & (freqs < (f_c + bw))
        return P[m].sum()

    P_c = band_power(f0, bw)
    P_r = band_power(f1, bw)

    Rp = P_r / (P_r + P_c + 1e-12)
    dP = (P_r - P_c) / (P_r + P_c + 1e-12)

    return {
        "R_prime": Rp,
        "dP": dP,
        "f_car": f0, "f_red": f1,
        "bw": bw, "freqs": freqs, "P": P
    }

def permutation_pvalue(A_vals, B_vals, B=20000):
    """One-sided: H1: mean(A) > mean(B)."""
    A_vals = np.asarray(A_vals)
    B_vals = np.asarray(B_vals)
    obs = A_vals.mean() - B_vals.mean()
    all_vals = np.concatenate([A_vals, B_vals])
    nA = A_vals.size
    cnt = 0
    for _ in range(B):
        rng.shuffle(all_vals)
        a = all_vals[:nA]
        b = all_vals[nA:]
        if (a.mean() - b.mean()) >= obs:
            cnt += 1
    return (cnt + 1) / (B + 1), obs

# -----------------------------
# 3) Main sweep
# -----------------------------
all_summaries = []
examples = []

for Delta_n in Delta_n_list:
    redshift_pct = 100*(1 - n0/(n0 + Delta_n))

    for gate_w in gate_width_list:
        center = step_time + offset_after_step
        Rp_step, Rp_null = [], []
        dP_step, dP_null = [], []
        saved = False
        ex = {}

        for k in range(M):
            # STEP
            fld, n_t, _ = synthesize_field(t, Delta_n, step_time, noise_sigma=noise_sigma, theta=theta)
            fld_g = apply_time_gate(fld, t, center, gate_w)
            met = spectrum_metrics(fld_g, Delta_n)
            Rp_step.append(met["R_prime"])
            dP_step.append(met["dP"])

            # NULL
            fld0, n0_t, _ = synthesize_field(t, 0.0, step_time, noise_sigma=noise_sigma, theta=theta)
            fld0_g = apply_time_gate(fld0, t, center, gate_w)
            met0 = spectrum_metrics(fld0_g, 0.0)
            Rp_null.append(met0["R_prime"])
            dP_null.append(met0["dP"])

            if not saved:
                ex = dict(Delta_n=Delta_n, gate_w=gate_w, center=center, redshift_pct=redshift_pct,
                          step=dict(field=fld_g, n=n_t, met=met),
                          null=dict(field=fld0_g, n=n0_t, met=met0))
                saved = True

        Rp_step = np.array(Rp_step); Rp_null = np.array(Rp_null)
        dP_step = np.array(dP_step); dP_null = np.array(dP_null)

        # High-res permutation tests
        p_Rp, diff_Rp = permutation_pvalue(Rp_step, Rp_null, B=perm_B)
        p_dP, diff_dP = permutation_pvalue(dP_step, dP_null, B=perm_B)

        lift_Rp = 100*(Rp_step.mean()/(Rp_null.mean()+1e-12) - 1)
        lift_dP = 100*(dP_step.mean()/(abs(dP_null.mean())+1e-12) - 1)  # relative to |null| mean

        all_summaries.append(dict(
            Delta_n=Delta_n,
            redshift_pct_theory=redshift_pct,
            gate_width=gate_w,
            mean_Rp_step=float(Rp_step.mean()),
            mean_Rp_null=float(Rp_null.mean()),
            diff_Rp=float(diff_Rp),
            p_Rp=float(p_Rp),
            mean_dP_step=float(dP_step.mean()),
            mean_dP_null=float(dP_null.mean()),
            diff_dP=float(diff_dP),
            p_dP=float(p_dP),
            lift_Rp=float(lift_Rp),
            trials=M
        ))
        examples.append(ex)

# -----------------------------
# 4) Plots (a few illustrative cases)
# -----------------------------
to_show = []
seen = set()
for ex in examples:
    key = (ex["Delta_n"], ex["gate_w"])
    if key not in seen:
        to_show.append(ex)
        seen.add(key)
    if len(to_show) >= 3:
        break

for ex in to_show:
    Delta_n = ex["Delta_n"]; gate_w = ex["gate_w"]
    redshift_pct = ex["redshift_pct"]

    # Spectra
    for key in ["step", "null"]:
        freqs = ex[key]["met"]["freqs"]; P = ex[key]["met"]["P"]
        fcar = ex[key]["met"]["f_car"]; fred = ex[key]["met"]["f_red"]
        bw = ex[key]["met"]["bw"]
        plt.figure()
        plt.plot(freqs, P)
        plt.axvline(fcar, linestyle="--", label="carrier")
        plt.axvline(fred, linestyle="--", label="red")
        plt.title(f"Spectrum ({key}) | Δn={Delta_n:.3f}, gate={gate_w:.3f}s, red≈{redshift_pct:.1f}%, bw≈{bw:.2f}")
        plt.xlabel("frequency"); plt.ylabel("power")
        plt.legend(); plt.show()

    # Short distributions for this setting
    Rp_s, Rp_n, dP_s, dP_n = [], [], [], []
    for _ in range(160):
        fld,_,_ = synthesize_field(t, Delta_n, step_time, noise_sigma=noise_sigma, theta=theta)
        fld_g = apply_time_gate(fld, t, ex["center"], gate_w)
        m = spectrum_metrics(fld_g, Delta_n); Rp_s.append(m["R_prime"]); dP_s.append(m["dP"])
        fld0,_,_ = synthesize_field(t, 0.0, step_time, noise_sigma=noise_sigma, theta=theta)
        fld0_g = apply_time_gate(fld0, t, ex["center"], gate_w)
        m0 = spectrum_metrics(fld0_g, 0.0); Rp_n.append(m0["R_prime"]); dP_n.append(m0["dP"])

    plt.figure()
    bins = np.linspace(0.0, 1.0, 60)
    plt.hist(Rp_s, bins=bins, alpha=0.7, density=True, label="step")
    plt.hist(Rp_n, bins=bins, alpha=0.7, density=True, label="null")
    plt.title(f"R' distributions | Δn={Delta_n:.3f}, gate={gate_w:.3f}s")
    plt.xlabel("R'"); plt.ylabel("density"); plt.legend(); plt.show()

    plt.figure()
    bins = np.linspace(-1.0, 1.0, 60)
    plt.hist(dP_s, bins=bins, alpha=0.7, density=True, label="step")
    plt.hist(dP_n, bins=bins, alpha=0.7, density=True, label="null")
    plt.title(f"ΔP distributions | Δn={Delta_n:.3f}, gate={gate_w:.3f}s")
    plt.xlabel("ΔP = (P_red - P_car)/(P_red + P_car)"); plt.ylabel("density"); plt.legend(); plt.show()

# -----------------------------
# 5) Printed summary
# -----------------------------
print("\n=== Temporal Refraction (Post-step + Hann + Adaptive BW) — Summary ===")
all_summaries = sorted(all_summaries, key=lambda d: (d["Delta_n"], d["gate_width"]))
for s in all_summaries:
    print(
        f"Δn={s['Delta_n']:.3f} | red≈{s['redshift_pct_theory']:.1f}% | gate={s['gate_width']:.3f}s | "
        f"mean R'_step={s['mean_Rp_step']:.4f} vs null={s['mean_Rp_null']:.4f} | "
        f"ΔR'={s['diff_Rp']:.4f} | p_R'={s['p_Rp']:.4g} | "
        f"mean ΔP_step={s['mean_dP_step']:.4f} vs null={s['mean_dP_null']:.4f} | "
        f"Δ(ΔP)={s['diff_dP']:.4f} | p_ΔP={s['p_dP']:.4g} | "
        f"lift_R'={s['lift_Rp']:.2f}% | trials={s['trials']}"
    )
print("Notes: One-sided permutation tests for step > null on both metrics (R' and ΔP).")
