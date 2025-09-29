# ==================== 2D FDTD (TMz) — Strict TFSF (Ez/Hx) + CPML + PML Scan + Convergence + Bootstrap CI ====================
# - Strict TFSF: normal-incidence plane wave along +y. TMz => (Ez, Hx, Hy) with Hy_inc = 0, only Hx_inc nonzero; corrections at top/bottom edges.
# - Time boundary: D-continuity (E = D/eps(t)) both in 2D and 1D incident solver (consistency at temporal interface).
# - CPML absorbing boundaries.
# - PML reflection scan vs frequency (approx via early/late gating at a near-source probe).
# - Convergence (coarse/mid/fine) + Multi-Δn dose–response; M=80 seeds with paired sign-flip permutation, bootstrap CI (B=2000).
# - Colab-ready: only plt/print, no file I/O.

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Global / Physics -----------------------------
np.random.seed(2025)

c0   = 1.0
eps0 = 1.0
mu0  = 1.0

# Temporal slab parameters (applied globally in space)
T_STEP   = 2200          # up-step time index
SLAB_LEN = 1400          # plateau length; if 0 => single step, else two temporal boundaries (up & down)
SMOOTH_T = 12            # tanh smoothing (time steps)

# Source (plane wave CW with ramp), injected via strict TFSF
f0      = 0.10
omega0  = 2*np.pi*f0
SRC_AMP = 1.0
SRC_TAU = 500

# Grid / time (baseline; per-setup can override Nt)
Nx_base, Ny_base = 200, 200
S = 0.5
dx = dy = 1.0
dt = S*dx/c0
NT = 8000   # extend if you want longer late windows

# CPML parameters
NPML   = 10
CPML_m = 3.5
CPML_R = 1e-6

# TFSF window margins (rectangle; corrections only on top/bottom edges for +y incidence)
TFSF_MARGIN = 36

# Probe region (to the right of center, to emphasize propagation & mixing)
PROBE_X_OFFSET = 55
PROBE_X_W      = 15
PROBE_Y_HW     = 6

# External-time gates (late longer)
GATE_EARLY_OFFSET = 60
GATE_EARLY_LEN    = 500
GATE_LATE_OFFSET  = 1700
GATE_LATE_LEN     = 1200

# Stats
M_SEEDS = 80
PERM_B  = 5000
BOOT_B  = 2000

# Δn sweep (dose–response)
DELTA_N_LIST = [0.04, 0.06, 0.08, 0.10]  # adjust as needed
n_before = 1.50

# Convergence setups
SETUPS = [
    dict(name="coarse", Nx=160, Ny=160, Nt=7000),
    dict(name="mid",    Nx=200, Ny=200, Nt=8000),
    dict(name="fine",   Nx=260, Ny=260, Nt=10000),
]

MAKE_PLOTS = True

# ----------------------------- Helpers -----------------------------
def n_of_t_slab(n0, n1, n, t0, slab_len, smooth):
    if slab_len <= 0:
        s = 0.5*(1.0 + np.tanh((n - t0)/max(1,smooth)))
        return n0 + (n1 - n0)*s
    s_up   = 0.5*(1.0 + np.tanh((n - t0)/max(1,smooth)))
    s_down = 0.5*(1.0 + np.tanh(((t0 + slab_len) - n)/max(1,smooth)))
    plateau = s_up * s_down
    return n0 + (n1 - n0)*plateau

def hann_gate_1d(center, width, length):
    g = np.zeros(length)
    L = int(center - width//2); R = L + int(width)
    L = max(L, 0); R = min(R, length)
    m = R - L
    if m > 1:
        k = np.arange(m)
        g[L:R] = 0.5 - 0.5*np.cos(2*np.pi*k/(m-1))
    return g

def apply_gate_norm(x, g):
    xw = x * g
    nrm = np.sqrt(np.sum(g**2) + 1e-18)
    return xw / nrm

def rfft_powers(x, dt):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, d=dt)
    P = np.abs(X)**2
    return f, P

def disjoint_band_metrics(xw, dt, f0, f1, bw_scale=0.49):
    f, P = rfft_powers(xw, dt)
    sep = abs(f0 - f1)
    bw = bw_scale*sep if sep>0 else 0.0
    def band_power(fc):
        if bw == 0:
            return P[np.argmin(np.abs(f - fc))]
        m = (f >= fc - bw) & (f <= fc + bw)
        return P[m].sum()
    Pc = band_power(f0); Pr = band_power(f1)
    Rp = Pr/(Pr + Pc + 1e-18)
    dP = (Pr - Pc)/(Pr + Pc + 1e-18)
    return Rp, dP

def matched_filter_energy(xw, dt, f):
    t = np.arange(xw.size)*dt
    c = np.cos(2*np.pi*f*t); s = np.sin(2*np.pi*f*t)
    M = np.vstack([c,s]).T
    a,b = np.linalg.lstsq(M, xw, rcond=None)[0]
    return a*a + b*b

def inst_freq_demod(xw, dt, f_ref):
    t = np.arange(xw.size)*dt
    z = xw * np.exp(-1j*2*np.pi*f_ref*t)
    from scipy.signal import hilbert
    zh = hilbert(np.real(z)) + 1j*hilbert(np.imag(z))
    ph = np.unwrap(np.angle(zh))
    fi = (1.0/(2*np.pi*dt)) * np.diff(ph)
    if fi.size == 0:
        return 0.0
    fi = np.nan_to_num(fi, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.mean(fi))

def signflip_perm_pvalue(diffs, B=5000, alternative='greater'):
    diffs = np.asarray(diffs, float)
    diffs = np.nan_to_num(diffs, nan=0.0, posinf=0.0, neginf=0.0)
    if diffs.size == 0:
        return 1.0, 0.0
    obs = float(np.mean(diffs))
    n = diffs.size
    cnt = 0
    for _ in range(B):
        s = np.random.choice([-1,1], size=n)
        m = np.mean(s*diffs)
        if alternative == 'greater':
            if m >= obs: cnt += 1
        elif alternative == 'less':
            if m <= obs: cnt += 1
        else:
            if abs(m) >= abs(obs): cnt += 1
    return (cnt+1)/(B+1), obs

def bh_fdr(pvals, alpha=0.05):
    p = np.array(pvals, float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thr = alpha*(np.arange(1,m+1)/m)
    q = np.empty_like(p)
    prev = 1.0
    for i in range(m-1, -1, -1):
        qi = min(prev, m/float(i+1) * ranked[i])
        q[i] = qi; prev = qi
    qvals = np.empty_like(p); qvals[order] = q
    signif = np.zeros_like(p, bool)
    if np.any(ranked <= thr):
        k = np.where(ranked <= thr)[0].max()+1
        signif[order[:k]] = True
    return signif, qvals

def bootstrap_ci(arr, B=2000, alpha=0.05):
    arr = np.asarray(arr, float)
    n = arr.size
    if n == 0:
        return (0.0, 0.0)
    idx = np.random.randint(0, n, size=(B, n))
    samp = arr[idx].mean(axis=1)
    lo = np.percentile(samp, 100*(alpha/2))
    hi = np.percentile(samp, 100*(1-alpha/2))
    return float(lo), float(hi)

# ----------------------------- CPML Builder -----------------------------
def build_cpml(Nx, Ny, dt, dx, dy, npml=10, m=3.5, R=1e-6):
    Lx = npml*dx; Ly = npml*dy
    sigma_max_x = -(m+1)*np.log(R) / (2*Lx)
    sigma_max_y = -(m+1)*np.log(R) / (2*Ly)
    kappa_max = 7.0
    alpha_max = 0.05

    sigma_x = np.zeros(Nx); sigma_y = np.zeros(Ny)
    kappa_x = np.ones(Nx);  kappa_y = np.ones(Ny)
    alpha_x = np.zeros(Nx)+alpha_max; alpha_y = np.zeros(Ny)+alpha_max

    for i in range(npml):
        x = (npml - i)/npml
        sigma_x[i]      = sigma_max_x*(x**m)
        sigma_x[Nx-1-i] = sigma_max_x*(x**m)
        kappa_x[i]      = 1 + (kappa_max-1)*(x**m)
        kappa_x[Nx-1-i] = 1 + (kappa_max-1)*(x**m)
        y = (npml - i)/npml
        sigma_y[i]      = sigma_max_y*(y**m)
        sigma_y[Ny-1-i] = sigma_max_y*(y**m)
        kappa_y[i]      = 1 + (kappa_max-1)*(y**m)
        kappa_y[Ny-1-i] = 1 + (kappa_max-1)*(y**m)

    bx = np.exp(-(sigma_x/kappa_x + alpha_x)*dt)
    by = np.exp(-(sigma_y/kappa_y + alpha_y)*dt)
    cx = sigma_x * (bx - 1.0) / ( (sigma_x + kappa_x*alpha_x)*kappa_x + 1e-18 )
    cy = sigma_y * (by - 1.0) / ( (sigma_y + kappa_y*alpha_y)*kappa_y + 1e-18 )
    return dict(bx=bx, by=by, cx=cx, cy=cy, kx=kappa_x, ky=kappa_y)

# ----------------------------- 1D Aux FDTD for Incident (Strict TFSF) -----------------------------
# Plane wave along +y, TMz => Ez_inc(z), Hx_inc only (Hy_inc=0). Time-varying eps(t) with D-continuity.
def run_1d_incident(Ny, Nt, dt, dy, eps_t, f_src, amp=1.0, tau=SRC_TAU):
    Ez = np.zeros(Ny)        # Ez(j)
    Hx = np.zeros(Ny-1)      # Hx(j+1/2)
    Dz = np.zeros(Ny)
    inv_eps = 1.0/eps_t
    rec_Ez = []
    rec_Hx = []
    for n in range(Nt):
        # Hx update: Hx^{n+1/2} = Hx^{n-1/2} - (dt/mu0) * (Ez(j+1)-Ez(j))/dy
        Hx[:] -= (dt/mu0) * ( (Ez[1:] - Ez[:-1]) / dy )

        # D update: Dz^{n+1} = Dz^n + dt * (dHy/dx - dHx/dy), but in 1D along y, Hy=0 => Dz += -dt * dHx/dy
        dHx_dy = np.zeros_like(Ez)
        dHx_dy[1:-1] = (Hx[1:] - Hx[:-1]) / dy
        Dz += -dt * dHx_dy

        # Soft source at top j=0: add Dz source (consistent with D-continuity scheme)
        ramp = 1.0 - np.exp(-n/max(1,tau))
        s = amp * ramp * np.cos(2*np.pi*f_src*n*dt)
        Dz[0] += s

        # E from D / eps(t)
        Ez[:] = Dz * inv_eps[n]

        rec_Ez.append(Ez.copy())
        rec_Hx.append(Hx.copy())
    return np.array(rec_Ez), np.array(rec_Hx)  # shapes: (Nt, Ny), (Nt, Ny-1)

# ----------------------------- 2D FDTD (Strict TFSF Ez/Hx) -----------------------------
def run_fdtd_strict_tfsf(Delta_n, Nx, Ny, Nt, f_src=f0, bw_scale=0.49):
    n_after = n_before + Delta_n
    # eps(t)
    eps_t = np.array([eps0*(n_of_t_slab(n_before, n_after, n, T_STEP, SLAB_LEN, SMOOTH_T)**2) for n in range(Nt)], float)
    inv_eps_t = 1.0/eps_t

    # fields
    Ez = np.zeros((Nx, Ny))
    Hx = np.zeros((Nx, Ny-1))
    Hy = np.zeros((Nx-1, Ny))
    Dz = np.zeros((Nx, Ny))

    # CPML coeffs
    cp = build_cpml(Nx, Ny, dt, dx, dy, NPML, CPML_m, CPML_R)
    bx, by, cx, cy, kx, ky = cp["bx"], cp["by"], cp["cx"], cp["cy"], cp["kx"], cp["ky"]
    psi_Ezx_x = np.zeros((Nx, Ny))
    psi_Ezy_y = np.zeros((Nx, Ny))
    psi_Hxy_y = np.zeros((Nx, Ny-1))
    psi_Hyx_x = np.zeros((Nx-1, Ny))

    # TFSF window: only top/bottom edges corrections for +y incidence
    i1, i2 = TFSF_MARGIN, Nx - 1 - TFSF_MARGIN
    j1, j2 = TFSF_MARGIN, Ny - 1 - TFSF_MARGIN

    # 1D incident fields (strict consistency with D-continuity)
    Ez1D, Hx1D = run_1d_incident(Ny, Nt, dt, dy, eps_t, f_src, amp=SRC_AMP)

    # Probe region (central-right)
    cxm, cym = Nx//2, Ny//2
    px0 = max(cxm + PROBE_X_OFFSET, 1)
    px1 = min(px0 + PROBE_X_W, Nx-1)
    py0 = max(cym - PROBE_Y_HW, 1)
    py1 = min(cym + PROBE_Y_HW, Ny-1)

    probe = []

    for n in range(Nt):
        # --- H updates (with CPML) ---
        dEz_dy = (Ez[:,1:] - Ez[:,:-1]) / dy
        for j in range(Ny-1):
            psi_Hxy_y[:, j] = by[j]*psi_Hxy_y[:, j] + cy[j]*dEz_dy[:, j]
            Hx[:, j] -= (dt/mu0) * ( (1.0/ky[j])*dEz_dy[:, j] + psi_Hxy_y[:, j] )

        dEz_dx = (Ez[1:, :] - Ez[:-1, :]) / dx
        for i in range(Nx-1):
            psi_Hyx_x[i, :] = bx[i]*psi_Hyx_x[i, :] + cx[i]*dEz_dx[i, :]
            Hy[i, :] += (dt/mu0) * ( (1.0/kx[i])*dEz_dx[i, :] + psi_Hyx_x[i, :] )

        # --- Strict TFSF correction on H (top/bottom edges use Ez_inc) ---
        Ez_inc_top = Ez1D[n, j1]
        Ez_inc_bot = Ez1D[n, j2]
        Hx[i1:i2+1, j1-1] -= (dt/mu0) * ( Ez_inc_top / dy )
        Hx[i1:i2+1, j2]   += (dt/mu0) * ( Ez_inc_bot / dy )

        # --- D update (with CPML split terms) ---
        dHy_dx = np.zeros_like(Ez); dHy_dx[1:-1, :] = (Hy[1:, :] - Hy[:-1, :]) / dx
        for i in range(Nx):
            psi_Ezx_x[i, :] = bx[i]*psi_Ezx_x[i, :] + cx[i]*dHy_dx[i, :]
            dHy_dx[i, :] = (1.0/kx[i])*dHy_dx[i, :] + psi_Ezx_x[i, :]

        dHx_dy = np.zeros_like(Ez); dHx_dy[:, 1:-1] = (Hx[:, 1:] - Hx[:, :-1]) / dy
        for j in range(Ny):
            psi_Ezy_y[:, j] = by[j]*psi_Ezy_y[:, j] + cy[j]*dHx_dy[:, j]
            dHx_dy[:, j] = (1.0/ky[j])*dHx_dy[:, j] + psi_Ezy_y[:, j]

        Dz += dt * (dHy_dx - dHx_dy)

        # --- Strict TFSF correction on D (top/bottom edges use Hx_inc) ---
        # At top edge j=j1: Dz += dt * (+ dHy_inc/dx - dHx_inc/dy). For 1D along y, Hy_inc=0, so -dHx_inc/dy term applies.
        # Approx: inject -dt * dHx_inc/dy along the ring. Discretize via difference of 1D Hx at ring-adjacent positions.
        dHx_inc_top = (Hx1D[n, j1] - Hx1D[n, j1-1]) / dy if 0 < j1 <= Hx1D.shape[1]-1 else 0.0
        dHx_inc_bot = (Hx1D[n, j2] - Hx1D[n, j2-1]) / dy if 0 < j2 <= Hx1D.shape[1]-1 else 0.0
        Dz[i1:i2+1, j1] += -dt * dHx_inc_top
        Dz[i1:i2+1, j2] += +dt * dHx_inc_bot   # note sign due to orientation at bottom ring

        # --- Time boundary (D continuity): E = D / eps(t) ---
        Ez[:, :] = Dz * inv_eps_t[n]

        # --- Record probe ---
        box = Ez[px0:px1, py0:py1]
        probe.append(np.nanmean(box))

    probe = np.array(probe)
    probe = np.nan_to_num(probe, nan=0.0, posinf=0.0, neginf=0.0)
    return probe, eps_t

# ----------------------------- PML Reflection Scan (approx) -----------------------------
def pml_reflection_scan(freqs, Nx=200, Ny=200, Nt=6000):
    # No temporal change (Delta_n=0). Inject CW via strict TFSF, measure a probe near source.
    # Early gate (before round-trip) ~ incident; late gate (after reflection arrives) ~ reflected. R ≈ |A_late / A_early|.
    n0 = n_before; Delta_n = 0.0
    R_list = []
    for f in freqs:
        probe, _ = run_fdtd_strict_tfsf(Delta_n, Nx, Ny, Nt, f_src=f)
        T = len(probe)
        # crude estimate of round-trip index: pick a probe near top, but we only have central-right probe.
        # Use two windows separated sufficiently:
        gE = hann_gate_1d(center=800, width=400, length=T)
        gL = hann_gate_1d(center=Nt-1200, width=800, length=T)
        xE = apply_gate_norm(probe, gE)
        xL = apply_gate_norm(probe, gL)
        # amplitude at f
        def tone_amp(x, f):
            t = np.arange(x.size)*dt
            c = np.cos(2*np.pi*f*t); s = np.sin(2*np.pi*f*t)
            M = np.vstack([c,s]).T
            a,b = np.linalg.lstsq(M, x, rcond=None)[0]
            return np.sqrt(a*a + b*b)
        A_inc = tone_amp(xE, f) + 1e-18
        A_ref = tone_amp(xL, f)
        R_list.append(abs(A_ref/A_inc))
    return np.array(R_list)

# ----------------------------- Experiment: Convergence + Dose–Response + Stats -----------------------------
def run_dose_response_and_convergence():
    rows = []   # collect per-setup & per-Δn stats
    dose_points = []  # for plotting dose–response on mid setup

    for setup in SETUPS:
        name, Nx, Ny, Nt = setup["name"], setup["Nx"], setup["Ny"], setup["Nt"]
        print(f"\n=== Setup: {name} (Nx={Nx}, Ny={Ny}, Nt={Nt}) ===")
        for Delta_n in DELTA_N_LIST:
            fred = f0 * (n_before/(n_before+Delta_n))
            probe, eps_t = run_fdtd_strict_tfsf(Delta_n, Nx, Ny, Nt, f_src=f0)
            T = len(probe)

            # early/late gate centers
            if SLAB_LEN > 0:
                early_c = T_STEP + GATE_EARLY_OFFSET + GATE_EARLY_LEN//2
                late_c  = T_STEP + SLAB_LEN + GATE_LATE_OFFSET + GATE_LATE_LEN//2
            else:
                early_c = T_STEP + GATE_EARLY_OFFSET + GATE_EARLY_LEN//2
                late_c  = T_STEP + GATE_LATE_OFFSET + GATE_LATE_LEN//2
            early_c = int(np.clip(early_c, 0, T-1))
            late_c  = int(np.clip(late_c,  0, T-1))

            # paired seeds via gate jitter + tiny measurement noise
            Rp_diff=[]; Rpp_diff=[]; dP_diff=[]; IF_diff=[]
            for s in range(M_SEEDS):
                je = np.random.randint(-8, 9)
                jl = np.random.randint(-12, 13)
                gE = hann_gate_1d(early_c+je, GATE_EARLY_LEN, T)
                gL = hann_gate_1d(late_c +jl, GATE_LATE_LEN,  T)
                xE = apply_gate_norm(probe + 0.001*np.random.randn(T), gE)
                xL = apply_gate_norm(probe + 0.001*np.random.randn(T), gL)

                RpE, dPE = disjoint_band_metrics(xE, dt, f0, fred, bw_scale=0.35)  # conservative bands
                RpL, dPL = disjoint_band_metrics(xL, dt, f0, fred, bw_scale=0.35)
                EcE, ErE = matched_filter_energy(xE, dt, f0), matched_filter_energy(xE, dt, fred)
                EcL, ErL = matched_filter_energy(xL, dt, f0), matched_filter_energy(xL, dt, fred)
                RppE = ErE/(ErE+EcE+1e-18); RppL = ErL/(ErL+EcL+1e-18)
                IFe  = inst_freq_demod(xE, dt, f_ref=fred)
                IFl  = inst_freq_demod(xL, dt, f_ref=fred)

                Rp_diff.append(RpE-RpL)
                Rpp_diff.append(RppE-RppL)
                dP_diff.append(dPE-dPL)
                IF_diff.append(IFe-IFl)

            Rp_diff  = np.array(Rp_diff);  Rpp_diff = np.array(Rpp_diff)
            dP_diff  = np.array(dP_diff);  IF_diff  = np.array(IF_diff)

            # permutation p-values
            p_Rp,  obs_Rp  = signflip_perm_pvalue(Rp_diff,  B=PERM_B, alternative='greater')
            p_Rpp, obs_Rpp = signflip_perm_pvalue(Rpp_diff, B=PERM_B, alternative='greater')
            p_dP,  obs_dP  = signflip_perm_pvalue(dP_diff,  B=PERM_B, alternative='greater')
            p_IF,  obs_IF  = signflip_perm_pvalue(IF_diff,  B=PERM_B, alternative='two-sided')

            # bootstrap CI
            ci_Rp  = bootstrap_ci(Rp_diff,  B=BOOT_B)
            ci_Rpp = bootstrap_ci(Rpp_diff, B=BOOT_B)
            ci_dP  = bootstrap_ci(dP_diff,  B=BOOT_B)
            ci_IF  = bootstrap_ci(IF_diff,  B=BOOT_B)

            rows.append(dict(setup=name, Delta_n=Delta_n,
                             mean_Rp=float(np.mean(Rp_diff)),  p_Rp=float(p_Rp),  ci_Rp=ci_Rp,
                             mean_Rpp=float(np.mean(Rpp_diff)),p_Rpp=float(p_Rpp),ci_Rpp=ci_Rpp,
                             mean_dP=float(np.mean(dP_diff)),  p_dP=float(p_dP),  ci_dP=ci_dP,
                             mean_IF=float(np.mean(IF_diff)),  p_IF=float(p_IF),  ci_IF=ci_IF))

            print(f"Δn={Delta_n:.3f} | ΔR'={np.mean(Rp_diff):.4f} [{ci_Rp[0]:.4f},{ci_Rp[1]:.4f}] p={p_Rp:.3g} | "
                  f"ΔR''={np.mean(Rpp_diff):.4f} [{ci_Rpp[0]:.4f},{ci_Rpp[1]:.4f}] p={p_Rpp:.3g} | "
                  f"ΔΔP={np.mean(dP_diff):.4f} [{ci_dP[0]:.4f},{ci_dP[1]:.4f}] p={p_dP:.3g} | "
                  f"ΔIF={np.mean(IF_diff):.4f} [{ci_IF[0]:.4f},{ci_IF[1]:.4f}] p={p_IF:.3g}")

            # collect dose-point on mid setup for plotting
            if name == "mid":
                dose_points.append(dict(Delta_n=Delta_n,
                                        Rp_mean=np.mean(Rp_diff),  Rp_ci=ci_Rp,
                                        Rpp_mean=np.mean(Rpp_diff),Rpp_ci=ci_Rpp,
                                        dP_mean=np.mean(dP_diff),  dP_ci=ci_dP))

    # FDR across all (setups × Δn × {ΔR',ΔR'',ΔΔP,ΔIF})
    pvals = []
    for r in rows:
        pvals += [r["p_Rp"], r["p_Rpp"], r["p_dP"], r["p_IF"]]
    signif, qvals = bh_fdr(pvals, alpha=0.05)
    # assign back
    k = 0
    for r in rows:
        r["q_Rp"]  = float(qvals[k]);   r["sig_Rp"]  = bool(signif[k]);   k+=1
        r["q_Rpp"] = float(qvals[k]);   r["sig_Rpp"] = bool(signif[k]);   k+=1
        r["q_dP"]  = float(qvals[k]);   r["sig_dP"]  = bool(signif[k]);   k+=1
        r["q_IF"]  = float(qvals[k]);   r["sig_IF"]  = bool(signif[k]);   k+=1

    # print summary table
    print("\n=== FDR-controlled (BH, α=0.05) — M=80, perm_B=5000, Bstrap=2000 ===")
    header = "{:7s} {:8s} | {:>9s} {:>9s} {:>9s} {:>6s} || {:>9s} {:>9s} {:>9s} {:>6s}".format(
        "Δn","setup","ΔR'","q_R'","ΔR''","q_R''","ΔΔP","q_ΔP","ΔIF","q_IF"
    )
    print(header)
    for r in rows:
        print("{:7.3f} {:8s} | {:9.4f} {:9.3g} {:9.4f} {:9.3g} || {:9.4f} {:9.3g} {:9.4f} {:9.3g}".format(
            r["Delta_n"], r["setup"],
            r["mean_Rp"], r["q_Rp"],
            r["mean_Rpp"], r["q_Rpp"],
            r["mean_dP"], r["q_dP"],
            r["mean_IF"], r["q_IF"]
        ))

    # Dose–response plot (mid setup)
    if MAKE_PLOTS and len(dose_points)>0:
        dose_points = sorted(dose_points, key=lambda d: d["Delta_n"])
        xs = [d["Delta_n"] for d in dose_points]
        def eb(arr):
            return np.array([ [m-ci[0], ci[1]-m] for (m, ci) in arr ])
        Rp_m = [d["Rp_mean"] for d in dose_points]
        Rp_ci= [d["Rp_ci"]   for d in dose_points]
        Rpp_m= [d["Rpp_mean"] for d in dose_points]
        Rpp_ci=[d["Rpp_ci"]   for d in dose_points]
        dP_m = [d["dP_mean"] for d in dose_points]
        dP_ci= [d["dP_ci"]   for d in dose_points]

        plt.figure()
        y = np.array(Rp_m); e = eb(list(zip(Rp_m,Rp_ci)))
        plt.errorbar(xs, y, yerr=e.T, fmt='o-', label="ΔR'")
        y = np.array(Rpp_m); e = eb(list(zip(Rpp_m,Rpp_ci)))
        plt.errorbar(xs, y, yerr=e.T, fmt='s--', label="ΔR''")
        y = np.array(dP_m); e = eb(list(zip(dP_m,dP_ci)))
        plt.errorbar(xs, y, yerr=e.T, fmt='^-.' , label="ΔΔP")
        plt.xlabel("Δn"); plt.ylabel("effect (early − late)")
        plt.title("Dose–response (mid setup): effect vs Δn (mean ± 95% CI)")
        plt.legend(); plt.show()

    return rows

# ----------------------------- Run: PML scan + Dose–Response/Convergence -----------------------------
# 1) PML reflection scan (quick)
freqs = np.linspace(0.05, 0.35, 7)
print("Running PML reflection scan (approx)...")
Rmag = pml_reflection_scan(freqs, Nx=Nx_base, Ny=Ny_base, Nt=6000)
plt.figure()
plt.plot(freqs, 20*np.log10(Rmag+1e-12), marker='o')
plt.xlabel("frequency"); plt.ylabel("|R| (dB)")
plt.title("Approx. CPML reflection vs frequency")
plt.show()

# 2) Main experiment
rows = run_dose_response_and_convergence()
print("\nDone.")


