# ==== 2D FDTD (TMz) — CPML + Soft CW Source + M=80 Perm/FDR + Δn Sweep (stable Colab) ====

import numpy as np, matplotlib.pyplot as plt

# -----------------------------
# 0) Hyper-parameters
# -----------------------------
np.random.seed(2025)
c0=1.0; eps0=1.0; mu0=1.0

DELTA_N_LIST = [0.08, 0.10]   # 需要的话加 0.12
n_before = 1.50

T_STEP   = 2200
SLAB_LEN = 1400
SMOOTH_T = 12

# 软源：域内一条 y=y_src 的 CW + 缓启（更稳）
f0=0.10; omega0=2*np.pi*f0
SRC_AMP=1.0; SRC_TAU=500
y_src_frac = 1/3   # 源行位置占 Ny 的比例

# 网格/时间
Nx, Ny = 180, 180
S=0.5; dx=dy=1.0; dt = S*dx/c0
NT=7000

# CPML
NPML=10; CPML_m=3.5; CPML_R=1e-6

# 探针放更右
PROBE_X_OFFSET=60; PROBE_X_W=15; PROBE_Y_HW=6

# 门函数（更长 late）
GATE_EARLY_OFFSET=60;  GATE_EARLY_LEN=500
GATE_LATE_OFFSET =1700; GATE_LATE_LEN =1200

# 统计
M_SEEDS=80; PERM_B=5000
MAKE_PLOTS=False

# -----------------------------
# 1) Utils
# -----------------------------
def n_of_t_slab(n0,n1,n,t0,slab_len,smooth):
    if slab_len<=0:
        s=0.5*(1+np.tanh((n-t0)/max(1,smooth))); return n0+(n1-n0)*s
    s_up=0.5*(1+np.tanh((n-t0)/max(1,smooth)))
    s_dn=0.5*(1+np.tanh(((t0+slab_len)-n)/max(1,smooth)))
    return n0+(n1-n0)*(s_up*s_dn)

def hann_gate_1d(center,width,length):
    g=np.zeros(length); L=int(center-width//2); R=L+int(width)
    L=max(L,0); R=min(R,length); m=R-L
    if m>1:
        k=np.arange(m); g[L:R]=0.5-0.5*np.cos(2*np.pi*k/(m-1))
    return g

def apply_gate_norm(x,g):
    xw=x*g; nrm=np.sqrt(np.sum(g**2)+1e-18); return xw/nrm

def rfft_powers(x,dt):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(x.size,d=dt); P=np.abs(X)**2; return f,P

def disjoint_band_metrics(xw,dt,f0,f1):
    f,P=rfft_powers(xw,dt); sep=abs(f0-f1); bw=0.49*sep if sep>0 else 0.0
    def bp(fc):
        if bw==0: return P[np.argmin(np.abs(f-fc))]
        m=(f>=fc-bw)&(f<=fc+bw); return P[m].sum()
    Pc=bp(f0); Pr=bp(f1)
    Rp=Pr/(Pr+Pc+1e-18); dP=(Pr-Pc)/(Pr+Pc+1e-18)
    return Rp, dP

def matched_filter_energy(xw,dt,f):
    t=np.arange(xw.size)*dt; c=np.cos(2*np.pi*f*t); s=np.sin(2*np.pi*f*t)
    M=np.vstack([c,s]).T; a,b=np.linalg.lstsq(M,xw,rcond=None)[0]; return a*a+b*b

def inst_freq_demod(xw,dt,f_ref):
    t=np.arange(xw.size)*dt; z=xw*np.exp(-1j*2*np.pi*f_ref*t)
    from scipy.signal import hilbert
    zh=hilbert(np.real(z))+1j*hilbert(np.imag(z))
    ph=np.unwrap(np.angle(zh)); fi=(1/(2*np.pi*dt))*np.diff(ph)
    if fi.size==0 or not np.isfinite(fi).all(): return 0.0
    return float(np.nanmean(np.nan_to_num(fi)))

def signflip_perm_pvalue(diffs,B=5000,alternative='greater'):
    diffs=np.asarray(diffs,float)
    diffs=np.nan_to_num(diffs, nan=0.0, posinf=0.0, neginf=0.0)  # 防 NaN/Inf
    if diffs.size==0: return 1.0, 0.0
    obs=float(np.mean(diffs)); n=diffs.size; cnt=0
    for _ in range(B):
        s=np.random.choice([-1,1],size=n); m=np.mean(s*diffs)
        if alternative=='greater':
            if m>=obs: cnt+=1
        elif alternative=='less':
            if m<=obs: cnt+=1
        else:
            if abs(m)>=abs(obs): cnt+=1
    return (cnt+1)/(B+1), obs

def bh_fdr(pvals,alpha=0.05):
    p=np.array(pvals,float); m=p.size; order=np.argsort(p); ranked=p[order]
    thr=alpha*(np.arange(1,m+1)/m); q=np.empty_like(p); prev=1.0
    for i in range(m-1,-1,-1):
        qi=min(prev, m/float(i+1)*ranked[i]); q[i]=qi; prev=qi
    qvals=np.empty_like(p); qvals[order]=q
    signif=np.zeros_like(p,bool)
    if np.any(ranked<=thr): k=np.where(ranked<=thr)[0].max()+1; signif[order[:k]]=True
    return signif,qvals

# -----------------------------
# 2) CPML
# -----------------------------
def build_cpml(Nx,Ny,dt,dx,dy,npml=10,m=3.5,R=1e-6):
    Lx=npml*dx; Ly=npml*dy
    sig_max_x=-(m+1)*np.log(R)/(2*Lx); sig_max_y=-(m+1)*np.log(R)/(2*Ly)
    kmax=7.0; amax=0.05
    sx=np.zeros(Nx); sy=np.zeros(Ny)
    kx=np.ones(Nx);  ky=np.ones(Ny)
    ax=np.zeros(Nx)+amax; ay=np.zeros(Ny)+amax
    for i in range(npml):
        x=(npml-i)/npml; sx[i]=sx[Nx-1-i]=sig_max_x*(x**m); kx[i]=kx[Nx-1-i]=1+(kmax-1)*(x**m)
        y=(npml-i)/npml; sy[i]=sy[Ny-1-i]=sig_max_y*(y**m); ky[i]=ky[Ny-1-i]=1+(kmax-1)*(y**m)
    bx=np.exp(-(sx/kx+ax)*dt); by=np.exp(-(sy/ky+ay)*dt)
    cx=sx*(bx-1)/((sx+kx*ax)*kx+1e-18); cy=sy*(by-1)/((sy+ky*ay)*ky+1e-18)
    return dict(bx=bx,by=by,cx=cx,cy=cy,kx=kx,ky=ky)

# -----------------------------
# 3) 2D FDTD with CPML + soft source (稳定)
# -----------------------------
def run_fdtd_once(Delta_n):
    n_after=n_before+Delta_n
    # 时变 ε(t)
    eps_t=np.array([eps0*(n_of_t_slab(n_before,n_after,n,T_STEP,SLAB_LEN,SMOOTH_T)**2) for n in range(NT)], float)
    inv_eps=1.0/eps_t

    Ez=np.zeros((Nx,Ny)); Hx=np.zeros((Nx,Ny-1)); Hy=np.zeros((Nx-1,Ny)); Dz=np.zeros((Nx,Ny))
    cp=build_cpml(Nx,Ny,dt,dx,dy,NPML,CPML_m,CPML_R)
    bx,by,cx,cy,kx,ky = cp["bx"],cp["by"],cp["cx"],cp["cy"],cp["kx"],cp["ky"]
    psi_Ezx_x=np.zeros((Nx,Ny)); psi_Ezy_y=np.zeros((Nx,Ny))
    psi_Hxy_y=np.zeros((Nx,Ny-1)); psi_Hyx_x=np.zeros((Nx-1,Ny))

    y_src=int(Ny*y_src_frac)

    # 探针
    cxm, cym = Nx//2, Ny//2
    px0=cxm+PROBE_X_OFFSET; px1=px0+PROBE_X_W
    py0=cym-PROBE_Y_HW;     py1=cym+PROBE_Y_HW
    px0=max(px0,1); px1=min(px1,Nx-1); py0=max(py0,1); py1=min(py1,Ny-1)

    probe=[]

    for n in range(NT):
        # H 更新 + CPML
        dEz_dy=(Ez[:,1:]-Ez[:,:-1])/dy
        for j in range(Ny-1):
            psi_Hxy_y[:,j]=by[j]*psi_Hxy_y[:,j]+cy[j]*dEz_dy[:,j]
            Hx[:,j]-=(dt/mu0)*((1.0/ky[j])*dEz_dy[:,j]+psi_Hxy_y[:,j])

        dEz_dx=(Ez[1:,:]-Ez[:-1,:])/dx
        for i in range(Nx-1):
            psi_Hyx_x[i,:]=bx[i]*psi_Hyx_x[i,:]+cx[i]*dEz_dx[i,:]
            Hy[i,:]+=(dt/mu0)*((1.0/kx[i])*dEz_dx[i,:]+psi_Hyx_x[i,:])

        # D 更新 + CPML
        dHy_dx=np.zeros_like(Ez); dHy_dx[1:-1,:]=(Hy[1:,:]-Hy[:-1,:])/dx
        for i in range(Nx):
            psi_Ezx_x[i,:]=bx[i]*psi_Ezx_x[i,:]+cx[i]*dHy_dx[i,:]
            dHy_dx[i,:]=(1.0/kx[i])*dHy_dx[i,:]+psi_Ezx_x[i,:]

        dHx_dy=np.zeros_like(Ez); dHx_dy[:,1:-1]=(Hx[:,1:]-Hx[:,:-1])/dy
        for j in range(Ny):
            psi_Ezy_y[:,j]=by[j]*psi_Ezy_y[:,j]+cy[j]*dHx_dy[:,j]
            dHx_dy[:,j]=(1.0/ky[j])*dHx_dy[:,j]+psi_Ezy_y[:,j]

        Dz += dt*(dHy_dx - dHx_dy)

        # 软源注入在 Dz（整列或整行都可；这里用一行）
        ramp = 1.0 - np.exp(-n/max(1,SRC_TAU))
        s = SRC_AMP * ramp * np.cos(omega0 * n * dt)
        Dz[:, y_src] += s

        # 时间边界：E = D / ε(t)
        Ez[:,:] = Dz * inv_eps[n]

        # 记录探针
        box = Ez[px0:px1, py0:py1]
        probe.append(np.nanmean(box))

    probe=np.array(probe); probe=np.nan_to_num(probe, nan=0.0, posinf=0.0, neginf=0.0)
    return probe, eps_t

# -----------------------------
# 4) Experiment: Δn sweep + M=80 seeds + perm + FDR
# -----------------------------
def run_experiment():
    all_rows=[]
    for Delta_n in DELTA_N_LIST:
        fred = f0 * (n_before/(n_before+Delta_n))
        probe, eps_t = run_fdtd_once(Delta_n)
        T=len(probe)

        # 窗口中心
        if SLAB_LEN>0:
            early_c = T_STEP + GATE_EARLY_OFFSET + GATE_EARLY_LEN//2
            late_c  = T_STEP + SLAB_LEN + GATE_LATE_OFFSET + GATE_LATE_LEN//2
        else:
            early_c = T_STEP + GATE_EARLY_OFFSET + GATE_EARLY_LEN//2
            late_c  = T_STEP + GATE_LATE_OFFSET + GATE_LATE_LEN//2

        # 保证窗口落在 [0,T)
        early_c = int(np.clip(early_c, 0, T-1))
        late_c  = int(np.clip(late_c,  0, T-1))

        Rp_diff=[]; Rpp_diff=[]; dP_diff=[]; IF_diff=[]
        for s in range(M_SEEDS):
            je = np.random.randint(-8,9); jl = np.random.randint(-12,13)
            gE = hann_gate_1d(early_c+je, GATE_EARLY_LEN, T)
            gL = hann_gate_1d(late_c +jl, GATE_LATE_LEN,  T)

            # 小测量噪声 + 归一
            xE = apply_gate_norm(probe + 0.001*np.random.randn(T), gE)
            xL = apply_gate_norm(probe + 0.001*np.random.randn(T), gL)
            xE = np.nan_to_num(xE); xL = np.nan_to_num(xL)

            RpE,dPE = disjoint_band_metrics(xE, dt, f0, fred)
            RpL,dPL = disjoint_band_metrics(xL, dt, f0, fred)
            EcE,ErE = matched_filter_energy(xE,dt,f0), matched_filter_energy(xE,dt,fred)
            EcL,ErL = matched_filter_energy(xL,dt,f0), matched_filter_energy(xL,dt,fred)
            RppE = ErE/(ErE+EcE+1e-18); RppL = ErL/(ErL+EcL+1e-18)
            IFe  = inst_freq_demod(xE, dt, f_ref=fred)
            IFl  = inst_freq_demod(xL, dt, f_ref=fred)

            Rp_diff.append(RpE-RpL); Rpp_diff.append(RppE-RppL)
            dP_diff.append(dPE-dPL);  IF_diff.append(IFe-IFl)

        # 置换 & 汇总
        p_Rp,obs_Rp   = signflip_perm_pvalue(Rp_diff,  B=PERM_B, alternative='greater')
        p_Rpp,obs_Rpp = signflip_perm_pvalue(Rpp_diff, B=PERM_B, alternative='greater')
        p_dP,obs_dP   = signflip_perm_pvalue(dP_diff,  B=PERM_B, alternative='greater')
        p_IF,obs_IF   = signflip_perm_pvalue(IF_diff,  B=PERM_B, alternative='two-sided')

        print(f"\n--- Δn={Delta_n:.3f} (fred={fred:.4f}) — seeds={M_SEEDS}, perm_B={PERM_B} ---")
        print("metric  mean    p")
        print(f"ΔR'   {np.mean(Rp_diff):7.4f}  {p_Rp:8.3g}")
        print(f"ΔR''  {np.mean(Rpp_diff):7.4f}  {p_Rpp:8.3g}")
        print(f"ΔΔP   {np.mean(dP_diff):7.4f}  {p_dP:8.3g}")
        print(f"ΔIF   {np.mean(IF_diff):7.4f}  {p_IF:8.3g}")

        all_rows += [
            dict(Delta_n=Delta_n, metric="ΔR'",  mean=float(np.mean(Rp_diff)),  p=float(p_Rp)),
            dict(Delta_n=Delta_n, metric="ΔR''", mean=float(np.mean(Rpp_diff)), p=float(p_Rpp)),
            dict(Delta_n=Delta_n, metric="ΔΔP",  mean=float(np.mean(dP_diff)),  p=float(p_dP)),
            dict(Delta_n=Delta_n, metric="ΔIF",  mean=float(np.mean(IF_diff)),  p=float(p_IF)),
        ]

        if MAKE_PLOTS:
            gE = hann_gate_1d(early_c, GATE_EARLY_LEN, T)
            gL = hann_gate_1d(late_c,  GATE_LATE_LEN,  T)
            plt.figure()
            plt.plot(probe, lw=1.0); plt.axvline(T_STEP, color='k', ls='--', label='step up')
            if SLAB_LEN>0: plt.axvline(T_STEP+SLAB_LEN, color='k', ls='--', label='step down')
            plt.plot(gE/np.max(gE+1e-18), alpha=0.6, label='early gate (scaled)')
            plt.plot(gL/np.max(gL+1e-18), alpha=0.6, label='late gate (scaled)')
            plt.title(f"Probe (Δn={Delta_n:.3f})"); plt.legend(); plt.show()

    # FDR
    pvals=[r["p"] for r in all_rows]; signif,qvals=bh_fdr(pvals, alpha=0.05)
    for r,q,s in zip(all_rows,qvals,signif): r["q"]=float(q); r["signif"]=bool(s)

    print("\n=== FDR-controlled results (BH, α=0.05) — M=80, perm_B=5000 ===")
    print("{:7s} {:6s} {:10s} {:8s} {:8s} {:6s}".format("Δn","metric","mean","p","q","sig"))
    for r in all_rows:
        print("{:7.3f} {:6s} {:10.4f} {:8.3g} {:8.3g} {:6s}".format(
            r["Delta_n"], r["metric"], r["mean"], r["p"], r["q"], str(r["signif"])
        ))

    return all_rows

# -----------------------------
# 5) Run
# -----------------------------
rows = run_experiment()

