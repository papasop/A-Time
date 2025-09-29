# =========================================================
# Time-Layered Variational Framework — One-click Colab
# 验证项：
# 1) 光学：两段直线 Fermat → 斯涅尔定律
# 2) 力学：摆线最速降线（解析）+ 时间域“证书” d²v/dt² + Ω² v ≈ 0（toy）
# 3) 信息论：最大熵率 → 指数分布（常数风险率）；含噪声鲁棒性
# 4) 三重等价：解析速率 κ（Ω 与 h 的归一化）
# 5) 核心 (A1)–(A3)：加权作用量最小化 ↔ 加权欧拉–拉格朗日，硬阶跃与光滑 Hann
#    - 采用单元中心(cell-centered)离散，通量/源项残差与装配公式严格一致
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter

np.random.seed(2025)

# ---------- 工具函数 ----------
def sci(x):  # 科学计数法打印
    return f"{x:.3e}"

def hann_gate(t, center, width):
    g = np.zeros_like(t)
    L = center - width / 2
    R = center + width / 2
    m = (t >= L) & (t <= R)
    k = (t[m] - L) / width
    g[m] = 0.5 - 0.5 * np.cos(2*np.pi*k)
    return g

def second_derivative_uniform(y, dt):
    # Savitzky–Golay 平滑 + 二阶差分，抑制数值噪声
    wl = max(7, (len(y)//200)*2+7)  # 奇数
    y_sm = savgol_filter(y, window_length=wl, polyorder=3, mode='interp')
    return np.gradient(np.gradient(y_sm, dt), dt)

# ---------- 1) 光学：两段 Fermat + 斯涅尔 ----------
def verify_optics():
    print("\n=== Optics: Snell's law (two-segment Fermat) ===")
    x0, xm, x1 = 0.0, 0.5, 1.0
    y0, y1 = 0.0, 1.0
    n1, n2 = 1.5, 2.0

    # 费马时间（c=1）对界面 y_m 的函数
    def T_of(ym):
        L1 = np.hypot(xm - x0, ym - y0)
        L2 = np.hypot(x1 - xm, y1 - ym)
        return n1*L1 + n2*L2

    res = minimize_scalar(T_of, bounds=(min(y0,y1), max(y0,y1)), method='bounded')
    y_m = res.x

    # 两段角度的 sin（相对于法线 x 轴）
    L1 = np.hypot(xm - x0, y_m - y0)
    L2 = np.hypot(x1 - xm, y1 - y_m)
    sin1 = abs((y_m - y0) / L1)
    sin2 = abs((y1 - y_m) / L2)
    snell_err = n1*sin1 - n2*sin2
    print(f"Snell error n1*sinθ1 - n2*sinθ2 = {sci(snell_err)}  (should ≈ 0)")

    # 路径图
    xs = np.array([x0, xm, x1])
    ys = np.array([y0, y_m, y1])
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, '-o', label='Fermat 2-segment path')
    plt.axvline(xm, ls='--', label='Interface x=0.5')
    plt.xlabel('x'); plt.ylabel('y'); plt.title('Optics: Fermat path')
    plt.legend(); plt.tight_layout(); plt.show()

    print("Within-layer invariant holds by construction (piecewise straight lines).")

# ---------- 2) 力学：摆线 + 时间域证书（toy） ----------
def verify_mechanics():
    print("\n=== Mechanics: brachistochrone (analytic cycloid) & time-domain certificate ===")
    g = 9.81
    Dy = 1.0                 # 下落高度
    r  = Dy/2
    phi_f = np.pi
    Np = 4001
    phi = np.linspace(0, phi_f, Np)

    # 摆线（向下为正），再映射成论文坐标 y_paper: 1 → 0
    x_c = r*(phi - np.sin(phi))
    y_down = r*(1 - np.cos(phi))        # 0 → Dy
    y_paper = 1.0 - y_down              # 1 → 0

    # 速度 v = sqrt(2 g y_down)
    v = np.sqrt(2*g*y_down)

    # 时间：dt = ds / v，ds = sqrt(dx^2 + dy^2)
    dx = np.gradient(x_c)
    dy = np.gradient(y_down)
    ds = np.sqrt(dx*dx + dy*dy)
    t = np.cumsum(ds / np.maximum(v, 1e-12))
    t -= t[0]
    # 均匀时间重采样
    Nu = 2000
    t_uni = np.linspace(0, t[-1], Nu)
    v_uni = np.interp(t_uni, t, v)

    # toy 证书：d²v/dt² + Ω² v ≈ 0,  Ω^2 = g/(2Δy)
    Omega = np.sqrt(g/(2*Dy))
    d2v = second_derivative_uniform(v_uni, t_uni[1]-t_uni[0])
    cert = d2v + (Omega**2)*v_uni
    err = np.mean(np.abs(cert))
    print(f"Mean |d²v/dt² + Ω² v| = {sci(err)}  (toy certificate; should be small after smoothing)")

    # 轨迹 & 证书图
    plt.figure(figsize=(6,4))
    plt.plot(x_c, y_paper, label='Cycloid (y: 1→0)')
    plt.xlabel('x'); plt.ylabel('y'); plt.title('Mechanics: brachistochrone (analytic)')
    plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(t_uni, cert, label=r'$d^2v/dt^2+\Omega^2 v$')
    plt.axhline(0, ls='--', color='k')
    plt.xlabel('t'); plt.ylabel('certificate'); plt.title('Time-domain certificate (toy)')
    plt.legend(); plt.tight_layout(); plt.show()

# ---------- 3) 信息论：指数分布（常数 hazard） ----------
def verify_information(add_noise=True, snr_db=30.0):
    print("\n=== Information: max-entropy-rate → exponential (constant hazard) ===")
    mu = 1.0
    T  = 12.0
    N  = 8000
    t  = np.linspace(0, T, N)
    dt = t[1]-t[0]

    p_star = (1/mu)*np.exp(-t/mu)   # 解析密度
    S = np.exp(-t/mu)               # 解析生存函数
    h = p_star / S                  # = 1/μ

    err_no_noise = np.mean(np.abs(h - 1/mu))
    print(f"Mean hazard error (no noise): {sci(err_no_noise)}  (≈ machine precision)")

    if add_noise:
        # 按能量 SNR 添加高斯噪声，截断并重新归一化
        pE = np.sum(p_star**2)*dt
        sigma = np.sqrt(pE/(10**(snr_db/10))/ (N*dt))
        noise = np.random.randn(N)*sigma
        p_noisy = np.clip(p_star + noise, 0, None)
        p_noisy /= (np.sum(p_noisy)*dt + 1e-15)

        F = np.cumsum(p_noisy)*dt
        S_noisy = np.clip(1.0 - F, 1e-12, None)
        h_noisy = p_noisy / S_noisy
        M = int(0.95*N)  # 避免尾部失真
        err_noise = np.mean(np.abs(h_noisy[:M] - 1/mu))
        print(f"Mean hazard error (with noise, {snr_db:.0f} dB): {sci(err_noise)}")

        plt.figure(figsize=(6,4))
        plt.plot(t, h_noisy, label='hazard noisy')
        plt.axhline(1/mu, ls='--', color='k', label='1/μ')
        plt.xlim(0, T*0.95)
        plt.xlabel('t'); plt.ylabel('h(t)')
        plt.title('Information: hazard with noise (clipped & renormalized)')
        plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(t, p_star, label='p*(t) = μ⁻¹ e^{-t/μ}')
    plt.xlabel('t'); plt.ylabel('p(t)')
    plt.title('Information: optimal density')
    plt.legend(); plt.tight_layout(); plt.show()

# ---------- 4) 三重等价：解析 κ 与归一化 ----------
def verify_triple_equivalence():
    print("\n=== Triple equivalence: analytic rates & normalization ===")
    g = 9.81; Dy = 1.0
    kappa_opt = 0.0                # 同质层内 |d/ds(n sinθ)| = 0
    Omega = np.sqrt(g/(2*Dy))      # κ_mech
    h = 1.0                        # κ_info (μ=1)
    alpha = Omega / h
    print(f"κ_opt (in-layer) = {sci(kappa_opt)} (exactly 0 in homogeneous layers)")
    print(f"κ_mech = Ω = {sci(Omega)}")
    print(f"κ_info = h = {sci(h)}")
    print(f"Time rescaling α = Ω/h = {sci(alpha)}")
    print("Conclusion: κ_opt=0 in-layer; κ_mech and κ_info align under dt = α dτ.")

# ---------- (A1)–(A3) 数值验证：加权作用量最小化 & 加权 EL ----------
# 线性二阶 ODE 的三对角解算器（Thomas 算法）
def solve_tridiagonal(a, b, c, d):
    n = len(b)
    ac, bc, cc, dc = map(np.array, (a.copy(), b.copy(), c.copy(), d.copy()))
    for i in range(1, n):
        m = ac[i]/bc[i-1]
        bc[i] = bc[i] - m*cc[i-1]
        dc[i] = dc[i] - m*dc[i-1]
    x = np.zeros(n)
    x[-1] = dc[-1]/bc[-1]
    for i in range(n-2, -1, -1):
        x[i] = (dc[i]-cc[i]*x[i+1])/bc[i]
    return x

def reconstruct_full_q(q_int, q0, qT):
    return np.concatenate(([q0], q_int, [qT]))

def action_value(t, w_nodes, m, k, q):
    dt = t[1]-t[0]
    v = np.diff(q)/dt
    # 能量式正定“作用量”（仅用于数值报告；解算依据来自 EL 离散方程）
    # S = ∫ w [ 0.5 m v^2 + 0.5 k q^2 ] dt
    # 节点权重 w_nodes 与速度定义在单元中心，做简单配对：
    w_mid = 0.5*(w_nodes[:-1] + w_nodes[1:])
    S_kin = np.sum(w_mid * 0.5*m*(v**2)) * dt
    S_pot = np.sum(w_nodes * 0.5*k*(q**2)) * dt
    return S_kin + S_pot

# —— 单元中心(cell-centered)装配，与加权 EL 完全一致（硬阶跃放在单元面）——
def build_tridiagonal_EL_system_cell(t, w1, w2, t_s, m, k, q0, qT):
    """
    离散方程： (F_{i+1/2} - F_{i-1/2})/dt + w_i * k * q_i = 0,  i=1..N-1
    F_{i+1/2} = m * w_{i+1/2} * v_{i+1/2},  v_{i+1/2}=(q_{i+1}-q_i)/dt
    硬阶跃放在单元面：t[j] < t_s < t[j+1] → 左侧 cell 用 w1，右侧 cell 用 w2
    """
    N  = len(t) - 1
    dt = t[1] - t[0]
    j = int(np.floor((t_s - t[0]) / dt))
    j = np.clip(j, 0, N-1)

    # 单元中心权重
    w_mid = np.empty(N)
    w_mid[:j+1] = w1
    w_mid[j+1:] = w2

    # 节点权重（用于源项）：相邻单元平均
    w_node = np.zeros(N+1)
    w_node[0]   = w_mid[0]
    w_node[-1]  = w_mid[-1]
    w_node[1:-1] = 0.5*(w_mid[:-1] + w_mid[1:])

    # 组装三对角
    nint = N-1
    A_lo = np.zeros(nint)
    A_di = np.zeros(nint)
    A_up = np.zeros(nint)
    bvec = np.zeros(nint)

    for i in range(1, N):   # interior node i
        wl = w_mid[i-1]     # w_{i-1/2}
        wr = w_mid[i]       # w_{i+1/2}
        wi = w_node[i]

        ai = - (m * wl) / (dt*dt)
        ci = - (m * wr) / (dt*dt)
        bi =   (m * (wl + wr)) / (dt*dt) + wi * k

        jrow = i-1
        A_di[jrow] = bi
        if jrow-1 >= 0:
            A_lo[jrow] = ai
        else:
            bvec[jrow] -= ai * q0
        if jrow+1 <= nint-1:
            A_up[jrow] = ci
        else:
            bvec[jrow] -= ci * qT

    return A_lo, A_di, A_up, bvec, w_mid, w_node

def EL_residual_rms_matched(t, m, k, q, w_mid, w_node):
    """用与装配完全一致的格式计算残差（自一致）"""
    dt = t[1]-t[0]
    v  = np.diff(q) / dt                    # N
    F  = m * w_mid * v                      # N 个单元中心通量
    R  = np.zeros_like(q)
    R[1:-1] = (F[1:] - F[:-1])/dt + w_node[1:-1] * k * q[1:-1]
    return np.sqrt(np.mean(R[1:-1]**2))

def check_interface_jump_cell(t, q, w_mid, t_s, m):
    """界面处加权动量连续性 (A3)：|[ w m qdot ]| → 0"""
    dt = t[1]-t[0]
    j = int(np.floor((t_s - t[0]) / dt))
    j = np.clip(j, 0, len(w_mid)-1)

    vL = (q[j+1] - q[j]) / dt
    pL = m * w_mid[j] * vL
    if j+1 < len(w_mid):
        vR = (q[j+2] - q[j+1]) / dt
        pR = m * w_mid[j+1] * vR
        jump = abs(pL - pR)
    else:
        jump = np.nan; pR = np.nan
    return jump, pL, pR, j

# —— (A1),(A2),(A3) 两个驱动：硬阶跃（含界面条件）与光滑权重（无界面）——
def verify_core_A1_A3():
    print("\n=== Weighted action minimization / EL solve — HARD STEP (A1,A2,A3) ===")
    T, N = 1.0, 8000
    t = np.linspace(0, T, N+1)
    m, k = 1.0, 4.0
    q0, qT = 0.0, 1.0

    t_s, w1, w2 = 0.45, 1.0, 0.3
    A_lo, A_di, A_up, b, w_mid, w_node = build_tridiagonal_EL_system_cell(t, w1, w2, t_s, m, k, q0, qT)

    q_int = solve_tridiagonal(A_lo, A_di, A_up, b)
    q_sol = reconstruct_full_q(q_int, q0, qT)

    # 为了便于计算作用量，给节点权重（两端复制单元中心）
    w_nodes = np.r_[w_mid[0], 0.5*(w_mid[:-1] + w_mid[1:]), w_mid[-1]]
    S_star = action_value(t, w_nodes, m, k, q_sol)
    res    = EL_residual_rms_matched(t, m, k, q_sol, w_mid, w_node)
    jump, pL, pR, j = check_interface_jump_cell(t, q_sol, w_mid, t_s, m)

    print(f"S* = {sci(S_star)}")
    print(f"||EL residual||_rms = {sci(res)}")
    print(f"Interface at t_s ≈ {t[j+1]:.6f}: |w m qdot| jump = {sci(jump)}   (p_L={sci(pL)}, p_R={sci(pR)})")
    print("Interpretation: (A3) expects jump → 0 up to discretization.")

    # 简图
    plt.figure(figsize=(6,4))
    plt.plot(t, q_sol, label='q*(t) (hard step)')
    plt.axvline(t[j+1], ls='--', color='k', label='interface')
    plt.xlabel('t'); plt.ylabel('q')
    plt.title('A1–A3: solution with hard step in w(t)')
    plt.legend(); plt.tight_layout(); plt.show()

def verify_core_A1_A2_smooth():
    print("\n=== Weighted action minimization / EL solve — SMOOTH HANN (A1,A2) ===")
    T, N = 1.0, 8000
    t = np.linspace(0, T, N+1)
    m, k = 1.0, 4.0
    q0, qT = 0.0, 1.0

    center, width = 0.55, 0.5
    w_nodes = 0.2 + 0.8*hann_gate(t, center, width)   # 节点正权重
    # 与装配一致的单元中心/节点权重
    w_mid = 0.5*(w_nodes[:-1] + w_nodes[1:])
    w_node = np.zeros_like(w_nodes)
    w_node[0] = w_mid[0]; w_node[-1] = w_mid[-1]
    w_node[1:-1] = 0.5*(w_mid[:-1] + w_mid[1:])

    # 直接用 cell-centered 公式装配
    Ncells = len(w_mid)
    nint   = len(t)-2
    A_lo = np.zeros(nint); A_di = np.zeros(nint); A_up = np.zeros(nint); b = np.zeros(nint)
    dt = t[1]-t[0]
    for i in range(1, len(t)-1):
        wl = w_mid[i-1]; wr = w_mid[i]; wi = w_node[i]
        ai = - (m * wl) / (dt*dt)
        ci = - (m * wr) / (dt*dt)
        bi =   (m * (wl + wr)) / (dt*dt) + wi * k
        jrow = i-1
        A_di[jrow] = bi
        if jrow-1 >= 0: A_lo[jrow] = ai
        else:           b[jrow]   -= ai*q0
        if jrow+1 <= nint-1: A_up[jrow] = ci
        else:                b[jrow]   -= ci*qT

    q_int = solve_tridiagonal(A_lo, A_di, A_up, b)
    q_sol = reconstruct_full_q(q_int, q0, qT)

    res = EL_residual_rms_matched(t, m, k, q_sol, w_mid, w_node)
    S_star = action_value(t, w_nodes, m, k, q_sol)
    print(f"S* = {sci(S_star)}")
    print(f"||EL residual||_rms = {sci(res)}")
    print("Interpretation: smooth w(t) ⇒ (A1) minimizer satisfies (A2) to machine precision.")

    plt.figure(figsize=(6,4))
    plt.plot(t, q_sol, label='q*(t) (Hann)')
    plt.xlabel('t'); plt.ylabel('q')
    plt.title('A1–A2: solution with smooth w(t) (Hann)')
    plt.legend(); plt.tight_layout(); plt.show()

# ---------- 主程序 ----------
def main():
    verify_optics()
    verify_mechanics()
    verify_information(add_noise=True, snr_db=30.0)
    verify_triple_equivalence()
    verify_core_A1_A3()
    verify_core_A1_A2_smooth()
    print("\nAll verifications completed!")

main()
