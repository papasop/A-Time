# =========================================================
# GL2 (order-4) variational integrator for weighted EL ODE
#   L = 1/2 m qdot^2 + V(q),   EL: d/dt (m w qdot) - w dV/dq = 0
#   => qddot = (k/m) q - (w'/w) qdot   (V=1/2 k q^2 示例；可替换成一般 V)
# - Smooth w(t): 演示四阶收敛
# - Hard step  : 对齐界面并用 A3: p=m w qdot 连续，重置右侧初速
# =========================================================

import numpy as np

# ---------- utilities ----------
def sci(x): return f"{x:.6e}"

# ---------- problem setup ----------
m = 1.0
k = 10.0                     # 可调；适中避免刚性
def V(q):      return 0.5*k*q*q
def dVdq(q):   return k*q

# (1) 平滑权重（处处C^∞）——用于4阶收敛验证
def w_smooth(t, a=0.3):      # w(t)=1+a sin(2πt) > 0
    return 1.0 + a*np.sin(2*np.pi*t)
def wdot_smooth(t, a=0.3):
    return a*(2*np.pi)*np.cos(2*np.pi*t)

# (2) 硬截止（分段常数）——用于A3界面条件验证
def w_hard(t, t_s=0.5, wL=1.0, wR=0.2):
    return np.where(t < t_s, wL, wR)

# 加权EL对应的一阶系统 y=[q,v]
# y' = f(t,y) with q' = v
# v' = (k/m) q - (w'/w) v
def f_EL(t, y, w_fun, wdot_fun=None):
    q, v = y
    w  = w_fun(t)
    if wdot_fun is None:
        # 分段常数场景（硬截止子区间内），w' = 0
        wp = 0.0
    else:
        wp = wdot_fun(t)
    return np.array([v, (k/m)*q - (wp/max(w,1e-30))*v], dtype=float)

# ---------- GL2（Gauss–Legendre 2-stage, order 4）一步 ----------
# Butcher tableau:
# c1,2 = 1/2 ± sqrt(3)/6;  b1=b2=1/2
# A = [[1/4, 1/4 - s], [1/4 + s, 1/4]],  s = sqrt(3)/6
c1 = 0.5 - np.sqrt(3)/6.0
c2 = 0.5 + np.sqrt(3)/6.0
b1 = b2 = 0.5
s  = np.sqrt(3)/6.0
A11, A12 = 0.25, 0.25 - s
A21, A22 = 0.25 + s, 0.25

def gl2_step(tn, yn, h, w_fun, wdot_fun):
    """
    Implicit GL2 step for y' = f(t,y) with 2D state y=[q,v].
    Newton solve for K = [k1, k2], each ki in R^2.
    """
    # 初值：用显式Euler做个粗猜
    f0 = f_EL(tn, yn, w_fun, wdot_fun)
    K1 = f0.copy()
    K2 = f0.copy()

    for _ in range(20):
        Y1 = yn + h*(A11*K1 + A12*K2)
        Y2 = yn + h*(A21*K1 + A22*K2)
        F1 = f_EL(tn + c1*h, Y1, w_fun, wdot_fun)
        F2 = f_EL(tn + c2*h, Y2, w_fun, wdot_fun)

        # 残差：G(K)=0
        R1 = K1 - F1
        R2 = K2 - F2
        res = np.linalg.norm(np.hstack([R1,R2]), ord=2)
        if res < 1e-12:
            break

        # 雅可比： dF/dY = Jf(t,Y)
        # Jf = [[0, 1],[k/m, -(w'/w)]], 评估在 Y1, Y2 时刻
        # dR1/dK1 = I - h*A11*Jf(t1),  dR1/dK2 = -h*A12*Jf(t1)
        # dR2/dK1 = -h*A21*Jf(t2),     dR2/dK2 = I - h*A22*Jf(t2)
        def Jf(t, y):
            q, v = y
            w  = w_fun(t)
            if wdot_fun is None: wp = 0.0
            else: wp = wdot_fun(t)
            return np.array([[0.0, 1.0],
                             [k/m, -wp/max(w,1e-30)]], dtype=float)

        J1 = Jf(tn + c1*h, Y1)
        J2 = Jf(tn + c2*h, Y2)

        I = np.eye(2)
        M11 = I - h*A11*J1
        M12 =   - h*A12*J1
        M21 =   - h*A21*J2
        M22 = I - h*A22*J2

        # 4x4 线性系统求解增量 dK
        M = np.block([[M11, M12],
                      [M21, M22]])
        rhs = -np.hstack([R1, R2])
        dK  = np.linalg.solve(M, rhs)
        K1 += dK[:2]
        K2 += dK[2:]

    yn1 = yn + h*(b1*K1 + b2*K2)
    return yn1

def gl2_solve(t0, t1, y0, N, w_fun, wdot_fun):
    t = np.linspace(t0, t1, N+1)
    h = (t1 - t0)/N
    Y = np.zeros((N+1, 2), dtype=float)
    Y[0] = y0
    for n in range(N):
        Y[n+1] = gl2_step(t[n], Y[n], h, w_fun, wdot_fun)
    return t, Y

# ---------- diagnostics ----------
def residual_EL(t, Y, w_fun, wdot_fun):
    """R = d/dt (m w v) - w k q，用中心差分近似 d/dt(...)"""
    q = Y[:,0]; v = Y[:,1]
    w = w_fun(t)
    if wdot_fun is None:
        wp = np.zeros_like(t)
    else:
        wp = wdot_fun(t)
    p = m*w*v
    # 中心差分
    dt = t[1]-t[0]
    dp = np.zeros_like(p)
    dp[1:-1] = (p[2:] - p[:-2])/(2*dt)
    dp[0]  = (p[1]-p[0])/dt
    dp[-1] = (p[-1]-p[-2])/dt
    R = dp - w*(k*q)
    return R

# =========================================================
# (A) 平滑 w：四阶收敛验证（与高分辨率参考解比较）
# =========================================================
def test_smooth_convergence():
    print("=== Smooth w(t): order-4 convergence test (GL2) ===")
    t0, t1 = 0.0, 1.0
    y0 = np.array([0.0, 1.0])   # 初值：q(0)=0, v(0)=1
    # 参考解（很细的步长）
    N_ref = 32768
    t_ref, Y_ref = gl2_solve(t0, t1, y0, N_ref, w_smooth, wdot_smooth)
    q_ref = Y_ref[:,0]; v_ref = Y_ref[:,1]

    for N in [128, 256, 512, 1024, 2048]:
        tN, YN = gl2_solve(t0, t1, y0, N, w_smooth, wdot_smooth)
        # 与参考对齐（下采样）
        step = N_ref//N
        q_err = np.linalg.norm(YN[:,0] - q_ref[::step], ord=2)/np.sqrt(N+1)
        v_err = np.linalg.norm(YN[:,1] - v_ref[::step], ord=2)/np.sqrt(N+1)
        print(f"N={N:5d},  L2(q)={sci(q_err)},  L2(v)={sci(v_err)}")
    print("期望：误差 ~ O(N^{-4})，相邻 N 翻倍时误差约降 ~16 倍。\n")

# =========================================================
# (B) 硬截止：界面对齐 + A3 动量连续
# =========================================================
def test_hard_interface():
    print("=== Hard step w(t): interface-aligned + momentum continuity (A3) ===")
    t0, ts, t1 = 0.0, 0.5, 1.0
    wL, wR = 1.0, 0.2
    # 左段
    N1 = 1024
    y0 = np.array([0.0, 1.0])
    t1a, Y1 = gl2_solve(t0, ts, y0, N1, lambda t: wL, None)
    qL, vL = Y1[-1,0], Y1[-1,1]
    pL = m*wL*vL

    # A3: p 连续 => v_right0 = (wL/wR) v_left
    vR0 = (wL/wR)*vL
    yR0 = np.array([qL, vR0])

    # 右段
    N2 = 1024
    t2b, Y2 = gl2_solve(ts, t1, yR0, N2, lambda t: wR, None)
    pR0 = m*wR*Y2[0,1]
    jump = abs(pR0 - pL)

    # 汇总
    t_full = np.concatenate([t1a, t2b])
    Y_full = np.vstack([Y1, Y2])
    R = residual_EL(t_full, Y_full, lambda t: w_hard(t, ts, wL, wR), None)
    print(f"接口动量跳变 |Δp| = {sci(jump)}   (A3 期望≈0)")
    print(f"EL 残差 RMS  = {sci(np.sqrt(np.mean(R**2)))}  （分段内 w'=0，仅分段计算）\n")

# =========================================================
# (C) 快速 sanity：EL 残差 + 能量型量纲检查
# =========================================================
def quick_sanity():
    print("=== Quick sanity on smooth w ===")
    t0, t1 = 0.0, 1.0
    y0 = np.array([0.0, 1.0])
    N  = 2048
    t, Y = gl2_solve(t0, t1, y0, N, w_smooth, wdot_smooth)
    R = residual_EL(t, Y, w_smooth, wdot_smooth)
    print(f"EL residual RMS (smooth) = {sci(np.sqrt(np.mean(R**2)))}\n")

# ---------------- run all ----------------
if __name__ == "__main__":
    test_smooth_convergence()
    test_hard_interface()
    quick_sanity()

