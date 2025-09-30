import numpy as np

# ---------- utilities ----------
def sci(x): return f"{x:.6e}"

# ---------- problem setup for UTH ----------
m = 1.0
k = 10.0  # 可调；适中避免刚性
kappa_U = 1.0  # 非零以启用动态U
def V(q): return 0.5 * k * q * q  # 势能
def dVdq(q): return k * q
def V_U(U): return 0.0  # 简单起见，设为0；可调整如0.5 * U**2
def dV_U_dU(U): return 0.0
# W(U)：平滑权重作为U的函数
a = 0.3  # 可调，小值减小非线性
def W(U): return 1.0 + a * np.sin(2 * np.pi * U)
def dW_dU(U): return a * (2 * np.pi) * np.cos(2 * np.pi * U)
def d2W_dU2(U): return -a * (2 * np.pi)**2 * np.sin(2 * np.pi * U)
# L(q, v) = 1/2 m v^2 - V(q)
def L(q, v): return 0.5 * m * v**2 - V(q)

# UTH系统的一阶形式 y = [q, v, U, p_U]，p_U = kappa_U * dot U
# y' = f(t, y) (t独立，但保留为通用)
def f_UTH(t, y):
    q, v, U, p_U = y
    W_val = W(U)
    dW_dU_val = dW_dU(U)
    dot_q = v
    # dot_v = - (1/m) dV/dq - (v / W) * dW/dU * (p_U / kappa_U)
    dot_v = - (1.0 / m) * dVdq(q) - (v / max(W_val, 1e-30)) * dW_dU_val * (p_U / kappa_U)
    dot_U = p_U / kappa_U
    # dot_p_U = - dW/dU * L(q,v) + dV_U/dU
    dot_p_U = - dW_dU_val * L(q, v) + dV_U_dU(U)
    return np.array([dot_q, dot_v, dot_U, dot_p_U], dtype=float)

# ---------- GL2一步 (扩展到4D状态) ----------
c1 = 0.5 - np.sqrt(3)/6.0
c2 = 0.5 + np.sqrt(3)/6.0
b1 = b2 = 0.5
s = np.sqrt(3)/6.0
A11, A12 = 0.25, 0.25 - s
A21, A22 = 0.25 + s, 0.25

def gl2_step_UTH(tn, yn, h):
    f0 = f_UTH(tn, yn)
    K1 = f0.copy()
    K2 = f0.copy()
    for iter in range(30):  # 增加迭代次数以确保收敛
        Y1 = yn + h * (A11 * K1 + A12 * K2)
        Y2 = yn + h * (A21 * K1 + A22 * K2)
        F1 = f_UTH(tn + c1 * h, Y1)
        F2 = f_UTH(tn + c2 * h, Y2)
        R1 = K1 - F1
        R2 = K2 - F2
        res = np.linalg.norm(np.hstack([R1, R2]), ord=2)
        if res < 1e-12:
            break
        # 解析雅可比 Jf_UTH (4x4)
        def Jf_UTH(t, y):
            q, v, U, p_U = y
            W_val = W(U)
            dW_dU_val = dW_dU(U)
            d2W_dU2_val = d2W_dU2(U)
            J = np.zeros((4, 4))
            # row 0: dot_q = v
            J[0, 1] = 1.0
            # row 1: dot_v
            J[1, 0] = - (k / m)  # ∂/∂q: - (1/m) k q 的导数
            J[1, 1] = - (dW_dU_val / max(W_val, 1e-30)) * (p_U / kappa_U)  # ∂/∂v
            # ∂/∂U: - v * (p_U / kappa_U) * (d2W W - dW^2) / W^2
            partial_dWW = (d2W_dU2_val * W_val - dW_dU_val**2) / W_val**2
            J[1, 2] = - v * (p_U / kappa_U) * partial_dWW
            J[1, 3] = - (v / W_val) * dW_dU_val * (1.0 / kappa_U)  # ∂/∂p_U
            # row 2: dot_U = p_U / kappa_U
            J[2, 3] = 1.0 / kappa_U
            # row 3: dot_p_U
            J[3, 0] = - dW_dU_val * (- k * q)  # ∂L/∂q = -k q
            J[3, 1] = - dW_dU_val * (m * v)  # ∂L/∂v = m v
            J[3, 2] = - d2W_dU2_val * L(q, v) + 0.0  # d2V_U=0
            return J
        J1 = Jf_UTH(tn + c1 * h, Y1)
        J2 = Jf_UTH(tn + c2 * h, Y2)
        I = np.eye(4)
        M11 = I - h * A11 * J1
        M12 = - h * A12 * J1
        M21 = - h * A21 * J2
        M22 = I - h * A22 * J2
        M = np.block([[M11, M12], [M21, M22]])
        rhs = -np.hstack([R1, R2])
        dK = np.linalg.solve(M, rhs)
        K1 += dK[:4]
        K2 += dK[4:]
    else:
        print(f"Warning: Newton did not converge after 30 iters, res={res}")
    yn1 = yn + h * (b1 * K1 + b2 * K2)
    return yn1

def gl2_solve_UTH(t0, t1, y0, N):
    t = np.linspace(t0, t1, N+1)
    h = (t1 - t0) / N
    Y = np.zeros((N+1, 4), dtype=float)
    Y[0] = y0
    for n in range(N):
        Y[n+1] = gl2_step_UTH(t[n], Y[n], h)
    return t, Y

# ---------- test convergence ----------
def test_UTH_convergence():
    print("=== UTH with smooth W(U): order-4 convergence test (GL2) ===")
    t0, t1 = 0.0, 1.0
    # 初值: q(0)=0, v(0)=1, U(0)=0, p_U(0)=kappa_U * dot U(0)=kappa_U *1 (假设初始 dot U=1)
    y0 = np.array([0.0, 1.0, 0.0, kappa_U * 1.0])
    # 参考解（细步长）
    N_ref = 32768
    t_ref, Y_ref = gl2_solve_UTH(t0, t1, y0, N_ref)
    q_ref = Y_ref[:,0]; v_ref = Y_ref[:,1]; U_ref = Y_ref[:,2]; p_U_ref = Y_ref[:,3]
    for N in [128, 256, 512, 1024, 2048]:
        tN, YN = gl2_solve_UTH(t0, t1, y0, N)
        step = N_ref // N
        q_err = np.linalg.norm(YN[:,0] - q_ref[::step], ord=2) / np.sqrt(N+1)
        v_err = np.linalg.norm(YN[:,1] - v_ref[::step], ord=2) / np.sqrt(N+1)
        U_err = np.linalg.norm(YN[:,2] - U_ref[::step], ord=2) / np.sqrt(N+1)
        p_U_err = np.linalg.norm(YN[:,3] - p_U_ref[::step], ord=2) / np.sqrt(N+1)
        print(f"N={N:5d}, L2(q)={sci(q_err)}, L2(v)={sci(v_err)}, L2(U)={sci(U_err)}, L2(p_U)={sci(p_U_err)}")
    print("期望：误差 ~ O(N^{-4})，相邻 N 翻倍时误差约降 ~16 倍。")

if __name__ == "__main__":
    test_UTH_convergence()
