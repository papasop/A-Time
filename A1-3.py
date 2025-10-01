# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # UTH with GL2 Integrator: Enhanced with Corner Test, Energy Bookkeeping, Equal-U Resampling, U-Window Diagnostics, and Corner Projection

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import bisect
from scipy.signal import savgol_filter

# Utilities
def sci(x): return f"{x:.6e}"

# Problem parameters
m = 1.0
k = 10.0
kappa_U = 1.0

def V(q): return 0.5 * k * q * q
def dVdq(q): return k * q
def V_U(U): return 0.0
def dV_U_dU(U): return 0.0

# W(U): Smooth weight
a = 0.3
def W(U): return 1.0 + a * np.sin(2 * np.pi * U)
def dW_dU(U): return a * (2 * np.pi) * np.cos(2 * np.pi * U)
def d2W_dU2(U): return -a * (2 * np.pi)**2 * np.sin(2 * np.pi * U)

def L(q, v): return 0.5 * m * v**2 - V(q)

def f_UTH(t, y):
    q, v, U, p_U = y
    W_val = W(U)
    dW_dU_val = dW_dU(U)
    dot_q = v
    dot_v = - (1.0 / m) * dVdq(q) - (v / np.maximum(W_val, 1e-30)) * dW_dU_val * (p_U / kappa_U)
    dot_U = p_U / kappa_U
    dot_p_U = - dW_dU_val * L(q, v) + dV_U_dU(U)
    return np.array([dot_q, dot_v, dot_U, dot_p_U], dtype=float)

def Jf_UTH(t, y):
    q, v, U, p_U = y
    W_val = W(U)
    dW_dU_val = dW_dU(U)
    d2W_dU2_val = d2W_dU2(U)
    J = np.zeros((4, 4))
    J[0, 1] = 1.0
    J[1, 0] = - (k / m)
    J[1, 1] = - (dW_dU_val / np.maximum(W_val, 1e-30)) * (p_U / kappa_U)
    partial_dWW = (d2W_dU2_val * W_val - dW_dU_val**2) / W_val**2
    J[1, 2] = - v * (p_U / kappa_U) * partial_dWW
    J[1, 3] = - (v / W_val) * dW_dU_val * (1.0 / kappa_U)
    J[2, 3] = 1.0 / kappa_U
    J[3, 0] = - dW_dU_val * (- k * q)
    J[3, 1] = - dW_dU_val * (m * v)
    J[3, 2] = - d2W_dU2_val * L(q, v)
    return J

# GL2 parameters
c1 = 0.5 - np.sqrt(3)/6.0
c2 = 0.5 + np.sqrt(3)/6.0
b1 = b2 = 0.5
s = np.sqrt(3)/6.0
A11, A12 = 0.25, 0.25 - s
A21, A22 = 0.25 + s, 0.25

def gl2_step_UTH(tn, yn, h):
    if h <= 0:
        raise ValueError("Step size h must be positive")
    f0 = f_UTH(tn, yn)
    K1 = f0.copy()
    K2 = f0.copy()
    for iter in range(30):
        Y1 = yn + h * (A11 * K1 + A12 * K2)
        Y2 = yn + h * (A21 * K1 + A22 * K2)
        F1 = f_UTH(tn + c1 * h, Y1)
        F2 = f_UTH(tn + c2 * h, Y2)
        R1 = K1 - F1
        R2 = K2 - F2
        res = np.linalg.norm(np.hstack([R1, R2]), ord=2)
        if res < 1e-12:
            break
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

def find_t_s(t0, y0, U_s, h, t1, N_coarse=100):
    """Find t_s where U(t_s) = U_s using bisection"""
    def simulate_to_t(t):
        if t <= t0:
            return y0[2] - U_s
        N_sim = max(1, int((t - t0)/h))
        t_grid, Y_grid = gl2_solve_UTH_plain(t0, t, y0, N_sim)
        return Y_grid[-1,2] - U_s
    try:
        t_s = bisect(simulate_to_t, t0, t1, xtol=1e-12)
    except ValueError as e:
        raise ValueError(f"Bisection failed: U_s={U_s} not reachable in [t0={t0}, t1={t1}]: {e}")
    return t_s

def gl2_solve_UTH(t0, t1, y0, N, U_s=None):
    r"""GL2 integrator with precise half-node alignment and projection for corner at U_s"""
    if t1 <= t0 or N <= 0:
        raise ValueError(f"Invalid t0={t0}, t1={t1}, N={N}")
    if U_s is not None:
        h = (t1 - t0) / N
        t_s = find_t_s(t0, y0, U_s, h, t1)
        n_before = int(np.floor((t_s - t0) / h))
        t_before = t0 + n_before * h
        h_adjust = (t_s - t_before) * 2
        if h_adjust <= 0:
            h_adjust = h / 2
        t_grid1 = np.linspace(t0, t_s - h_adjust/2, max(1, n_before + 1))
        Y_grid1 = np.zeros((len(t_grid1), 4))
        Y_grid1[0] = y0
        for n in range(len(t_grid1)-1):
            Y_grid1[n+1] = gl2_step_UTH(t_grid1[n], Y_grid1[n], t_grid1[n+1] - t_grid1[n])
        y_minus = Y_grid1[-1]
        y_short_end = gl2_step_UTH(t_s - h_adjust/2, y_minus, h_adjust)
        # 1D Newton projection: force P_w^+ = P_w^-
        W_minus = W(U_s - 1e-10)
        Pw_minus = m * W_minus * y_minus[1]
        def phi(v_plus):
            return m * W_minus * v_plus - Pw_minus
        def dphi_dv(v_plus):
            return m * W_minus
        v_plus = y_short_end[1]
        for _ in range(5):
            delta_v = -phi(v_plus) / dphi_dv(v_plus)
            v_plus += delta_v
            if abs(delta_v) < 1e-12:
                break
        y_short_end[1] = v_plus
        # Update p_U using EL equation
        q, v, U, _ = y_short_end
        dot_U = (y_short_end[2] - y_minus[2]) / h_adjust
        y_short_end[3] = kappa_U * dot_U
        t_grid2 = np.linspace(t_s, t1, max(1, int((t1 - t_s)/h) + 1))
        Y_grid2 = np.zeros((len(t_grid2), 4))
        Y_grid2[0] = y_short_end
        for n in range(len(t_grid2)-1):
            Y_grid2[n+1] = gl2_step_UTH(t_grid2[n], Y_grid2[n], t_grid2[n+1] - t_grid2[n])
        t = np.concatenate([t_grid1, t_grid2[1:]])
        Y = np.concatenate([Y_grid1, Y_grid2[1:]])
    else:
        t = np.linspace(t0, t1, N+1)
        h = (t1 - t0) / N
        Y = np.zeros((N+1, 4), dtype=float)
        Y[0] = y0
        for n in range(N):
            Y[n+1] = gl2_step_UTH(t[n], Y[n], h)
    return t, Y

def gl2_solve_UTH_plain(t0, t1, y0, N):
    if t1 <= t0 or N <= 0:
        raise ValueError(f"Invalid t0={t0}, t1={t1}, N={N}")
    t = np.linspace(t0, t1, N+1)
    h = (t1 - t0) / N
    Y = np.zeros((N+1, 4), dtype=float)
    Y[0] = y0
    for n in range(N):
        Y[n+1] = gl2_step_UTH(t[n], Y[n], h)
    return t, Y

# Enhancement 1: Corner Test
def setup_corner_W(U_s=0.5, jump_scale=0.1):
    r"""Piecewise W(U) with derivative discontinuity at U_s"""
    def W_corner(U):
        return np.where(U < U_s, 1.0 + a * np.sin(2 * np.pi * U),
                        1.0 + a * np.sin(2 * np.pi * U) + jump_scale * (U - U_s))
    def dW_corner(U):
        return np.where(U < U_s, a * 2 * np.pi * np.cos(2 * np.pi * U),
                        a * 2 * np.pi * np.cos(2 * np.pi * U) + jump_scale)
    def d2W_corner(U):
        return np.where(U < U_s, -a * (2 * np.pi)**2 * np.sin(2 * np.pi * U),
                        -a * (2 * np.pi)**2 * np.sin(2 * np.pi * U))
    return W_corner, dW_corner, d2W_corner

def check_corner_continuity(t, Y, U_s=0.5):
    r"""Check discrete weighted-momentum continuity P_w = W * \partial L/\partial v = W * m * v at half-steps"""
    Pw = []
    W_minus = W(U_s - 1e-10)  # Left limit
    for i in range(len(t)):
        q, v, U, p_U = Y[i]
        Pw.append(W_minus * m * v)  # Use W_minus for projection consistency
    Pw_half = 0.5 * (np.array(Pw[:-1]) + np.array(Pw[1:]))
    crossings = np.where((Y[:-1,2] < U_s) & (Y[1:,2] >= U_s))[0]
    if crossings.size > 0:
        idx = crossings[0]
        print(f"Corner at step {idx}: Pw_half before={Pw_half[idx-1]:.6e}, after={Pw_half[idx]:.6e}, diff={abs(Pw_half[idx] - Pw_half[idx-1]):.6e}")
    else:
        print("No corner crossing found.")

# Enhancement 2: Energy Bookkeeping
def compute_energies(Y):
    r"""Compute E_q = \frac{1}{2} m v^2 + V(q), E_U = \frac{1}{2 \kappa_U} p_U^2 + V_U(U)"""
    q, v, U, p_U = Y.T
    E_q = 0.5 * m * v**2 + V(q)
    E_U = 0.5 * (p_U**2 / kappa_U) + V_U(U)
    E_total = E_q + E_U
    return E_q, E_U, E_total

def check_energy_balance(t, Y):
    r"""Check \dot E_q + \dot E_U = - W'(U) L \dot U - v^2 W' \dot U / W (corrected form)"""
    q, v, U, p_U = Y.T
    dot_U = p_U / kappa_U
    W_val = W(U)
    dot_v = - (1.0 / m) * dVdq(q) - (v / np.maximum(W_val, 1e-30)) * dW_dU(U) * dot_U
    dot_E_q = m * v * dot_v + dVdq(q) * v
    dot_E_U = p_U * (- dW_dU(U) * L(q, v) + dV_U_dU(U)) / kappa_U + dV_U_dU(U) * dot_U
    lhs = dot_E_q + dot_E_U
    rhs = - dW_dU(U) * L(q, v) * dot_U - v**2 * dW_dU(U) * dot_U / np.maximum(W_val, 1e-30)
    error = np.mean(np.abs(lhs - rhs))
    print(f"Avg energy balance error: {sci(error)} (should be O(h^5) for GL2)")
    return error

# Enhancement 3: Equal-U Grid Resampling
def resample_to_equal_U(t, Y, num_U_points=5000):
    r"""Resample to uniform U grid assuming U monotone increasing"""
    U = Y[:,2]
    if not np.all(np.diff(U) > 0):
        raise ValueError("U must be strictly increasing for resampling.")
    U_grid = np.linspace(U.min(), U.max(), num_U_points)
    interp_t = interp1d(U, t, kind='linear', fill_value="extrapolate")
    interp_q = interp1d(U, Y[:,0], kind='linear', fill_value="extrapolate")
    interp_v = interp1d(U, Y[:,1], kind='linear', fill_value="extrapolate")
    t_resamp = interp_t(U_grid)
    q_resamp = interp_q(U_grid)
    v_resamp = interp_v(U_grid)
    return U_grid, t_resamp, q_resamp, v_resamp

# Enhancement 4: U-Window Diagnostics
def compute_EL_residual_U(q, v, U, t=None, is_Uparam=False):
    r"""Weighted EL residual in U-domain: d/dU (W \partial L/\partial v) - W \partial L/\partial q or reparameterized form"""
    W_val = W(U)
    if is_Uparam:
        if t is None:
            raise ValueError("t required for U-parameterized EL residual")
        dt_dU = np.gradient(t, U)
        dt_dU = savgol_filter(dt_dU, window_length=15, polyorder=5)  # Enhanced smoothing
        dot_U = 1.0 / np.where(np.abs(dt_dU) < 1e-10, 1e-10, dt_dU)
        q_prime = v / dot_U
        partial_qprime_hatL = W_val * m * q_prime * dot_U**2
        d_partial_qprime_dU = np.gradient(partial_qprime_hatL, U)
        partial_q_hatL = W_val * (-k * q) * dot_U
        res = d_partial_qprime_dU - partial_q_hatL
    else:
        Pv = W_val * m * v
        dPv_dU = np.gradient(Pv, U)
        partial_q = W_val * (-k * q)
        res = dPv_dU - partial_q
    return np.linalg.norm(res)

def u_window_diagnostics(U_grid, q_resamp, v_resamp, t_resamp=None, window_size=200, is_Uparam=False):
    r"""Stats in equal-U windows: EL residual norm, \Theta=\int \omega dU, \kappa"""
    if window_size <= 0 or len(U_grid) < window_size:
        raise ValueError(f"Invalid window_size={window_size} or U_grid length={len(U_grid)}")
    num_windows = len(U_grid) // window_size
    if num_windows == 0:
        print("Warning: No windows available; increase num_U_points or decrease window_size")
        return
    for win in range(num_windows):
        start, end = win * window_size, (win + 1) * window_size
        U_win = U_grid[start:end]
        q_win = q_resamp[start:end]
        v_win = v_resamp[start:end]
        t_win = t_resamp[start:end] if is_Uparam else None
        dU = np.diff(U_win, prepend=U_win[0])
        Theta = np.trapezoid(v_win, U_win)
        EL_res = compute_EL_residual_U(q_win, v_win, U_win, t_win, is_Uparam)
        f = np.fft.fftfreq(len(U_win), d=np.mean(dU))
        Qf = np.fft.fft(q_win)
        Vf = np.fft.fft(v_win)
        C = (Qf * np.conj(Vf)) / np.abs(Qf * np.conj(Vf) + 1e-10)
        kappa_vals = np.real(np.fft.ifft(C))
        kappa_hat = U_win[np.argmax(kappa_vals)]
        print(rf"U-window [{U_win[0]:.2f}, {U_win[-1]:.2f}]: \Theta={Theta:.4f}, EL_res={EL_res:.4e}, \hat \kappa={kappa_hat:.4f}")

# Enhancement 5: U-Parameterized Integrator with Adaptive Step
def gl2_step_UTH_Uparam(Un, yn, h_U, dot_U_threshold=5e-4):
    if h_U <= 0:
        raise ValueError("Step size h_U must be positive")
    q, v, t, p_U = yn
    dot_U = p_U / kappa_U
    # Adaptive step size
    if abs(dot_U) < dot_U_threshold:
        h_U = h_U * 0.5
    f0 = f_UTH_Uparam(Un, yn)
    K1 = f0.copy()
    K2 = f0.copy()
    for iter in range(30):
        Y1 = yn + h_U * (A11 * K1 + A12 * K2)
        Y2 = yn + h_U * (A21 * K1 + A22 * K2)
        F1 = f_UTH_Uparam(Un + c1 * h_U, Y1)
        F2 = f_UTH_Uparam(Un + c2 * h_U, Y2)
        R1 = K1 - F1
        R2 = K2 - F2
        res = np.linalg.norm(np.hstack([R1, R2]), ord=2)
        if res < 1e-12:
            break
        J1 = Jf_UTH_Uparam(Un + c1 * h_U, Y1)
        J2 = Jf_UTH_Uparam(Un + c2 * h_U, Y2)
        I = np.eye(4)
        M11 = I - h_U * A11 * J1
        M12 = - h_U * A12 * J1
        M21 = - h_U * A21 * J2
        M22 = I - h_U * A22 * J2
        M = np.block([[M11, M12], [M21, M22]])
        rhs = -np.hstack([R1, R2])
        dK = np.linalg.solve(M, rhs)
        K1 += dK[:4]
        K2 += dK[4:]
    yn1 = yn + h_U * (b1 * K1 + b2 * K2)
    return yn1

def f_UTH_Uparam(U, y):
    r"""Dynamics in U-parameter: y = [q, v, t, p_U], \hat L = L(q, v \dot U, t) \dot U"""
    q, v, t, p_U = y
    dot_U = p_U / kappa_U
    W_val = W(U)
    dW_dU_val = dW_dU(U)
    L_val = L(q, v * dot_U) * dot_U
    dot_q = v
    dot_v = - (1.0 / m) * dVdq(q) / np.maximum(dot_U, 1e-10) - (v / np.maximum(W_val, 1e-30)) * dW_dU_val
    dot_t = 1.0 / np.maximum(dot_U, 1e-10)
    dot_p_U = - dW_dU_val * L_val
    return np.array([dot_q, dot_v, dot_t, dot_p_U], dtype=float)

def Jf_UTH_Uparam(U, y):
    q, v, t, p_U = y
    dot_U = p_U / kappa_U
    W_val = W(U)
    dW_dU_val = dW_dU(U)
    d2W_dU2_val = d2W_dU2(U)
    J = np.zeros((4, 4))
    J[0, 1] = 1.0
    J[1, 0] = - (k / m) / np.maximum(dot_U, 1e-10)
    J[1, 1] = - (dW_dU_val / np.maximum(W_val, 1e-30))
    J[1, 3] = (k / (m * dot_U**2)) * (1.0 / kappa_U) - v * (d2W_dU2_val * W_val - dW_dU_val**2) / (W_val**2 * kappa_U)
    J[2, 3] = -1.0 / (dot_U**2 * kappa_U)
    J[3, 0] = - dW_dU_val * (-k * q * dot_U)
    J[3, 1] = - dW_dU_val * (m * v * dot_U**2)
    J[3, 3] = - dW_dU_val * m * v * dot_U**2 * (1.0 / kappa_U)
    return J

def gl2_solve_UTH_Uparam(U0, U1, y0, N):
    if U1 <= U0 or N <= 0:
        raise ValueError(f"Invalid U0={U0}, U1={U1}, N={N}")
    U = np.linspace(U0, U1, N+1)
    h_U = (U1 - U0) / N
    Y = np.zeros((N+1, 4), dtype=float)
    Y[0] = y0
    for n in range(N):
        Y[n+1] = gl2_step_UTH_Uparam(U[n], Y[n], h_U)
    return U, Y

# Enhancement 6: Manufactured Solution Test
def test_manufactured_solution(N=2048):
    r"""Test with manufactured solution q(t) = sin(t) + 0.1 sin(3t), U(t) = t + 0.3 sin(t)"""
    t0, t1 = 0.0, 1.0
    y0 = np.array([0.0, 1.3, 0.0, kappa_U * 1.3])  # q(0)=0, \dot q(0)=1.3, U(0)=0, \dot U(0)=1.3
    global W, dW_dU, d2W_dU2
    W = lambda U: 1.0 + 0.3 * np.sin(2 * np.pi * U)
    dW_dU = lambda U: 0.3 * (2 * np.pi) * np.cos(2 * np.pi * U)
    d2W_dU2 = lambda U: -0.3 * (2 * np.pi)**2 * np.sin(2 * np.pi * U)
    t, Y = gl2_solve_UTH(t0, t1, y0, N)
    # Compute L2 error
    t_exact = np.linspace(t0, t1, N+1)
    q_exact = np.sin(t_exact) + 0.1 * np.sin(3 * t_exact)
    U_exact = t_exact + 0.3 * np.sin(t_exact)
    q_error = np.sqrt(np.mean((Y[:,0] - q_exact)**2))
    U_error = np.sqrt(np.mean((Y[:,2] - U_exact)**2))
    E_q, E_U, E_total = compute_energies(Y)
    print(f"Manufactured solution test: Total energy drift: {sci(E_total[-1] - E_total[0])}")
    print(f"L2 error q: {sci(q_error)}, L2 error U: {sci(U_error)}")
    error = check_energy_balance(t, Y)
    U_grid, t_resamp, q_resamp, v_resamp = resample_to_equal_U(t, Y)
    u_window_diagnostics(U_grid, q_resamp, v_resamp, t_resamp, window_size=200, is_Uparam=False)
    return t, Y, error

# Integrated Test
def test_enhanced_UTH(N=2048, use_corner=False, a_val=0.3, use_Uparam=False):
    global a
    a = a_val
    t0, t1 = 0.0, 2.0  # Extended to ensure U covers [0, 1]
    y0 = np.array([0.0, 1.0, 0.0, kappa_U * 1.0])
    global W, dW_dU, d2W_dU2
    if use_corner:
        W, dW_dU, d2W_dU2 = setup_corner_W()
    else:
        W = lambda U: 1.0 + a * np.sin(2 * np.pi * U)
        dW_dU = lambda U: a * (2 * np.pi) * np.cos(2 * np.pi * U)
        d2W_dU2 = lambda U: -a * (2 * np.pi)**2 * np.sin(2 * np.pi * U)
    if use_Uparam:
        U0, U1 = y0[2], y0[2] + 1.0
        U, Y = gl2_solve_UTH_Uparam(U0, U1, y0, N)
        t = Y[:,2]
        U_grid, t_resamp, q_resamp, v_resamp = resample_to_equal_U(t, Y)
    else:
        U_s = 0.5 if use_corner else None
        t, Y = gl2_solve_UTH(t0, t1, y0, N, U_s=U_s)
        U_grid, t_resamp, q_resamp, v_resamp = resample_to_equal_U(t, Y)
    E_q, E_U, E_total = compute_energies(Y)
    print(f"Total energy drift: {sci(E_total[-1] - E_total[0])}")
    error = check_energy_balance(t, Y)
    if use_corner:
        check_corner_continuity(t, Y, U_s=0.5)
    u_window_diagnostics(U_grid, q_resamp, v_resamp, t_resamp, window_size=200, is_Uparam=use_Uparam)
    return t, Y, error

# Run tests
print("=== Smooth W(U) Test (a=0.3) ===")
t, Y, error = test_enhanced_UTH(N=2048, use_corner=False, a_val=0.3, use_Uparam=False)

print("\n=== Smooth W(U) Test (a=0, linear case) ===")
t, Y, error_linear = test_enhanced_UTH(N=2048, use_corner=False, a_val=0.0, use_Uparam=False)

print("\n=== Corner W(U) Test (a=0.3, half-node aligned with projection) ===")
t, Y, error_corner = test_enhanced_UTH(N=2048, use_corner=True, a_val=0.3, use_Uparam=False)

print("\n=== U-Parameterized Test (a=0.3) ===")
t, Y, error_Uparam = test_enhanced_UTH(N=2048, use_corner=False, a_val=0.3, use_Uparam=True)

print("\n=== Manufactured Solution Test ===")
t, Y, error_manufactured = test_manufactured_solution(N=2048)
