import numpy as np
import matplotlib.pyplot as plt

def sci(x): 
    return f"{x:.6e}"

# ---------- Thomas tridiagonal solver ----------
def thomas_tridiag(a, b, c, d):
    n = len(b)
    cp = np.empty(n)
    dp = np.empty(n)
    x = np.empty(n)
    beta = b[0]
    cp[0] = c[0]/beta
    dp[0] = d[0]/beta
    for i in range(1, n):
        beta = b[i] - a[i]*cp[i-1]
        cp[i] = c[i]/beta if i < n-1 else 0.0
        dp[i] = (d[i] - a[i]*dp[i-1])/beta
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

# ---------- Half-step weights ----------
def half_weights(w):
    return 0.5*(w[:-1] + w[1:])

# ---------- High-order derivative (4th-order central difference) ----------
def high_order_qdot(q, dt):
    qdot = np.zeros_like(q)
    qdot[2:-2] = (-q[4:] + 8*q[3:-1] - 8*q[1:-3] + q[:-4]) / (12*dt)
    qdot[0] = (q[1] - q[0]) / dt
    qdot[1] = (q[2] - q[0]) / (2*dt)
    qdot[-2] = (q[-1] - q[-3]) / (2*dt)
    qdot[-1] = (q[-1] - q[-2]) / dt
    return qdot

# ---------- Build EL system (corrected for linear case) ----------
def build_EL_halfstep_system(t, w, m=1.0, k=5.0, q0=0.0, qT=1.0):
    N = len(t)
    dt = t[1] - t[0]
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)
    b[0] = 1.0; d[0] = q0
    b[-1] = 1.0; d[-1] = qT
    wh = half_weights(w)
    for i in range(1, N-1):
        w_imh = wh[i-1]
        w_iph = wh[i]
        a[i] = (m / dt**2) * w_imh
        b[i] = -(m / dt**2) * (w_imh + w_iph) - w[i] * k
        c[i] = (m / dt**2) * w_iph
        d[i] = 0.0
    d[1] -= a[1] * q0; a[1] = 0.0
    d[-2] -= c[-2] * qT; c[-2] = 0.0
    return a, b, c, d

# ---------- Action and residual (corrected for linear case) ----------
def action_and_residual_halfstep(q, t, w, m=1.0, k=5.0):
    N = len(t)
    dt = t[1] - t[0]
    wh = half_weights(w)
    qdot_h = high_order_qdot(q, dt)[:-1]
    p_h = m * wh * qdot_h
    R = np.zeros(N)
    R[1:-1] = (p_h[1:] - p_h[:-1]) / dt - w[1:-1] * k * q[1:-1]
    R[0] = R[-1] = 0.0
    qdot = high_order_qdot(q, dt)
    L_vals = 0.5 * m * qdot**2 + 0.5 * k * q**2
    S = np.trapz(w * L_vals, t)
    rms = np.sqrt(np.mean(R[1:-1]**2))
    return S, rms, R, p_h

# ---------- Exact solution for linear case ----------
def exact_solution(t, m=1.0, k=5.0, q0=0.0, qT=1.0):
    omega = np.sqrt(k/m)
    return q0 + (qT - q0) * np.sinh(omega * t) / np.sinh(omega)

# ---------- Nonlinear iteration solver ----------
def nonlinear_iterate(t, w, m=1.0, k=5.0, alpha=0.1, q0=0.0, qT=1.0, tol=1e-14, max_iter=1000):
    N = len(t)
    dt = t[1] - t[0]
    q = exact_solution(t, m, k, q0, qT)  # Better initial guess
    for _ in range(max_iter):
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        d = np.zeros(N)
        b[0] = 1.0; d[0] = q0
        b[-1] = 1.0; d[-1] = qT
        wh = half_weights(w)
        for i in range(1, N-1):
            w_imh = wh[i-1]
            w_iph = wh[i]
            a[i] = (m / dt**2) * w_imh
            b[i] = -(m / dt**2) * (w_imh + w_iph) - w[i] * (k + 4 * alpha * q[i]**2)
            c[i] = (m / dt**2) * w_iph
            d[i] = -w[i] * 4 * alpha * q[i]**3
        d[1] -= a[1] * q0; a[1] = 0.0
        d[-2] -= c[-2] * qT; c[-2] = 0.0
        q_new = thomas_tridiag(a, b, c, d)
        delta = np.max(np.abs(q_new - q))
        if delta < tol:
            break
        # Adaptive damping
        damping = min(1.0, 0.5 / max(delta, 1e-10))
        q = q + damping * (q_new - q)
    return q

# ---------- Nonlinear residual calculation ----------
def nonlinear_residual(q, t, w, m=1.0, k=5.0, alpha=0.1):
    N = len(t)
    dt = t[1] - t[0]
    wh = half_weights(w)
    qdot_h = high_order_qdot(q, dt)[:-1]
    p_h = m * wh * qdot_h
    R = np.zeros(N)
    R[1:-1] = (p_h[1:] - p_h[:-1]) / dt - w[1:-1] * (-(k * q[1:-1] + 4 * alpha * q[1:-1]**3))
    R[0] = R[-1] = 0.0
    rms = np.sqrt(np.mean(R[1:-1]**2))
    return R, rms

# ---------- Test 1: Nonlinear Potential (with iteration) ----------
print("=== Nonlinear Potential Test ===")
def nonlinear_L(q, qdot, t):
    V = 0.5 * 5.0 * q**2 + 0.1 * q**4
    T = 0.5 * 1.0 * qdot**2
    return T - V

N = 64001  # Further increased for better resolution
T = 1.0
t = np.linspace(0.0, T, N)
w = np.ones_like(t)
q0, qT = 0.0, 1.0
q_nonlin = nonlinear_iterate(t, w, m=1.0, k=5.0, alpha=0.1, q0=q0, qT=qT)
qdot = high_order_qdot(q_nonlin, t[1] - t[0])
L_vals = nonlinear_L(q_nonlin, qdot, t)
S_nonlin = np.trapz(w * L_vals, t)
R_nonlin, rms_nonlin = nonlinear_residual(q_nonlin, t, w)
print(f"S* = {sci(S_nonlin)}")
print(f"||EL residual||_rms = {sci(rms_nonlin)}")
print("Note: Using Newton iteration with adaptive damping, high-order derivative, and finer grid.\n")

# ---------- Test 2: Multiple Interfaces (with smooth weights) ----------
print("=== Multiple Interfaces Test ===")
def build_w_multiple_smooth(t, ts_list, w_min=1e-3, width=0.005):  # Adjusted width
    w = np.full_like(t, w_min)
    dt = t[1] - t[0]
    for ts in ts_list:
        mask = (t >= ts - width) & (t <= ts + width)
        tau = (t[mask] - (ts - width)) / (2 * width)
        w[mask] = w_min + (1.0 - w_min) * (0.5 - 0.5 * np.cos(2 * np.pi * tau))
    return w

ts_list = [0.3, 0.7]
w_multi = build_w_multiple_smooth(t, ts_list, w_min=1e-3, width=0.005)
a, b, c, d = build_EL_halfstep_system(t, w_multi, m=1.0, k=5.0, q0=q0, qT=qT)
q_multi = thomas_tridiag(a, b, c, d)
S_multi, rms_multi, R_multi, p_h_multi = action_and_residual_halfstep(q_multi, t, w_multi)
print(f"S* = {sci(S_multi)}")
print(f"||EL residual||_rms = {sci(rms_multi)}")
dt = t[1] - t[0]
for ts in ts_list:
    i_s = int(round(ts / dt - 0.5))
    if 0 < i_s < len(p_h_multi):
        jump = abs(p_h_multi[i_s] - p_h_multi[i_s-1])
        print(f"Jump at ts={ts:.3f}: {sci(jump)} (should ≈0 for continuity)")
print("\n")

# ---------- Test 3: Corrected Convergence Analysis ----------
print("=== Corrected Convergence Analysis (Error vs Exact Solution) ===")
Ns = [101, 201, 401, 801, 1601, 3201, 6401]
errors = []
for N in Ns:
    t_conv = np.linspace(0, T, N)
    w_conv = np.ones_like(t_conv)
    a, b, c, d = build_EL_halfstep_system(t_conv, w_conv, m=1.0, k=5.0, q0=0, qT=1)
    q_conv = thomas_tridiag(a, b, c, d)
    q_exact = exact_solution(t_conv)
    error = np.sqrt(np.mean((q_conv - q_exact)**2))
    dt = T / (N - 1)
    errors.append(error)
    print(f"N={N}, dt={sci(dt)}, Error={sci(error)}")

# Estimate orders
orders = []
for i in range(1, len(errors)):
    order = np.log2(errors[i-1] / errors[i])
    orders.append(order)
    print(f"Order between N={Ns[i-1]} and N={Ns[i]}: {order:.2f}")
print(f"Average order: {np.mean(orders):.2f} (expect ~2 for second-order method)\n")

# ---------- Visualization ----------
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(t, q_nonlin, label='q(t) - Nonlinear')
plt.plot(t, exact_solution(t), '--', label='q(t) - Exact (Linear)')
plt.title('Nonlinear vs Exact Linear Solution')
plt.xlabel('t')
plt.ylabel('q')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, w_multi)
plt.title('Weight w(t) - Multiple Interfaces')
plt.xlabel('t')
plt.ylabel('w')
for ts in ts_list:
    plt.axvline(ts, color='r', linestyle='--', alpha=0.5)

plt.subplot(2, 2, 3)
plt.loglog([1/n for n in Ns], errors, 'o-', label='Error')
plt.title('Convergence: Error vs 1/N')
plt.xlabel('1/N')
plt.ylabel('Error')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.semilogy(Ns, errors, 'o-', label='Error')
plt.title('Error vs N')
plt.xlabel('N')
plt.ylabel('Error')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ---------- Residual and Momentum Distribution Plots ----------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t, R_multi)
plt.title('Residual R(t) - Multiple Interfaces')
plt.xlabel('t')
plt.ylabel('R')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t[:-1], p_h_multi)
plt.title('Momentum p_h(t) - Multiple Interfaces')
plt.xlabel('t')
plt.ylabel('p_h')
for ts in ts_list:
    plt.axvline(ts, color='r', linestyle='--', alpha=0.5)
plt.grid(True)
plt.tight_layout()
plt.show()
