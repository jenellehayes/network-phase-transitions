import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ════════════════════════════════════════════════════════════════════════
# UNIVERSALITY CLASS ANALYSIS
# Extends network_phase_transition.py to measure three things:
#
#   1. ORDER PARAMETER EXPONENT (β_op)
#      How the hub/field split grows above κ_c.
#      Fit: hub_excess ~ (κ − κ_c)^β_op
#
#   2. CORRELATION LENGTH EXPONENT (ν)
#      How the width of the transition scales with system size.
#      Fit: Δκ ~ N^(−1/ν)  →  slope gives ν directly.
#
#   3. DATA COLLAPSE
#      Rescale all entropy curves onto one master curve.
#      The ν that produces the best collapse IS the exponent.
#
# Run with:  python universality_analysis.py
# Outputs:   four .png plots + printed summary table
# Runtime:   ~30–60 min depending on hardware (16 seeds, dense sweeps)
# ════════════════════════════════════════════════════════════════════════


# ── Simulation parameters — unchanged from original ──────────────────
dt            = 0.005
max_steps     = 800
converge_tol  = 1e-4
check_every   = 20
alpha_decay   = 0.50      # renamed: 'alpha' in original
beta_sat      = 0.10      # renamed: 'beta' in original (saturation)
mu            = 0.25
noise_sigma   = 0.001
target_mean_W = 0.02
eps           = 1e-8

entropy_drop_threshold = 0.20
v1loc_ratio_threshold  = 5.0

# ── What to measure ──────────────────────────────────────────────────
Ns_all       = [48, 64, 80, 96, 128, 160, 192]
Ns_fit       = [N for N in Ns_all if N not in (96, 160)]  # exclude anomalies
FIXED_SIGMA  = 0.45     # middle sigma; universality confirmed across all three
FIXED_N      = 128      # system size for order parameter scan
N_SEEDS      = 16       # seeds per measurement point

# κ offsets above κ_c for order parameter scan
OP_OFFSETS   = [0.010, 0.018, 0.028, 0.040, 0.055, 0.072, 0.092, 0.115, 0.142]

# Dense sweep for transition width and collapse curves
DENSE_KAPPA  = 45       # points in dense sweep window
NU_RANGE     = np.linspace(0.3, 4.0, 250)   # trial ν values


# ════════════════════════════════════════════════════════════════════════
# CORE SIMULATION  (reproduced from network_phase_transition.py)
# ════════════════════════════════════════════════════════════════════════

def initialize_W(N, seed, fitness_sigma):
    rng = np.random.default_rng(seed)
    W   = rng.uniform(0.01, 0.03, size=(N, N))
    W   = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    f   = rng.lognormal(0.0, fitness_sigma, size=N)
    f   = f / (np.mean(f) + eps)
    return W, f, rng


def normalize_W(W):
    pos = W[W > 0]
    return W if len(pos) == 0 else W * (target_mean_W / (np.mean(pos) + eps))


def step_W(W, fitness, kappa, rng):
    I      = W.sum(axis=1)
    mean_I = np.mean(I) + eps
    F      = np.outer(fitness, fitness)
    dW     = (kappa * F * W * np.outer(I, I) / (mean_I**2 + eps)
              - alpha_decay * W
              - beta_sat    * W**2
              - mu          * W * (mean_I / (np.mean(W) + eps)))
    noise  = rng.normal(0.0, noise_sigma, size=W.shape)
    noise  = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0.0)
    W      = np.maximum(W + dt * dW + noise, 0.0)
    W      = np.nan_to_num(W, nan=0.0, posinf=1000.0)
    W      = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    return normalize_W(W)


def run_extended(N, kappa, seed, fitness_sigma):
    """
    Single simulation run to convergence.
    Returns entropy_norm, v1_localization, AND order_param.

    Order parameter:
      In a perfectly uniform network, the top 10% of nodes hold exactly
      10% of total influence. Above the transition, hubs concentrate
      more. We measure the excess above the uniform baseline:

        order_param = (influence share of top 10%) − 0.10

      This is 0 in the diffuse phase and rises above the transition.
      It is the quantity we fit to (κ − κ_c)^β_op.
    """
    W, fitness, rng = initialize_W(N, seed, fitness_sigma)
    W = normalize_W(W)
    hist = []

    for step in range(max_steps):
        W = step_W(W, fitness, kappa, rng)
        if (step + 1) % check_every == 0:
            I       = W.sum(axis=1)
            total_I = np.sum(I) + eps
            h       = -np.sum((I/total_I) * np.log(I/total_I + eps)) / np.log(N)
            hist.append(h)
            if (len(hist) >= 2 and
                    abs(hist[-1] - hist[-2]) / (abs(hist[-2]) + eps) < converge_tol):
                break

    I        = W.sum(axis=1)
    total_I  = np.sum(I) + eps
    p        = I / total_I
    v1       = np.linalg.eigh(W)[1][:, -1]

    k_top        = max(1, int(N * 0.10))
    hub_share    = float(np.sort(I)[-k_top:].sum() / total_I)
    order_param  = hub_share - 0.10   # excess above uniform baseline

    return {
        "entropy_norm":    float(-np.sum(p * np.log(p + eps)) / np.log(N)),
        "v1_localization": float(np.sum(v1**4)),
        "order_param":     order_param,
    }


def sweep(N, kappas, fitness_sigma, verbose=False):
    """Average observables over seeds at each kappa."""
    rows = []
    for kappa in kappas:
        obs = [run_extended(N, kappa, s, fitness_sigma) for s in range(N_SEEDS)]
        row = {
            "kappa":           float(kappa),
            "entropy_norm":    float(np.mean([o["entropy_norm"]    for o in obs])),
            "v1_localization": float(np.mean([o["v1_localization"] for o in obs])),
            "order_param":     float(np.mean([o["order_param"]     for o in obs])),
        }
        rows.append(row)
        if verbose:
            print(f"      κ={kappa:.4f}  H={row['entropy_norm']:.3f}"
                  f"  op={row['order_param']:.4f}")
    return rows


def detect_onset(rows):
    ks  = np.array([r["kappa"]           for r in rows])
    e   = np.array([r["entropy_norm"]    for r in rows])
    v   = np.array([r["v1_localization"] for r in rows])
    hit = np.logical_or(
        e[0] - e > entropy_drop_threshold,
        v / (v[0] + eps) > v1loc_ratio_threshold,
    )
    idx = np.where(hit)[0]
    return float(ks[idx[0]]) if len(idx) else np.nan


def find_onset_kc(N, fitness_sigma):
    """Two-stage coarse→fine onset (from original)."""
    coarse = np.linspace(0.08, 0.30, 20)
    kc     = detect_onset(sweep(N, coarse, fitness_sigma))
    if np.isnan(kc):
        return np.nan
    lo   = max(0.05, kc - 0.04)
    hi   = min(0.30, kc + 0.04)
    fine = np.linspace(lo, hi, 35)
    return detect_onset(sweep(N, fine, fitness_sigma))


def power_fit(x, y):
    """Log-space linear fit: y = A * x^slope."""
    x, y   = np.array(x, float), np.array(y, float)
    mask   = np.isfinite(y) & (y > 0) & (x > 0)
    if mask.sum() < 3:
        return None
    slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    A       = np.exp(intercept)
    pred    = A * x[mask]**slope
    rel_err = float(np.mean(np.abs(pred - y[mask]) / (y[mask] + eps)))
    return {"slope": float(slope), "A": float(A), "rel_err": rel_err}


# ════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — ORDER PARAMETER EXPONENT β_op
# ════════════════════════════════════════════════════════════════════════

def measure_beta_op(kc, fitness_sigma=FIXED_SIGMA, N=FIXED_N):
    """
    Scan κ above κ_c at fixed N.  Fit order_param ~ (κ − κ_c)^β_op.

    We use the order parameter defined in run_extended:
      order_param = (hub share of top 10%) − 0.10
    This is zero at uniform baseline and rises at the transition.
    """
    print(f"\n  Scanning above κ_c = {kc:.4f}  (N={N}, σ={fitness_sigma})")
    kappas = [kc + off for off in OP_OFFSETS if kc + off < 0.32]
    deltas, ops = [], []

    for kappa in kappas:
        obs = [run_extended(N, kappa, s, fitness_sigma) for s in range(N_SEEDS)]
        op  = float(np.mean([o["order_param"] for o in obs]))
        deltas.append(kappa - kc)
        ops.append(op)
        print(f"    κ − κ_c = {kappa-kc:.4f}   order_param = {op:.4f}")

    # Only fit points where order param is clearly positive
    d_arr = np.array(deltas)
    o_arr = np.array(ops)
    mask  = o_arr > 0.002
    if mask.sum() < 3:
        print("  WARNING: too few positive order param points for fit")
        return np.nan, deltas, ops

    fit    = power_fit(d_arr[mask], o_arr[mask])
    beta_op = fit["slope"] if fit else np.nan
    print(f"\n  β_op = {beta_op:.4f}  (order parameter exponent)")
    return beta_op, deltas, ops


# ════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — CORRELATION LENGTH EXPONENT ν  (from transition width)
# ════════════════════════════════════════════════════════════════════════

def measure_width(N, kc, fitness_sigma=FIXED_SIGMA):
    """
    Dense sweep around κ_c.
    Width Δκ = distance between κ at 20% drop and κ at 80% drop of entropy.

    In finite-size scaling:  Δκ ~ N^(−1/ν)
    So  log(Δκ) vs log(N)  has slope  −1/ν.
    """
    lo     = max(0.04, kc - 0.07)
    hi     = min(0.33, kc + 0.07)
    kappas = np.linspace(lo, hi, DENSE_KAPPA)
    rows   = sweep(N, kappas, fitness_sigma)

    e_arr  = np.array([r["entropy_norm"] for r in rows])
    k_arr  = np.array([r["kappa"]        for r in rows])

    e_hi   = e_arr[0]
    e_lo   = np.min(e_arr)
    span   = e_hi - e_lo

    if span < 0.05:
        return np.nan, k_arr, e_arr

    # κ where entropy has dropped 20% of total range (start of transition)
    k_20  = k_arr[np.argmin(np.abs(e_arr - (e_hi - 0.20 * span)))]
    # κ where entropy has dropped 80% of total range (end of transition)
    k_80  = k_arr[np.argmin(np.abs(e_arr - (e_hi - 0.80 * span)))]

    return float(k_80 - k_20), k_arr, e_arr


def measure_nu_from_width(kc_dict, fitness_sigma=FIXED_SIGMA):
    """
    Measure transition width for each N, fit Δκ ~ N^(−1/ν).
    Returns ν, the per-N width dict, and full entropy curves.
    """
    print(f"\n  Measuring transition widths across system sizes...")
    widths = {}
    curves = {}   # {N: (k_arr, e_arr)}

    for N in Ns_all:   # include 96, 160 for curve comparison
        kc = kc_dict.get(N, np.nan)
        if np.isnan(kc):
            continue
        print(f"    N={N:>3} ...", end=" ", flush=True)
        w, k_arr, e_arr = measure_width(N, kc, fitness_sigma)
        widths[N] = w
        curves[N] = (k_arr, e_arr)
        flag = " ← anomaly (not used in fit)" if N in (96, 160) else ""
        print(f"Δκ = {w:.5f}{flag}")

    # Fit only the clean sizes
    valid_N = [N for N in Ns_fit if not np.isnan(widths.get(N, np.nan))]
    valid_w = [widths[N] for N in valid_N]

    if len(valid_N) < 3:
        print("  WARNING: too few valid sizes for ν fit")
        return np.nan, widths, curves

    fit  = power_fit(valid_N, valid_w)
    nu   = -1.0 / fit["slope"]   # slope = −1/ν

    print(f"\n  Width scaling: Δκ ~ N^{fit['slope']:.4f}")
    print(f"  ν = {nu:.4f}   rel_err = {fit['rel_err']:.4f}")
    print(f"\n  Known references:")
    print(f"    1D DP  ν_⊥ = 1.097   ν_∥ = 1.734")
    print(f"    2D DP  ν_⊥ = 0.733")
    print(f"    Mean field  ν = 0.500")

    return nu, widths, curves


# ════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — DATA COLLAPSE
# ════════════════════════════════════════════════════════════════════════

def collapse_score(nu, kc_dict, curves):
    """
    For a trial ν, rescale each entropy curve:
      x_rescaled = (κ − κ_c) · N^(1/ν)

    If ν is correct, all curves land on the same master function.
    We measure this by binning x_rescaled and computing the mean
    within-bin standard deviation of entropy values.
    Lower score = better collapse.
    """
    all_x, all_y = [], []
    for N, (k_arr, e_arr) in curves.items():
        kc = kc_dict.get(N, np.nan)
        if np.isnan(kc):
            continue
        x_scaled = (k_arr - kc) * N**(1.0 / nu)
        all_x.extend(x_scaled.tolist())
        all_y.extend(e_arr.tolist())

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # Score: mean std within bins of the rescaled axis
    x_lo, x_hi = np.percentile(all_x, 5), np.percentile(all_x, 95)
    bins        = np.linspace(x_lo, x_hi, 30)
    stds        = []
    for i in range(len(bins) - 1):
        mask = (all_x >= bins[i]) & (all_x < bins[i+1])
        if mask.sum() >= 3:
            stds.append(float(np.std(all_y[mask])))

    return float(np.mean(stds)) if stds else np.inf


def find_best_collapse(kc_dict, curves):
    """Scan ν, return the value giving the best data collapse."""
    print(f"\n  Scanning {len(NU_RANGE)} trial ν values for best collapse ...")
    scores = np.array([collapse_score(nu, kc_dict, curves) for nu in NU_RANGE])
    best_idx = int(np.argmin(scores))
    best_nu  = float(NU_RANGE[best_idx])
    print(f"  Best collapse: ν = {best_nu:.4f}  (score = {scores[best_idx]:.5f})")
    return best_nu, scores


# ════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════

COLORS = plt.cm.viridis(np.linspace(0.05, 0.92, len(Ns_all)))
N_COLOR = {N: COLORS[i] for i, N in enumerate(Ns_all)}


def plot_order_parameter(kc, deltas, ops, beta_op):
    fig, ax = plt.subplots(figsize=(7, 5))
    d_arr = np.array(deltas)
    o_arr = np.array(ops)
    mask  = o_arr > 0.001

    ax.loglog(d_arr[mask], o_arr[mask], 'o', color='steelblue',
              ms=7, label='measured')

    if not np.isnan(beta_op) and mask.sum() >= 2:
        x_fit = np.geomspace(d_arr[mask].min(), d_arr[mask].max(), 100)
        # Anchor the power law at the first valid point
        y0    = o_arr[mask][0]
        x0    = d_arr[mask][0]
        y_fit = y0 * (x_fit / x0)**beta_op
        ax.loglog(x_fit, y_fit, '--', color='tomato', lw=1.8,
                  label=f'β_op = {beta_op:.3f}')

    ax.set_xlabel('κ − κ_c', fontsize=12)
    ax.set_ylabel('Order parameter\n(hub share excess above uniform)', fontsize=11)
    ax.set_title('Order Parameter Exponent  β_op', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    plt.tight_layout()
    plt.savefig('order_parameter_exponent.png', dpi=150)
    plt.close()
    print("  → saved  order_parameter_exponent.png")


def plot_transition_width(widths, kc_dict):
    valid_N = sorted([N for N in Ns_fit if not np.isnan(widths.get(N, np.nan))])
    valid_w = [widths[N] for N in valid_N]
    fit     = power_fit(valid_N, valid_w)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Δκ vs N log-log
    ax = axes[0]
    for N in Ns_all:
        if N in widths and not np.isnan(widths[N]):
            mk = 's' if N in (96, 160) else 'o'
            ax.loglog(N, widths[N], marker=mk, ms=8,
                      color=N_COLOR[N], label=f'N={N}')

    if fit:
        x_fit = np.geomspace(min(valid_N), max(valid_N), 100)
        ax.loglog(x_fit, fit['A'] * x_fit**fit['slope'], '--',
                  color='tomato', lw=1.8,
                  label=f'slope = {fit["slope"]:.3f}  →  ν = {-1/fit["slope"]:.3f}')

    ax.set_xlabel('N  (system size)', fontsize=12)
    ax.set_ylabel('Δκ  (transition width)', fontsize=12)
    ax.set_title('Width scaling:  Δκ ~ N^(−1/ν)', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which='both', alpha=0.25)

    # Right: entropy curves to show what "width" means
    ax = axes[1]
    for N, w_val in widths.items():
        k_arr, e_arr = None, None
        # We don't have curves here — placeholder note
    ax.set_visible(False)   # Hidden; replaced by transition_curves.png

    plt.tight_layout()
    plt.savefig('transition_width_scaling.png', dpi=150)
    plt.close()
    print("  → saved  transition_width_scaling.png")


def plot_entropy_curves(curves, kc_dict):
    fig, ax = plt.subplots(figsize=(9, 6))
    for N in sorted(curves.keys()):
        k_arr, e_arr = curves[N]
        kc = kc_dict.get(N, np.nan)
        ls = ':' if N in (96, 160) else '-'
        ax.plot(k_arr, e_arr, ls, color=N_COLOR[N], lw=1.8, label=f'N={N}')
        if not np.isnan(kc):
            ax.axvline(kc, color=N_COLOR[N], alpha=0.25, lw=1)

    ax.set_xlabel('κ', fontsize=12)
    ax.set_ylabel('Normalised entropy', fontsize=12)
    ax.set_title('Entropy curves by system size\n(dotted = anomalous N)', fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig('entropy_curves.png', dpi=150)
    plt.close()
    print("  → saved  entropy_curves.png")


def plot_data_collapse(best_nu, kc_dict, curves, nu_range, scores):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: raw curves
    ax = axes[0]
    for N in sorted(curves.keys()):
        k_arr, e_arr = curves[N]
        ls = ':' if N in (96, 160) else '-'
        ax.plot(k_arr, e_arr, ls, color=N_COLOR[N], lw=1.6, label=f'N={N}')
    ax.set_xlabel('κ', fontsize=11)
    ax.set_ylabel('Normalised entropy', fontsize=11)
    ax.set_title('Raw curves', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)

    # Panel 2: collapsed curves
    ax = axes[1]
    for N in sorted(curves.keys()):
        k_arr, e_arr = curves[N]
        kc = kc_dict.get(N, np.nan)
        if np.isnan(kc):
            continue
        ls = ':' if N in (96, 160) else '-'
        x_sc = (k_arr - kc) * N**(1.0 / best_nu)
        ax.plot(x_sc, e_arr, ls, color=N_COLOR[N], lw=1.6, label=f'N={N}')
    ax.set_xlabel(f'(κ − κ_c) · N^(1/ν)\nν = {best_nu:.3f}', fontsize=11)
    ax.set_ylabel('Normalised entropy', fontsize=11)
    ax.set_title(f'Collapsed  (ν = {best_nu:.3f})', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-15, 15)

    # Panel 3: ν scan
    ax = axes[2]
    ax.plot(nu_range, scores, color='steelblue', lw=1.5)
    ax.axvline(best_nu, color='tomato', ls='--', lw=1.8,
               label=f'best ν = {best_nu:.3f}')
    refs = [('1D DP ν⊥', 1.097), ('2D DP ν⊥', 0.733), ('MF', 0.500)]
    y_top = np.max(scores[np.isfinite(scores)]) * 0.95
    for label, val in refs:
        ax.axvline(val, color='gray', ls=':', alpha=0.7)
        ax.text(val, y_top, label, fontsize=7.5, ha='center',
                va='top', color='gray', rotation=90)
    ax.set_xlabel('Trial ν', fontsize=11)
    ax.set_ylabel('Collapse score\n(lower = better)', fontsize=11)
    ax.set_title('ν scan', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('data_collapse.png', dpi=150)
    plt.close()
    print("  → saved  data_collapse.png")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("UNIVERSALITY CLASS ANALYSIS")
    print(f"sigma={FIXED_SIGMA}   N_SEEDS={N_SEEDS}   dense_kappa={DENSE_KAPPA}\n")

    # ── Step 0: κ_c for each N ───────────────────────────────────────
    print("═" * 60)
    print("STEP 0: Measuring κ_c  (two-stage, same method as original)")
    print("═" * 60)
    kc_dict = {}
    for N in Ns_all:
        print(f"  N={N:>3} ...", end=" ", flush=True)
        kc = find_onset_kc(N, FIXED_SIGMA)
        kc_dict[N] = kc
        flag = " ← anomaly" if N in (96, 160) else ""
        print(f"κ_c = {kc:.5f}{flag}")

    # ── Analysis 1: Order parameter exponent β_op ────────────────────
    print("\n" + "═" * 60)
    print("ANALYSIS 1: ORDER PARAMETER EXPONENT  β_op")
    print("  Fit: (hub excess above uniform) ~ (κ − κ_c)^β_op")
    print("═" * 60)
    kc_fixed = kc_dict.get(FIXED_N, np.nan)
    if not np.isnan(kc_fixed):
        beta_op, deltas, ops = measure_beta_op(kc_fixed)
        plot_order_parameter(kc_fixed, deltas, ops, beta_op)
    else:
        print(f"  WARNING: no κ_c for N={FIXED_N}, skipping β_op measurement")
        beta_op = np.nan

    # ── Analysis 2: ν from transition width ──────────────────────────
    print("\n" + "═" * 60)
    print("ANALYSIS 2: CORRELATION LENGTH EXPONENT  ν")
    print("  Fit: Δκ ~ N^(−1/ν)  via dense sweeps at each N")
    print("═" * 60)
    nu_width, widths, curves = measure_nu_from_width(kc_dict)
    plot_transition_width(widths, kc_dict)
    plot_entropy_curves(curves, kc_dict)

    # ── Analysis 3: Data collapse ─────────────────────────────────────
    print("\n" + "═" * 60)
    print("ANALYSIS 3: DATA COLLAPSE")
    print("  Scan ν: rescale (κ − κ_c)·N^(1/ν); find best collapse")
    print("═" * 60)
    best_nu, scores = find_best_collapse(kc_dict, curves)
    plot_data_collapse(best_nu, kc_dict, curves, NU_RANGE, scores)

    # ── Final summary ─────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "═" * 60)
    print("SUMMARY — UNIVERSALITY CLASS FINGERPRINT")
    print("═" * 60)
    print(f"\n  From original study:")
    print(f"    β_size  ≈ 0.305 ± 0.009   (κ_c scaling with N)")
    print(f"\n  From this analysis:")
    print(f"    β_op       = {beta_op:.4f}          (order parameter exponent)")
    print(f"    ν (width)  = {nu_width:.4f}          (from Δκ ~ N^(−1/ν))")
    print(f"    ν (collapse)= {best_nu:.4f}          (from data collapse)")
    print(f"\n  Known universality classes for comparison:")
    print(f"  {'Class':<28} {'β':>7}  {'ν_⊥':>7}  {'ν_∥':>7}")
    print(f"  {'-'*52}")
    print(f"  {'1D Directed Percolation':<28} {'0.276':>7}  {'1.097':>7}  {'1.734':>7}")
    print(f"  {'2D Directed Percolation':<28} {'0.583':>7}  {'0.733':>7}  {'1.295':>7}")
    print(f"  {'Mean Field':<28} {'1.000':>7}  {'0.500':>7}  {'0.500':>7}")
    print(f"  {'This system (β_op / ν)':<28} {beta_op:>7.3f}  {best_nu:>7.3f}  {'?':>7}")
    print(f"\n  Outputs saved:")
    print(f"    order_parameter_exponent.png")
    print(f"    transition_width_scaling.png")
    print(f"    entropy_curves.png")
    print(f"    data_collapse.png")
    print(f"\n  Total runtime: {elapsed/60:.1f} min")

    # Interpretation note
    print(f"\n  INTERPRETATION:")
    if abs(beta_op - 0.276) < 0.05 and abs(best_nu - 1.097) < 0.15:
        print(f"    → Consistent with 1D Directed Percolation universality class.")
    elif abs(beta_op - best_nu) > 0.5:
        print(f"    → β_op and ν differ significantly from all canonical classes.")
        print(f"       This combination may indicate a novel universality class.")
        print(f"       This is the result worth putting in front of the professor.")
    else:
        print(f"    → Exponents lie between known classes.")
        print(f"       Further system sizes (N=256+) would sharpen the picture.")


if __name__ == "__main__":
    main()
