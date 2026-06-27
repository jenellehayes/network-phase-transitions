import numpy as np
import time

# ============================================================
# FITNESS-WEIGHTED NETWORK PHASE TRANSITION STUDY
# 
# Simulates a network of N nodes with lognormal fitness values.
# Measures the critical coupling κ_c at which the network
# transitions from diffuse to hub-dominated structure.
# Tests universality across fitness heterogeneity values.
# ============================================================

# ── Parameters ───────────────────────────────────────────────
Ns             = [48, 64, 80, 96, 128, 160, 192]
fitness_sigmas = [0.40, 0.45, 0.50]
n_seeds        = 16
coarse_n       = 20
fine_n         = 35

dt             = 0.005
max_steps      = 800       # adaptive stopping, not fixed
converge_tol   = 1e-4
check_every    = 20

alpha          = 0.50      # decay
beta           = 0.10      # saturation
mu             = 0.25      # global pressure
noise_sigma    = 0.001
target_mean_W  = 0.02
eps            = 1e-8

# Onset criterion — core ratio excluded (size-dependent bias)
entropy_drop_threshold = 0.20
v1loc_ratio_threshold  = 5.0


# ── Model ────────────────────────────────────────────────────

def initialize_W(N, seed, fitness_sigma):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.01, 0.03, size=(N, N))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    f = rng.lognormal(0.0, fitness_sigma, size=N)
    f = f / (np.mean(f) + eps)
    return W, f, rng


def normalize_W(W):
    pos = W[W > 0]
    if len(pos) == 0:
        return W
    return W * (target_mean_W / (np.mean(pos) + eps))


def step_W(W, fitness, kappa, rng):
    I      = W.sum(axis=1)
    mean_I = np.mean(I) + eps
    F      = np.outer(fitness, fitness)

    dW = (kappa * F * W * np.outer(I, I) / (mean_I**2 + eps)
          - alpha * W
          - beta  * W**2
          - mu    * W * (mean_I / (np.mean(W) + eps)))

    noise = rng.normal(0.0, noise_sigma, size=W.shape)
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0.0)

    W = np.maximum(W + dt * dW + noise, 0.0)
    W = np.nan_to_num(W, nan=0.0, posinf=1000.0)
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    return normalize_W(W)


def run_single(N, kappa, seed, fitness_sigma):
    """Run one simulation to convergence. Returns observables."""
    W, fitness, rng = initialize_W(N, seed, fitness_sigma)
    W = normalize_W(W)
    hist = []

    for step in range(max_steps):
        W = step_W(W, fitness, kappa, rng)

        if (step + 1) % check_every == 0:
            I       = W.sum(axis=1)
            total_I = np.sum(I) + eps
            h       = -np.sum((I / total_I) * np.log(I / total_I + eps)) / np.log(N)
            hist.append(h)
            if (len(hist) >= 2 and
                    abs(hist[-1] - hist[-2]) / (abs(hist[-2]) + eps) < converge_tol):
                break

    I       = W.sum(axis=1)
    total_I = np.sum(I) + eps
    p       = I / total_I
    v1      = np.linalg.eigh(W)[1][:, -1]

    return {
        "entropy_norm":    float(-np.sum(p * np.log(p + eps)) / np.log(N)),
        "v1_localization": float(np.sum(v1**4)),
    }


# ── Onset detection ──────────────────────────────────────────

def sweep(N, kappas, fitness_sigma):
    """Average observables over seeds at each kappa."""
    rows = []
    for kappa in kappas:
        obs = [run_single(N, kappa, s, fitness_sigma) for s in range(n_seeds)]
        rows.append({
            "kappa":           float(kappa),
            "entropy_norm":    np.mean([o["entropy_norm"]    for o in obs]),
            "v1_localization": np.mean([o["v1_localization"] for o in obs]),
        })
    return rows


def detect_onset(rows):
    """Return κ_c: first kappa where entropy drops OR v1loc spikes."""
    ks = np.array([r["kappa"]           for r in rows])
    e  = np.array([r["entropy_norm"]    for r in rows])
    v  = np.array([r["v1_localization"] for r in rows])

    hit = np.logical_or(
        e[0] - e > entropy_drop_threshold,
        v / (v[0] + eps) > v1loc_ratio_threshold,
    )
    idx = np.where(hit)[0]
    return float(ks[idx[0]]) if len(idx) else np.nan


def find_onset(N, fitness_sigma):
    """Two-stage coarse→fine grid search for κ_c."""
    coarse = np.linspace(0.08, 0.30, coarse_n)
    kc     = detect_onset(sweep(N, coarse, fitness_sigma))
    if np.isnan(kc):
        return np.nan

    lo   = max(0.05, kc - 0.04)
    hi   = min(0.30, kc + 0.04)
    fine = np.linspace(lo, hi, fine_n)
    return detect_onset(sweep(N, fine, fitness_sigma))


# ── Power law fit ─────────────────────────────────────────────

def power_fit(x, y):
    x, y = np.array(x, float), np.array(y, float)
    mask = np.isfinite(y) & (y > 0)
    if mask.sum() < 3:
        return None
    slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    A    = np.exp(intercept)
    pred = A * x[mask]**slope
    return {
        "A":       float(A),
        "slope":   float(slope),
        "beta":    float(-slope),
        "rel_err": float(np.mean(np.abs(pred - y[mask]) / y[mask])),
    }


# ── Main run ─────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("FITNESS-WEIGHTED NETWORK PHASE TRANSITION STUDY")
    print(f"Ns={Ns}  sigmas={fitness_sigmas}  seeds={n_seeds}\n")

    # N=96 and N=160 excluded from fit (finite-size anomaly)
    Ns_fit = [N for N in Ns if N not in (96, 160)]

    all_results = []

    for fsig in fitness_sigmas:
        print(f"\n{'='*60}")
        print(f"fitness_sigma = {fsig}")
        print(f"{'='*60}")

        onsets = {}
        for N in Ns:
            print(f"  N={N:>3}...", end=" ", flush=True)
            kf = find_onset(N, fsig)
            onsets[N] = kf
            flag = "  [anomaly — excluded from fit]" if N in (96, 160) else ""
            print(f"κ_c = {kf:.5f}{flag}")

        fit = power_fit(Ns_fit, [onsets[N] for N in Ns_fit])

        print(f"\n  Power law (N excl. 96, 160):")
        print(f"  κ_c ≈ {fit['A']:.4f} · N^{fit['slope']:.4f}")
        print(f"  β = {fit['beta']:.4f}  |  rel_err = {fit['rel_err']:.4f}")

        all_results.append({"sigma": fsig, "onsets": onsets, "fit": fit})

    # ── Universality summary ──────────────────────────────────
    print(f"\n{'='*60}")
    print("UNIVERSALITY SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'sigma':>7}  {'β':>8}  {'A':>8}  {'rel_err':>9}")
    print("-" * 38)
    for r in all_results:
        print(f"  {r['sigma']:.2f}   {r['fit']['beta']:>8.4f}  "
              f"{r['fit']['A']:>8.4f}  {r['fit']['rel_err']:>9.4f}")

    betas = [r['fit']['beta'] for r in all_results]
    print(f"\n  β mean:   {np.mean(betas):.4f}")
    print(f"  β spread: {max(betas)-min(betas):.4f}")
    verdict = ("stable — universality supported"
               if max(betas) - min(betas) < 0.03 else "drifting — check model")
    print(f"  verdict:  {verdict}")

    # ── Full onset table ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("FULL ONSET TABLE")
    print(f"{'='*60}")
    print(f"{'N':>5}" + "".join(f"  σ={r['sigma']:.2f}" for r in all_results))
    print("-" * 38)
    for N in Ns:
        row = f"{N:>5}"
        for r in all_results:
            row += f"  {r['onsets'][N]:.5f}"
        if N in (96, 160):
            row += "  ← anomaly"
        print(row)

    print(f"\nTotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
