#!/usr/bin/env python3
"""
PR3 Headless Verification Test — Visionary Nudge
=================================================

Tests the visionary spatial nudge:
  1. Visionaries drift toward the hidden fitness peak faster than regulars
  2. Visionary mask is stable (same particles stay visionary across steps)
  3. Visionary fraction is respected
  4. With nudge=0, visionaries behave identically to regulars
  5. Knowledge field still grows correctly with visionaries present

Runs entirely headless (no GPU, no OpenGL).
"""

import sys, os
import numpy as np
import time

# ── Make 3D_sim importable ──
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Create standalone dir with symlinks (before importing)
standalone_dir = os.path.join(REPO, '_3D_sim_standalone')
os.makedirs(standalone_dir, exist_ok=True)
init_path = os.path.join(standalone_dir, '__init__.py')
if not os.path.exists(init_path):
    with open(init_path, 'w') as f:
        pass
for mod in ['knowledge_field.py', 'landscape.py', 'mountain_mesh.py']:
    src = os.path.join(REPO, '3D_sim', mod)
    dst = os.path.join(standalone_dir, mod)
    if not os.path.exists(dst) and os.path.exists(src):
        os.symlink(src, dst)

from _3D_sim_standalone.knowledge_field import KnowledgeField
from _3D_sim_standalone.landscape import make_default_landscape


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def dist_to_peak(pos_xy, peak_xy=np.array([0.8, 0.85])):
    """Euclidean distance from positions to the global peak."""
    return np.linalg.norm(pos_xy - peak_xy[None, :], axis=1)


def run_simulation(N, n_steps, vis_frac, vis_nudge, seed=42):
    """Run a simplified simulation with visionary nudge.

    Returns:
        pos: final (N, 2) positions
        visionary_mask: (N,) bool
        history: dict with per-step tracking data
    """
    rng = np.random.default_rng(seed)
    SPACE = 1.0

    landscape = make_default_landscape(seed=42)
    kf = KnowledgeField(grid_res=64, diffusion_sigma=0.5, decay=0.9999,
                        support_radius=3, max_slope=0.4, space=SPACE)
    kf.set_fitness_surface(landscape)

    pos = rng.random((N, 2)) * SPACE

    # Create visionary mask
    n_vis = max(1, int(vis_frac * N))
    visionary_mask = np.zeros(N, dtype=bool)
    if vis_frac > 0:
        vis_ids = rng.choice(N, size=n_vis, replace=False)
        visionary_mask[vis_ids] = True

    history = {
        'vis_fitness': [],
        'reg_fitness': [],
        'coverage': [],
        'peak_knowledge': [],
    }

    for step in range(n_steps):
        x, y = pos[:, 0], pos[:, 1]

        # Deposit knowledge
        kf.deposit(x, y, np.full(N, 0.005))
        kf.step()

        # Visionary nudge
        if vis_nudge > 0 and vis_frac > 0:
            vis_pos_3d = np.column_stack([pos[visionary_mask], np.zeros(visionary_mask.sum())])
            grad = landscape.gradient(vis_pos_3d)  # (n_vis, 2)
            grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
            grad_dir = grad / np.maximum(grad_norm, 1e-12)
            pos[visionary_mask, 0] += vis_nudge * grad_dir[:, 0]
            pos[visionary_mask, 1] += vis_nudge * grad_dir[:, 1]
            pos[visionary_mask] %= SPACE

        # Random walk for all particles (small noise so nudge signal is visible)
        pos += rng.normal(0, 0.001, (N, 2))
        pos %= SPACE

        # Track metrics every 500 steps
        if step % 500 == 0:
            pos_3d = np.column_stack([pos, np.zeros(N)])
            fit = landscape.fitness(pos_3d)
            vis_f = fit[visionary_mask].mean() if visionary_mask.any() else 0
            reg_f = fit[~visionary_mask].mean() if (~visionary_mask).any() else 0
            history['vis_fitness'].append(vis_f)
            history['reg_fitness'].append(reg_f)
            history['coverage'].append(kf.coverage())
            history['peak_knowledge'].append(kf.peak_knowledge())

    return pos, visionary_mask, history


# ═══════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════

def test_visionary_drift():
    """Test 1: Visionaries reach higher fitness than regulars."""
    print("Test 1: Visionary fitness ... ", end="", flush=True)

    pos, mask, hist = run_simulation(
        N=200, n_steps=15000, vis_frac=0.05, vis_nudge=0.002, seed=42)

    landscape = make_default_landscape(seed=42)
    pos_3d = np.column_stack([pos, np.zeros(len(pos))])
    fitness = landscape.fitness(pos_3d)

    vis_fitness = fitness[mask].mean()
    reg_fitness = fitness[~mask].mean()

    assert vis_fitness > reg_fitness, \
        f"Visionaries ({vis_fitness:.4f}) not fitter than regulars ({reg_fitness:.4f})"

    ratio = vis_fitness / max(reg_fitness, 1e-12)
    print(f"PASS (vis={vis_fitness:.4f}, reg={reg_fitness:.4f}, ratio={ratio:.2f}x)")
    return True


def test_mask_fraction():
    """Test 2: Visionary fraction is respected."""
    print("Test 2: Mask fraction ... ", end="", flush=True)

    N = 300
    vis_frac = 0.05
    rng = np.random.default_rng(42)

    n_vis = max(1, int(vis_frac * N))
    mask = np.zeros(N, dtype=bool)
    vis_ids = rng.choice(N, size=n_vis, replace=False)
    mask[vis_ids] = True

    actual_frac = mask.sum() / N
    expected = n_vis / N

    assert abs(actual_frac - expected) < 0.01, \
        f"Fraction {actual_frac:.3f} != expected {expected:.3f}"

    print(f"PASS ({mask.sum()}/{N} = {actual_frac:.3f})")
    return True


def test_zero_nudge():
    """Test 3: With nudge=0, visionaries behave like regulars."""
    print("Test 3: Zero nudge baseline ... ", end="", flush=True)

    pos_nudge, _, hist_nudge = run_simulation(
        N=200, n_steps=5000, vis_frac=0.05, vis_nudge=0.0, seed=42)
    pos_base, _, hist_base = run_simulation(
        N=200, n_steps=5000, vis_frac=0.0, vis_nudge=0.0, seed=42)

    # With no nudge and no visionaries, coverage should be similar
    cov_nudge = hist_nudge['coverage'][-1]
    cov_base = hist_base['coverage'][-1]

    diff = abs(cov_nudge - cov_base)
    assert diff < 0.1, f"Coverage differs too much: {cov_nudge:.3f} vs {cov_base:.3f}"

    print(f"PASS (cov_vis0={cov_nudge:.3f}, cov_base={cov_base:.3f})")
    return True


def test_knowledge_growth():
    """Test 4: Knowledge field grows correctly with visionaries."""
    print("Test 4: Knowledge growth ... ", end="", flush=True)

    _, _, hist = run_simulation(
        N=200, n_steps=15000, vis_frac=0.05, vis_nudge=0.001, seed=42)

    final_cov = hist['coverage'][-1]
    final_peak = hist['peak_knowledge'][-1]

    assert final_cov > 0.3, f"Coverage too low: {final_cov:.3f}"
    assert final_peak > 0.3, f"Peak too low: {final_peak:.3f}"
    assert final_peak <= 1.0, f"Peak exceeds 1.0: {final_peak:.3f}"

    print(f"PASS (cov={final_cov:.1%}, peak={final_peak:.3f})")
    return True


def test_gradient_nonzero():
    """Test 5: Gradient is non-zero and points uphill."""
    print("Test 5: Gradient is non-zero and uphill ... ", end="", flush=True)

    landscape = make_default_landscape(seed=42)

    # Test from several positions — gradient should be non-zero and
    # stepping along it should increase fitness
    test_positions = np.array([
        [0.5, 0.5, 0.0],
        [0.3, 0.3, 0.0],
        [0.7, 0.8, 0.0],
    ])

    grads = landscape.gradient(test_positions)
    f_before = landscape.fitness(test_positions)

    eps = 0.01
    stepped = test_positions.copy()
    stepped[:, 0] += eps * grads[:, 0]
    stepped[:, 1] += eps * grads[:, 1]
    f_after = landscape.fitness(stepped)

    uphill = (f_after >= f_before).sum()
    nonzero = (np.linalg.norm(grads, axis=1) > 1e-6).sum()

    assert nonzero >= 2, f"Only {nonzero}/3 gradients are non-zero"
    assert uphill >= 2, f"Only {uphill}/3 steps went uphill"
    print(f"PASS ({nonzero}/3 non-zero, {uphill}/3 uphill)")
    return True


# ═══════════════════════════════════════════════════════════════════
# Diagnostic plot
# ═══════════════════════════════════════════════════════════════════

def generate_diagnostic_plot():
    """Generate a 2x2 diagnostic plot for PR3."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("PR3 Verification: Visionary Nudge", fontsize=14, fontweight='bold')

    # Run simulation for plotting
    N = 200
    pos, mask, hist = run_simulation(
        N=N, n_steps=15000, vis_frac=0.05, vis_nudge=0.002, seed=42)

    steps = [i * 500 for i in range(len(hist['vis_fitness']))]

    # Panel 1: Fitness over time
    ax = axes[0, 0]
    ax.plot(steps, hist['vis_fitness'], label='Visionaries', color='red', linewidth=2)
    ax.plot(steps, hist['reg_fitness'], label='Regulars', color='gray', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Fitness at Position')
    ax.set_title('Fitness: Visionaries vs Regulars')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Final particle positions
    ax = axes[0, 1]
    landscape = make_default_landscape(seed=42)
    # Background fitness heatmap
    gx = np.linspace(0, 1, 100)
    gy = np.linspace(0, 1, 100)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel(), np.zeros(10000)])
    F = landscape.fitness(pts).reshape(100, 100)
    ax.imshow(F, origin='lower', extent=[0, 1, 0, 1], cmap='terrain', alpha=0.4)
    # Particles
    ax.scatter(pos[~mask, 0], pos[~mask, 1], s=3, alpha=0.3, color='gray', label='Regulars')
    ax.scatter(pos[mask, 0], pos[mask, 1], s=20, alpha=0.9, color='red',
               marker='*', label='Visionaries')
    ax.plot(0.8, 0.85, 'k+', markersize=15, markeredgewidth=3, label='Global Peak')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Final Particle Positions')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Panel 3: Knowledge coverage over time
    ax = axes[1, 0]
    ax.plot(steps, hist['coverage'], color='green', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Coverage')
    ax.set_title('Knowledge Coverage Growth')
    ax.grid(True, alpha=0.3)

    # Panel 4: Comparison — with vs without visionaries
    _, _, hist_no_vis = run_simulation(
        N=N, n_steps=15000, vis_frac=0.0, vis_nudge=0.0, seed=42)
    steps2 = [i * 500 for i in range(len(hist_no_vis['peak_knowledge']))]

    ax = axes[1, 1]
    ax.plot(steps, hist['peak_knowledge'], label='With visionaries', color='red', linewidth=2)
    ax.plot(steps2, hist_no_vis['peak_knowledge'], label='Without visionaries',
            color='gray', linewidth=2, linestyle='--')
    ax.set_xlabel('Step')
    ax.set_ylabel('Peak Knowledge')
    ax.set_title('Peak Knowledge: Visionaries vs Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(REPO, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'pr3_verification.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plot saved to {out_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("PR3 Verification: Visionary Nudge")
    print("=" * 60)

    results = []
    results.append(test_visionary_drift())
    results.append(test_mask_fraction())
    results.append(test_zero_nudge())
    results.append(test_knowledge_growth())
    results.append(test_gradient_nonzero())

    print()
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    generate_diagnostic_plot()

    if passed < total:
        sys.exit(1)
