#!/usr/bin/env python3
"""
Headless verification test for PR1: Knowledge Field
====================================================

Tests the KnowledgeField and Landscape classes without any GPU/OpenGL
dependencies.  Verifies:

1. Landscape produces correct multi-peak fitness values
2. KnowledgeField accumulates knowledge correctly
3. Fitness ceiling is enforced
4. Structural support constraint works
5. Decay and diffusion work
6. Visionary particles drift toward the global peak
7. Timing: ~5 min experiment at 60fps = ~18,000 steps

Generates a diagnostic plot to results/pr1_verification.png
"""

import sys
import os
import shutil
import numpy as np

# Add project root to path
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, _project_root)

# Create standalone copies (no relative imports)
_standalone_dir = os.path.join(_project_root, '_3D_sim_standalone')
if not os.path.exists(_standalone_dir):
    os.makedirs(_standalone_dir)
_src_dir = os.path.join(_project_root, '3D_sim')
for _fname in ['knowledge_field.py', 'landscape.py']:
    shutil.copy2(os.path.join(_src_dir, _fname), os.path.join(_standalone_dir, _fname))
with open(os.path.join(_standalone_dir, '__init__.py'), 'w') as _f:
    _f.write('')

from _3D_sim_standalone.landscape import RuggedLandscape, make_default_landscape
from _3D_sim_standalone.knowledge_field import KnowledgeField


def test_landscape():
    """Test that the landscape produces expected peaks."""
    landscape = make_default_landscape(seed=42)

    # Sample at the known global peak center
    peak_pos = np.array([[0.8, 0.85]])
    f_peak = landscape.fitness(peak_pos)[0]

    # Sample at the foothill
    foot_pos = np.array([[0.3, 0.35]])
    f_foot = landscape.fitness(foot_pos)[0]

    # Sample at a low point
    low_pos = np.array([[0.0, 0.0]])
    f_low = landscape.fitness(low_pos)[0]

    print(f"  Peak fitness at (0.8, 0.85): {f_peak:.3f}")
    print(f"  Foothill at (0.3, 0.35):     {f_foot:.3f}")
    print(f"  Low point at (0.0, 0.0):     {f_low:.3f}")

    assert f_peak > f_foot, "Global peak should be higher than foothill"
    assert f_foot > f_low, "Foothill should be higher than low point"
    print("  PASS: Landscape peaks are correctly ordered")

    # Test gradient
    grad = landscape.gradient(peak_pos)
    print(f"  Gradient at peak: ({grad[0, 0]:.4f}, {grad[0, 1]:.4f})")
    # Gradient should be near zero at the peak
    assert np.linalg.norm(grad) < 1.0, "Gradient should be small near peak"
    print("  PASS: Gradient is small at peak")


def test_knowledge_field_basics():
    """Test basic KnowledgeField operations."""
    kf = KnowledgeField(grid_res=32, diffusion_sigma=0.5, decay=0.999,
                         support_radius=3, max_slope=0.4)
    landscape = make_default_landscape(seed=42)
    kf.set_fitness_surface(landscape)

    # Initially empty
    assert kf.grid.max() == 0.0, "Grid should start empty"
    assert kf.coverage() == 0.0, "Coverage should start at 0"
    print("  PASS: Initial state is empty")

    # Deposit at a single point
    x = np.array([0.5])
    y = np.array([0.5])
    amounts = np.array([0.1])
    growth = kf.deposit(x, y, amounts)
    assert growth[0] > 0, "Should have positive growth"
    assert kf.grid.max() > 0, "Grid should have knowledge"
    print(f"  PASS: Deposit works (growth={growth[0]:.4f})")

    # Fitness ceiling
    kf2 = KnowledgeField(grid_res=32, diffusion_sigma=0.0, decay=1.0,
                          support_radius=0, max_slope=10.0)
    kf2.set_fitness_surface(landscape)

    # Deposit a lot at a low-fitness point
    x_low = np.array([0.0])
    y_low = np.array([0.0])
    for _ in range(1000):
        kf2.deposit(x_low, y_low, np.array([0.1]))
    cx, cy = kf2._pos_to_cell(x_low, y_low)
    ceiling = kf2._fitness_grid[cx[0], cy[0]]
    actual = kf2.grid[cx[0], cy[0]]
    print(f"  Ceiling at (0,0): {ceiling:.4f}, actual: {actual:.4f}")
    assert actual <= ceiling + 1e-6, "Knowledge should not exceed fitness ceiling"
    print("  PASS: Fitness ceiling enforced")


def test_structural_support():
    """Test that the structural support constraint prevents thin spires."""
    kf = KnowledgeField(grid_res=32, diffusion_sigma=0.0, decay=1.0,
                         support_radius=3, max_slope=0.1)
    # No fitness ceiling (set very high)
    kf._fitness_grid = np.ones((32, 32), dtype=np.float64)

    # Deposit a lot at one point
    x = np.array([0.5])
    y = np.array([0.5])
    for _ in range(100):
        kf.deposit(x, y, np.array([0.5]))

    cx, cy = kf._pos_to_cell(x, y)
    center_h = kf.grid[cx[0], cy[0]]
    # With max_slope=0.1 and support_radius=3, the center can only be
    # 0.1 above the local mean, which is limited by the base
    print(f"  Center height after 100 deposits: {center_h:.4f}")
    assert center_h < 0.5, "Support constraint should limit thin spires"
    print("  PASS: Structural support limits thin spires")


def test_decay_and_diffusion():
    """Test that decay and diffusion work correctly."""
    kf = KnowledgeField(grid_res=32, diffusion_sigma=1.0, decay=0.99)
    kf._fitness_grid = np.ones((32, 32), dtype=np.float64)

    # Deposit at center
    x = np.array([0.5])
    y = np.array([0.5])
    kf.deposit(x, y, np.array([0.5]))
    peak_before = kf.grid.max()

    # Run 100 steps of decay+diffusion
    for _ in range(100):
        kf.step()
    peak_after = kf.grid.max()

    print(f"  Peak before: {peak_before:.4f}, after 100 steps: {peak_after:.6f}")
    assert peak_after < peak_before, "Decay should reduce peak"
    assert peak_after > 0, "Should not fully decay in 100 steps"
    print("  PASS: Decay and diffusion work")


def test_full_simulation(n_steps=18000, n_particles=300):
    """Simulate the full knowledge accumulation process.

    Returns history arrays for diagnostic plotting.
    """
    print(f"\n  Running {n_steps} steps with {n_particles} particles...")
    landscape = make_default_landscape(seed=42)
    kf = KnowledgeField(grid_res=64, diffusion_sigma=0.5, decay=0.9999,
                         support_radius=3, max_slope=0.4)
    kf.set_fitness_surface(landscape)

    rng = np.random.default_rng(42)
    SPACE = 1.0

    # Initialize particles uniformly
    pos = rng.uniform(0, SPACE, (n_particles, 3))

    # First n_vis are visionaries
    vis_frac = 0.02
    n_vis = max(1, int(vis_frac * n_particles))
    vis_nudge = 0.001
    write_rate = 0.005
    step_size = 0.003

    # History tracking
    record_every = 100
    hist_steps = []
    hist_coverage = []
    hist_peak = []
    hist_vis_dist = []
    hist_reg_dist = []

    # Global peak location
    peak_x, peak_y = 0.8, 0.85

    for step in range(n_steps):
        x = pos[:, 0]
        y = pos[:, 1]

        # Deposit knowledge
        amounts = np.full(n_particles, write_rate, dtype=np.float64)
        kf.deposit(x, y, amounts)

        # Visionary nudge
        vis_mask = np.zeros(n_particles, dtype=bool)
        vis_mask[:n_vis] = True
        probes = np.column_stack([x[vis_mask], y[vis_mask]])
        vis_grad = landscape.gradient(probes)
        mag = np.linalg.norm(vis_grad, axis=1, keepdims=True)
        mag = np.maximum(mag, 1e-10)
        vis_dir = vis_grad / mag
        pos[vis_mask, 0] = (pos[vis_mask, 0] + vis_nudge * vis_dir[:, 0]) % SPACE
        pos[vis_mask, 1] = (pos[vis_mask, 1] + vis_nudge * vis_dir[:, 1]) % SPACE

        # Random walk for all particles (simulating base physics)
        noise = rng.normal(0, step_size, (n_particles, 3))
        pos = (pos + noise) % SPACE

        # Diffuse and decay
        kf.step()

        # Record
        if step % record_every == 0:
            hist_steps.append(step)
            hist_coverage.append(kf.coverage())
            hist_peak.append(kf.peak_knowledge())

            # Distance to global peak
            vis_d = np.sqrt((pos[vis_mask, 0] - peak_x)**2 +
                            (pos[vis_mask, 1] - peak_y)**2)
            reg_d = np.sqrt((pos[~vis_mask, 0] - peak_x)**2 +
                            (pos[~vis_mask, 1] - peak_y)**2)
            hist_vis_dist.append(vis_d.mean())
            hist_reg_dist.append(reg_d.mean())

    print(f"  Final coverage: {kf.coverage():.1%}")
    print(f"  Final peak:     {kf.peak_knowledge():.3f}")
    print(f"  Vis mean dist to peak:  {hist_vis_dist[-1]:.3f}")
    print(f"  Reg mean dist to peak:  {hist_reg_dist[-1]:.3f}")

    # Assertions
    assert kf.coverage() > 0.1, f"Coverage should be >10%, got {kf.coverage():.1%}"
    print("  PASS: Coverage > 10%")

    assert kf.peak_knowledge() > 0.3, f"Peak should be >0.3, got {kf.peak_knowledge():.3f}"
    print("  PASS: Peak knowledge > 0.3")

    vis_drift = hist_vis_dist[0] - hist_vis_dist[-1]
    reg_drift = hist_reg_dist[0] - hist_reg_dist[-1]
    print(f"  Visionary drift toward peak: {vis_drift:.3f}")
    print(f"  Regular drift toward peak:   {reg_drift:.3f}")
    assert vis_drift > reg_drift, "Visionaries should drift more toward peak"
    print("  PASS: Visionaries drift more toward peak than regulars")

    return (np.array(hist_steps), np.array(hist_coverage),
            np.array(hist_peak), np.array(hist_vis_dist),
            np.array(hist_reg_dist), kf)


def generate_diagnostic_plot(steps, coverage, peak, vis_dist, reg_dist, kf):
    """Generate a diagnostic plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('PR1 Knowledge Field Verification', fontsize=14, fontweight='bold')

    # Coverage and peak
    ax = axes[0, 0]
    ax.plot(steps, coverage, 'b-', label='Coverage', linewidth=2)
    ax.plot(steps, peak, 'r-', label='Peak Knowledge', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title('Knowledge Growth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Visionary vs regular drift
    ax = axes[0, 1]
    ax.plot(steps, vis_dist, 'r-', label='Visionaries', linewidth=2)
    ax.plot(steps, reg_dist, 'b-', label='Regulars', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean dist to peak')
    ax.set_title('Drift Toward Global Peak')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Knowledge surface
    ax = axes[1, 0]
    im = ax.imshow(kf.grid.T, origin='lower', extent=[0, 1, 0, 1],
                    cmap='YlGn', vmin=0, vmax=1)
    ax.set_title('Final Knowledge Surface')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)

    # Fitness ceiling
    ax = axes[1, 1]
    if kf._fitness_grid is not None:
        im = ax.imshow(kf._fitness_grid.T, origin='lower', extent=[0, 1, 0, 1],
                        cmap='Blues', vmin=0, vmax=1)
        ax.set_title('Hidden Fitness Landscape')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/pr1_verification.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved: results/pr1_verification.png")


def _cleanup():
    if os.path.exists(_standalone_dir):
        shutil.rmtree(_standalone_dir)


if __name__ == '__main__':
    print("=" * 60)
    print("PR1 Knowledge Field Verification Test")
    print("=" * 60)

    print("\n1. Landscape tests:")
    test_landscape()

    print("\n2. Knowledge field basics:")
    test_knowledge_field_basics()

    print("\n3. Structural support:")
    test_structural_support()

    print("\n4. Decay and diffusion:")
    test_decay_and_diffusion()

    print("\n5. Full simulation (18,000 steps):")
    results = test_full_simulation()

    print("\n6. Generating diagnostic plot:")
    generate_diagnostic_plot(*results)

    # Cleanup
    _cleanup()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
