# Knowledge Manifold Integration Plan

This plan details how to integrate the Knowledge Manifold into the base Social Dynamics Simulator. The core insight is that **the spatial $(x,y)$ domain is the skill landscape**, and preferences remain pure social signals. 

The integration introduces the Knowledge Manifold as a sibling to the existing Spatial Memory Field. Both fields operate on the same spatial grid with accumulation, decay, and blur, but with different read/write semantics and constraints.

The integration is broken down into three self-contained PRs.

## PR 1: The Knowledge Field (Environmental Layer)

This PR introduces the environmental data structure without changing particle physics.

**What to add:**
1. **`Landscape` class**: A static 2D hidden fitness landscape $F(x,y)$ over the spatial domain $[0, SPACE)^2$.
2. **`KnowledgeField` class**: A scalar grid $M(x,y)$ over the spatial domain.
   - Shares the grid resolution ($G \times G$) of the spatial memory field.
   - **Write**: Accumulates a constant `write_rate` from particles at their spatial locations (using `np.add.at`, exactly like `memory_field`).
   - **Constraints**: Applies the fitness ceiling ($M \leq F$) and the structural support max-slope constraint.
   - **Decay & Blur**: Applies exponential decay and Gaussian blur (same logic as `memory_field`).

**Why this is safe:** It runs entirely in parallel to the base simulation. The field updates, but particles don't read from it yet.

## PR 2: Reward-Modulated Social Learning (The Feedback Loop)

This PR closes the loop: particles read the knowledge field to compute a reward, which then weights their social influence.

**What to add:**
1. **Reward Computation**: Before physics, each particle reads its height on the knowledge grid: $h_i = M(x_i, y_i)$.
   - Compute reward: $R_i = \text{exponential\_scale}(h_i) \cdot (1 + \text{growth\_bonus})$.
   - Update per-particle reward EMA: $\bar{R}_i$.
2. **Social Weighting**: In the `step()` function (where `prefs[:] = (1.0 - social) * prefs + social * nbr_mean` occurs), modify the neighbor mean calculation.
   - Instead of an unweighted mean, weight each neighbor $j$ by $w_{ij} = 1 + \rho \bar{R}_j$.
   - This makes high-knowledge particles more influential in shaping their neighbors' preferences.

**Why this works:** High-reward particles attract others to adopt their preferences. The base physics naturally moves particles toward neighbors with similar preferences. Thus, social alignment drives spatial clustering near productive regions.

## PR 3: The Visionary Nudge & 3D Visualization

This PR adds the final active exploration mechanism and the visual representation.

**What to add:**
1. **Visionary Nudge**: During the physics update, identify a small fraction of particles (`visionary_fraction`).
   - Add a small velocity vector pointing along the hidden gradient $\nabla F(x_i, y_i)$ to their movement.
   - Regular particles remain blind to $F$.
2. **3D Visualization**: When rendering, use the 2D spatial coordinates $(x,y)$ for the horizontal plane, and map the particle's $z$-coordinate to its knowledge height $M(x_i, y_i)$.
   - Render the knowledge grid as a solid mesh.
   - Render the hidden fitness landscape as a wireframe ghost.

**The Result:** Particles physically "walk" on the growing knowledge mountain, driven purely by base preference physics and reward-weighted social influence.

## Design Principles

- **Spatial position is skill**: Particles move spatially; preferences remain pure social signals.
- **No explicit gradient climbing**: Regular particles find peaks via social alignment, not by computing $\nabla M$.
- **Unified architecture**: The knowledge field is architecturally identical to the spatial memory field (accumulation, decay, blur), with two additional constraints (fitness ceiling, structural support) and a scalar reward output instead of a preference vector output.
