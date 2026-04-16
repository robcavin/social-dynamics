"""
Knowledge Field
================

A scalar spatial field over [0, L)^2 that records accumulated knowledge.
Architecturally a sibling of the spatial memory field (Section 8 of
``simulator_overview.pdf``), sharing the same grid, accumulation, decay,
and blur pattern, but with different semantics:

    Spatial Memory Field          Knowledge Field
    ─────────────────────         ──────────────────────
    Shape: (G, G, K)              Shape: (G, G)
    Stores: preference vectors    Stores: scalar knowledge height
    Write: deposit p_i^(d)        Write: deposit constant write_rate
    Read: modulate preferences    Read: return height (→ reward in PR2)
    Ceiling: none                 Ceiling: hidden fitness F(x, y)
    Support: none                 Support: max-slope constraint

Coordinate convention
---------------------
- Spatial positions x, y ∈ [0, L)  where L = SPACE (default 1.0)
- Grid cell mapping:  cell_x = floor(x / L * G) mod G
                      cell_y = floor(y / L * G) mod G
  This matches the spatial memory field's ``inv_cell = G / SPACE``.
- Knowledge height ∈ [0, 1]  (normalised against fitness range)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter


class KnowledgeField:
    """Persistent scalar knowledge surface over the spatial domain.

    Parameters
    ----------
    grid_res : int
        Resolution of the square knowledge grid (G × G).
    diffusion_sigma : float
        Gaussian blur sigma (in grid cells) for spatial diffusion.
    decay : float
        Multiplicative decay per step (1.0 = no decay).
    support_radius : int
        Radius (in cells) of the neighbourhood for the structural
        support constraint.
    max_slope : float
        Maximum allowed height above the local neighbourhood mean.
        Lower values require broader bases for tall peaks.
    space : float
        Domain size (default 1.0, matching ``grid3d.SPACE``).
    """

    def __init__(self, grid_res=64, diffusion_sigma=1.0, decay=0.9999,
                 support_radius=3, max_slope=0.4, space=1.0):
        self.G = grid_res
        self.diffusion_sigma = diffusion_sigma
        self.decay = decay
        self.support_radius = support_radius
        self.max_slope = max_slope
        self.space = space

        # The knowledge surface: scalar height at each grid cell
        self.grid = np.zeros((grid_res, grid_res), dtype=np.float64)

        # Hidden fitness ceiling (set via set_fitness_surface)
        self._fitness_grid = None   # (G, G) normalised to [0, 1]
        self._f_min = 0.0
        self._f_max = 1.0

    # ── Fitness ceiling ───────────────────────────────────────────

    def set_fitness_surface(self, landscape, f_min=None, f_max=None):
        """Pre-compute the hidden fitness at every grid cell.

        Parameters
        ----------
        landscape : object with ``.fitness(positions)`` method
            The hidden fitness landscape.  ``positions`` is (N, 2+).
        f_min, f_max : float or None
            If None, computed from the grid sampling.
        """
        G = self.G
        L = self.space
        # Sample on cell centres
        xs = np.linspace(0, L, G, endpoint=False) + L / (2 * G)
        ys = np.linspace(0, L, G, endpoint=False) + L / (2 * G)
        gx, gy = np.meshgrid(xs, ys, indexing='ij')
        probes = np.column_stack([gx.ravel(), gy.ravel()])

        raw_fitness = landscape.fitness(probes).reshape(G, G)

        if f_min is None:
            f_min = raw_fitness.min()
        if f_max is None:
            f_max = raw_fitness.max()

        self._f_min = f_min
        self._f_max = f_max
        span = max(f_max - f_min, 1e-12)
        self._fitness_grid = (raw_fitness - f_min) / span   # [0, 1]

    # ── Reset ─────────────────────────────────────────────────────

    def reset(self):
        """Zero out the knowledge grid."""
        self.grid[:] = 0.0

    # ── Coordinate helpers ────────────────────────────────────────

    def _pos_to_cell(self, x, y):
        """Convert spatial positions to grid cell indices.

        Uses the same mapping as the spatial memory field:
            inv_cell = G / SPACE
            cx = floor(x * inv_cell) mod G

        Parameters
        ----------
        x, y : ndarray (N,)
            Spatial coordinates in [0, SPACE).

        Returns
        -------
        cx, cy : ndarray (N,) int
            Grid cell indices, clamped to [0, G-1].
        """
        G = self.G
        inv_cell = G / self.space
        cx = (x * inv_cell).astype(int) % G
        cy = (y * inv_cell).astype(int) % G
        return cx, cy

    # ── Read ──────────────────────────────────────────────────────

    def sample(self, x, y):
        """Read the knowledge height at spatial positions.

        Parameters
        ----------
        x, y : ndarray (N,)
            Spatial coordinates in [0, SPACE).

        Returns
        -------
        heights : ndarray (N,) float64  in [0, 1]
        """
        cx, cy = self._pos_to_cell(x, y)
        return self.grid[cx, cy]

    def sample_fitness(self, x, y):
        """Read the hidden fitness ceiling at spatial positions.

        Returns
        -------
        fitness : ndarray (N,) float64  in [0, 1]
        """
        if self._fitness_grid is None:
            return np.zeros_like(x)
        cx, cy = self._pos_to_cell(x, y)
        return self._fitness_grid[cx, cy]

    # ── Write ─────────────────────────────────────────────────────

    def deposit(self, x, y, amounts):
        """Raise the knowledge surface at particle positions.

        Mirrors the spatial memory field's write step::

            np.add.at(self.grid, (cx, cy), amounts)

        Then applies the fitness ceiling and structural support
        constraints (the two additions over the base spatial field).

        Parameters
        ----------
        x, y : ndarray (N,)
            Spatial coordinates.
        amounts : ndarray (N,) or float
            How much to raise at each position.

        Returns
        -------
        actual_growth : ndarray (N,) float64
            How much the surface actually rose at each particle's cell.
        """
        cx, cy = self._pos_to_cell(x, y)

        before = self.grid[cx, cy].copy()

        # Accumulate (same as spatial memory field write)
        np.add.at(self.grid, (cx, cy), amounts)

        # Fitness ceiling (new constraint)
        if self._fitness_grid is not None:
            np.minimum(self.grid, self._fitness_grid, out=self.grid)

        # Structural support (new constraint)
        self._apply_support_constraint()

        # Safety clamp
        np.clip(self.grid, 0.0, 1.0, out=self.grid)

        after = self.grid[cx, cy]
        actual_growth = np.maximum(after - before, 0.0)
        return actual_growth

    def _apply_support_constraint(self):
        """Enforce structural support: height ≤ local_mean + max_slope.

        Prevents infinitely thin spires.  A cell can only be
        ``max_slope`` above the average of its neighbourhood.
        Broad bases support higher peaks.
        """
        r = self.support_radius
        if r <= 0:
            return
        size = 2 * r + 1
        local_mean = uniform_filter(self.grid, size=size, mode='wrap')
        max_allowed = local_mean + self.max_slope
        np.minimum(self.grid, max_allowed, out=self.grid)
        np.clip(self.grid, 0.0, 1.0, out=self.grid)

    # ── Decay and diffusion ───────────────────────────────────────

    def step(self):
        """Apply one step of decay + spatial diffusion.

        Mirrors the spatial memory field's decay and blur steps::

            M ← γ · M                      (decay)
            M ← M * G(σ)                   (blur)

        Then re-applies the ceiling and support constraints, since
        blur can push values above the ceiling.
        """
        # Decay (same as spatial memory field)
        if self.decay < 1.0:
            self.grid *= self.decay

        # Diffuse (Gaussian blur with periodic boundary)
        if self.diffusion_sigma > 0:
            self.grid = gaussian_filter(
                self.grid, sigma=self.diffusion_sigma, mode='wrap')

        # Re-apply constraints after diffusion
        self._apply_support_constraint()
        if self._fitness_grid is not None:
            np.minimum(self.grid, self._fitness_grid, out=self.grid)
        np.clip(self.grid, 0.0, 1.0, out=self.grid)

    # ── Metrics ───────────────────────────────────────────────────

    def coverage(self):
        """Fraction of the fitness landscape that has been discovered.

        Returns the ratio of total knowledge volume to total fitness
        volume.  Analogous to asking "how much of the spatial memory
        field has been written to?"
        """
        if self._fitness_grid is None:
            return 0.0
        fitness_vol = self._fitness_grid.sum()
        if fitness_vol < 1e-12:
            return 0.0
        return self.grid.sum() / fitness_vol

    def peak_knowledge(self):
        """Maximum knowledge height anywhere on the grid."""
        return self.grid.max()

    def peak_location(self):
        """Grid cell of the highest knowledge point (as spatial coords).

        Returns
        -------
        x, y : float
            Spatial coordinates of the peak in [0, SPACE).
        """
        idx = np.unravel_index(self.grid.argmax(), self.grid.shape)
        G = self.G
        L = self.space
        x = (idx[0] + 0.5) * L / G
        y = (idx[1] + 0.5) * L / G
        return x, y
