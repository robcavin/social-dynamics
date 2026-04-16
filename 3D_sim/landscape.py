"""
Fitness Landscape
=================

Defines the *hidden* fitness surface over the spatial domain [0, L)^2.
This is the ceiling of the knowledge field — the maximum possible
knowledge at each point in space.

The landscape is a sum of Gaussian peaks plus spectral noise,
creating a rugged terrain with multiple local optima, ridges,
and narrow viable paths.

Coordinate convention
---------------------
- Spatial positions x, y ∈ [0, L)  where L = SPACE = 1.0
- Fitness values are non-negative; normalised to [0, 1] by the
  KnowledgeField when it pre-computes the ceiling grid.
"""

import numpy as np


class RuggedLandscape:
    """Multi-peak fitness landscape with spectral noise.

    Parameters
    ----------
    peaks : list of dict
        Each dict has ``center`` (2,), ``height`` float, ``sigma`` float.
        Centers are in spatial coordinates [0, 1).
    noise_frequencies : list of (freq, amplitude) tuples
        Spectral noise components added to the base peaks.
    seed : int
        Random seed for reproducible noise phases.
    """

    def __init__(self, peaks, noise_frequencies=None, seed=42):
        self.peaks = peaks
        self.noise_frequencies = noise_frequencies or []
        self._rng = np.random.default_rng(seed)
        n_freq = len(self.noise_frequencies)
        self._phases_x = self._rng.uniform(0, 2 * np.pi, n_freq)
        self._phases_y = self._rng.uniform(0, 2 * np.pi, n_freq)
        self._freq_dirs = self._rng.uniform(0, 2 * np.pi, n_freq)

    def fitness(self, positions):
        """Evaluate fitness at spatial positions.

        Parameters
        ----------
        positions : ndarray (N, 2) or (N, 3)
            Spatial coordinates.  Only columns 0 and 1 (x, y) are used.

        Returns
        -------
        f : ndarray (N,)
        """
        x = positions[:, 0]
        y = positions[:, 1]
        f = np.zeros(len(positions), dtype=np.float64)

        for peak in self.peaks:
            cx, cy = peak['center'][0], peak['center'][1]
            h = peak['height']
            s = peak['sigma']
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            f += h * np.exp(-dist_sq / (2.0 * s ** 2))

        for i, (freq, amp) in enumerate(self.noise_frequencies):
            dx = np.cos(self._freq_dirs[i])
            dy = np.sin(self._freq_dirs[i])
            proj = x * dx + y * dy
            f += amp * np.sin(2 * np.pi * freq * proj + self._phases_x[i])

        return f

    def gradient(self, positions, eps=1e-4):
        """Numerical gradient via central differences.

        Returns
        -------
        grad : ndarray (N, 2)
        """
        N = len(positions)
        grad = np.zeros((N, 2), dtype=np.float64)
        for dim in range(2):
            p_plus = positions.copy()
            p_minus = positions.copy()
            p_plus[:, dim] += eps
            p_minus[:, dim] -= eps
            grad[:, dim] = (self.fitness(p_plus) -
                            self.fitness(p_minus)) / (2 * eps)
        return grad


def make_default_landscape(seed=42):
    """Create a rugged landscape with multiple peaks and spectral noise.

    Peak centers are in spatial coordinates [0, 1).  The landscape has:
    - A global summit (hard to reach, narrow)
    - Several foothills (easy local optima / traps)
    - High-frequency noise creating ridges and narrow paths
    """
    peaks = [
        # Global summit — high but narrow
        {'center': [0.8, 0.85], 'height': 1.0, 'sigma': 0.12},
        # Attractive foothill — wide, easy to find
        {'center': [0.3, 0.35], 'height': 0.6, 'sigma': 0.20},
        # Secondary foothill
        {'center': [0.65, 0.25], 'height': 0.5, 'sigma': 0.15},
        # Small ridge
        {'center': [0.15, 0.75], 'height': 0.35, 'sigma': 0.10},
    ]

    noise_frequencies = [
        (2.0, 0.08),    # Low frequency — broad undulations
        (3.5, 0.06),
        (5.0, 0.05),    # Medium frequency — ridges
        (7.0, 0.04),
        (9.0, 0.03),    # High frequency — fine texture
        (11.0, 0.02),
        (13.0, 0.015),
    ]

    return RuggedLandscape(peaks, noise_frequencies, seed=seed)
