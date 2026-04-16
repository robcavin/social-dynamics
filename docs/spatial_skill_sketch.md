# Spatial Skill: A Minimal Knowledge Manifold Integration

**Core Insight**: In the base model, particles have spatial positions $(x,y)$ and preference vectors $\mathbf{p}$. Rather than hijacking preference dimensions to represent "skill," we can treat the $(x,y)$ spatial domain itself as the "skill landscape." Preferences return to being pure social signals.

This drastically simplifies the integration. The Knowledge Manifold becomes an environmental feature (a spatial memory field) that influences particles *only* through existing social dynamics channels.

Here is the step-by-step sketch of the minimal integration.

## Step 1: The Environment (The Spatial Grid)

We introduce a 2D grid $M(x,y)$ over the existing spatial domain $[0,1)^2$.
- We also define a static hidden fitness landscape $F(x,y)$ over the same domain.
- **Visualization**: Since the simulation is 3D, we map particle $(x,y)$ to the grid, and constrain their $z$-coordinate to match the knowledge height: $z_i = M(x_i, y_i)$. Particles literally "walk on the mountain."

## Step 2: The Write Operation (Research)

At the end of each simulation step, particles deposit knowledge into the grid at their current $(x,y)$ location.
- **Deposit**: $M(x,y) \leftarrow M(x,y) + \text{write\_rate}$
- **Fitness Ceiling**: $M(x,y) \leftarrow \min(M(x,y), F(x,y))$
- **Structural Support**: Apply a max-slope constraint so isolated spikes collapse into broad hills.
- **Diffusion**: Apply a small Gaussian blur to spread knowledge locally.

*This is essentially a spatial memory field (as described in Richard's base docs) with two new constraints (ceiling and slope).*

## Step 3: The Read Operation (Reward)

At the start of each step, particles read the knowledge height at their location to compute their reward.
- $R_i = \text{exponential\_scale}(M(x_i, y_i))$
- We maintain an exponential moving average of this reward, $\bar{R}_i$.

## Step 4: Reward-Modulated Social Learning (The Feedback Loop)

This is the only modification to the base particle dynamics. We use the reward to bias the existing social learning rule.

In the base model, preferences shift toward the unweighted mean of neighbors:
$$ \mathbf{p}_i \leftarrow (1 - \eta)\mathbf{p}_i + \eta \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{p}_j $$

We simply weight this sum by the neighbor's reward:
$$ w_{ij} = 1 + \rho \bar{R}_j $$
$$ \mathbf{p}_i \leftarrow (1 - \eta)\mathbf{p}_i + \eta \sum_{j \in \mathcal{N}(i)} \frac{w_{ij}}{\sum_k w_{ik}} \mathbf{p}_j $$

**Emergent behavior**: High-knowledge regions produce high-reward particles. These particles become highly influential in the social learning phase. Because they share similar preferences (having learned from each other), they attract other particles to adopt their preferences, which in turn causes those particles to navigate toward the same spatial (skill) regions.

## Step 5: The Visionary Nudge (Optional but helpful)

To ensure the organization doesn't get permanently stuck on local optima, a small fraction of particles (~2%) are designated as "visionaries."
- During the spatial movement update, visionaries get a small additional velocity vector pointing along the gradient of the hidden landscape: $\mathbf{v}_{\text{vis}} \propto \nabla F(x_i, y_i)$.
- Regular particles move *only* via the base model's preference-directed physics.

## Summary of Simplifications

Compared to the current `feature/knowledge-manifold` branch, this approach:
1. **Frees up preferences**: $p_0, p_1$ are no longer hijacked for $(x,y)$ movement. They return to being $K$-dimensional social signals.
2. **Removes gradient climbing**: Regular particles no longer explicitly climb $\nabla M$. They arrive at peaks purely because social learning aligns their preferences with successful peers, causing them to move together spatially.
3. **Removes adaptive noise**: The base model's inherent spatial exploration (driven by preference conflicts and repulsion) is sufficient.

The entire "Mountain Mode" reduces to just a specialized spatial memory field that weights the existing social learning rule.
