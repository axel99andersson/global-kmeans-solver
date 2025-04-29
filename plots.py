import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Create figure
fig, axs = plt.subplots(1, 4, figsize=(24, 6))

# Define points of the true feasible set (convex hull)
feasible_set = np.array([
    [2, 2],
    [3, 1.5],
    [3.5, 2.5],
    [2.8, 3.5],
    [2, 3]
])

# Define initial simplex (outer approximation)
simplex = np.array([
    [1, 1],
    [5, 1],
    [3, 5]
])

# Mark Z^N (solution to relaxed problem)
Z_N = np.array([3.8, 2])
# Mark Z_hat^N (projection onto feasible set)
Z_hat_N = np.array([3, 2.5])

# Normal vector for cutting plane
normal_vec = Z_N - Z_hat_N
normal_vec = normal_vec / np.linalg.norm(normal_vec)

# Equation of cutting plane: normal_vec . (x - Z_hat_N) = 0
x_vals = np.linspace(1, 5, 100)
y_vals = (-normal_vec[0]*(x_vals - Z_hat_N[0])/normal_vec[1]) + Z_hat_N[1]

# --- Step 1: Initial setup ---
axs[0].add_patch(Polygon(feasible_set, closed=True, facecolor='lightgreen', edgecolor='green', alpha=0.5))
axs[0].add_patch(Polygon(simplex, closed=True, facecolor='none', edgecolor='blue', linestyle='--', linewidth=2))
axs[0].plot(Z_N[0], Z_N[1], 'rx', markersize=10)
axs[0].set_title('Step 1: Relaxed solution $Z^N$')
axs[0].set_xlim(0, 6)
axs[0].set_ylim(0, 6)
axs[0].grid(True)

# --- Step 2: Projection onto feasible set ---
axs[1].add_patch(Polygon(feasible_set, closed=True, facecolor='lightgreen', edgecolor='green', alpha=0.5))
axs[1].add_patch(Polygon(simplex, closed=True, facecolor='none', edgecolor='blue', linestyle='--', linewidth=2))
axs[1].plot(Z_N[0], Z_N[1], 'rx', markersize=10)
axs[1].plot(Z_hat_N[0], Z_hat_N[1], 'ko', markersize=8)
axs[1].set_title('Step 2: Find closest feasible $\hat{Z}^N$')
axs[1].set_xlim(0, 6)
axs[1].set_ylim(0, 6)
axs[1].grid(True)

# --- Step 3: Add cutting plane ---
axs[2].add_patch(Polygon(feasible_set, closed=True, facecolor='lightgreen', edgecolor='green', alpha=0.5))
axs[2].add_patch(Polygon(simplex, closed=True, facecolor='none', edgecolor='blue', linestyle='--', linewidth=2))
axs[2].plot(Z_N[0], Z_N[1], 'rx', markersize=10)
axs[2].plot(Z_hat_N[0], Z_hat_N[1], 'ko', markersize=8)
axs[2].plot(x_vals, y_vals, 'm-', linewidth=2)
axs[2].arrow(Z_hat_N[0], Z_hat_N[1], 0.8*normal_vec[0], 0.8*normal_vec[1], head_width=0.1, head_length=0.1, fc='magenta', ec='magenta')
axs[2].set_title('Step 3: Cutting plane added')
axs[2].set_xlim(0, 6)
axs[2].set_ylim(0, 6)
axs[2].grid(True)

# --- Step 4: Feasible region after cut ---
axs[3].add_patch(Polygon(feasible_set, closed=True, facecolor='lightgreen', edgecolor='green', alpha=0.3))
axs[3].add_patch(Polygon(simplex, closed=True, facecolor='none', edgecolor='blue', linestyle='--', linewidth=2))
axs[3].plot(Z_hat_N[0], Z_hat_N[1], 'ko', markersize=8)
axs[3].plot(x_vals, y_vals, 'm-', linewidth=2)

# Shade feasible and infeasible regions clearly
x_fill = np.linspace(1, 5, 500)
y_cut = (-normal_vec[0]*(x_fill - Z_hat_N[0])/normal_vec[1]) + Z_hat_N[1]
axs[3].fill_between(x_fill, 0, y_cut, color='red', alpha=0.3, label='Cut-off region')
axs[3].fill_between(x_fill, y_cut, 6, color='lightblue', alpha=0.5, label='Remaining feasible region')

axs[3].set_title('Step 4: Updated feasible region')
axs[3].set_xlim(0, 6)
axs[3].set_ylim(0, 6)
axs[3].grid(True)
axs[3].legend()

for ax in axs:
    ax.set_xlabel('$Z_1$')
    ax.set_ylabel('$Z_2$')

plt.suptitle('Visualization of Relaxed Solution, Projection, Cutting Plane, and New Feasible Region', fontsize=16)
plt.tight_layout()
plt.show()