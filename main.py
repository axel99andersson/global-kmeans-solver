import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog

np.random.seed(42)

def generate_data(n_points=6):
    cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(n_points//2, 2))
    cluster2 = np.random.normal(loc=[3, 3], scale=0.5, size=(n_points//2, 2))
    X = np.vstack([cluster1, cluster2])
    return X

def evaluate_objective(X, Z, d=2, k=2):
    """Evaluate the concave objective function at Z."""
    obj = 0
    for j in range(k):
        coord = Z[:d, j]
        weight = Z[d, j]
        if weight > 0:
            obj += np.sum(coord**2) / weight
    return obj

def create_initial_halfspaces(X, d=2, k=2):
    """Create initial box constraints for Z."""
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    halfspaces = []

    for j in range(k):
        for i in range(d):
            h1 = np.zeros((d+1)*k)
            h1[i + j*(d+1)] = -1  # -z_ij <= -min
            halfspaces.append((h1, -mins[i]))
            h2 = np.zeros((d+1)*k)
            h2[i + j*(d+1)] = 1   # z_ij <= max
            halfspaces.append((h2, maxs[i]))

        # Weights between 1 and n
        h3 = np.zeros((d+1)*k)
        h3[d + j*(d+1)] = -1
        halfspaces.append((h3, -1))
        h4 = np.zeros((d+1)*k)
        h4[d + j*(d+1)] = 1
        halfspaces.append((h4, len(X)))

    return halfspaces

def find_interior_point(halfspaces):
    """Find a strictly interior point."""
    A = np.array([hs[0] for hs in halfspaces])
    b = np.array([hs[1] for hs in halfspaces])

    n_var = A.shape[1]

    # Introduce slack variable s (scalar) to maximize
    c = np.zeros(n_var + 1)
    c[-1] = -1  # maximize slack

    # Constraints: A x + s * ||A_i|| <= b
    A_slack = np.hstack([A, np.linalg.norm(A, axis=1).reshape(-1,1)])
    bounds = [(None, None)] * n_var + [(0, None)]  # slack must be positive

    res = linprog(c, A_ub=A_slack, b_ub=b, bounds=bounds, method="highs")

    if res.success:
        interior_point = res.x[:-1]
        return interior_point
    else:
        raise ValueError("Failed to find an interior point!")

def enumerate_vertices(halfspaces):
    """Enumerate vertices from halfspaces."""
    A = np.array([hs[0] for hs in halfspaces])
    b = np.array([hs[1] for hs in halfspaces])

    center = find_interior_point(halfspaces)
    try:
        hs = HalfspaceIntersection(np.hstack([A, b[:, None]]), center)
        return hs.intersections
    except Exception as e:
        print(e)
        return []

def assign_clusters(X, Z, d=2, k=2):
    """Assign each data point to nearest centroid."""
    centroids = Z[:d, :] / Z[d, :]
    distances = np.linalg.norm(X[:, None, :] - centroids.T[None, :, :], axis=2)
    assignment = np.argmin(distances, axis=1)
    return assignment

def add_cutting_plane(X, Z_star, assignment, halfspaces, d=2, k=2):
    """Add a new cutting plane to refine the feasible region."""
    n = len(X)
    Z_proj = np.zeros((d+1, k))
    for j in range(k):
        idx = np.where(assignment == j)[0]
        if len(idx) > 0:
            Z_proj[:d,j] = np.sum(X[idx], axis=0)
            Z_proj[d,j] = len(idx)

    diff = (Z_star - Z_proj).flatten()
    rhs = np.dot(diff, Z_proj.flatten())
    halfspaces.append((diff, rhs))
    return halfspaces

def cutting_plane_algorithm(X, epsilon=1e-3, k=2):
    n, d = X.shape

    halfspaces = create_initial_halfspaces(X, d, k)

    lower_bound = -np.inf
    upper_bound = np.inf
    iteration = 0

    while upper_bound - lower_bound > epsilon:
        iteration += 1
        vertices = enumerate_vertices(halfspaces)
        if len(vertices) == 0:
            print("No feasible vertices left!")
            break

        best_obj = -np.inf
        best_vertex = None

        for v in vertices:
            Z = v.reshape(d+1, k)
            obj = evaluate_objective(X, Z, d, k)
            if obj > best_obj:
                best_obj = obj
                best_vertex = Z

        # best_obj approximates the concave function g(Z)
        c0 = np.sum(np.sum(X**2, axis=1))  # constant term
        current_lower_bound = c0 - best_obj

        assignment = assign_clusters(X, best_vertex, d, k)
        centroids = best_vertex[:d, :] / best_vertex[d, :]
        assignment_cost = np.sum([
            np.linalg.norm(X[i] - centroids[assignment[i]])**2 for i in range(n)
        ])
        current_upper_bound = assignment_cost

        lower_bound = max(lower_bound, current_lower_bound)
        upper_bound = min(upper_bound, current_upper_bound)

        print(f"Iteration {iteration}: lower bound = {lower_bound:.4f}, upper bound = {upper_bound:.4f}")

        halfspaces = add_cutting_plane(X, best_vertex, assignment, halfspaces, d, k)

    return assignment

# Main execution
if __name__ == "__main__":
    X = generate_data()
    assignment = cutting_plane_algorithm(X, epsilon=1e-2)

    plt.scatter(X[:,0], X[:,1], c=assignment, cmap='coolwarm')
    plt.title("Globally Optimal Clustering (Tiny Example)")
    plt.show()
