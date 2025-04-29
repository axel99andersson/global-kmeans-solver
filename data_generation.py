import numpy as np

def generate_data(n_clusters, n_points, proximity=5):
    data = []
    for k in range(n_clusters):
        cluster = np.random.normal(loc=[max((k-1)*proximity, 0), k*proximity], scale=1.0, size=n_points // n_clusters)
        data.append(cluster)

    X = np.vstack(data)
    return X