import numpy as np

# values out of the head
R = 0.4
r = 0.2
MARGIN = 1.2 # +20%

def torus_sdf(points, R, r):
    q = np.stack(
        [np.linalg.norm(points[:, [0,2]], axis=-1) - R, points[:, 1]],
        axis=-1
    )

    return np.linalg.norm(q, axis=-1) - r

def generate_dataset():

    rng = np.random.default_rng(0)
    bound = (R + r) * MARGIN

    points = rng.uniform(-bound, bound, size=(100_000, 3))
    return points
