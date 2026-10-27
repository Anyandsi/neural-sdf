import numpy as np

# values out of the head
R = 0.4
r = 0.2
MARGIN = 1.2 # +20%
BOUND = (R + r) * MARGIN

def torus_sdf(points):
    q = np.stack(
        [np.linalg.norm(points[:, [0,2]], axis=-1) - R, points[:, 1]],
        axis=-1
    )

    return np.linalg.norm(q, axis=-1) - r

def generate_dataset():
    rng = np.random.default_rng(0)

    points = rng.uniform(-BOUND, BOUND, size=(100_000, 3))
    return points
