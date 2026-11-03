import trimesh
import warnings
import joblib
import numpy as np
from mesh_to_sdf import sample_sdf_near_surface

memory = joblib.Memory("cache", verbose=0)

@memory.cache
def generate_dataset(filepath: str):
    print("Loading " + filepath)
    mesh = trimesh.load_mesh(filepath)

    if not mesh.is_watertight:
        warnings.warn(filepath + " mesh is not watertight!")
    
    points, sdf = sample_sdf_near_surface(mesh, number_of_points=500000)

    # shuffle
    rng = np.random.default_rng(0)
    shuffle_idx = rng.permutation(len(points))

    return points[shuffle_idx], sdf[shuffle_idx]
