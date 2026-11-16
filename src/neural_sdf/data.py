import trimesh
import warnings
import joblib
import mesh_to_sdf
import numpy as np

memory = joblib.Memory("cache", verbose=0)

@memory.cache
# return two shuffled data sets
# sampled points within the unit sphere and their ground truth sdf values
# and sampled on-surface points with their normal vectors
def generate_dataset(filepath: str, num_near_points: int, num_surface_points: int):
    print("Loading " + filepath)
    mesh = trimesh.load_mesh(filepath)

    if not mesh.is_watertight:
        warnings.warn(filepath + " mesh is not watertight!")
    
    near_points, near_sdf = mesh_to_sdf.sample_sdf_near_surface(mesh, number_of_points=num_near_points)

    # sample additional set of points on the mesh and get their normals
    # these are later used to do an additional pass for Neumann condition
    mesh = mesh_to_sdf.utils.scale_to_unit_sphere(mesh)
    surface_points, f_idx = trimesh.sample.sample_surface(mesh, count=num_surface_points)
    surface_normals = mesh.face_normals[f_idx]

    # shuffle
    rng = np.random.default_rng(0)

    near_shuffle_idx = rng.permutation(num_near_points)
    surface_shuffle_idx = rng.permutation(num_surface_points)

    return dict(
        near_points=near_points[near_shuffle_idx],
        near_sdf=near_sdf[near_shuffle_idx],
        surface_points=surface_points[surface_shuffle_idx],
        surface_normals=surface_normals[surface_shuffle_idx],
    )
