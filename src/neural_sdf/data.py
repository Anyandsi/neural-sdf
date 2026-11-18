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
def generate_dataset(filepath: str, num_near_points: int, num_surface_points: int, num_far_points: int):
    rng = np.random.default_rng(0)
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

    # uniform points over the whole [-1, 1] cube, so the corners outside
    # the unit sphere are supervised too: otherwise Fourier features create floaters there
    far_points = rng.uniform(-1.0, 1.0, size=(num_far_points, 3)).astype(np.float32)
    far_sdf = mesh_to_sdf.mesh_to_sdf(mesh, far_points).astype(np.float32)

    sdf_points = np.concatenate([near_points, far_points])
    sdf_values = np.concatenate([near_sdf, far_sdf])

    # shuffle
    sdf_shuffle_idx = rng.permutation(len(sdf_points))
    surface_shuffle_idx = rng.permutation(num_surface_points)

    return dict(
        sdf_points=sdf_points[sdf_shuffle_idx],
        sdf_values=sdf_values[sdf_shuffle_idx],
        surface_points=surface_points[surface_shuffle_idx],
        surface_normals=surface_normals[surface_shuffle_idx]
    )
