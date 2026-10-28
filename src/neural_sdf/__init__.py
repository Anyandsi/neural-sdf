import torch
import trimesh
import pymeshlab
import polyscope as ps
import numpy as np
from neural_sdf.train import train_model
from neural_sdf.model import SDFNet
from skimage.measure import marching_cubes
from mesh_to_sdf.utils import scale_to_unit_sphere

def main() -> None:
    device="cuda"
    bound = 1.0 # match the normalization in mesh_to_sdf step
    res = 256

    # meshes to train on
    BUNNY_PATH = "data/bunny_watertight.obj"
    bunny_mesh = trimesh.load_mesh(BUNNY_PATH)

    train_model(BUNNY_PATH, device)

    model = SDFNet()
    model.load_state_dict(torch.load("model.pt"))
    model.to(device)
    model.eval()

    # visualize
    with torch.no_grad():
        xs = np.linspace(-bound, bound, res, dtype=np.float32)
        grid = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1).reshape(-1, 3)
        out = []

        for chunk in np.array_split(grid, 64):
            out.append(model(torch.from_numpy(chunk).to(device)).squeeze(-1).cpu().numpy())
    
    sdf_grid = np.concatenate(out).reshape(res, res, res)
    r_vertices, r_faces, _, _ = marching_cubes(sdf_grid, level=0.0, spacing=(xs[1] - xs[0],) * 3)
    r_vertices += xs[0]

    # add original for comparison
    bunny_mesh = scale_to_unit_sphere(bunny_mesh)
    
    ps.init()
    ps.register_surface_mesh("reconstructed mesh", r_vertices, r_faces)
    ps.register_surface_mesh("original mesh", bunny_mesh.vertices, bunny_mesh.faces).translate([2.5, 0, 0])
    ps.show()