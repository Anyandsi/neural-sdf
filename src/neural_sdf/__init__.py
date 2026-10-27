import torch
import trimesh
import pymeshlab
import polyscope as ps
import numpy as np
from neural_sdf.train import train_model
from neural_sdf.model import SDFNet
from skimage.measure import marching_cubes

def main() -> None:
    device="cuda"
    bound = 1.0 # match the normalization in mesh_to_sdf step
    res = 256

    # meshes to train on
    BUNNY_PATH = "data/bunny_watertight.obj"

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
    print("predictions: ", sdf_grid.min(), sdf_grid.max())
    verts, faces, _, _ = marching_cubes(sdf_grid, level=0.0, spacing=(xs[1] - xs[0],) * 3)
    verts += xs[0]

    ps.init()
    ps.register_surface_mesh("reconstructed mesh", verts, faces)
    ps.show()