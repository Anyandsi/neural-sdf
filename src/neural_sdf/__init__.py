import torch
import polyscope as ps
import numpy as np
from neural_sdf.train import train_model
from neural_sdf.model import SDFNet
from neural_sdf.data import BOUND
from skimage.measure import marching_cubes

def main() -> None:
    # train_model()

    model = SDFNet()
    model.load_state_dict(torch.load("torus.pt"))
    model.eval()

    # visualize
    grid_resolution = 50

    xs = np.linspace(-BOUND, BOUND, grid_resolution)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    grid_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    grid_tensor = torch.from_numpy(grid_points).float()
    chunks = []
    with torch.no_grad():
        for chunk in torch.split(grid_tensor, 65536):
            chunks.append(model(chunk))

    volume = torch.cat(chunks).numpy().reshape(grid_resolution, grid_resolution, grid_resolution)

    spacing = xs[1] - xs[0]
    verts, faces, normals, values = marching_cubes(volume, level=0.0, spacing=(spacing, spacing, spacing))
    verts -= BOUND

    ps.init()
    ps.register_surface_mesh("learned torus", verts, faces)
    ps.show()