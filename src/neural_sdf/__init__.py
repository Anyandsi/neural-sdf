import torch
import trimesh
import argparse
import polyscope as ps
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
from neural_sdf.train import train_model
from neural_sdf.model import SDFNet
from neural_sdf.data import prepare_mesh
from skimage.measure import marching_cubes
from mesh_to_sdf.utils import scale_to_unit_sphere
from neural_sdf.data import MESH_OPTIONS

MODELS_DIR = Path("models")

def main() -> None:
    parser = argparse.ArgumentParser()

    # either train a new model from scratch and overwrite it [train], then show comparison
    # or just show comparison [display] (model must be there!)
    parser.add_argument(
    "command",
    choices=["train", "display"],
    help="Operation mode. [train] will create a new model or overwrite the existing one "
         "before displaying the results. [display] will only display the result from exisitng model (file must exist under models/)." ,
)

    # choose mesh from the available options, bunny is default
    parser.add_argument("--mesh", choices=MESH_OPTIONS, default="bunny", help="Mesh to train on or display. [bunny] is default option")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    mesh_path = str(prepare_mesh(args.mesh))
    model_path = MODELS_DIR / f"{args.mesh}.pt"

    if args.command == "train":
        train_and_display(mesh_path, model_path)
    else:
        display(mesh_path, model_path)


def train_and_display(mesh_path, model_path):
    MODELS_DIR.mkdir(exist_ok=True)
    train_model(mesh_path, "cuda", model_path)
    display(mesh_path, model_path)


def display(mesh_path, model_path):
    device = "cuda"
    bound = 1.0  # match the normalization in mesh_to_sdf step
    res = 256

    if not model_path.exists():
        raise SystemExit(f"{model_path} not found, run uv run neural-sdf train --mesh {model_path.stem}")

    model = SDFNet()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # visualize reconstructed mesh alongside the original
    with torch.no_grad():
        xs = np.linspace(-bound, bound, res, dtype=np.float32)
        grid = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1).reshape(-1, 3)
        out = []

        for chunk in np.array_split(grid, 64):
            out.append(model(torch.from_numpy(chunk).to(device)).squeeze(-1).cpu().numpy())
    
    sdf_grid = np.concatenate(out).reshape(res, res, res)
    r_vertices, r_faces, _, _ = marching_cubes(sdf_grid, level=0.0, spacing=(xs[1] - xs[0],) * 3)
    r_vertices += xs[0]

    original_mesh = trimesh.load_mesh(mesh_path)
    original_mesh = scale_to_unit_sphere(original_mesh)
    reconstructed_mesh = trimesh.Trimesh(vertices=r_vertices, faces=r_faces)

    print("chamfer: ", chamfer_distance(reconstructed_mesh, original_mesh, 500_000))
    
    ps.init()
    ps.register_surface_mesh("reconstructed mesh", r_vertices, r_faces)
    ps.register_surface_mesh("original mesh", original_mesh.vertices, original_mesh.faces).translate([2.5, 0, 0])
    ps.show()


def chamfer_distance(mesh1, mesh2, n_sampled_points):
    points1, _ = trimesh.sample.sample_surface(mesh1, n_sampled_points)
    points2, _ = trimesh.sample.sample_surface(mesh2, n_sampled_points)

    tree1 = KDTree(points1)
    dists1, _ = tree1.query(points2)
    tree2 = KDTree(points2)
    dists2, _ = tree2.query(points1)

    return np.mean(dists1) + np.mean(dists2)
