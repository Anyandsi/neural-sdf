import polyscope as ps
import trimesh

def main() -> None:
    mesh = trimesh.load("data/stanford-bunny.obj", force="mesh")
    print(mesh.vertices.shape, mesh.faces.shape, mesh.is_watertight)

    ps.init()
    ps.register_surface_mesh("bunny", mesh.vertices, mesh.faces)
    ps.show()
