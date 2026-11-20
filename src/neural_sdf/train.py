import torch
from neural_sdf.model import SDFNet
from neural_sdf.data import generate_dataset
from pathlib import Path

LEARNING_RATE = 0.0002
BATCH_SIZE = 32768
SURFACE_BATCH_SIZE = BATCH_SIZE // 4
EPOCHS = 2000
WARMUP_EPOCHS = 20
DELTA = 0.1

SDF_WEIGHT = 1.0
EIKONAL_WEIGHT = 0.1
NEUMANN_WEIGHT = 0.1
DIRICHLET_WEIGHT = 1.0

NUM_NEAR_POINTS = 500000
NUM_FAR_POINTS = 50000
NUM_SURFACE_POINTS = 100000

# we only care about accurate numeric predictions close to the surface
def clamped_l1(pred, y):
    return (torch.clamp(pred, -DELTA, DELTA) - torch.clamp(y, -DELTA, DELTA)).abs().mean()

# gradients should ideally be of magnitude 1 everywhere
def eikonal(grad):
    return ((grad.norm(dim=-1) - 1)**2).mean()

# direction of the gradient should ideally be the same as normals direction at that point
# this and the next term require a separate set of points sampled exclusively on the mesh
def neumann(grad, normal_y):
    return (1 - torch.nn.functional.cosine_similarity(grad, normal_y, dim=-1)).mean()

# the points sampled on the mesh must return 0
def dirichlet(pred):
    return pred.abs().mean()

def test_loop(X_all, y_all, model, loss_fn):
    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(X_all), y_all).item()
    print(f"Avg loss: {test_loss:>8f}")


def train_loop(X_sdf_all, y_sdf_all, X_surface_all, y_normals_all, model, loss_fn, optimizer, batch_size):
    model.train()
    n = X_sdf_all.shape[0]
    perm = torch.randperm(n, device=X_sdf_all.device)

    for start in range(0, n, batch_size):
        # separately get next batch of sdf points
        # and a random batch of surface points
        # compute their predictions and gradients 
        idx = perm[start:start + batch_size]
        X_sdf, y_sdf = X_sdf_all[idx], y_sdf_all[idx]

        surface_idx = torch.randint(0, X_surface_all.shape[0], (SURFACE_BATCH_SIZE,), device=X_surface_all.device)
        X_surface, y_normals = X_surface_all[surface_idx], y_normals_all[surface_idx]
        
        # single forward pass over both batches, then split back
        n_sdf = X_sdf.shape[0]
        pred, grad = forward_with_grad(model, torch.cat([X_sdf, X_surface]))
        pred_sdf, pred_surface = pred[:n_sdf], pred[n_sdf:]
        grad_sdf, grad_surface = grad[:n_sdf], grad[n_sdf:]

        # compute loss terms

        sdf_loss = loss_fn(pred_sdf, y_sdf)
        eik_loss = eikonal(grad_sdf)
        neu_loss = neumann(grad_surface, y_normals)
        dir_loss = dirichlet(pred_surface)

        if(start == 0):
            print(f"sdf {sdf_loss.item():.4f} | eik {eik_loss.item():.4f} | neu {neu_loss.item():.4f} | dir {dir_loss.item():.4f}")

        loss =  SDF_WEIGHT * sdf_loss + EIKONAL_WEIGHT * eik_loss + NEUMANN_WEIGHT * neu_loss + DIRICHLET_WEIGHT * dir_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

def forward_with_grad(model, X):
    X.requires_grad = True
    pred = model(X)
    grad = torch.autograd.grad(pred, X, grad_outputs=torch.ones_like(pred), create_graph=True)[0]

    return pred, grad


def train_model(filepath: str, device: str, model_path: Path):
    torch.manual_seed(0)
    print("Training the model")
    dataset = generate_dataset(filepath=filepath, num_near_points=NUM_NEAR_POINTS, num_surface_points=NUM_SURFACE_POINTS, num_far_points=NUM_FAR_POINTS)

    n_train = int(0.8 * len(dataset["sdf_points"])) # since we only test l1 on sdf points

    train_sdf_points, test_sdf_points = dataset["sdf_points"][:n_train], dataset["sdf_points"][n_train:]
    train_sdf_values, test_sdf_values = dataset["sdf_values"][:n_train], dataset["sdf_values"][n_train:]

    train_sdf_points_tensor = torch.from_numpy(train_sdf_points).float()
    test_sdf_points_tensor = torch.from_numpy(test_sdf_points).float()
    train_sdf_values_tensor = torch.from_numpy(train_sdf_values).float().unsqueeze(1)
    test_sdf_values_tensor = torch.from_numpy(test_sdf_values).float().unsqueeze(1)

    # only test on clamped l1, use point/normal pairs only during training
    train_points_surface_tensor = torch.from_numpy(dataset["surface_points"]).float()
    train_normals_surface_tensor = torch.from_numpy(dataset["surface_normals"]).float()

    train_sdf_X, train_sdf_y = train_sdf_points_tensor.to(device), train_sdf_values_tensor.to(device)
    train_surface_X, train_surface_y = train_points_surface_tensor.to(device), train_normals_surface_tensor.to(device)
    test_sdf_X,  test_sdf_y  = test_sdf_points_tensor.to(device),  test_sdf_values_tensor.to(device)

    model = SDFNet()
    model.to(device)
    loss = clamped_l1

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    for epoch in range(EPOCHS):
        train_loop(train_sdf_X, train_sdf_y, train_surface_X, train_surface_y, model, loss, optimizer, BATCH_SIZE)
        scheduler.step()
        if epoch % 10 == 0:
            print("EPOCH #", epoch, scheduler.get_last_lr()[0])
            test_loop(test_sdf_X, test_sdf_y, model, loss)
    
    print("Training complete! Saving the model")
    torch.save(model.state_dict(), model_path)
