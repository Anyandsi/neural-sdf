import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from neural_sdf.model import SDFNet
from neural_sdf.data import generate_dataset

LEARNING_RATE = 0.0001
BATCH_SIZE = 8192
SURFACE_BATCH_SIZE = BATCH_SIZE // 4
EPOCHS = 2000
DELTA = 0.1

SDF_WEIGHT = 1.0
EIKONAL_WEIGHT = 0.1
NEUMANN_WEIGHT = 0.1
DIRICHLET_WEIGHT = 1.0

NUM_NEAR_POINTS = 500000
NUM_SURFACE_POINTS = 100000

# loss terms

def clamped_l1(pred, y):
    return (torch.clamp(pred, -DELTA, DELTA) - torch.clamp(y, -DELTA, DELTA)).abs().mean()

def eikonal(grad):
    return ((grad.norm(dim=-1) - 1)**2).mean()

def neumann(grad, normal_y):
    return (1 - torch.nn.functional.cosine_similarity(grad, normal_y, dim=-1)).mean()

def dirichlet(pred):
    return pred.abs().mean()

def test_loop(X_all, y_all, model, loss_fn):
    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(X_all), y_all).item()
    print(f"Avg loss: {test_loss:>8f}")


def train_loop(X_near_all, y_near_all, X_surface_all, y_normals_all, model, loss_fn, optimizer, batch_size):
    model.train()
    n = X_near_all.shape[0]
    perm = torch.randperm(n, device=X_near_all.device)

    for start in range(0, n, batch_size):
        # separately get next batch of near surface points
        # and a random batch of surface points
        # compute their predictions and gradients 
        idx = perm[start:start + batch_size]
        X_near, y_near = X_near_all[idx], y_near_all[idx]

        surface_idx = torch.randint(0, X_surface_all.shape[0], (SURFACE_BATCH_SIZE,))
        X_surface, y_normals = X_surface_all[surface_idx], y_normals_all[surface_idx]
        
        # single forward pass over both batches, then split back
        n_near = X_near.shape[0]
        pred, grad = forward_with_grad(model, torch.cat([X_near, X_surface]))
        pred_near, pred_surface = pred[:n_near], pred[n_near:]
        grad_near, grad_surface = grad[:n_near], grad[n_near:]

        # compute loss terms

        sdf_loss = loss_fn(pred_near, y_near)
        eik_loss = eikonal(grad_near)
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


def train_model(filepath: str, device):
    print("Training the model")
    dataset = generate_dataset(filepath=filepath, num_near_points=NUM_NEAR_POINTS, num_surface_points=NUM_SURFACE_POINTS)

    n_train = int(0.8 * NUM_NEAR_POINTS) # since we only test l1 on near points

    train_points_near, test_points_near = dataset["near_points"][:n_train], dataset["near_points"][n_train:]
    train_sdf_near, test_sdf_near = dataset["near_sdf"][:n_train], dataset["near_sdf"][n_train:]

    train_points_near_tensor = torch.from_numpy(train_points_near).float()
    test_points_near_tensor = torch.from_numpy(test_points_near).float()
    train_sdf_near_tensor = torch.from_numpy(train_sdf_near).float().unsqueeze(1)
    test_sdf_near_tensor = torch.from_numpy(test_sdf_near).float().unsqueeze(1)

    # only test on clamped l1, use point/normal pairs only during training
    train_points_surface_tensor = torch.from_numpy(dataset["surface_points"]).float()
    train_normals_surface_tensor = torch.from_numpy(dataset["surface_normals"]).float()

    train_near_X, train_near_y = train_points_near_tensor.to(device), train_sdf_near_tensor.to(device)
    train_surface_X, train_surface_y = train_points_surface_tensor.to(device), train_normals_surface_tensor.to(device)
    test_near_X,  test_near_y  = test_points_near_tensor.to(device),  test_sdf_near_tensor.to(device)

    model = SDFNet()
    model.to(device)
    loss = clamped_l1
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loop(train_near_X, train_near_y, train_surface_X, train_surface_y, model, loss, optimizer, BATCH_SIZE)
        if epoch % 10 == 0:
            print("EPOCH #", epoch)
            test_loop(test_near_X, test_near_y, model, loss)
    
    print("Training complete! Saving the model")
    torch.save(model.state_dict(), "model.pt")
