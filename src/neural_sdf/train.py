import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from neural_sdf.model import SDFNet
from neural_sdf.data import generate_dataset

LEARNING_RATE = 0.0001
BATCH_SIZE = 8192
EPOCHS = 10000
DELTA = 0.1

def clamped_l1(pred, y):
    return (torch.clamp(pred, -DELTA, DELTA) - torch.clamp(y, -DELTA, DELTA)).abs().mean()

def test_loop(X_all, y_all, model, loss_fn):
    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(X_all), y_all).item()
    print(f"Avg loss: {test_loss:>8f}")


def train_loop(X_all, y_all, model, loss_fn, optimizer, batch_size):
    model.train()
    n = X_all.shape[0]
    perm = torch.randperm(n, device=X_all.device)

    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        X, y = X_all[idx], y_all[idx]

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_model(filepath: str, device):
    print("Training the model")
    points, sdf = generate_dataset(filepath)

    n_train = int(0.8 * len(points))

    train_points, test_points = points[:n_train], points[n_train:]
    train_sdf, test_sdf = sdf[:n_train], sdf[n_train:]

    train_points_tensor = torch.from_numpy(train_points).float()
    test_points_tensor = torch.from_numpy(test_points).float()
    train_sdf_tensor = torch.from_numpy(train_sdf).float().unsqueeze(1)
    test_sdf_tensor = torch.from_numpy(test_sdf).float().unsqueeze(1)

    train_X, train_y = train_points_tensor.to(device), train_sdf_tensor.to(device)
    test_X,  test_y  = test_points_tensor.to(device),  test_sdf_tensor.to(device)

    model = SDFNet()
    model.to(device)
    loss = clamped_l1
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loop(train_X, train_y, model, loss, optimizer, BATCH_SIZE)
        if epoch % 10 == 0:
            print("EPOCH #", epoch)
            test_loop(test_X, test_y, model, loss)
    
    print("Training complete! Saving the model")
    torch.save(model.state_dict(), "model.pt")
