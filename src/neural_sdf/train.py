import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from neural_sdf.model import SDFNet
from neural_sdf.data import generate_dataset
import numpy as np

# values from head
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 120

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    model.train()

    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()

    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")


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

    train_dataset = TensorDataset(train_points_tensor.to(device), train_sdf_tensor.to(device))
    test_dataset = TensorDataset(test_points_tensor.to(device), test_sdf_tensor.to(device))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = SDFNet()
    model.to(device)
    loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print("EPOCH #", epoch)
        train_loop(dataloader=train_dataloader, model=model, loss_fn=loss, optimizer=optimizer, batch_size=BATCH_SIZE)
        test_loop(dataloader=test_dataloader, model=model, loss_fn=loss)
    
    print("Training complete! Saving the model")
    torch.save(model.state_dict(), "model.pt")
