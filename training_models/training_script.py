import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

plt.style.use('ggplot')


def get_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
    }

def load_dataloaders(root: Path, transforms, batch_size=32):
    train_dir = root / "src" / "data" / "train"
    val_dir   = root / "src" / "data" / "valid"
    #test_dir  = root / "src" / "data" / "test"

    train_ds = datasets.ImageFolder(train_dir, transforms["train"])
    val_ds   = datasets.ImageFolder(val_dir, transforms["val"])
    #test_ds  = datasets.ImageFolder(test_dir, transforms["test"])

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val":   DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        #"test":  DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    }

    return loaders, len(train_ds.classes), train_ds.class_to_idx


def build_model(num_classes, device):
    model = models.resnet50(weights="DEFAULT")

    for p in model.parameters():
        p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)
    )

    return model.to(device)

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if training:
                optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total

def train(model, loaders, criterion, optimizer, device, epochs=25):
    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = run_epoch(
            model, loaders["val"], criterion, None, device
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, "best_model.pt")

        history.append((train_loss, val_loss, train_acc, val_acc))

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train L {train_loss:.4f} A {train_acc:.3f} | "
            f"Val L {val_loss:.4f} A {val_acc:.3f}"
        )

    return history

def plot_performance(history):
    plt.figure(figsize=(10, 7))
    history = np.array(history)
    plt.plot(history[:,0:2])
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.savefig('loss_curve.png')
    plt.show()


    plt.figure(figsize=(10, 7))
    plt.plot(history[:,2:4])
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_curve.png')
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(__file__).resolve().parents[1]
    transforms = get_transforms()
    loaders, num_classes, class_to_idx = load_dataloaders(root, transforms)

    model = build_model(num_classes, device)
    criterion = nn.NLLLoss()

    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)

    history = train(model, loaders, criterion, optimizer, device)
    torch.save(history, root / "_history.pt")

if __name__ == "__main__":
    main()