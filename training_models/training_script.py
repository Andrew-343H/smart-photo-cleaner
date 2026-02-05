import torch
import torch.nn as nn
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import datetime
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

def main():
    # Support function
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
    )

        return model.to(device)
    
    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            # Send batch on device (GPU/CPU)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Zero your gradients for every batch!
            optimizer.zero_grad(set_to_none=True)

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                last_loss = running_loss / 100 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def plot_loss_curve(history, out_path='loss_curve.png'):
        history = np.array(history)  # shape (n_epochs, 2)
        epochs = np.arange(1, history.shape[0] + 1)

        plt.figure(figsize=(8,5))
        plt.plot(epochs, history[:, 0], marker='o', label='Train Loss')
        plt.plot(epochs, history[:, 1], marker='o', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per epoch')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parents[1]
    # Create data loaders for our datasets; shuffle for training, not for validation
    train_dir = root / "src" / "data" / "train"
    val_dir   = root / "src" / "data" / "valid"
    transforms_dict = get_transforms()

    train_ds = datasets.ImageFolder(train_dir, transforms_dict["train"])
    val_ds   = datasets.ImageFolder(val_dir, transforms_dict["val"])
    num_classes = len(train_ds.classes)

    batch_size = 32
    num_workers = os.cpu_count() - 1
    pin_memory = True if device.type == "cuda" else False
    training_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    validation_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    model = build_model(num_classes, device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0
    # how many epoch you want to add to model
    EPOCHS = 50

    start_epoch = 0
    best_vloss = float('inf')
    resume_path = None #"model_20240201_120000_ep7.pth"
    history = []

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_vloss = checkpoint['best_vloss']
        start_epoch = checkpoint['epoch'] + 1

    epoch_number = start_epoch

    for _ in range(EPOCHS):
        print(f'EPOCH {epoch_number + 1}')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                # Model on device, so val batch on device too
                vinputs = vinputs.to(device, non_blocking=True)
                vlabels = vlabels.to(device, non_blocking=True)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        history.append([avg_loss, avg_vloss])

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            'Training vs. Validation Loss',
            {'Training': avg_loss, 'Validation': avg_vloss},
            epoch_number + 1
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            # Save checkpoint
            model_path = 'best_model.pt'
            torch.save(model, model_path)

        epoch_number += 1

    plot_loss_curve(history, out_path='loss_curve.png')



if __name__ == "__main__":
    main()