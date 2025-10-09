import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from tqdm import tqdm


# Configuration
CONFIG = {
    "data_dir": "../input/dog-breed-identification/",
    "batch_size": 32,
    "test_batch_size": 32,
    "epochs": 20,
    "lr": 0.001,
    "gamma": 0.7,
    "seed": 42,
    "log_interval": 10,
    "save_model": True,
    "output_dir": "../output/kaggle/working/",
    "num_workers": 4,
}


class DogBreedDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Create label mapping
        self.breeds = sorted(self.data["breed"].unique())
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(self.breeds)}
        self.idx_to_breed = {idx: breed for breed, idx in self.breed_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx]["id"]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        label = self.breed_to_idx[self.data.iloc[idx]["breed"]]

        if self.transform:
            image = self.transform(image)

        return image, label


class LeNet(nn.Module):
    """LeNet-style CNN adapted for dog breed classification"""

    def __init__(self, num_classes=120):
        super(LeNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Calculate feature size after conv layers
        # Input: 224x224 -> conv1+pool: 112x112 -> conv2+pool: 56x56 -> conv3+pool: 28x28
        self.feature_size = 128 * 28 * 28

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flatten
        x = torch.flatten(x, 1)

        # FC layers
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


def train(model, device, train_loader, optimizer, epoch, train_losses, train_accs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % CONFIG["log_interval"] == 0:
            pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    print(f"Train Epoch: {epoch} \tLoss: {epoch_loss:.4f} \tAccuracy: {epoch_acc:.2f}%")


def test(model, device, test_loader, test_losses, test_accs):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.2f}%)\n"
    )

    return np.array(all_preds), np.array(all_targets)


def plot_training_curves(
    train_losses, train_accs, test_losses, test_accs, save_path="training_curves.png"
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(train_losses, label="Train Loss", marker="o")
    ax1.plot(test_losses, label="Test Loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Test Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy curve
    ax2.plot(train_accs, label="Train Accuracy", marker="o")
    ax2.plot(test_accs, label="Test Accuracy", marker="s")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Test Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true, y_pred, class_names, save_path="confusion_matrix.png", top_n=20
):
    # Plot only top N classes for readability
    cm = confusion_matrix(y_true, y_pred)

    # Get top N classes by frequency
    class_counts = np.bincount(y_true)
    top_classes = np.argsort(class_counts)[-top_n:]

    # Filter confusion matrix
    cm_filtered = cm[np.ix_(top_classes, top_classes)]
    filtered_names = [class_names[i] for i in top_classes]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_filtered,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=filtered_names,
        yticklabels=filtered_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Top {top_n} Classes)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def compute_saliency_map(model, image, target_class, device):
    model.eval()
    image = image.unsqueeze(0).to(device)
    image.requires_grad = True

    output = model(image)
    model.zero_grad()

    # Backward pass
    output[0, target_class].backward()

    # Get gradients
    saliency = image.grad.data.abs()
    saliency = saliency.squeeze().cpu()
    saliency = saliency.max(dim=0)[0]  # Take max across color channels

    return saliency.numpy()


def visualize_saliency(
    model, dataset, device, num_samples=5, save_path="saliency_maps.png"
):
    model.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        image, label = dataset[idx]

        # Compute saliency
        saliency = compute_saliency_map(model, image, label, device)

        # Denormalize image for display
        img_display = image.cpu().numpy().transpose(1, 2, 0)
        img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array(
            [0.485, 0.456, 0.406]
        )
        img_display = np.clip(img_display, 0, 1)

        # Plot original image
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f"Original - Class: {dataset.dataset.idx_to_breed[label]}")
        axes[i, 0].axis("off")

        # Plot saliency map
        axes[i, 1].imshow(saliency, cmap="hot")
        axes[i, 1].set_title("Saliency Map")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saliency maps saved to {save_path}")
    plt.close()


def main():
    # Setup
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Data augmentation and normalization (ImageNet statistics)
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    train_csv = os.path.join(CONFIG["data_dir"], "labels.csv")
    train_img_dir = os.path.join(CONFIG["data_dir"], "train")

    full_dataset = DogBreedDataset(train_csv, train_img_dir, transform=train_transform)

    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )

    # Update validation dataset transform
    val_dataset.dataset.transform = test_transform

    train_kwargs = {"batch_size": CONFIG["batch_size"], "shuffle": True}
    test_kwargs = {"batch_size": CONFIG["test_batch_size"], "shuffle": False}

    if torch.cuda.is_available():
        cuda_kwargs = {"num_workers": CONFIG["num_workers"], "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **test_kwargs)

    # Model setup
    num_classes = len(full_dataset.breeds)
    print(f"Number of dog breeds: {num_classes}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    model = LeNet(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = StepLR(optimizer, step_size=5, gamma=CONFIG["gamma"])

    # Training
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    print(f"\nStarting training for {CONFIG['epochs']} epochs...\n")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train(model, device, train_loader, optimizer, epoch, train_losses, train_accs)
        preds, targets = test(model, device, val_loader, test_losses, test_accs)
        scheduler.step()

    # Visualizations
    print("\nGenerating visualizations...")

    plot_training_curves(
        train_losses,
        train_accs,
        test_losses,
        test_accs,
        save_path=os.path.join(CONFIG["output_dir"], "training_curves.png"),
    )

    plot_confusion_matrix(
        targets,
        preds,
        full_dataset.breeds,
        save_path=os.path.join(CONFIG["output_dir"], "confusion_matrix.png"),
    )

    visualize_saliency(
        model,
        val_dataset,
        device,
        num_samples=5,
        save_path=os.path.join(CONFIG["output_dir"], "saliency_maps.png"),
    )

    # Save model
    if CONFIG["save_model"]:
        model_path = os.path.join(CONFIG["output_dir"], "dog_breed_lenet.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print("\nTraining completed!")
    print(f"Final Results:")
    print(f"  Train Accuracy: {train_accs[-1]:.2f}%")
    print(f"  Test Accuracy: {test_accs[-1]:.2f}%")


if __name__ == "__main__":
    main()
