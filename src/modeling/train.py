from pathlib import Path

import typer
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights
from torchvision.datasets import ImageFolder
import torch.optim as optim

import src.config as config
# import wandb

app = typer.Typer()

@app.command()
def main(
    image_path: Path = PROCESSED_DATA_DIR / "pizza_hamburger_hotdog_20_percent",
    model_path: Path = MODELS_DIR / "vgg16.pth", # make sure this matches 'models/vgg16.pth',
    model_save_path: Path = MODELS_DIR,
    num_epochs: int = 20,
    batch_size: int = config.BATCH_SIZE,
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    #features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    #labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    #model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = VGG16_Weights.DEFAULT
    transform = weights.transforms()

    train_dir = image_path / 'train'
    valid_dir = image_path / 'valid'

    train_data = ImageFolder(train_dir, transform=transform)
    valid_data = ImageFolder(valid_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config.BATCH_SIZE, shuffle=True)

    model = torch.load(model_path)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.3)
    criterion = nn.CrossEntropyLoss()

    #wandb.watch(model, criterion, log="all", log_freq=10)

    best_val_accuracy = 0

    overall_progress_bar = tqdm(total=num_epochs, desc='Overall Training Progress', position=0, leave=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #wandb.log({"train_batch_loss": loss.item()})

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        #wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "epoch": epoch + 1})

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                #wandb.log({"val_batch_loss": loss.item()})

        val_loss = val_running_loss / len(valid_loader)
        val_accuracy = 100 * val_correct / val_total
        #wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "epoch": epoch + 1})

        # Save the model if the validation accuracy is the best we've seen so far.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_save_path = model_save_path / 'vgg16_new.pth'
            torch.save(model, model_save_path)

        overall_progress_bar.set_postfix({
            'Best Val Accuracy': f'{best_val_accuracy:.2f}%',
            'Current Train Accuracy': f'{train_accuracy:.2f}%',
            'Current Val Accuracy': f'{val_accuracy:.2f}%'
        })
        overall_progress_bar.update(1)

    overall_progress_bar.close()





    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # logger.info("Training some model...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
