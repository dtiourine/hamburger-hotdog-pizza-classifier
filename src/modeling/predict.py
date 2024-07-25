from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
import src.config as config
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights
from torchvision.datasets import ImageFolder

import pandas as pd

app = typer.Typer()



@app.command()
def main(
    image_path: Path = PROCESSED_DATA_DIR / "pizza_hamburger_hotdog_20_percent",
    model_path: Path = MODELS_DIR / "vgg16.pth",  # make sure this matches 'models/vgg16.pth',
    criterion: torch.nn.CrossEntropyLoss = nn.CrossEntropyLoss(),
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv", # Modify this later
):
    logger.info("Loading model from {}", model_path)
    model = torch.load(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    weights = VGG16_Weights.DEFAULT
    transform = weights.transforms()

    test_dir = image_path / 'test'
    test_data = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=True)

    predictions = []
    filenames = []

    with torch.no_grad():
        correct = 0
        total = 0

        for inputs, labels in tqdm(test_loader, desc="Processing Batches"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Collect predictions
            predictions.extend(predicted.cpu().numpy())
            filenames.extend([test_data.imgs[i][0] for i in range(len(labels))])

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f'Test Accuracy: {accuracy:.2f}%')

    # Save predictions to CSV
    df = pd.DataFrame({
        'filename': filenames,
        'predicted_label': predictions
    })
    df.to_csv(predictions_path, index=False)
    logger.info(f'Predictions saved to {predictions_path}')


if __name__ == "__main__":
    app()
