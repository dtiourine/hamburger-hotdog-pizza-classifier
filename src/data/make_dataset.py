from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import torchvision.datasets as datasets
import pathlib
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# def get_subset(image_path=data_path,
#                data_splits=["train", "test"],
#                target_classes=["pizza", "hot_dog", "hamburger"],
#                amount=0.1,
#                seed=42):
#     random.seed(42)
#     label_splits = {}
#
#     # Get labels
#     for data_split in data_splits:
#         print(f"[INFO] Creating image split for: {data_split}...")
#         label_path = data_dir / "food-101" / "meta" / f"{data_split}.txt"
#         with open(label_path, "r") as f:
#             labels = [line.strip("\n") for line in f.readlines() if line.split("/")[0] in target_classes]
#
#             # Get random subset of target classes image ID's
#         number_to_sample = round(amount * len(labels))
#         print(f"[INFO] Getting random subset of {number_to_sample} images for {data_split}...")
#         sampled_images = random.sample(labels, k=number_to_sample)
#
#         # Apply full paths
#         image_paths = [pathlib.Path(str(image_path / sample_image) + ".jpg") for sample_image in sampled_images]
#         label_splits[data_split] = image_paths
#     return label_splits
#
#
# label_splits = get_subset(amount=amount_to_get)
# label_splits["train"][:10]

def download_food101_dataset(data_dir=RAW_DATA_DIR):
    logger.info(f"Downloading Food101 dataset to {data_Dir}")
    train_data = datasets.Food101(root=data_dir, split="train", download=True)
    test_data = datasets.Food101(root=data_dir, split="test", download=True)
    logger.success(f"Finished downloading Food101 dataset to {data_dir}")

@app.command()
def main():

    logger.info("Downloading Food101 dataset to data/raw")
    #download_food101_dataset()



if __name__ == "__main__":
    app()
