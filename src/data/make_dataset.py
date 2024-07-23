from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import torchvision.datasets as datasets
import pathlib
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

import shutil
import random

app = typer.Typer()

def get_subset(image_path = RAW_DATA_DIR / "food-101" / "images",
               data_splits = ["train", "test"],
               target_classes = ["pizza", "hot_dog", "hamburger"],
               amount=0.1,
               seed=42,
               valid_ratio=0.4):
    random.seed(42)
    label_splits = {}

    # Get labels
    for data_split in data_splits:
        print(f"[INFO] Creating image split for: {data_split}...")
        label_path = RAW_DATA_DIR / "food-101" / "meta" / f"{data_split}.txt"
        with open(label_path, "r") as f:
            labels = [line.strip("\n") for line in f.readlines() if line.split("/")[0] in target_classes]

            # Get random subset of target classes image ID's
        number_to_sample = round(amount * len(labels))
        sampled_images = random.sample(labels, k=number_to_sample)

        # Apply full paths
        image_paths = [pathlib.Path(str(image_path / sample_image) + ".jpg") for sample_image in sampled_images]

        if data_split == "train":
            split_idx = int((1 - valid_ratio) * len(image_paths))
            random.shuffle(image_paths)
            label_splits["valid"] = image_paths[split_idx:]
            label_splits["train"] = image_paths[:split_idx]
    else:
        label_splits[data_split] = image_paths
    return label_splits

def download_food101_dataset(data_dir=RAW_DATA_DIR):
    logger.info(f"Downloading Food101 dataset to {data_dir}")
    train_data = datasets.Food101(root=data_dir, split="train", download=True)
    test_data = datasets.Food101(root=data_dir, split="test", download=True)
    logger.success(f"Finished downloading Food101 dataset to {data_dir}")

def get_hhp_subset(amount_to_get=0.1):
    logger.info(f"Getting Hamburger-HotDog-Pizza subset of {amount_to_get:.2f}...")
    label_splits = get_subset(amount=amount_to_get)

    target_dir_name =  PROCESSED_DATA_DIR / f"pizza_hamburger_hotdog_{str(int(amount_to_get * 100))}_percent"
    target_dir = pathlib.Path(target_dir_name)
    logger.info(f"Copy/Pasting Hamburger-HotDog-Pizza subset to {target_dir_name}")

    if not target_dir.exists():
        logger.info(f"Creating directory: '{target_dir_name}'")
        target_dir.mkdir(parents=True, exist_ok=True)

    for image_split in label_splits.keys():
        # Add tqdm for progress tracking on image copying
        for image_path in tqdm(label_splits[image_split], desc=f"Copying images in {image_split}", unit="file"):
            image_path = pathlib.Path(image_path)
            dest_dir = target_dir / image_split / image_path.parent.stem / image_path.name

            if not dest_dir.parent.is_dir():
                dest_dir.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"[INFO] Copying {image_path} to {dest_dir}...")
            shutil.copy2(image_path, dest_dir)

@app.command()
def main():
    download_food101_dataset()
    get_hhp_subset(amount_to_get=0.2)
    get_hhp_subset(amount_to_get=1)

if __name__ == "__main__":
    app()
