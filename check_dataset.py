
from pathlib import Path

dataset_path = Path.home() / "Downloads" / "archive" / "dataset"

print("Dataset found:", dataset_path.exists())

for folder in ["TRAIN", "TEST"]:
    for label in ["O", "R"]:
        path = dataset_path / folder / label

        if path.exists():
            images = [
                file for file in path.iterdir()
                if file.suffix.lower() in
                [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
            ]

            print(f"{folder}/{label}: {len(images)} images")
        else:
            print(f"Folder not found: {path}")
