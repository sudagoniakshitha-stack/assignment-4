from pathlib import Path
import json

# Path to your existing dataset
DATASET_PATH = Path(
    r"C:\Users\Sudagoni Akshitha\OneDrive\Desktop\Garbage Image classification\garbage_classification"
)

# Project directory
PROJECT_PATH = Path(__file__).resolve().parent.parent
REPORTS_PATH = PROJECT_PATH / "reports"

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def inspect_dataset():
    if not DATASET_PATH.exists():
        print("ERROR: Dataset folder not found.")
        print(f"Checked location: {DATASET_PATH}")
        print("Please verify the folder path.")
        return

    print("Dataset folder found successfully!")
    print(f"Location: {DATASET_PATH}")

    # Identify class folders
    class_folders = sorted(
        folder for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
    )

    if not class_folders:
        print("\nNo category folders found directly inside the dataset.")
        print("We need to inspect the folder structure further.")
        return

    results = {}
    total_images = 0

    print("\nIMAGE COUNT BY CATEGORY")
    print("-" * 45)

    for folder in class_folders:
        image_count = sum(
            1
            for file in folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        results[folder.name] = image_count
        total_images += image_count

        print(f"{folder.name:<25} {image_count:>7}")

    print("-" * 45)
    print(f"{'Total images':<25} {total_images:>7}")
    print(f"Number of category folders: {len(class_folders)}")

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    report_file = REPORTS_PATH / "dataset_inspection.json"

    report = {
        "dataset_path": str(DATASET_PATH),
        "number_of_classes": len(class_folders),
        "total_images": total_images,
        "class_image_counts": results
    }

    report_file.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print(f"\nInspection report saved to:\n{report_file}")


if __name__ == "__main__":
    inspect_dataset()