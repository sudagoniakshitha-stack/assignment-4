from pathlib import Path
import json
from PIL import Image

DATASET_PATH = Path(
    r"C:\Users\Sudagoni Akshitha\OneDrive\Desktop\Garbage Image classification\garbage_classification"
)

PROJECT_PATH = Path(__file__).resolve().parent.parent
REPORTS_PATH = PROJECT_PATH / "reports"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def check_image_quality():
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        return

    class_folders = sorted(
        folder for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
    )

    total_checked = 0
    valid_images = 0
    invalid_images = []
    class_results = {}

    print("Checking image quality...")
    print("Please wait until the scan finishes.\n")

    for folder in class_folders:
        class_checked = 0
        class_valid = 0
        class_invalid = 0

        for file_path in folder.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            total_checked += 1
            class_checked += 1

            try:
                with Image.open(file_path) as image:
                    image.verify()

                # Reopen the image and force complete decoding.
                with Image.open(file_path) as image:
                    image.load()

                valid_images += 1
                class_valid += 1

            except Exception as error:
                class_invalid += 1

                invalid_images.append({
                    "path": str(file_path),
                    "error": str(error)
                })

        class_results[folder.name] = {
            "checked": class_checked,
            "valid": class_valid,
            "invalid": class_invalid
        }

        print(
            f"{folder.name:<20} "
            f"Checked: {class_checked:>5} | "
            f"Valid: {class_valid:>5} | "
            f"Invalid: {class_invalid:>5}"
        )

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset_path": str(DATASET_PATH),
        "total_checked": total_checked,
        "valid_images": valid_images,
        "invalid_images_count": len(invalid_images),
        "class_results": class_results,
        "invalid_images": invalid_images
    }

    report_path = REPORTS_PATH / "image_quality_report.json"

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 55)
    print(f"Total checked: {total_checked}")
    print(f"Valid images: {valid_images}")
    print(f"Invalid images: {len(invalid_images)}")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    check_image_quality()