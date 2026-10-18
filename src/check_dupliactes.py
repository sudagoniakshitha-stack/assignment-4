from pathlib import Path
from collections import defaultdict
import hashlib
import json

DATASET_PATH = Path(
    r"C:\Users\Sudagoni Akshitha\OneDrive\Desktop\Garbage Image classification\garbage_classification"
)

PROJECT_PATH = Path(__file__).resolve().parent.parent
REPORTS_PATH = PROJECT_PATH / "reports"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def calculate_hash(file_path):
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def check_duplicates():
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        return

    hashes = defaultdict(list)
    total_images = 0

    print("Checking for exact duplicate images...")
    print("Please wait until the scan finishes.\n")

    for folder in sorted(DATASET_PATH.iterdir()):
        if not folder.is_dir():
            continue

        for file_path in folder.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                total_images += 1
                file_hash = calculate_hash(file_path)
                hashes[file_hash].append(str(file_path))

    duplicate_groups = [
        paths for paths in hashes.values()
        if len(paths) > 1
    ]

    duplicate_file_count = sum(
        len(paths) for paths in duplicate_groups
    )

    results = {
        "total_images": total_images,
        "unique_image_contents": len(hashes),
        "duplicate_groups": len(duplicate_groups),
        "files_in_duplicate_groups": duplicate_file_count,
        "duplicates": duplicate_groups
    }

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_PATH / "duplicate_report.json"

    report_path.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8"
    )

    print("=" * 50)
    print(f"Total images: {total_images}")
    print(f"Unique image contents: {len(hashes)}")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Files in duplicate groups: {duplicate_file_count}")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    check_duplicates()