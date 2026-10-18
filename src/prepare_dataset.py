from pathlib import Path
from collections import defaultdict
import hashlib
import json
import pandas as pd

DATASET_PATH = Path(
    r"C:\Users\Sudagoni Akshitha\OneDrive\Desktop\Garbage Image classification\garbage_classification"
)

PROJECT_PATH = Path(__file__).resolve().parent.parent
REPORTS_PATH = PROJECT_PATH / "reports"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_hash(path):
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    classes = sorted(
        folder for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
    )

    if len(classes) != 12:
        raise ValueError(f"Expected 12 classes, found {len(classes)}")

    groups = defaultdict(list)

    for folder in classes:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                groups[get_hash(path)].append({
                    "image_path": str(path),
                    "class_name": folder.name,
                    "sha256": get_hash(path)
                })

    retained = []
    excluded = []

    for group in groups.values():
        labels = {item["class_name"] for item in group}

        if len(labels) > 1:
            for item in group:
                excluded.append({
                    **item,
                    "reason": "Duplicate across different classes"
                })
        else:
            retained.append(group[0])

            for item in group[1:]:
                excluded.append({
                    **item,
                    "reason": "Duplicate within the same class"
                })

    retained.sort(key=lambda x: (x["class_name"], x["image_path"]))

    df = pd.DataFrame(retained)
    excluded_df = pd.DataFrame(
        excluded,
        columns=["image_path", "class_name", "sha256", "reason"]
    )

    if df["class_name"].nunique() != 12:
        raise ValueError("At least one class has no retained images.")

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    df.to_csv(REPORTS_PATH / "dataset_manifest_clean.csv", index=False)
    excluded_df.to_csv(REPORTS_PATH / "excluded_duplicates.csv", index=False)

    summary = {
        "original_images": sum(len(g) for g in groups.values()),
        "unique_image_contents": len(groups),
        "retained_images": len(df),
        "excluded_images": len(excluded_df),
        "number_of_classes": int(df["class_name"].nunique()),
        "class_counts": {
            str(k): int(v)
            for k, v in df["class_name"].value_counts().sort_index().items()
        }
    }

    (REPORTS_PATH / "dataset_cleaning_summary.json").write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    print("Dataset preparation completed.")
    print(json.dumps(summary, indent=4))
    print(f"\nReports saved in: {REPORTS_PATH}")


if __name__ == "__main__":
    main()