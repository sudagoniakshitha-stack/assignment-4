from pathlib import Path
from collections import Counter
import random
from PIL import Image

import matplotlib.pyplot as plt
import pandas as pd


DATASET_PATH = Path(
    r"C:\Users\Sudagoni Akshitha\OneDrive\Desktop\Garbage Image classification\garbage_classification"
)

PROJECT_PATH = Path(__file__).resolve().parent.parent
REPORTS_PATH = PROJECT_PATH / "reports"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def build_dataset_table():
    records = []

    for class_folder in sorted(DATASET_PATH.iterdir()):
        if not class_folder.is_dir():
            continue

        for image_path in class_folder.rglob("*"):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                records.append({
                    "image_path": str(image_path),
                    "class_name": class_folder.name,
                    "extension": image_path.suffix.lower()
                })

    return pd.DataFrame(records)

def plot_sample_images(df):
    classes = sorted(df["class_name"].unique())

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    for i, class_name in enumerate(classes):
        class_images = df[df["class_name"] == class_name]

        image_path = random.choice(
            class_images["image_path"].tolist()
        )

        with Image.open(image_path) as img:
            axes[i].imshow(img.convert("RGB"))

        axes[i].set_title(class_name)
        axes[i].axis("off")

    plt.suptitle("Sample Images from 12 Garbage Categories")
    plt.tight_layout()

    output_path = REPORTS_PATH / "sample_images.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Sample images saved to: {output_path}")

def analyze_image_dimensions(df):
    from PIL import Image

    dimensions = []

    for image_path in df["image_path"]:
        try:
            with Image.open(image_path) as img:
                width, height = img.size

            dimensions.append({
                "image_path": image_path,
                "width": width,
                "height": height
            })

        except Exception as e:
            print(f"Could not read {image_path}: {e}")

    dimensions_df = pd.DataFrame(dimensions)

    dimensions_df.to_csv(
        REPORTS_PATH / "image_dimensions.csv",
        index=False
    )

    print("\nIMAGE DIMENSIONS SUMMARY")
    print("=" * 40)
    print(dimensions_df[["width", "height"]].describe())

    plt.figure(figsize=(8, 6))
    plt.scatter(
        dimensions_df["width"],
        dimensions_df["height"],
        alpha=0.4
    )
    plt.title("Original Image Dimensions")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.tight_layout()

    output_path = REPORTS_PATH / "image_dimensions.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Dimensions data saved to: {REPORTS_PATH / 'image_dimensions.csv'}")
    print(f"Dimensions plot saved to: {output_path}")

def analyze_color_distribution(df, sample_size=100):
    from PIL import Image
    import numpy as np

    sample_df = df.sample(
        n=min(sample_size, len(df)),
        random_state=42
    )

    channel_values = {
        "Red": [],
        "Green": [],
        "Blue": []
    }

    for image_path in sample_df["image_path"]:
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((100, 100))

        pixels = np.asarray(img)

        for i, channel in enumerate(["Red", "Green", "Blue"]):
            channel_values[channel].extend(
                pixels[:, :, i].ravel()
            )

    plt.figure(figsize=(10, 6))

    for channel, values in channel_values.items():
        plt.hist(
            values,
            bins=50,
            alpha=0.5,
            label=channel
        )

    plt.title("RGB Pixel Intensity Distribution")
    plt.xlabel("Pixel Intensity (0–255)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    output_path = REPORTS_PATH / "rgb_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"RGB distribution saved to: {output_path}")

def analyze_class_imbalance(df):
    counts = df["class_name"].value_counts().sort_values(
        ascending=False
    )

    total = counts.sum()
    percentages = (counts / total * 100).round(2)

    imbalance_df = pd.DataFrame({
        "class_name": counts.index,
        "image_count": counts.values,
        "percentage": percentages.values
    })

    imbalance_df.to_csv(
        REPORTS_PATH / "class_imbalance.csv",
        index=False
    )

    print("\nCLASS IMBALANCE ANALYSIS")
    print("=" * 45)
    print(imbalance_df.to_string(index=False))

    imbalance_ratio = counts.max() / counts.min()

    print(f"\nLargest class: {counts.idxmax()} ({counts.max()} images)")
    print(f"Smallest class: {counts.idxmin()} ({counts.min()} images)")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

    plt.figure(figsize=(12, 6))

    counts.sort_values().plot(kind="barh")

    plt.title("Class Imbalance Analysis")
    plt.xlabel("Number of Images")
    plt.ylabel("Garbage Category")
    plt.tight_layout()

    output_path = REPORTS_PATH / "class_imbalance.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nClass imbalance graph saved to: {output_path}")

def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_PATH}"
        )

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    df = build_dataset_table()
    plot_sample_images(df)
    analyze_image_dimensions(df)
    analyze_color_distribution(df)
    analyze_class_imbalance(df)

    if df.empty:
        raise ValueError("No supported image files were found.")

    counts = (
        df.groupby("class_name")
        .size()
        .sort_values(ascending=False)
    )

    print("\nDATASET SUMMARY")
    print("=" * 45)
    print(f"Total images: {len(df)}")
    print(f"Number of classes: {df['class_name'].nunique()}")

    print("\nIMAGES PER CLASS")
    print(counts.to_string())

    df.to_csv(
        REPORTS_PATH / "dataset_manifest_initial.csv",
        index=False
    )

    counts.rename("image_count").to_csv(
        REPORTS_PATH / "class_distribution.csv",
        header=True
    )

    plt.figure(figsize=(12, 6))
    counts.sort_values().plot(kind="barh")

    plt.title("Garbage Dataset: Images per Class")
    plt.xlabel("Number of Images")
    plt.ylabel("Class")

    plt.tight_layout()
    plt.savefig(
        REPORTS_PATH / "class_distribution.png",
        dpi=150
    )
    plt.close()

    print("\nReports saved:")
    print(REPORTS_PATH / "dataset_manifest_initial.csv")
    print(REPORTS_PATH / "class_distribution.csv")
    print(REPORTS_PATH / "class_distribution.png")


if __name__ == "__main__":
    main()