from pathlib import Path
from PIL import Image, ImageDraw
import hashlib

DATASET_PATH = Path(
    r"C:\Users\Sudagoni Akshitha\OneDrive\Desktop\Garbage Image classification\garbage_classification"
)

PROJECT_PATH = Path(__file__).resolve().parent.parent
REPORTS_PATH = PROJECT_PATH / "reports"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def calculate_hash(file_path):
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def find_cross_class_duplicates():
    hash_groups = {}

    for class_folder in sorted(DATASET_PATH.iterdir()):
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        for image_path in class_folder.rglob("*"):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                image_hash = calculate_hash(image_path)

                hash_groups.setdefault(image_hash, []).append(
                    (class_name, image_path)
                )

    cross_class_groups = []

    for group in hash_groups.values():
        class_names = {class_name for class_name, _ in group}

        if len(class_names) > 1:
            cross_class_groups.append(group)

    if not cross_class_groups:
        print("No cross-class duplicate groups found.")
        return

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_PATH / "cross_class_duplicates"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Cross-class duplicate groups: {len(cross_class_groups)}")

    for group_number, group in enumerate(
        cross_class_groups, start=1
    ):
        print(f"\nDuplicate group {group_number}:")

        images = []

        for class_name, image_path in group:
            print(f"Class: {class_name}")
            print(f"File:  {image_path}")

            with Image.open(image_path) as image:
                images.append(
                    (class_name, image.copy().convert("RGB"))
                )

        tile_width = 300
        tile_height = 340

        comparison = Image.new(
            "RGB",
            (tile_width * len(images), tile_height),
            "white"
        )

        draw = ImageDraw.Draw(comparison)

        for index, (class_name, image) in enumerate(images):
            image.thumbnail((280, 280))

            x = index * tile_width + 10
            y = 40

            comparison.paste(image, (x, y))

            draw.text(
                (x, 10),
                class_name,
                fill="black"
            )

        comparison.save(
            output_path / f"group_{group_number}.jpg"
        )

    print(f"\nComparison images saved to: {output_path}")


if __name__ == "__main__":
    find_cross_class_duplicates()