from pathlib import Path
import pandas as pd
import tensorflow as tf

PROJECT_PATH = Path(__file__).resolve().parent.parent
REPORTS_PATH = PROJECT_PATH / "reports"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def load_dataset(manifest_name, shuffle=False):
    manifest_path = REPORTS_PATH / manifest_name

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing file: {manifest_path}")

    df = pd.read_csv(manifest_path)
    paths = df["image_path"].astype(str).to_numpy()
    labels = df["class_name"].astype(str).to_numpy()

    classes = sorted(
        pd.read_csv(REPORTS_PATH / "dataset_manifest_clean.csv")["class_name"].unique()
    )
    class_to_index = {name: i for i, name in enumerate(classes)}
    encoded_labels = [class_to_index[label] for label in labels]

    dataset = tf.data.Dataset.from_tensor_slices((paths, encoded_labels))

    if shuffle:
        dataset = dataset.shuffle(
            min(len(df), 5000),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    def process_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(
            image, channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        return image, label

    dataset = dataset.map(
        process_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset, classes, len(df)


def main():
    for filename in [
        "train_manifest.csv",
        "validation_manifest.csv",
        "test_manifest.csv"
    ]:
        dataset, classes, count = load_dataset(
            filename,
            shuffle=filename == "train_manifest.csv"
        )

        labels_df = pd.read_csv(REPORTS_PATH / filename)
        label_indices = labels_df["class_name"].map(
            {name: i for i, name in enumerate(classes)}
        )

        images, _ = next(iter(dataset))

        print(f"\n{filename}")
        print(f"Images: {count}")
        print(f"Classes: {len(classes)}")
        print(f"Batch shape: {images.shape}")
        print(f"Pixel range: {float(tf.reduce_min(images)):.1f} to "
              f"{float(tf.reduce_max(images)):.1f}")
        print(f"Label range: {label_indices.min()} to {label_indices.max()}")
        print(f"Unique labels: {label_indices.nunique()}")


if __name__ == "__main__":
    main()