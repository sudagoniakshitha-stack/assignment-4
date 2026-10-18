import json
from pathlib import Path

import tensorflow as tf
import pandas as pd

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_CSV = REPORTS_DIR / "train_manifest.csv"
VAL_CSV = REPORTS_DIR / "validation_manifest.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("Loading dataset manifests...")


def load_manifest(csv_path):
    df = pd.read_csv(csv_path)

    required_columns = {"image_path", "class_name"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{csv_path.name} must contain columns "
            f"{required_columns}. Found: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            "image_path": "filepath",
            "class_name": "label",
        }
    )

    df["filepath"] = df["filepath"].astype(str)
    df["label"] = df["label"].astype(str)

    missing = df.loc[
        ~df["filepath"].map(lambda p: Path(p).is_file())
    ]

    if not missing.empty:
        raise FileNotFoundError(
            f"{len(missing)} image paths are missing in {csv_path.name}. "
            f"Example: {missing.iloc[0]['filepath']}"
        )

    return df


train_df = load_manifest(TRAIN_CSV)
val_df = load_manifest(VAL_CSV)

class_names = sorted(train_df["label"].unique())
num_classes = len(class_names)
class_to_index = {name: i for i, name in enumerate(class_names)}

if set(val_df["label"].unique()) != set(class_names):
    raise ValueError("Training and validation classes do not match.")

print("Classes:", class_names)
print("Training images:", len(train_df))
print("Validation images:", len(val_df))


def make_dataset(df, training=False):
    paths = df["filepath"].to_numpy()
    labels = df["label"].map(class_to_index).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 5000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)

        # EfficientNetB0 includes input rescaling internally.
        image = tf.cast(image, tf.float32)

        label = tf.one_hot(label, num_classes)

        return image, label

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)

print("Building EfficientNetB0 model...")

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax",
    name="predictions",
)(x)

model = models.Model(
    inputs,
    outputs,
    name="EfficientNetB0_Garbage",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

best_model_path = MODELS_DIR / "efficientnetb0_best.keras"
final_model_path = MODELS_DIR / "efficientnetb0_final.keras"
history_path = REPORTS_DIR / "efficientnetb0_training_history.csv"
summary_path = REPORTS_DIR / "efficientnetb0_training_summary.json"

callbacks = [
    ModelCheckpoint(
        str(best_model_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    CSVLogger(str(history_path)),
]

print("Starting EfficientNetB0 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save(final_model_path)

best_epoch = int(
    max(
        range(len(history.history["val_accuracy"])),
        key=lambda i: history.history["val_accuracy"][i],
    )
) + 1

summary = {
    "model": "EfficientNetB0",
    "classes": class_names,
    "training_images": len(train_df),
    "validation_images": len(val_df),
    "epochs_completed": len(history.history["loss"]),
    "best_epoch": best_epoch,
    "best_validation_accuracy": float(
        max(history.history["val_accuracy"])
    ),
    "best_validation_loss": float(
        min(history.history["val_loss"])
    ),
}

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\nEfficientNetB0 training completed.")
print(
    f"Best validation accuracy: "
    f"{summary['best_validation_accuracy']:.4f}"
)
print(f"Best epoch: {best_epoch}")
print("Best model saved to:", best_model_path)
print("Training history saved to:", history_path)