import json
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
)

from prepare_images import load_dataset


PROJECT_PATH = Path(__file__).resolve().parent.parent
REPORTS_PATH = PROJECT_PATH / "reports"
MODELS_PATH = PROJECT_PATH / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

tf.keras.utils.set_random_seed(SEED)


def build_cnn(num_classes):
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),

        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),

        layers.Dropout(0.4),

        layers.Dense(128, activation="relu"),

        layers.Dense(num_classes, activation="softmax"),
    ], name="Baseline_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    print("Loading training dataset...")

    train_ds, classes, train_count = load_dataset(
        "train_manifest.csv",
        shuffle=True,
    )

    val_ds, val_classes, val_count = load_dataset(
        "validation_manifest.csv",
    )

    if classes != val_classes:
        raise ValueError("Training and validation class order differs.")

    print("Classes:", classes)
    print("Training images:", train_count)
    print("Validation images:", val_count)

    model = build_cnn(len(classes))
    model.summary()

    best_model_path = MODELS_PATH / "baseline_cnn_best.keras"

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

        CSVLogger(
            str(REPORTS_PATH / "baseline_cnn_training.csv")
        ),
    ]

    print("\nStarting baseline CNN training...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODELS_PATH / "baseline_cnn_final.keras")

    best_epoch = int(
        max(
            range(len(history.history["val_accuracy"])),
            key=lambda i: history.history["val_accuracy"][i],
        )
    ) + 1

    summary = {
        "model": "Baseline CNN",
        "classes": classes,
        "training_images": int(train_count),
        "validation_images": int(val_count),
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "best_validation_loss": float(
            min(history.history["val_loss"])
        ),
    }

    with open(
        REPORTS_PATH / "baseline_cnn_training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nBaseline CNN training completed.")
    print(
        f"Best validation accuracy: "
        f"{summary['best_validation_accuracy']:.4f}"
    )
    print("Best epoch:", best_epoch)
    print("Best model saved to:", best_model_path)


if __name__ == "__main__":
    main()