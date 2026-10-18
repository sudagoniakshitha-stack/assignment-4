from pathlib import Path
import json
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from prepare_images import load_dataset

PROJECT_PATH = Path(__file__).resolve().parent.parent
REPORTS_PATH = PROJECT_PATH / "reports"
MODELS_PATH = PROJECT_PATH / "models"
IMAGE_SIZE = (224, 224)
SEED = 42
BATCH_SIZE = 32
EPOCHS = 10

tf.keras.utils.set_random_seed(SEED)


def main():
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    train_ds, classes, train_count = load_dataset(
        "train_manifest.csv", shuffle=True
    )
    val_ds, _, val_count = load_dataset("validation_manifest.csv")

    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1)
    ])

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(classes), activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODELS_PATH / "mobilenetv2_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(
            REPORTS_PATH / "mobilenetv2_training.csv"
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(MODELS_PATH / "mobilenetv2_final.keras")

    result = {
        "model": "MobileNetV2",
        "classes": classes,
        "training_images": train_count,
        "validation_images": val_count,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        )
    }

    (REPORTS_PATH / "mobilenetv2_training_summary.json").write_text(
        json.dumps(result, indent=4),
        encoding="utf-8"
    )

    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
    