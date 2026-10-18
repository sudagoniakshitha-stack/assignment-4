from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

PROJECT_PATH = Path(__file__).resolve().parent.parent
REPORTS_PATH = PROJECT_PATH / "reports"
MODELS_PATH = PROJECT_PATH / "models"

MODEL_FILES = {
    "Baseline CNN": "baseline_cnn_best.keras",
    "MobileNetV2": "mobilenetv2_best.keras",
    "ResNet50": "resnet50_best.keras",
    "EfficientNetB0": "efficientnetb0_best.keras",
}


def main():
    from prepare_images import load_dataset

    test_dataset, class_names, test_count = load_dataset(
        "test_manifest.csv",
        shuffle=False,
    )

    results = []

    print(f"Test images: {test_count}")
    print(f"Classes: {len(class_names)}")

    for model_name, model_file in MODEL_FILES.items():
        model_path = MODELS_PATH / model_file

        if not model_path.exists():
            print(f"\nSkipping {model_name}: model file not found.")
            continue

        print(f"\nEvaluating {model_name}...")

        model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

        y_true = []
        for _, labels in test_dataset:
            y_true.extend(labels.numpy().tolist())

        y_true = np.array(y_true)

        probabilities = model.predict(test_dataset, verbose=1)
        y_pred = np.argmax(probabilities, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        macro_f1 = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        report_path = REPORTS_PATH / (
            model_name.lower().replace(" ", "_") + "_evaluation.json"
        )

        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names,
        )

        fig, ax = plt.subplots(figsize=(12, 10))
        display.plot(
            ax=ax,
            xticks_rotation=90,
            colorbar=False,
        )
        ax.set_title(f"{model_name} - Confusion Matrix")
        fig.tight_layout()

        cm_path = REPORTS_PATH / (
            model_name.lower().replace(" ", "_")
            + "_confusion_matrix.png"
        )

        fig.savefig(cm_path, dpi=150)
        plt.close(fig)

        results.append({
            "Model": model_name,
            "Test Accuracy": accuracy,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
        })

        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Macro precision: {precision:.4f}")
        print(f"Macro recall: {recall:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")
        print(f"Weighted F1-score: {weighted_f1:.4f}")

        del model
        tf.keras.backend.clear_session()

    if not results:
        print("No models were evaluated.")
        return

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(
        by="Macro F1",
        ascending=False,
    )

    comparison_path = REPORTS_PATH / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n========== MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))
    print(f"\nComparison saved to: {comparison_path}")

    best_model = comparison.iloc[0]
    print(f"\nBest model based on Macro F1: {best_model['Model']}")
    print(f"Best Macro F1-score: {best_model['Macro F1']:.4f}")


if __name__ == "__main__":
    main()