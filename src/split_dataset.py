from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_PATH = Path(__file__).resolve().parent.parent
REPORTS_PATH = PROJECT_PATH / "reports"

MANIFEST_PATH = REPORTS_PATH / "dataset_manifest_clean.csv"
RANDOM_STATE = 42

def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)

    required = {"image_path", "class_name"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain: {required}")

    if df.empty or df["class_name"].nunique() != 12:
        raise ValueError("Expected a non-empty dataset with 12 classes.")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["class_name"],
        random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["class_name"],
        random_state=RANDOM_STATE
    )

    splits = {
        "train": train_df,
        "validation": val_df,
        "test": test_df
    }

    for name, data in splits.items():
        data = data.sort_values(
            ["class_name", "image_path"]
        ).reset_index(drop=True)

        data.to_csv(
            REPORTS_PATH / f"{name}_manifest.csv",
            index=False
        )

        print(f"\n{name.upper()}: {len(data)} images")
        print(data["class_name"].value_counts().sort_index().to_string())

    print("\nDataset splitting completed.")
    print(f"Total images: {sum(len(data) for data in splits.values())}")
    print(f"Reports saved in: {REPORTS_PATH}")


if __name__ == "__main__":
    main()