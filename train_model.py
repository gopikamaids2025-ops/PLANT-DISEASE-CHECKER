
"""Train a plant disease classifier using PlantVillage images.

Usage:
    python train_model.py --data_dir data/PlantVillage --max_per_class 300
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)
from sklearn.model_selection import train_test_split

from features import extract_features


IMAGE_TYPES = {".jpg", ".jpeg", ".png"}


def load_dataset(data_dir, max_per_class):

    data_path = Path(data_dir)

    if not data_path.exists():
        raise SystemExit(f"Dataset folder not found: {data_path}")

    # Automatically locate the color folder.
    if (data_path / "color").is_dir():
        data_path = data_path / "color"

    folders = sorted(
        p for p in data_path.iterdir() if p.is_dir()
    )

    if len(folders) < 2:
        raise SystemExit(
            f"Expected at least 2 class folders inside {data_path}"
        )

    print(f"\nDataset location: {data_path}")
    print(f"Number of classes found: {len(folders)}")

    X = []
    y = []
    class_names = []

    for folder in folders:

        files = sorted(
            f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_TYPES
        )[:max_per_class]

        print(f"Reading {len(files):4d} images from {folder.name}")

        successful = 0

        for file in files:
            try:
                features = extract_features(file)

                if features is None:
                    continue

                X.append(features)
                y.append(len(class_names))
                successful += 1

            except Exception as error:
                print(f"Skipping {file.name}: {error}")

        if successful > 0:
            class_names.append(folder.name)

    if len(X) == 0:
        raise SystemExit(
            "\nNo images were loaded. Check your image files "
            "and features.py."
        )

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise SystemExit(
            f"Invalid feature array shape: {X.shape}. "
            "Expected a 2D feature array."
        )

    print(f"\nTotal images loaded: {len(X)}")
    print(f"Feature count per image: {X.shape[1]}")

    return X, y, class_names


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="data/PlantVillage"
    )

    parser.add_argument(
        "--max_per_class",
        type=int,
        default=300
    )

    args = parser.parse_args()

    if args.max_per_class < 1:
        raise SystemExit("--max_per_class must be at least 1")

    # 1. Load images.
    X, y, class_names = load_dataset(
        args.data_dir,
        args.max_per_class
    )

    print("\nClasses:", class_names)

    # 2. Check class counts.
    counts = np.bincount(y, minlength=len(class_names))

    for name, count in zip(class_names, counts):
        print(f"{name}: {count} images")

    if len(class_names) < 2 or np.min(counts) < 2:
        raise SystemExit(
            "Each of at least two classes needs 2 or more images."
        )

    # 3. Split dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining images: {len(X_train)}")
    print(f"Testing images: {len(X_test)}")

    # 4. Train model.
    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 5. Evaluate model.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    baseline = (
        np.bincount(y_test).max() / len(y_test)
    )

    print("\n========== MODEL RESULTS ==========")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Baseline accuracy: {baseline:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0
        )
    )

    # 6. Save confusion matrix.
    Path("outputs").mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 16))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
        display_labels=class_names,
        xticks_rotation=90,
        cmap="Blues",
        ax=ax,
        colorbar=False
    )

    ax.set_title("Plant Disease Classification - Confusion Matrix")

    fig.tight_layout()

    fig.savefig(
        "outputs/confusion_matrix.png",
        dpi=150
    )

    plt.close(fig)

    print("\nSaved: outputs/confusion_matrix.png")

    # 7. Save trained model.
    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "class_names": class_names
        },
        "models/plant_model.joblib"
    )

    print("Saved: models/plant_model.joblib")

    print("\n========== TRAINING COMPLETED ==========")


if __name__ == "__main__":
    main()
