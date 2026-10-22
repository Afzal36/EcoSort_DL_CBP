
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = Path("trashnet-dataset-main")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "waste_classifier.keras"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    expected_classes = [
        "cardboard", "glass", "metal",
        "paper", "plastic", "trash"
    ]

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR.resolve()}"
        )

    paths, labels = [], []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_name in expected_classes:
        folder = DATA_DIR / class_name

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing category folder: {folder}"
            )

        images = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]

        print(f"{class_name}: {len(images)} images")

        for image_path in images:
            paths.append(str(image_path.resolve()))
            labels.append(class_name)

    if len(paths) == 0:
        raise ValueError("No images found in dataset-resized.")

    if min(labels.count(c) for c in expected_classes) < 3:
        raise ValueError(
            "Each category needs at least 3 images for splitting."
        )

    print(f"\nTotal images: {len(paths)}")

    # Split: 70% train, 15% validation, 15% test.
    train_paths, remaining_paths, train_labels, remaining_labels = (
        train_test_split(
            paths,
            labels,
            test_size=0.30,
            random_state=SEED,
            stratify=labels,
        )
    )

    val_paths, test_paths, val_labels, test_labels = (
        train_test_split(
            remaining_paths,
            remaining_labels,
            test_size=0.50,
            random_state=SEED,
            stratify=remaining_labels,
        )
    )

    print(
        f"Train: {len(train_paths)} | "
        f"Validation: {len(val_paths)} | "
        f"Test: {len(test_paths)}"
    )

    # MobileNetV2 preprocessing scales images to [-1, 1].
    train_augment = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.85, 1.15),
        fill_mode="nearest",
    )

    normal_preprocess = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    import pandas as pd

    def make_generator(image_paths, image_labels, augment=False):
        frame = pd.DataFrame({
            "filepath": image_paths,
            "label": image_labels,
        })

        generator = train_augment if augment else normal_preprocess

        return generator.flow_from_dataframe(
            dataframe=frame,
            x_col="filepath",
            y_col="label",
            classes=expected_classes,
            class_mode="categorical",
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=augment,
            seed=SEED,
            validate_filenames=True,
        )

    train_gen = make_generator(train_paths, train_labels, True)
    val_gen = make_generator(val_paths, val_labels)
    test_gen = make_generator(test_paths, test_labels)

    # Balance classes based on training data only.
    train_ids = np.array([
        expected_classes.index(label) for label in train_labels
    ])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(expected_classes)),
        y=train_ids,
    )

    class_weights = {
        index: float(weight)
        for index, weight in enumerate(weights)
    }

    print("\nClass weights:")
    for index, class_name in enumerate(expected_classes):
        print(f"{class_name}: {class_weights[index]:.3f}")

    # Load the pretrained feature extractor.
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(6, activation="softmax")(x)

    model = Model(inputs, outputs)

    MODEL_DIR.mkdir(exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print("\nPHASE 1: Train classification head")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=12,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print("\nPHASE 2: Fine-tune final MobileNetV2 layers")

    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Keep batch-normalization layers frozen for stability.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Evaluate the best checkpoint on the test set.
    model = tf.keras.models.load_model(MODEL_PATH)
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)

    test_gen.reset()
    probabilities = model.predict(test_gen, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    actual = test_gen.classes

    print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")
    print("\nClassification report:")
    report_text = classification_report(
        actual,
        predictions,
        labels=list(range(6)),
        target_names=expected_classes,
        zero_division=0,
    )
    print(report_text)

    print("Confusion matrix:")
    print(confusion_matrix(actual, predictions, labels=list(range(6))))

    with open(MODEL_DIR / "class_names.json", "w") as file:
        json.dump(expected_classes, file, indent=2)

    with open(MODEL_DIR / "test_metrics.json", "w") as file:
        json.dump({
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "classification_report": report_text,
        }, file, indent=2)

    print(f"\nBest model saved to: {MODEL_PATH.resolve()}")
    print("Training complete!")


if __name__ == "__main__":
    main()
