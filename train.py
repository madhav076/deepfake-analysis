import os
import shutil
from typing import Tuple

import tensorflow as tf
from PIL import Image, ImageFile
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "deepfake_model.h5"
IMG_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
EPOCHS = 5
SEED = 123


def check_dataset_dir(dataset_dir: str = DATASET_DIR) -> None:
    """
    Ensure dataset directory exists and contains the expected subfolders.

    Expected structure:
        dataset/
            real/
            fake/
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Missing dataset folder: '{dataset_dir}'. Create it with 'real/' and 'fake/' subfolders."
        )

    required = ["real", "fake"]
    missing = [d for d in required if not os.path.isdir(os.path.join(dataset_dir, d))]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset subfolder(s): {', '.join(missing)}. Expected: real/ and fake/."
        )


def _is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_validated_dataset(
    source_dir: str = DATASET_DIR,
) -> str:
    """
    Copy valid images into a validated directory.

    This prevents training from crashing due to corrupted/invalid image files.
    """
    # Use a stable directory name so it can be reused.
    # We recreate it every run so stale/corrupted files can't linger.
    validated_dir = os.path.join(source_dir, "validated_tf")
    if os.path.isdir(validated_dir):
        shutil.rmtree(validated_dir)
    os.makedirs(validated_dir, exist_ok=True)

    for class_name in ["real", "fake"]:
        os.makedirs(os.path.join(validated_dir, class_name), exist_ok=True)

        src_class_dir = os.path.join(source_dir, class_name)
        dst_class_dir = os.path.join(validated_dir, class_name)

        for filename in os.listdir(src_class_dir):
            if not _is_image_file(filename):
                continue

            src_path = os.path.join(src_class_dir, filename)
            dst_path = os.path.join(dst_class_dir, filename)

            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                with Image.open(src_path) as img:
                    img.verify()  # verifies file integrity without fully decoding
                shutil.copy2(src_path, dst_path)
            except Exception:
                # Skip corrupted/unreadable files.
                print(f"Skipping invalid image: {src_path}")

    return validated_dir


def build_model() -> tf.keras.Model:
    """
    Build a MobileNetV2 transfer-learning model for binary classification.

    - Uses MobileNetV2 pretrained on ImageNet.
    - Freezes base layers.
    - Adds:
        GlobalAveragePooling2D
        Dense(128, relu)
        Dense(1, sigmoid)
    """
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base.trainable = False  # Freeze base layers (transfer learning)

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=outputs, name="deepfake_mobilenetv2")
    model.compile(
        optimizer=Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


class PrintTrainLoss(tf.keras.callbacks.Callback):
    """Print training loss at the end of each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            # Requirement: print training loss for every epoch.
            print(f"Epoch {epoch + 1}/{EPOCHS} - Training loss: {loss:.6f}")


def main() -> None:
    check_dataset_dir(DATASET_DIR)

    dataset_for_training = build_validated_dataset(DATASET_DIR)

    # Data generator does:
    # - Resize to IMG_SIZE using target_size
    # - Normalize images by rescaling to [0, 1] with rescale=1./255
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VAL_SPLIT,
    )

    # IMPORTANT label mapping:
    # - The generator uses class_mode='binary' and class order to decide label values.
    # - Requirement says: prediction > 0.5 -> Fake
    # - So we make class 'fake' become label 1, and 'real' become label 0.
    #   That is achieved by ordering classes as ['real', 'fake'].
    train_gen = datagen.flow_from_directory(
        dataset_for_training,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=SEED,
        classes=["real", "fake"],
    )
    val_gen = datagen.flow_from_directory(
        dataset_for_training,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=SEED,
        classes=["real", "fake"],
    )

    model = build_model()

    print("Starting training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[PrintTrainLoss()],
        verbose=0,  # keep output clean; callback prints per-epoch loss
    )

    # Save the trained model as requested.
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()

