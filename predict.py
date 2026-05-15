import os
import sys
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "deepfake_model.h5"
IMG_SIZE: Tuple[int, int] = (224, 224)


def load_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """Load the saved Keras model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'. Train first using train.py."
        )
    return tf.keras.models.load_model(model_path, compile=False)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL image to match training:
    - Resize to (224, 224)
    - Normalize by dividing by 255
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    arr = np.asarray(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)
    return arr


def predict_image(image: Image.Image, model: tf.keras.Model) -> Tuple[str, float]:
    """
    Predict "Real" or "Fake" from a PIL image.

    Returns:
        (label, prediction_value)
    """
    x = preprocess_image(image)
    pred = model.predict(x, verbose=0)
    pred_value = float(pred[0][0])

    # Requirement:
    # if prediction > 0.5 → Fake
    # else → Real
    label = "Fake" if pred_value > 0.5 else "Real"
    return label, pred_value


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        raise SystemExit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        raise SystemExit(1)

    model = load_model()

    try:
        image = Image.open(image_path)
    except Exception:
        print(f"Invalid image file: {image_path}")
        raise SystemExit(1)

    label, pred_value = predict_image(image, model=model)
    print(f"Prediction value: {pred_value:.6f}")
    print(f"Result: {label}")


if __name__ == "__main__":
    main()

