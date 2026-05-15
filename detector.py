import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_model.pth")


def get_device() -> torch.device:
    """
    Selects GPU if available, otherwise falls back to CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms() -> transforms.Compose:
    """
    Image preprocessing pipeline for inference.

    - Resize image to 224x224
    - Convert to tensor
    - Normalize with ImageNet statistics (matches training)
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(
    model_path: str = MODEL_PATH,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load the trained ResNet18-based model for deepfake detection.
    """
    if device is None:
        device = get_device()

    # Create the same architecture used during training
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)  # 2 classes: Fake, Real

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_image(
    image: Image.Image,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> tuple[str, float]:
    """
    Predict whether an input image is "Real" or "Fake".

    Args:
        image: PIL Image to classify.
        model: Optional preloaded model. If None, it will be loaded from disk.
        device: Optional torch.device. If None, it will be auto-selected.

    Returns:
        A tuple of:
        - label: "Real" or "Fake"
        - confidence: confidence percentage (0–100)
    """
    if device is None:
        device = get_device()

    if model is None:
        model = load_model(device=device)

    transform = build_transforms()

    # Apply preprocessing transforms and add batch dimension
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Disable gradient computation for inference
    with torch.no_grad():
        outputs = model(input_tensor)

        # Convert logits to probabilities
        probs = torch.softmax(outputs, dim=1)

        # Get predicted class and its confidence
        conf, predicted = torch.max(probs, 1)

    # During training, ImageFolder will typically assign:
    #   fake -> 0, real -> 1   (alphabetical order of folder names)
    # Adjust the mapping here if your folder names differ.
    idx_to_label = {0: "Fake", 1: "Real"}

    label = idx_to_label.get(predicted.item(), "Unknown")
    confidence = float(conf.item() * 100.0)

    return label, confidence


def generate_gradcam(
    image: Image.Image,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Image.Image:

    if device is None:
        device = get_device()

    if model is None:
        model = load_model(device=device)

    model.eval()

    transform = build_transforms()
    input_tensor = transform(image).unsqueeze(0).to(device)

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    outputs = model(input_tensor)
    pred_class = outputs.argmax(dim=1)

    model.zero_grad()
    outputs[0, pred_class].backward()

    fh.remove()
    bh.remove()

    act = activations[0].detach()
    grad = gradients[0].detach()

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1)

    cam = torch.relu(cam)

    cam = cam.squeeze().cpu().numpy()

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Convert to image
    cam = np.uint8(cam * 255)

    cam_img = Image.fromarray(cam)

    orig = image.convert("RGB")
    width, height = orig.size

    cam_img = cam_img.resize((width, height), Image.BILINEAR)
    cam_arr = np.array(cam_img)

    heatmap = np.zeros((height, width, 3), dtype=np.uint8)
    heatmap[:, :, 0] = cam_arr
    heatmap[:, :, 1] = (cam_arr * 0.5).astype(np.uint8)

    heatmap_img = Image.fromarray(heatmap)

    overlay = Image.blend(orig, heatmap_img, alpha=0.4)

    return overlay
