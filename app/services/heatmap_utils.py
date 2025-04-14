import torch
import cv2
import numpy as np
import torch.nn.functional as F
from typing import Tuple

def generate_heatmap(image_tensor, model) -> np.ndarray:
    """Generate a heatmap from the model's activations and gradients."""
    output = model(image_tensor)
    model.zero_grad()
    output[0].backward()

    gradients = model.get_activations_gradient()
    activations = model.get_activations()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    return heatmap.cpu().detach().numpy()

def create_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Create an overlay of the heatmap on the original image."""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = np.uint8(heatmap_colored * alpha + image * (1 - alpha))
    return overlay
