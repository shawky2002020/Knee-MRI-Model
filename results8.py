from enum import Enum
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from modelBackbone import modelBackbone , TransferLearningModel
from dataloader import class_names
from typing import List
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from datetime import datetime
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import shutil

class Status(Enum):
    ACL = "ACL"
    ACL_MENISCUS = "ACL and meniscus"
    MENISCUS = "meniscus"
    NORMAL = "normal"

serial_counter = 0

# Setup custom folder path (include full path to the directory)
image_folder_path = Path("F:\\Graduation_Project\\Sample\\Sara")

# Check if the folder exists
if image_folder_path.is_dir():
    print(f"{image_folder_path} exists, ready for use.")
else:
    print(f"{image_folder_path} does not exist. Please check the folder path.")

# Create transform pipeline to resize image
custom_image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
        transforms.ToTensor()
])
# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ACL Models
acl_models = {
    "sagittal": TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device),
    "coronal": TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device),
    "axial": TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device)
}

# Define Meniscus Models
meniscus_models = {
    "sagittal": TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device),
    "coronal": TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device),
    "axial": TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device)
}

# Load state dictionaries
for view in ["sagittal", "coronal", "axial"]:
    acl_models[view].load_state_dict(torch.load(f"acl_{view}.pth", map_location=device))
    meniscus_models[view].load_state_dict(torch.load(f"meniscus_{view}.pth", map_location=device))

# Set all models to evaluation mode
for model in acl_models.values():
    model.eval()
for model in meniscus_models.values():
    model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# Create the timestamped output directory and CSV data container
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_folder = f"output_results_{timestamp}"
os.makedirs(output_folder, exist_ok=True)

csv_data = []

def generate_heatmap(image_tensor, model):
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

# logistic regression coefficients ( tune or train )
# Format: [axial_weight, coronal_weight, sagittal_weight]
logreg_acl_weights = np.array([0.3, 0.3, 0.5])
logreg_meniscus_weights = np.array([0.3, 0.5, 0.3])
acl_intercept = -0.5
meniscus_intercept = -0.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def aggregate_logistic(probabilities, weights, intercept):
    weighted_sum = np.dot(probabilities, weights) + intercept
    return sigmoid(weighted_sum)

def classify_knee_condition_multiview(image_path_dict, save_path=None, row=None, col=None):
    acl_probs = []
    meniscus_probs = []
    view_acl_probs = {}
    view_meniscus_probs = {}
    images = {}
    image_tensors = {}


    # Load images and run through model per view
    for view_type, path in image_path_dict.items():
        image = Image.open(path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        images[view_type] = np.array(image)
        image_tensors[view_type] = image_tensor

        # Get probabilities
        acl_output = acl_models[view_type](image_tensor)
        meniscus_output = meniscus_models[view_type](image_tensor)

        acl_prob = torch.sigmoid(acl_output).item()
        meniscus_prob = torch.sigmoid(meniscus_output).item()

        acl_probs.append(acl_prob)
        meniscus_probs.append(meniscus_prob)

        view_acl_probs[view_type] = 1 - acl_prob  # Prob of being ACL tear
        view_meniscus_probs[view_type] = 1 - meniscus_prob  # Prob of being meniscus tear


    # Combine using logistic regression
    final_acl_prob = aggregate_logistic(np.array(acl_probs), logreg_acl_weights, acl_intercept)
    final_meniscus_prob = aggregate_logistic(np.array(meniscus_probs), logreg_meniscus_weights, meniscus_intercept)

    # Apply decision thresholds
    acl_result = 1 if final_acl_prob < 0.51 else 0
    meniscus_result = 1 if final_meniscus_prob < 0.51 else 0

    # Final diagnosis
    if acl_result and meniscus_result:
        status = Status.ACL_MENISCUS
        title = f"Diagnosis: {status.value} | ACL: {1-final_acl_prob:.2f}, Meniscus: {1-final_meniscus_prob:.2f}"
    elif acl_result:
        status = Status.ACL
        title = f"Diagnosis: {status.value} | ACL: {1-final_acl_prob:.2f}"
    elif meniscus_result:
        status = Status.MENISCUS
        title = f"Diagnosis: {status.value} | Meniscus: {1-final_meniscus_prob:.2f}"
    else:
        status = Status.NORMAL
        title = f"Diagnosis: {status.value} | Normal probability (ACL model: {final_acl_prob:.2f}, Meniscus model: {final_meniscus_prob:.2f})"

    print(f"Final ACL Prob: {1-final_acl_prob:.2f}, Meniscus Prob: {1-final_meniscus_prob:.2f}, Diagnosis: {status.value}")

    # Select the view with the most confident abnormality
    selected_view = None
    if status == Status.ACL:
        selected_view = max(view_acl_probs, key=view_acl_probs.get)
    elif status == Status.MENISCUS:
        selected_view = max(view_meniscus_probs, key=view_meniscus_probs.get)
    elif status == Status.ACL_MENISCUS:
        # Take the view with the highest of both conditions
        combined_scores = {
            v: max(view_acl_probs[v], view_meniscus_probs[v])
            for v in image_path_dict
        }
        selected_view = max(combined_scores, key=combined_scores.get)
    elif status == Status.NORMAL:
        # Default to any view, e.g., sagittal
        selected_view = "sagittal"

    # Original image
    img = np.array(image)

    # Prepare heatmaps only if needed
    heatmap_acl = heatmap_meniscus = overlay_acl = overlay_men = None             
    # Generate heatmaps only for selected view
    img = images[selected_view]
    image_tensor = image_tensors[selected_view]
    overlays = []

    if status in [Status.ACL, Status.ACL_MENISCUS]:
        heatmap_acl = generate_heatmap(image_tensor, acl_models[selected_view])
        heatmap_acl = cv2.resize(heatmap_acl, (img.shape[1], img.shape[0]))
        heatmap_acl_colored = cv2.applyColorMap(np.uint8(255 * heatmap_acl), cv2.COLORMAP_JET)
        overlay_acl = np.uint8(heatmap_acl_colored * 0.4 + img * 0.6)
        overlays.append(("ACL Heatmap", overlay_acl))

    if status in [Status.MENISCUS, Status.ACL_MENISCUS]:
        heatmap_meniscus = generate_heatmap(image_tensor, meniscus_models[selected_view])
        heatmap_meniscus = cv2.resize(heatmap_meniscus, (img.shape[1], img.shape[0]))
        heatmap_men_colored = cv2.applyColorMap(np.uint8(255 * heatmap_meniscus), cv2.COLORMAP_JET)
        overlay_men = np.uint8(heatmap_men_colored * 0.4 + img * 0.6)
        overlays.append(("Meniscus Heatmap", overlay_men))

    # Plotting
    fig_cols = 1 + len(overlays)
    fig, axs = plt.subplots(1, fig_cols, figsize=(4 * fig_cols, 4))
    fig.suptitle(f"{selected_view.upper()} View\n{title}", fontsize=12, fontweight='bold', wrap=True)

    if fig_cols == 1:
        axs.imshow(img)
        axs.set_title("Original")
        axs.axis("off")
    else:
        axs[0].imshow(img)
        axs[0].set_title("Original")
        axs[0].axis("off")
        for i, (label, heatmap) in enumerate(overlays):
            axs[i + 1].imshow(heatmap)
            axs[i + 1].set_title(label)
            axs[i + 1].axis("off")

    plt.tight_layout()
    
    if save_path is not None:
        global serial_counter
        base_name = Path(image_path_dict[selected_view]).stem
        filename = f"{serial_counter:03}_{selected_view}_{base_name}_{status.value.replace(' ', '_')}.png"
        full_save_path = os.path.join(save_path, filename)
        plt.savefig(full_save_path)
        print(f"Saved result to: {full_save_path}")

        csv_data.append({
            "Filename": filename,
            "View": selected_view,
            "Row": row,
            "Column": col,
            "Diagnosis": status.value,
            "ACL_Probability": round(1-final_acl_prob, 4),
            "Meniscus_Probability": round(1-final_meniscus_prob, 4)
        })
        serial_counter += 1

    plt.show()
                                                                                                                                                
    # Optionally, return or save details
    return {"final_acl_prob": 1-final_acl_prob, "final_meniscus_prob": 1-final_meniscus_prob,"Diagnosis": status.value,"View": selected_view,"ImageTensor": image_tensor,"Image": img,"Type": status
    }

# At the end of your full run, save the CSV
def save_diagnosis_csv(csv_path):
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    print(f"CSV report saved to: {csv_path}")

axial_files = sorted(Path("F:\\Graduation_Project\\Sample\\try\\axial2").glob("*.png"))
coronal_files = sorted(Path("F:\\Graduation_Project\\Sample\\try\\coronal2").glob("*.png"))
sagittal_files = sorted(Path("F:\\Graduation_Project\\Sample\\try\\sagittal2").glob("*.png"))

# Assuming the same filenames exist in all views
num_images = min(len(axial_files), len(coronal_files), len(sagittal_files))
# At the end of your full run, add this to find and display the most abnormal sample.
max_prob = -1
most_abnormal_sample = None
most_abnormal_status = None
most_abnormal_view = None
most_abnormal_image_tensor = None
most_abnormal_image = None
most_abnormal_type = None

# Loop through all images to identify the one with the highest probability
for idx in range(num_images):
    image_path_dict = {
        "axial": axial_files[idx],
        "coronal": coronal_files[idx],
        "sagittal": sagittal_files[idx]
    }

    result = classify_knee_condition_multiview(image_path_dict, save_path=output_folder, row=None, col=None)
    prob = max(result["final_acl_prob"], result["final_meniscus_prob"])

    # Keep track of the image with the highest probability
    if prob > max_prob:
        max_prob = prob
        most_abnormal_sample = image_path_dict
        most_abnormal_status = result["Diagnosis"]
        most_abnormal_view = result["View"]
        most_abnormal_image_tensor = result["ImageTensor"]
        most_abnormal_image = result["Image"]
        most_abnormal_type = result["Type"]

# Now, you can save the image with the highest probability along with its heatmap(s)
print(f"Most abnormal sample found: {most_abnormal_sample}")
print(f"Diagnosis: {most_abnormal_status} with Probability: {max_prob}")


if most_abnormal_sample:
    print("\n\n=== Most Confident Abnormality Case ===")
    
    # Run the original function (saves image normally)
    classify_knee_condition_multiview(
        image_path_dict = most_abnormal_sample,
        save_path=output_folder,
        row=None,
        col=None
    )
    
# Save CSV report
save_diagnosis_csv(os.path.join(output_folder, "diagnosis_report.csv"))