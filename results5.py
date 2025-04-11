from enum import Enum
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from app.modelBackbone import modelBackbone , TransferLearningModel
from dataloaderPngT import class_names
from typing import List
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from datetime import datetime
import os
import pandas as pd

class Status(Enum):
    ACL = "ACL"
    ACL_MENIISCUS = "ACL and meniscus"
    MENISCUS = "meniscus"
    NORMAL = "normal"

serial_counter = 0

# Setup custom folder path (include full path to the directory)
image_folder_path = Path("D:\\Knee_MRI\\sample")

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

def slice_image_by_edges(image_path, min_slice_dim=80, top_margin=50):
    """
    Slices the input MRI image into a grid and passes each slice individually for predictions.
    """
    # Load the image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    # Resize the image if necessary (optional step)
    max_width = 1000
    max_height = 1000
    height, width = image.shape[:2]

    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)  # Calculate scaling factor
        new_dim = (int(width * scale), int(height * scale))  # New dimensions after scaling
        image = cv2.resize(image, new_dim)  # Resize the image
    
    print(f"Loaded image: {image_path.name}")

    # Display the image for cropping
    roi = cv2.selectROI("Select Region to Crop", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:
        print("No region selected, proceeding with the full image.")
        roi = (0, 0, image.shape[1], image.shape[0])

    x, y, w, h = roi
    cropped_image = image[y:y+h, x:x+w]

    print(f"Cropped image size: {cropped_image.shape[:2]}")
    view_type = input("Enter view type (sagittal, coronal, axial): ").strip().lower()
    rows = int(input(f"Enter the number of rows for the cropped image (min {min_slice_dim}): "))
    cols = int(input(f"Enter the number of columns for the cropped image (min {min_slice_dim}): "))

    height, width, _ = cropped_image.shape
    slice_height = height // rows
    slice_width = width // cols

    # Extract slices and pass them individually for predictions
    for row in range(rows):
        for col in range(cols):
            y_start, y_end = row * slice_height, (row + 1) * slice_height
            x_start, x_end = col * slice_width, (col + 1) * slice_width

            slice_img = cropped_image[y_start:y_end, x_start:x_end]
            slice_img = cv2.resize(slice_img, (256, 256))  # Ensure consistent size
            
            # Save slice temporarily
            temp_slice_path = f"temp_slice_{row}_{col}.jpg"
            cv2.imwrite(temp_slice_path, slice_img)

            # Pass the cropped slice for prediction
            diagnosis = classify_knee_condition(image_path=temp_slice_path, view_type=view_type, save_path=output_folder, row=row, col=col)
            print(f"Slice [{row}, {col}]: {diagnosis}")

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

def classify_knee_condition(image_path, view_type, save_path=None, row=None, col=None):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get raw model outputs
    acl_output = acl_models[view_type](image_tensor)
    meniscus_output = meniscus_models[view_type](image_tensor)

    # Sigmoid probabilities
    acl_prob = torch.sigmoid(acl_output).item()
    meniscus_prob = torch.sigmoid(meniscus_output).item()

    # Binary classification based on threshold
    acl_result = 0 if acl_prob < 0.3 else 1
    meniscus_result = 0 if meniscus_prob < 0.3 else 1

    # Determine status
    if acl_result == 0 and meniscus_result == 0:
        status = Status.ACL_MENIISCUS
        title = f"Diagnosis: {status.value} | ACL: {1-acl_prob:.2f}, Meniscus: {1-meniscus_prob:.2f}"
    elif acl_result == 0:
        status = Status.ACL
        title = f"Diagnosis: {status.value} | ACL: {1-acl_prob:.2f}"
    elif meniscus_result == 0:
        status = Status.MENISCUS
        title = f"Diagnosis: {status.value} | Meniscus: {1-meniscus_prob:.2f}"
    else:
        status = Status.NORMAL
        title = f"Diagnosis: {status.value} | Normal probability (ACL model: {acl_prob:.2f}, Meniscus model: {meniscus_prob:.2f})"

    # Original image
    img = np.array(image)

    # Prepare heatmaps only if needed
    heatmap_acl = heatmap_meniscus = overlay_acl = overlay_men = None             
    overlays = []

    if status in [Status.ACL, Status.ACL_MENIISCUS]:
        heatmap_acl = generate_heatmap(image_tensor, acl_models[view_type])
        heatmap_acl = cv2.resize(heatmap_acl, (img.shape[1], img.shape[0]))
        heatmap_acl_colored = cv2.applyColorMap(np.uint8(255 * heatmap_acl), cv2.COLORMAP_JET)
        overlay_acl = np.uint8(heatmap_acl_colored * 0.4 + img * 0.6)
        overlays.append(("ACL Heatmap", overlay_acl))

    if status in [Status.MENISCUS, Status.ACL_MENIISCUS]:
        heatmap_meniscus = generate_heatmap(image_tensor, meniscus_models[view_type])
        heatmap_meniscus = cv2.resize(heatmap_meniscus, (img.shape[1], img.shape[0]))
        heatmap_men_colored = cv2.applyColorMap(np.uint8(255 * heatmap_meniscus), cv2.COLORMAP_JET)
        overlay_men = np.uint8(heatmap_men_colored * 0.4 + img * 0.6)
        overlays.append(("Meniscus Heatmap", overlay_men))

    # Plot dynamically based on overlays
    fig_cols = 1 + len(overlays)
    fig, axs = plt.subplots(1, fig_cols, figsize=(12, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold',wrap=True)

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

    # Save result image
    if save_path is not None:
        global serial_counter
        base_name = Path(image_path).stem
        filename = f"{serial_counter:03}_{base_name}_{status.value.replace(' ', '_')}.png"
        full_save_path = os.path.join(save_path, filename)
        plt.savefig(full_save_path)
        print(f"Saved result to: {full_save_path}")

        # Save CSV data
        csv_data.append({
            "Filename": filename,
            "Row": row,
            "Column": col,
            "Diagnosis": status.value,
            "ACL_Probability": round(1-acl_prob, 4),
            "Meniscus_Probability": round(1-meniscus_prob, 4)
        })

    plt.show()
    return status


# At the end of your full run, save the CSV
def save_diagnosis_csv(csv_path):
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    print(f"CSV report saved to: {csv_path}")


allowed_exts = {'.jpg', '.png'}

for image_file in image_folder_path.rglob("*"):
    if image_file.suffix.lower() in allowed_exts:
        #slice_image_by_edges(image_file)
        view_type = input("Enter view type (sagittal, coronal, axial): ").strip().lower()
        diagnosis = classify_knee_condition(
        image_file, 
        view_type, 
        save_path=output_folder, 
        row=None, 
        col=None
    )
        print(f"{image_file.name}: {diagnosis}")
        print(view_type)
        
save_diagnosis_csv(os.path.join(output_folder, "diagnosis_report.csv"))
