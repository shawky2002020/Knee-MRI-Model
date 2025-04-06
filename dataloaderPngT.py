import random
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
data_base_dir = Path("D:\\Knee_MRI\\data_set\\sagittal\\acl")
# Correct paths to ensure classes are specific to each modality

train_dir = data_base_dir / "train" 
valid_dir = data_base_dir / "valid" 
test_dir = data_base_dir / "test" 

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=str(train_dir), transform=train_transforms)
valid_data = datasets.ImageFolder(root=str(valid_dir), transform=test_transforms)
test_data = datasets.ImageFolder(root=str(test_dir), transform=test_transforms)

# Setup DataLoaders
BATCH_SIZE = 32
NUM_WORKERS = 4  # Adjust as needed

train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)

valid_dataloader = DataLoader(dataset=valid_data, 
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=False)

test_dataloader = DataLoader(dataset=test_data, 
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=False)


# Class names and dictionary for each modality

class_names = train_data.classes
class_dict = train_data.class_to_idx

# Define helper function to plot transformed images
def plot_transformed_images(image_paths, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
        plt.show()

# Main block for testing/debugging
if __name__ == "__main__":
    # Print basic information for debugging
    print(f"Device: {device}")
    print("Class Names:", class_names)
    
    # Print class mappings for each modality
    print("Class to Index Mapping:", class_dict)
    
    print(f"Train Data Size: {len(train_data)}")
    print(f"Validation Data Size: {len(valid_data)}")
    print(f"Test Data Size: {len(test_data)}")

    # Visualize transformed images (plot only once)
    image_path_list = list(data_base_dir.glob("*/*/*.png"))
    plot_transformed_images(image_path_list, transform=train_transforms, n=3)

    # Fetch one batch of data for testing
    img, label = next(iter(train_dataloader))
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
