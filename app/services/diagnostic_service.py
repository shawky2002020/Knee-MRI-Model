import os
from datetime import datetime
from pathlib import Path
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict
from cloudinary.uploader import upload as cloudinary_upload
from cloudinary.utils import cloudinary_url
import cloudinary
from dotenv import load_dotenv
from ..models import ModelLoader, Status  # Use ModelLoader and Status from models.py
from ..schemas import FinalResponse, Result
import cv2
import numpy as np
import torch.nn.functional as F
from .heatmap_utils import generate_heatmap, create_overlay
import io

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

class DiagnosticService:
    def __init__(self, output_base_dir: str = "output_results", user_id: str = "default_user", mri_scan: str = None , heat_map: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_base_dir = output_base_dir
        self.serial_counter = 0
        self.csv_data = []
        self.user_id = user_id  # Add user-specific folder organization
        self.mri_scan = mri_scan
        self.heat_map = heat_map
        
        # Use ModelLoader to load models
        self.model_loader = ModelLoader()
        
    def process_image(self, image_path: str, view_type: str) -> FinalResponse:
        """Process an image and return results with visualization"""
        # Use ModelLoader's predict method
        prediction = self.model_loader.predict(image_path, view_type)
        
        # Generate visualizations and upload to Cloudinary
        self.heat_map = self._save_visualization(
            image_path, view_type, prediction["acl_prob"], prediction["meniscus_prob"], Status(prediction["status"])
        )
        
        result_data = Result(
            status=prediction["status"],
            acl_prob=prediction["acl_prob"],
            meniscus_prob=prediction["meniscus_prob"]
        )

        # Save the report if user_id is provided
        report_url = self.save_report()

        # Construct and return the FinalResponse
        return FinalResponse(
            result=result_data,
            mri_scan=self.mri_scan,
            heat_map=self.heat_map,
            report=report_url
        )

    def _save_visualization(self, image_path, view_type, acl_prob, meniscus_prob, status: Status):
        """Generate and upload the visualization plot and MRI scan to Cloudinary."""
        # Generate filenames
        filename = f"{self.serial_counter:03}_{Path(image_path).stem}_{status.value.replace(' ', '_')}.png"
        user_folder = f"diagnostic_results/{self.user_id}"
        heatmap_folder = f"{user_folder}/heatmaps"

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.model_loader.transform(image).unsqueeze(0).to(self.device)
        img = np.array(image)

        # Prepare heatmaps and overlays
        overlays = []

        if status in [Status.ACL, Status.ACL_MENIISCUS]:
            heatmap_acl = generate_heatmap(image_tensor, self.model_loader.acl_models[view_type])
            overlay_acl = create_overlay(img, heatmap_acl)
            overlays.append(("ACL Heatmap", overlay_acl))

        if status in [Status.MENISCUS, Status.ACL_MENIISCUS]:
            heatmap_meniscus = generate_heatmap(image_tensor, self.model_loader.meniscus_models[view_type])
            overlay_men = create_overlay(img, heatmap_meniscus)
            overlays.append(("Meniscus Heatmap", overlay_men))

        # Plot dynamically based on overlays
        fig_cols = 1 + len(overlays)
        fig, axs = plt.subplots(1, fig_cols, figsize=(12, 6))
        fig.suptitle(f"Diagnosis: {status.value}", fontsize=14, fontweight='bold', wrap=True)

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

        # Save the visualization to an in-memory buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)

        # Upload visualization to Cloudinary
        upload_result = cloudinary_upload(buffer, folder=heatmap_folder, public_id=Path(filename).stem, resource_type="image")
        visualization_url = upload_result.get("secure_url")

        # Upload original MRI scan
        self.mri_scan = self.save_mri_scan(image_path)

        # Append data for CSV
        self.csv_data.append({
            "Attribute": "Filename",
            "Value": filename
        })
        self.csv_data.append({
            "Attribute": "Diagnosis",
            "Value": status.value
        })
        self.csv_data.append({
            "Attribute": "ACL_Probability",
            "Value": round(1 - acl_prob, 4)
        })
        self.csv_data.append({
            "Attribute": "Meniscus_Probability",
            "Value": round(1 - meniscus_prob, 4)
        })
        self.csv_data.append({
            "Attribute": "ViewType",
            "Value": view_type
        })
        self.csv_data.append({
            "Attribute": "Visualization_URL",
            "Value": visualization_url
        })
        self.csv_data.append({
            "Attribute": "Scan_URL",
            "Value": self.mri_scan
        })

        self.serial_counter += 1
        return visualization_url
    
    def save_mri_scan(self, image_path: str) -> str:
        """Upload an MRI scan to Cloudinary and return its URL."""
        user_folder = f"diagnostic_results/{self.user_id}/scans"
        upload_result = cloudinary_upload(image_path, folder=user_folder)
        return upload_result.get("secure_url")

    def save_report(self, filename: str = None) -> str:
        """Save diagnostic results to a CSV file and upload it to Cloudinary."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"diagnostic_report_{self.user_id}_{timestamp}.csv"
        
        # Create the CSV content in-memory
        buffer = io.StringIO()
        pd.DataFrame(self.csv_data).to_csv(buffer, index=False)
        buffer.seek(0)

        # Upload the CSV file to Cloudinary
        user_folder = f"diagnostic_results/{self.user_id}/reports"
        upload_result = cloudinary_upload(
            buffer, 
            folder=user_folder, 
            public_id=Path(filename).stem, 
            resource_type="raw", 
            format="csv"
        )
        report_url = upload_result.get("secure_url")

        return report_url

