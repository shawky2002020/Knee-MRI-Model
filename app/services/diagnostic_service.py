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

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

class DiagnosticService:
    def __init__(self, output_base_dir: str = "output_results", user_id: str = "default_user",mri_scan: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_base_dir = output_base_dir
        self.serial_counter = 0
        self.csv_data = []
        self.user_id = user_id  # Add user-specific folder organization
        self.mri_scan = mri_scan
        
        # Use ModelLoader to load models
        self.model_loader = ModelLoader()
        
        
    def create_output_dir(self):
        """No longer needed for Cloudinary uploads, kept for compatibility."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_folder = f"{self.output_base_dir}_{timestamp}"
        return self.output_folder

    def process_image(self, image_path: str, view_type: str) -> Dict:
        """Process an image and return results with visualization"""
        # Use ModelLoader's predict method
        prediction = self.model_loader.predict(image_path, view_type)
        
        # Generate visualizations and upload to Cloudinary
        visualization_url = self._save_visualization(
            image_path, view_type, prediction["acl_prob"], prediction["meniscus_prob"], Status(prediction["status"])
        )
        
        return {
            "status": prediction["status"],
            "acl_prob": prediction["acl_prob"],
            "meniscus_prob": prediction["meniscus_prob"],
            "visualization": visualization_url,
        }

    def _save_visualization(self, image_path, view_type, acl_prob, meniscus_prob, status: Status):
        """Generate and upload the visualization plot and MRI scan to Cloudinary"""
        # Generate filenames
        filename = f"{self.serial_counter:03}_{Path(image_path).stem}_{status.value.replace(' ', '_')}.png"
        user_folder = f"diagnostic_results/{self.user_id}"
        heatmap_folder = f"{user_folder}/heatmaps"
        scans_folder = f"{user_folder}/scans"

        # Save and upload visualization
        plt.savefig(filename)
        plt.close()
        upload_result = cloudinary_upload(filename, folder=heatmap_folder)
        visualization_url = upload_result.get("secure_url")
        os.remove(filename)

        # Upload original MRI scan
        self.mri_scan = self.save_mri_scan(image_path)

        # Append data for CSV in a vertical format
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

    def save_report(self):
        """Generate and upload the CSV report to Cloudinary"""
        csv_path = "diagnosis_report.csv"
        df = pd.DataFrame(self.csv_data)
        df.to_csv(csv_path, index=False)
        
        # Upload to Cloudinary
        user_folder = f"diagnostic_results/{self.user_id}/reports"
        upload_result = cloudinary_upload(csv_path, folder=user_folder, resource_type="raw")
        csv_url = upload_result.get("secure_url")
        
        # Clean up local file
        os.remove(csv_path)
        return csv_url
    
    def save_mri_scan(self, image_path: str) -> str:
        """Upload an MRI scan to Cloudinary and return its URL."""
        user_folder = f"diagnostic_results/{self.user_id}/scans"
        upload_result = cloudinary_upload(image_path, folder=user_folder)
        return upload_result.get("secure_url")