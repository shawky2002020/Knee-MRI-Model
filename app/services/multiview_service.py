import os
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import io
import base64
from enum import Enum

from ..models import ModelLoader, Status
from ..schemas import FinalResponse, Result
from .heatmap_utils import generate_heatmap, create_overlay
from cloudinary.uploader import upload as cloudinary_upload
from cloudinary.utils import cloudinary_url
import cloudinary
from dotenv import load_dotenv
load_dotenv()
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

class MultiViewDiagnosticService:
    def __init__(self, output_base_dir: str = "output_results", user_id: str = "default_user"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_base_dir = output_base_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_folder = f"{output_base_dir}_{self.timestamp}"
        os.makedirs(self.output_folder, exist_ok=True)
        self.serial_counter = 0
        self.csv_data = []
        self.user_id = user_id
        
        # Load models using ModelLoader
        self.model_loader = ModelLoader()
        
        # Logistic regression coefficients
        self.logreg_acl_weights = np.array([0.415, 0.257, 0.328])
        self.logreg_meniscus_weights = np.array([0.375, 0.298, 0.327])
        self.acl_intercept = -0.46
        self.meniscus_intercept = -0.62
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def aggregate_logistic(self, probabilities, weights, intercept):
        weighted_sum = np.dot(probabilities, weights) + intercept
        return self.sigmoid(weighted_sum)
    
    def process_multiview(self, image_paths: dict) -> FinalResponse:
        acl_probs = []
        meniscus_probs = []
        view_acl_probs = {}
        view_meniscus_probs = {}
        images = {}
        image_tensors = {}
        
        for view_type, paths in image_paths.items():
            if not isinstance(paths, list):
                paths = [paths]
            acl_probs_view = []
            meniscus_probs_view = []
            images_view = []
            image_tensors_view = []
            for path in paths:
                prediction = self.model_loader.predict(path, view_type)
                image = Image.open(path).convert("RGB")
                image_tensor = self.model_loader.transform(image).unsqueeze(0).to(self.device)
                images_view.append(np.array(image))
                image_tensors_view.append(image_tensor)
                acl_probs_view.append(1 - prediction["acl_prob"])
                meniscus_probs_view.append(1 - prediction["meniscus_prob"])
            # Aggregate per-view (mean)
            acl_prob = np.mean(acl_probs_view)
            meniscus_prob = np.mean(meniscus_probs_view)
            acl_probs.append(acl_prob)
            meniscus_probs.append(meniscus_prob)
            view_acl_probs[view_type] = 1 - acl_prob  # Prob of being ACL tear
            view_meniscus_probs[view_type] = 1 - meniscus_prob
            images[view_type] = images_view[0]  # Use the first image for visualization
            image_tensors[view_type] = image_tensors_view[0]
        
        # Combine using logistic regression
        final_acl_prob = self.aggregate_logistic(np.array(acl_probs), self.logreg_acl_weights, self.acl_intercept)
        final_meniscus_prob = self.aggregate_logistic(np.array(meniscus_probs), self.logreg_meniscus_weights, self.meniscus_intercept)
        
        # Apply decision thresholds
        acl_result = 1 if final_acl_prob < 0.51 else 0
        meniscus_result = 1 if final_meniscus_prob < 0.51 else 0

        if final_acl_prob < final_meniscus_prob - 0.09 and final_meniscus_prob > 0.48:
            meniscus_result = 0
        elif final_meniscus_prob < final_acl_prob - 0.1 and final_acl_prob >= 0.49:
            acl_result = 0
        
        # Determine final diagnosis
        if acl_result and meniscus_result:
            status = Status.ACL_MENIISCUS
        elif acl_result:
            status = Status.ACL
        elif meniscus_result:
            status = Status.MENISCUS
        else:
            status = Status.NORMAL
        
        # Select the view with the most confident abnormality
        selected_view = self._select_best_view(status, view_acl_probs, view_meniscus_probs, image_paths)
        
        # Generate heatmap for the selected view
        heatmap_url = self._generate_visualization(
            images[selected_view],
            image_tensors[selected_view],
            selected_view,
            1-final_acl_prob,
            1-final_meniscus_prob,
            status
        )
        
        # Save to CSV
        self._save_to_csv(selected_view, status, 1-final_acl_prob, 1-final_meniscus_prob)
        
        # Create response
        result = Result(
            status=status.value,
            acl_prob=1-final_acl_prob,
            meniscus_prob=1-final_meniscus_prob
        )
        
        # Generate report URL
        report_url = self._save_report()
        
        # Return final response
        return FinalResponse(
            result=result,
            mri_scan=selected_view,  # This is just the view type name
            heat_map=heatmap_url,    # This is now a Cloudinary URL
            report=report_url        # This is now a Cloudinary URL
        )
    
    def _select_best_view(self, status, view_acl_probs, view_meniscus_probs, image_paths):
        """Select the most informative view based on diagnosis"""
        if status == Status.ACL:
            return max(view_acl_probs, key=view_acl_probs.get)
        elif status == Status.MENISCUS:
            return max(view_meniscus_probs, key=view_meniscus_probs.get)
        elif status == Status.ACL_MENIISCUS:
            # Take the view with the highest of both conditions
            combined_scores = {
                v: max(view_acl_probs[v], view_meniscus_probs[v])
                for v in image_paths
            }
            return max(combined_scores, key=combined_scores.get)
        else:  # NORMAL
            return "sagittal"  # Default to sagittal for normal cases
    
    def _generate_visualization(self, img, image_tensor, view_type, acl_prob, meniscus_prob, status):
        """Generate visualization with heatmaps and upload to Cloudinary"""
        overlays = []
        
        if status in [Status.ACL, Status.ACL_MENIISCUS]:
            heatmap_acl = generate_heatmap(image_tensor, self.model_loader.acl_models[view_type])
            overlay_acl = create_overlay(img, heatmap_acl)
            overlays.append(("ACL Heatmap", overlay_acl))
        
        if status in [Status.MENISCUS, Status.ACL_MENIISCUS]:
            heatmap_meniscus = generate_heatmap(image_tensor, self.model_loader.meniscus_models[view_type])
            overlay_men = create_overlay(img, heatmap_meniscus)
            overlays.append(("Meniscus Heatmap", overlay_men))
        
        # Create visualization
        fig_cols = 1 + len(overlays)
        fig, axs = plt.subplots(1, fig_cols, figsize=(4 * fig_cols, 4))
        title = f"Diagnosis: {status.value} | ACL: {acl_prob:.2f}, Meniscus: {meniscus_prob:.2f}"
        fig.suptitle(f"{view_type.upper()} View\n{title}", fontsize=12, fontweight='bold', wrap=True)
        
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
        
        # Generate filename for Cloudinary
        filename = f"{self.serial_counter:03}_{view_type}_{status.value.replace(' ', '_')}"
        
        # Save to in-memory buffer and upload to Cloudinary
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        # Upload to Cloudinary
        upload_result = cloudinary_upload(
            buffer, 
            folder=f"diagnostic_results/{self.user_id}/heatmaps", 
            public_id=filename, 
            resource_type="image"
        )
        
        # Return the secure URL
        return upload_result.get("secure_url")
    
    def _save_report(self):
        """Save diagnosis report as CSV and upload to Cloudinary"""
        # Create CSV in memory
        buffer = io.StringIO()
        pd.DataFrame(self.csv_data).to_csv(buffer, index=False)
        buffer.seek(0)
        
        # Upload to Cloudinary
        upload_result = cloudinary_upload(
            buffer, 
            folder=f"diagnostic_results/{self.user_id}/reports", 
            public_id="diagnosis_report", 
            resource_type="raw", 
            format="csv"
        )
        
        # Return the secure URL
        return upload_result.get("secure_url")
    
    def _save_to_csv(self, view_type, status, acl_prob, meniscus_prob):
        """Save diagnosis data to CSV"""
        self.csv_data.append({
            "Filename": f"{self.serial_counter:03}_{view_type}_{status.value.replace(' ', '_')}.png",
            "View": view_type,
            "Diagnosis": status.value,
            "ACL_Probability": round(acl_prob, 4),
            "Meniscus_Probability": round(meniscus_prob, 4)
        })
        self.serial_counter += 1