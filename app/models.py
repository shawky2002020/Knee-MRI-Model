from enum import Enum
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from .modelBackbone import TransferLearningModel

class Status(Enum):
    ACL = "ACL"
    ACL_MENIISCUS = "ACL and meniscus"
    MENISCUS = "meniscus"
    NORMAL = "normal"

class ModelLoader:
    def __init__(self, model_dir="model_files", user_id: str = None):
        """Initialize the model loader, optionally with a user ID"""
        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.load_models()
        
    def load_models(self):
        # ACL Models
        self.acl_models = {
            "sagittal": self._load_model("acl_sagittal.pth"),
            "coronal": self._load_model("acl_coronal.pth"),
            "axial": self._load_model("acl_axial.pth")
        }
        
        # Meniscus Models
        self.meniscus_models = {
            "sagittal": self._load_model("meniscus_sagittal.pth"),
            "coronal": self._load_model("meniscus_coronal.pth"),
            "axial": self._load_model("meniscus_axial.pth")
        }
    
    def _load_model(self, model_name):
        model_path = self.model_dir / model_name
        model = TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def predict(self, image_path: str, view_type: str):
        """Perform prediction on the given image"""
        # Optionally log or handle user-specific logic using self.user_id
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        acl_output = self.acl_models[view_type](image_tensor)
        meniscus_output = self.meniscus_models[view_type](image_tensor)
        
        acl_prob = torch.sigmoid(acl_output).item()
        meniscus_prob = torch.sigmoid(meniscus_output).item()
        
        acl_result = 0 if acl_prob < 0.3 else 1
        meniscus_result = 0 if meniscus_prob < 0.3 else 1
        
        if acl_result == 0 and meniscus_result == 0:
            status = Status.ACL_MENIISCUS
        elif acl_result == 0:
            status = Status.ACL
        elif meniscus_result == 0:
            status = Status.MENISCUS
        else:
            status = Status.NORMAL
            
        return {
            "status": status.value,
            "acl_prob": 1 - acl_prob,
            "meniscus_prob": 1 - meniscus_prob
        }