import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"

class TransferLearningModel(nn.Module):
    def __init__(self, model_type: str, output_shape: int, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.model_type = model_type.lower()
        self.gradients = None
        self.activations = None

        # Select model backbone and modify for grayscale input
        if self.model_type == "alexnet":
            self.model = models.alexnet(weights='DEFAULT')
            self.model.features[0] = nn.Conv2d(3, self.model.features[0].out_channels, kernel_size=11, stride=4, padding=2)
            in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features, output_shape)
            self.target_layer = self.model.features[10]  # Last conv layer in AlexNet

        elif self.model_type in ["resnet18", "resnet50"]:
            self.model = models.__dict__[self.model_type](weights='DEFAULT')
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, output_shape)
            self.target_layer = self.model.layer4  # Final conv block in ResNet

        elif self.model_type == "effnetb0":
            self.model = models.efficientnet_b0(weights='DEFAULT')
            self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, output_shape)

        elif self.model_type == "effnetb5":
            self.model = models.efficientnet_b5(weights='DEFAULT')
            self.model.features[0][0] = nn.Conv2d(3, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, output_shape)

        elif self.model_type == "effnetv2_s":
            self.model = models.efficientnet_v2_s(weights='DEFAULT')
            self.model.conv_stem = nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            in_features = self.model.classifier[1].in_features  # Access the Linear layer in the Sequential classifier
            self.model.classifier = nn.Linear(in_features, output_shape)

        else:
            raise ValueError("Model type must be one of 'alexnet', 'resnet18', 'resnet50', 'effnetb0', 'effnetb5', 'effnetv2_s'.")

        # Additional dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze specific layers based on the model type
        if self.model_type == "alexnet":
            for param in self.model.classifier.parameters():
                param.requires_grad = True

        elif self.model_type in ["resnet18", "resnet50"]:
            for param in self.model.layer4.parameters():  # Unfreeze last layer of ResNet
                param.requires_grad = True

        elif self.model_type in ["effnetb0", "effnetb5"]:
            for param in self.model.classifier.parameters():  # Unfreeze classifier layer for EfficientNet
                param.requires_grad = True

        elif self.model_type in "effnetv2_s":
            for param in self.model.classifier.parameters():  # Unfreeze classifier layer for EfficientNetV2
                param.requires_grad = True

        # Register hook for gradients and activations
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = self.dropout(x)  # Apply dropout here
        return x

# Instantiate the model with chosen backbone and output shape
modelBackbone = TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device)

if __name__ == "__main__":

    # Model summary
    summary(modelBackbone, input_size=[1, 3, 256, 256])  # EfficientNetV2 models expect 256x256 input images

    # At the start of your training loop
    torch.cuda.empty_cache()

torch.cuda.empty_cache()  # Clears unused GPU memory